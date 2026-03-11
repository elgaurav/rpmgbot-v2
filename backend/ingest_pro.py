"""
Optimized Document Ingestion Pipeline
UPGRADED: bge-large-en-v1.5 embeddings, GPU-aware LlamaCPP, 1024-token chunks
- Must re-run after any model upgrade (vectors are incompatible across dimensions)
"""

import os
import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling_core.types.doc import PictureItem, TableItem, DoclingDocument
import config
import json

if not hasattr(config, 'PDF_OUTPUT_DIR'):
    config.PDF_OUTPUT_DIR = config.STATIC_DIR / "pdfs"
    print("⚠️  PDF_OUTPUT_DIR not in config, using default: static/pdfs")

def setup_directories():
    if config.STORAGE_DIR.exists():
        try:
            shutil.rmtree(config.STORAGE_DIR)
            print("🗑️  Removed old storage directory")
        except Exception as e:
            print(f"⚠️  Could not delete storage dir: {e}")

    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    config.IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output directories ready")

def setup_models():
    print("\n⚙️  Initializing AI Models...")

    # UPGRADED: bge-large-en-v1.5 with GPU support via device param.
    # embed_batch_size is configurable — lower it in config if you hit GPU OOM.
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=config.EMBED_MODEL_NAME,
        cache_folder=str(config.BASE_DIR / ".cache"),
        device="cuda" if config.EMBED_USE_GPU else "cpu",
        embed_batch_size=config.EMBED_BATCH_SIZE,
    )

    # UPGRADED: LlamaCPP with GPU layer offloading via n_gpu_layers.
    # Set LLM_GPU_LAYERS=0 in config.py for pure CPU inference.
    Settings.llm = LlamaCPP(
        model_path=config.LLM_MODEL_PATH,
        temperature=config.LLM_TEMPERATURE,
        max_new_tokens=config.LLM_MAX_TOKENS,
        context_window=config.LLM_CONTEXT_WINDOW,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": config.LLM_GPU_LAYERS},
        verbose=False,
    )

    # UPGRADED: chunk_size=1024, chunk_overlap=150 (was 512/50)
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP

    print(f"✅ Embedding model: {config.EMBED_MODEL_NAME} ({'GPU' if config.EMBED_USE_GPU else 'CPU'})")
    print(f"✅ Chunk size: {config.CHUNK_SIZE} (overlap: {config.CHUNK_OVERLAP})")
    print(f"✅ LLM GPU layers: {config.LLM_GPU_LAYERS}")

def setup_docling():
    """Configure Docling for PDF processing — unchanged from original."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = config.DOCLING_OCR
    pipeline_options.do_table_structure = config.DOCLING_TABLE_EXTRACTION
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = config.DOCLING_GENERATE_PICTURES
    pipeline_options.images_scale = config.DOCLING_IMAGE_SCALE

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    return converter

def extract_images_with_metadata(conv_result, doc_filename: str) -> dict:
    image_metadata = {}
    page_image_counts = {}

    for element, _level in conv_result.document.iterate_items():
        if isinstance(element, PictureItem):
            page_num = None
            if hasattr(element, 'prov') and element.prov:
                for prov_item in element.prov:
                    if hasattr(prov_item, 'page_no'):
                        page_num = prov_item.page_no
                        break

            if page_num is None:
                continue

            if page_num not in page_image_counts:
                page_image_counts[page_num] = 0
            page_image_counts[page_num] += 1
            fig_num = page_image_counts[page_num]

            image_name = f"{doc_filename}_page{page_num}_fig{fig_num}.png"
            image_path = config.IMAGE_OUTPUT_DIR / image_name

            img_data = element.get_image(conv_result.document)
            if img_data:
                with open(image_path, "wb") as f:
                    img_data.save(f, "PNG")

                image_metadata[image_name] = {
                    "page": page_num,
                    "figure_num": fig_num,
                    "document": doc_filename,
                    "path": str(image_path)
                }

    return image_metadata

def build_page_to_text_mapping(docling_doc: DoclingDocument) -> dict:
    page_text_map = {}

    for item, level in docling_doc.iterate_items():
        if hasattr(item, 'prov') and item.prov:
            for prov in item.prov:
                if hasattr(prov, 'page_no'):
                    page_num = prov.page_no

                    if isinstance(item, PictureItem):
                        continue

                    if hasattr(item, 'text') and item.text:
                        text = item.text
                    elif hasattr(item, 'export_to_markdown'):
                        try:
                            text = item.export_to_markdown(docling_doc)
                        except:
                            continue
                    else:
                        continue

                    if text and text.strip():
                        if page_num not in page_text_map:
                            page_text_map[page_num] = []
                        page_text_map[page_num].append(text)

    return {page: "\n".join(texts) for page, texts in page_text_map.items()}

def process_single_pdf(pdf_path: Path, converter: DocumentConverter) -> tuple:
    print(f"   📄 Processing: {pdf_path.name}")
    start = time.time()

    pdf_static_path = config.PDF_OUTPUT_DIR / pdf_path.name
    try:
        shutil.copy2(pdf_path, pdf_static_path)
        print(f"      📋 PDF copied to static directory")
    except Exception as e:
        print(f"      ⚠️  Could not copy PDF: {e}")

    conv_result = converter.convert(pdf_path)
    doc_filename = pdf_path.stem

    image_metadata = extract_images_with_metadata(conv_result, doc_filename)
    print(f"      ✅ {len(image_metadata)} images extracted with page metadata")

    page_text_map = build_page_to_text_mapping(conv_result.document)
    print(f"      ✅ {len(page_text_map)} pages mapped")

    md_text = conv_result.document.export_to_markdown()

    documents = []

    for page_num, page_text in page_text_map.items():
        if page_text.strip():
            doc = Document(
                text=page_text,
                metadata={
                    "file_name": pdf_path.name,
                    "source_path": str(pdf_path),
                    "page_label": str(page_num),
                    "page_number": page_num,
                    "document_stem": doc_filename
                }
            )
            documents.append(doc)

    if not documents:
        documents = [Document(
            text=md_text,
            metadata={
                "file_name": pdf_path.name,
                "source_path": str(pdf_path),
                "document_stem": doc_filename
            }
        )]

    elapsed = time.time() - start
    print(f"      ⏱️  Completed in {elapsed:.1f}s")

    return documents, image_metadata, page_text_map

def run_ingestion():
    print("\n" + "="*60)
    print("🚀 RPMG DOCUMENT INGESTION PIPELINE v4.0")
    print("   UPGRADED: bge-large + llama3.1:8b + 1024-token chunks")
    print("="*60)
    print(f"📂 Source: {config.DATA_DIR}")
    print(f"💾 Vector DB: Qdrant at {config.QDRANT_PATH}")
    print(f"🖼️  Images: {config.IMAGE_OUTPUT_DIR}")
    print(f"📋 PDFs: {config.PDF_OUTPUT_DIR}")
    print(f"🧠 Embedding: {config.EMBED_MODEL_NAME} ({'GPU' if config.EMBED_USE_GPU else 'CPU'})")
    print(f"📐 Vector dim: 1024 (was 384)")
    print(f"📝 Chunk size: {config.CHUNK_SIZE} tokens (was 512)")
    print("="*60 + "\n")

    start_time = time.time()

    setup_directories()
    setup_models()
    converter = setup_docling()

    pdf_files = list(config.DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDFs found in {config.DATA_DIR}")
        return

    print(f"\n📚 Found {len(pdf_files)} PDF(s)")

    all_documents = []
    all_image_metadata = {}
    all_page_maps = {}

    for pdf_path in pdf_files:
        docs, img_meta, page_map = process_single_pdf(pdf_path, converter)
        all_documents.extend(docs)
        all_image_metadata.update(img_meta)
        all_page_maps[pdf_path.stem] = page_map

    metadata_file = config.STORAGE_DIR / "image_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(all_image_metadata, f, indent=2)
    print(f"\n💾 Saved image metadata to {metadata_file}")

    print(f"\n🧠 Creating Qdrant vector database...")

    client = QdrantClient(path=str(config.QDRANT_PATH))
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="piping_docs"
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"   Embedding {len(all_documents)} document(s)...")
    embedding_start = time.time()

    index = VectorStoreIndex.from_documents(
        all_documents,
        storage_context=storage_context,
        show_progress=True
    )

    embedding_time = time.time() - embedding_start
    total_time = time.time() - start_time

    print("\n" + "="*60)
    print("✅ INGESTION COMPLETE!")
    print("="*60)
    print(f"📊 Statistics:")
    print(f"   PDFs processed: {len(pdf_files)}")
    print(f"   Documents created: {len(all_documents)}")
    print(f"   Images extracted: {len(all_image_metadata)}")
    print(f"   Vector dimensions: 1024 (bge-large)")
    print(f"   Chunk size: {config.CHUNK_SIZE} tokens")
    if all_image_metadata:
        example_img = list(all_image_metadata.keys())[0]
        print(f"   Image naming: {example_img}")
    print(f"   Embedding time: {embedding_time:.1f}s")
    print(f"   Total time: {total_time:.1f}s")
    print(f"\n⚡ Next steps:")
    print(f"   1. Restart your FastAPI server: python main.py")
    print(f"   2. The new index uses 1024-dim vectors — incompatible with old bge-small index")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_ingestion()
