#!/usr/bin/env python3
"""
Performance Diagnostic Tool for RPMG RAG Assistant
UPGRADED: Tests reflect bge-large-en-v1.5 embeddings, llama3.1:8b LLM, 8k context
"""

import time
import sys
from pathlib import Path


def test_llm_speed():
    """Test LlamaCPP model inference speed (replaces Ollama test)"""
    print("\n" + "="*60)
    print("TEST 1: LlamaCPP LLM Performance (llama3.1:8b)")
    print("="*60)

    try:
        from llama_index.llms.llama_cpp import LlamaCPP
        import config

        # UPGRADED: n_gpu_layers wired from config
        llm = LlamaCPP(
            model_path=config.LLM_MODEL_PATH,
            temperature=config.LLM_TEMPERATURE,
            max_new_tokens=50,
            context_window=config.LLM_CONTEXT_WINDOW,
            generate_kwargs={},
            model_kwargs={"n_gpu_layers": config.LLM_GPU_LAYERS},
            verbose=False,
        )

        test_prompt = "Explain corrosion in exactly 50 words."

        print(f"Model: {config.LLM_MODEL}")
        print(f"Model path: {config.LLM_MODEL_PATH}")
        print(f"GPU layers: {config.LLM_GPU_LAYERS} ({'GPU' if config.LLM_GPU_LAYERS > 0 else 'CPU'})")
        print(f"Context window: {config.LLM_CONTEXT_WINDOW} tokens")
        print(f"Prompt: '{test_prompt}'")
        print("Generating response...")

        start = time.time()
        response = llm.complete(test_prompt)
        elapsed = time.time() - start

        print(f"\n✅ Response generated in {elapsed:.2f}s")
        print(f"Response length: {len(str(response))} chars")
        print(f"Response preview: {str(response)[:200]}...")

        # UPGRADED: 8B model is heavier — adjust expectations
        if elapsed < 15:
            print(f"🚀 EXCELLENT - Fast GPU inference")
        elif elapsed < 40:
            print(f"✅ GOOD - Acceptable for 8B model")
        elif elapsed < 90:
            print(f"⚠️  SLOW - Check GPU layers (LLM_GPU_LAYERS in config)")
        else:
            print(f"❌ VERY SLOW - Likely running on CPU. Set LLM_GPU_LAYERS=999")

        return elapsed

    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Check LLM_MODEL_PATH in config.py points to your .gguf file")
        return None


def test_embedding_speed():
    """Test embedding generation speed with bge-large-en-v1.5"""
    print("\n" + "="*60)
    print("TEST 2: Embedding Performance (bge-large-en-v1.5)")
    print("="*60)

    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        import config

        # UPGRADED: GPU device and batch size from config
        embed_model = HuggingFaceEmbedding(
            model_name=config.EMBED_MODEL_NAME,
            cache_folder=str(config.BASE_DIR / ".cache"),
            device="cuda" if config.EMBED_USE_GPU else "cpu",
            embed_batch_size=config.EMBED_BATCH_SIZE,
        )

        test_text = "This is a test document for embedding performance." * 10

        print(f"Model: {config.EMBED_MODEL_NAME}")
        print(f"Device: {'GPU (cuda)' if config.EMBED_USE_GPU else 'CPU'}")
        print(f"Batch size: {config.EMBED_BATCH_SIZE}")
        print(f"Text length: {len(test_text)} chars")
        print("Generating embeddings...")

        # Warmup
        _ = embed_model.get_text_embedding(test_text)

        # Actual test
        start = time.time()
        for _ in range(10):
            embedding = embed_model.get_text_embedding(test_text)
        elapsed = time.time() - start
        avg_time = elapsed / 10

        print(f"\n✅ Average embedding time: {avg_time*1000:.1f}ms")
        # UPGRADED: confirm 1024-dim output for bge-large
        print(f"Embedding dimension: {len(embedding)} (expected 1024 for bge-large)")

        if len(embedding) != 1024:
            print(f"⚠️  WARNING: Expected 1024-dim but got {len(embedding)}-dim.")
            print(f"   Check EMBED_MODEL_NAME in config.py")

        if avg_time < 0.1:
            print("🚀 EXCELLENT - Very fast embeddings")
        elif avg_time < 0.5:
            print("✅ GOOD - Acceptable speed")
        else:
            print("⚠️  SLOW - Consider enabling GPU: set EMBED_USE_GPU=True in config")

        return avg_time

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_vector_store_speed():
    """Test vector store retrieval — UPGRADED to check for 1024-dim vectors"""
    print("\n" + "="*60)
    print("TEST 3: Vector Store Retrieval (1024-dim)")
    print("="*60)

    try:
        from qdrant_client import QdrantClient
        import config

        if not config.QDRANT_PATH.exists():
            print("⚠️  No index found. Run ingest_pro.py first.")
            return None

        client = QdrantClient(path=str(config.QDRANT_PATH))

        try:
            collection_info = client.get_collection("piping_docs")
            print(f"Collection: piping_docs")
            print(f"Vectors: {collection_info.points_count}")

            vector_size = collection_info.config.params.vectors.size
            print(f"Vector dimension: {vector_size}")

            # UPGRADED: warn if old 384-dim index is still present
            if vector_size == 384:
                print("⚠️  WARNING: Index uses 384-dim vectors (old bge-small).")
                print("   You must re-run ingest_pro.py to rebuild with 1024-dim bge-large vectors.")
                return None
            elif vector_size == 1024:
                print("✅ Index uses 1024-dim vectors (bge-large) — correct!")

            test_query = [0.1] * vector_size

            print("Running 10 search queries...")
            start = time.time()
            for _ in range(10):
                results = client.search(
                    collection_name="piping_docs",
                    query_vector=test_query,
                    limit=config.SIMILARITY_TOP_K
                )
            elapsed = time.time() - start
            avg_time = elapsed / 10

            print(f"\n✅ Average search time: {avg_time*1000:.1f}ms")
            # UPGRADED: Top-K is now 4
            print(f"Top-K: {config.SIMILARITY_TOP_K} (upgraded from 2)")

            if avg_time < 0.05:
                print("🚀 EXCELLENT - Very fast retrieval")
            elif avg_time < 0.2:
                print("✅ GOOD - Acceptable speed")
            else:
                print("⚠️  SLOW - Large index or slow disk")

            return avg_time

        except Exception as e:
            print(f"❌ Collection not found: {e}")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_image_metadata():
    """Test image metadata and page linking — unchanged logic"""
    print("\n" + "="*60)
    print("TEST 4: Image-Page Linking")
    print("="*60)

    try:
        import config
        import json

        metadata_file = config.STORAGE_DIR / "image_metadata.json"

        if not metadata_file.exists():
            print("⚠️  No image metadata found. Run ingest_pro.py first.")
            return None

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        print(f"✅ Loaded metadata for {len(metadata)} images")

        docs = {}
        for img_name, img_meta in metadata.items():
            doc = img_meta['document']
            if doc not in docs:
                docs[doc] = {'pages': set(), 'images': 0}
            docs[doc]['pages'].add(img_meta['page'])
            docs[doc]['images'] += 1

        print(f"\n📊 Documents with images:")
        for doc, info in docs.items():
            print(f"   {doc}:")
            print(f"      - {info['images']} images")
            print(f"      - Across {len(info['pages'])} pages")
            print(f"      - Pages: {sorted(list(info['pages']))[:5]}...")

        from engine import _get_images_for_pages

        test_doc = list(docs.keys())[0]
        test_pages = set(list(docs[test_doc]['pages'])[:2])

        print(f"\n🧪 Testing image filter:")
        print(f"   Document: {test_doc}")
        print(f"   Pages: {test_pages}")

        filtered_images = _get_images_for_pages(test_doc, test_pages)
        print(f"   ✅ Found {len(filtered_images)} images for these pages")

        return len(metadata)

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_pdf_links():
    """Test PDF source links — unchanged"""
    print("\n" + "="*60)
    print("TEST 5: PDF Source Links")
    print("="*60)

    try:
        import config

        if not config.PDF_OUTPUT_DIR.exists():
            print("⚠️  PDF output directory not found. Run ingest_pro.py first.")
            return None

        pdfs = list(config.PDF_OUTPUT_DIR.glob("*.pdf"))

        print(f"✅ Found {len(pdfs)} source PDFs")

        for pdf in pdfs:
            size_mb = pdf.stat().st_size / (1024 * 1024)
            print(f"   📄 {pdf.name} ({size_mb:.2f} MB)")
            example_link = f"{config.BASE_URL}/static/pdfs/{pdf.name}#page=5"
            print(f"      Link example: {example_link}")

        return len(pdfs)

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_full_pipeline():
    """Test full RAG pipeline — UPGRADED expectations for 8B model"""
    print("\n" + "="*60)
    print("TEST 6: Full RAG Pipeline")
    print("="*60)

    try:
        from engine import query_piping_data

        test_question = "What is corrosion and how does it affect piping systems?"

        print(f"Question: '{test_question}'")
        print("Running full RAG query...")

        start = time.time()
        result = query_piping_data(test_question)
        elapsed = time.time() - start

        print(f"\n✅ Query completed in {elapsed:.2f}s")
        print(f"Answer length: {len(result['answer'])} chars")
        print(f"Sources found: {len(result['sources'])}")
        print(f"Images found: {len(result['images'])}")

        print(f"\n📄 Sources with links:")
        for i, source in enumerate(result['sources'][:4], 1):  # UPGRADED: show up to 4
            print(f"   {i}. {source['file']} (page {source['page']})")
            if source.get('pdf_link'):
                print(f"      Link: {source['pdf_link']}")
            if source.get('score'):
                print(f"      Relevance: {source['score']}")

        if result['images']:
            print(f"\n🖼️  Images (filtered by page context):")
            for img in result['images'][:3]:
                print(f"   - {img}")

        if 'timing' in result:
            timing = result['timing']
            print(f"\n⏱️  Timing breakdown:")
            print(f"   - Retrieval: {timing.get('retrieval_seconds', 0):.2f}s")
            print(f"   - LLM: {timing.get('llm_seconds', 0):.2f}s")
            print(f"   - Total: {timing.get('total_seconds', 0):.2f}s")

        # UPGRADED: 8B model is slower — adjust thresholds
        if elapsed < 20:
            print(f"\n🚀 EXCELLENT - Great GPU inference speed!")
        elif elapsed < 60:
            print(f"\n✅ GOOD - Acceptable for 8B model on GPU")
        elif elapsed < 120:
            print(f"\n⚠️  SLOW - Check GPU setup. Try setting LLM_GPU_LAYERS=999")
        else:
            print(f"\n❌ VERY SLOW - Likely running on CPU.")
            print("   Set LLM_GPU_LAYERS=999 and EMBED_USE_GPU=True in config.py")

        return elapsed

    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure you've run ingest_pro.py first")
        return None


def main():
    print("\n" + "="*60)
    print("🔍 RPMG RAG ASSISTANT - PERFORMANCE DIAGNOSTICS v4.0")
    print("   UPGRADED: bge-large + llama3.1:8b + 1024-token chunks")
    print("="*60)

    try:
        import config
        print(f"\n📊 Current Configuration:")
        print(f"   LLM Model: {config.LLM_MODEL}")
        print(f"   LLM GPU Layers: {config.LLM_GPU_LAYERS} ({'GPU' if config.LLM_GPU_LAYERS > 0 else 'CPU'})")
        print(f"   Context Window: {config.LLM_CONTEXT_WINDOW} tokens")
        print(f"   Embedding Model: {config.EMBED_MODEL_NAME}")
        print(f"   Embedding Device: {'GPU' if config.EMBED_USE_GPU else 'CPU'}")
        print(f"   Embedding Batch Size: {config.EMBED_BATCH_SIZE}")
        print(f"   Top-K Retrieval: {config.SIMILARITY_TOP_K} (was 2)")
        print(f"   Max Tokens: {config.LLM_MAX_TOKENS}")
        print(f"   Chunk Size: {config.CHUNK_SIZE} tokens (was 512)")
        print(f"   Chunk Overlap: {config.CHUNK_OVERLAP} tokens (was 50)")
        print(f"\n🖼️  Image Configuration:")
        print(f"   Strict page matching: {config.IMAGE_PAGE_MATCH_STRICT}")
        print(f"   Max images per query: {config.MAX_IMAGES_PER_QUERY}")
    except Exception as e:
        print(f"❌ Could not load config: {e}")
        return

    results = {}

    results['llm'] = test_llm_speed()
    results['embedding'] = test_embedding_speed()
    results['vector_store'] = test_vector_store_speed()
    results['image_metadata'] = test_image_metadata()
    results['pdf_links'] = test_pdf_links()
    results['pipeline'] = test_full_pipeline()

    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)

    if results['llm']:
        print(f"LlamaCPP LLM (8B): {results['llm']:.2f}s")
    if results['embedding']:
        print(f"Embeddings (1024-dim): {results['embedding']*1000:.1f}ms avg")
    if results['vector_store']:
        print(f"Vector Search (Top-{config.SIMILARITY_TOP_K}): {results['vector_store']*1000:.1f}ms avg")
    if results['image_metadata']:
        print(f"Image Metadata: {results['image_metadata']} images indexed")
    if results['pdf_links']:
        print(f"Source PDFs: {results['pdf_links']} available")
    if results['pipeline']:
        print(f"Full Pipeline: {results['pipeline']:.2f}s")

        if results['pipeline'] < 60:
            print("\n🎉 System is performing well with upgraded models!")
        else:
            print("\n⚠️  System is slow. Check GPU configuration:")
            print("   - LLM_GPU_LAYERS should be 999 for full GPU inference")
            print("   - EMBED_USE_GPU should be True")
            print("   - Verify CUDA is installed: python -c \"import torch; print(torch.cuda.is_available())\"")

    print("\n✨ UPGRADE STATUS:")
    embed_ok = results.get('embedding') is not None
    print(f"   {'✅' if embed_ok else '❌'} bge-large-en-v1.5 (1024-dim): {'WORKING' if embed_ok else 'FAILED'}")
    llm_ok = results.get('llm') is not None
    print(f"   {'✅' if llm_ok else '❌'} llama3.1:8b LlamaCPP: {'WORKING' if llm_ok else 'FAILED'}")
    vs_ok = results.get('vector_store') is not None
    print(f"   {'✅' if vs_ok else '❌'} Qdrant 1024-dim index: {'WORKING' if vs_ok else 'NEEDS REINGESTION'}")

    print("\n💡 CPU-ONLY MODE: Set these in config.py:")
    print("   EMBED_USE_GPU  = False")
    print("   LLM_GPU_LAYERS = 0")
    print("   USE_GPU        = False")
    print("   EMBED_BATCH_SIZE = 8")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()