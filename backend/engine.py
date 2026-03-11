"""
Optimized RAG Engine for RPMG Assistant
UPGRADED: Higher-capacity models, larger context, GPU-aware LlamaCPP init
- bge-large-en-v1.5 embeddings (1024-dim)
- llama3.1:8b 4-bit quantized LLM
- Top-K 4, context window 8192
"""

import time
import json
import re
from pathlib import Path
from typing import Dict, List, Generator, Optional, Set
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import config

# ==================== GLOBAL INITIALIZATION ====================
print("🚀 Initializing RAG Engine...")

# UPGRADED: bge-large-en-v1.5 produces 1024-dim embeddings.
# device="cuda" is set when EMBED_USE_GPU=True — falls back gracefully if no GPU.
Settings.embed_model = HuggingFaceEmbedding(
    model_name=config.EMBED_MODEL_NAME,
    cache_folder=str(config.BASE_DIR / ".cache"),
    # UPGRADED: GPU acceleration for embeddings.
    # If EMBED_USE_GPU is False in config, this becomes "cpu".
    device="cuda" if config.EMBED_USE_GPU else "cpu",
    # UPGRADED: match batch size from config (was hardcoded 32, now configurable)
    embed_batch_size=config.EMBED_BATCH_SIZE,
)

# UPGRADED: LlamaCPP with n_gpu_layers controlling GPU offload.
# n_gpu_layers=999 puts all layers on GPU.
# n_gpu_layers=0 runs fully on CPU.
# This is controlled by LLM_GPU_LAYERS in config.py.
Settings.llm = LlamaCPP(
    model_path=config.LLM_MODEL_PATH,
    temperature=config.LLM_TEMPERATURE,
    max_new_tokens=config.LLM_MAX_TOKENS,
    # UPGRADED: 8192 context window (was 2048)
    context_window=config.LLM_CONTEXT_WINDOW,
    generate_kwargs={},
    # UPGRADED: GPU layer offloading — set to 0 in config for CPU-only
    model_kwargs={"n_gpu_layers": config.LLM_GPU_LAYERS},
    verbose=False,
)

Settings.chunk_size = config.CHUNK_SIZE
Settings.chunk_overlap = config.CHUNK_OVERLAP

print(f"✅ Using embedding model: {config.EMBED_MODEL_NAME}")
print(f"✅ Embedding device: {'GPU' if config.EMBED_USE_GPU else 'CPU'}")
print(f"✅ Using LLM: {config.LLM_MODEL}")
print(f"✅ LLM GPU layers: {config.LLM_GPU_LAYERS} ({'GPU' if config.LLM_GPU_LAYERS > 0 else 'CPU'})")
print(f"✅ Context window: {config.LLM_CONTEXT_WINDOW} tokens")

# ==================== QUERY CLASSIFICATION ====================

def is_casual_query(question: str) -> bool:
    """
    Determine if a query is casual (greeting/thanks) vs technical.
    Returns True if casual (no RAG needed), False if technical.
    """
    if not config.ENABLE_QUERY_CLASSIFICATION:
        return False

    question_lower = question.lower().strip()

    for pattern in config.CASUAL_QUERY_PATTERNS:
        if pattern in question_lower:
            if (question_lower == pattern or
                question_lower.startswith(pattern + " ") or
                question_lower.endswith(" " + pattern) or
                question_lower.startswith(pattern + "!") or
                question_lower.startswith(pattern + "?")):
                return True

    technical_keywords = [
        "pipe", "piping", "stress", "plot", "plan", "API", "ASME",
        "pressure", "temperature", "flange", "valve", "material",
        "corrosion", "thickness", "design", "analysis", "standard",
        "code", "specification", "weld", "joint", "support"
    ]

    if len(question_lower) < 15:
        has_technical = any(kw.lower() in question_lower for kw in technical_keywords)
        if not has_technical:
            return True

    return False

# ==================== SINGLETON PATTERN FOR QDRANT ====================

class QdrantManager:
    """Singleton manager for Qdrant client"""
    _instance: Optional['QdrantManager'] = None
    _client: Optional[QdrantClient] = None
    _index: Optional[VectorStoreIndex] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_client(self) -> QdrantClient:
        if self._client is None:
            print("📦 Creating Qdrant client...")
            self._client = QdrantClient(path=str(config.QDRANT_PATH))
            print("✅ Qdrant client ready")
        return self._client

    def get_index(self) -> VectorStoreIndex:
        if self._index is None:
            print("📦 Loading Qdrant index...")
            client = self.get_client()

            collections = client.get_collections().collections
            collection_exists = any(c.name == "piping_docs" for c in collections)

            if not collection_exists:
                raise FileNotFoundError(
                    "No index found! Run ingest_pro.py first."
                )

            vector_store = QdrantVectorStore(
                client=client,
                collection_name="piping_docs"
            )
            self._index = VectorStoreIndex.from_vector_store(vector_store)
            print("✅ Index loaded from Qdrant")

        return self._index

    def clear(self):
        if self._client is not None:
            try:
                self._client.close()
            except:
                pass
        self._client = None
        self._index = None
        print("🔄 Qdrant cache cleared")

_qdrant_manager = QdrantManager()

# ==================== IMAGE METADATA ====================

_IMAGE_METADATA_CACHE = None

def _load_image_metadata() -> dict:
    global _IMAGE_METADATA_CACHE

    if _IMAGE_METADATA_CACHE is None:
        metadata_file = config.STORAGE_DIR / "image_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                _IMAGE_METADATA_CACHE = json.load(f)
            print(f"📸 Loaded metadata for {len(_IMAGE_METADATA_CACHE)} images")
        else:
            _IMAGE_METADATA_CACHE = {}

    return _IMAGE_METADATA_CACHE

def _is_valid_page(page_label) -> bool:
    if not page_label:
        return False
    page_str = str(page_label).strip().upper()
    invalid_patterns = ['N/A', 'N.A.', 'NONE', 'NULL', '']
    if page_str in invalid_patterns:
        return False
    try:
        int(page_str)
        return True
    except ValueError:
        return len(page_str) > 0 and page_str not in invalid_patterns

def _get_images_for_pages(document_stem: str, page_numbers: Set[int]) -> List[str]:
    metadata = _load_image_metadata()
    matched_images = []

    expanded_pages = set()
    for page in page_numbers:
        expanded_pages.add(page)
        for offset in range(-config.IMAGE_ADJACENT_PAGES, config.IMAGE_ADJACENT_PAGES + 1):
            if offset != 0:
                expanded_pages.add(page + offset)

    for img_name, img_meta in metadata.items():
        if img_meta['document'] == document_stem and img_meta['page'] in expanded_pages:
            matched_images.append(img_name)

    if len(matched_images) > config.MAX_IMAGES_PER_QUERY:
        matched_images = matched_images[:config.MAX_IMAGES_PER_QUERY]

    return sorted(matched_images)

# ==================== QUERY FUNCTIONS ====================

def query_piping_data(question: str, stream: bool = False) -> Dict:
    """
    Query with smart classification.
    Casual queries → direct LLM, no RAG.
    Technical queries → full RAG pipeline.
    UPGRADED: Benefits from larger context window (8192) and stronger LLM (8B).
    """
    start_time = time.time()

    try:
        if is_casual_query(question):
            if config.VERBOSE:
                print(f"💬 Casual query detected: '{question}' - skipping RAG")

            llm = Settings.llm
            response = llm.complete(question)
            total_time = time.time() - start_time

            return {
                "answer": str(response),
                "sources": [],
                "images": [],
                "timing": {
                    "total_seconds": round(total_time, 2),
                    "retrieval_seconds": 0.0,
                    "llm_seconds": round(total_time, 2)
                },
                "query_type": "casual"
            }

        if config.VERBOSE:
            print(f"🔍 Technical query detected: '{question}' - using RAG")

        index = _qdrant_manager.get_index()

        # UPGRADED: similarity_top_k=4 (was 2) retrieves more context.
        # Safe because context window is now 8192 tokens.
        query_engine = index.as_query_engine(
            similarity_top_k=config.SIMILARITY_TOP_K,
            streaming=stream,
            verbose=config.VERBOSE
        )

        retrieval_start = time.time()
        response = query_engine.query(question)
        retrieval_time = time.time() - retrieval_start

        sources = []
        page_map = {}

        for node in response.source_nodes:
            file_name = node.metadata.get("file_name", "Unknown")
            page_label = node.metadata.get("page_label", None)
            page_number = node.metadata.get("page_number", None)
            doc_stem = node.metadata.get("document_stem", Path(file_name).stem)
            score = node.score if hasattr(node, 'score') else None

            if not _is_valid_page(page_label) or not page_number:
                continue

            if score and score < config.MIN_RELEVANCE_SCORE:
                if config.VERBOSE:
                    print(f"⚠️  Skipping low-relevance source (score: {score:.3f})")
                continue

            pdf_link = None
            if file_name != "Unknown" and hasattr(config, 'PDF_OUTPUT_DIR'):
                if page_number:
                    pdf_link = f"{config.BASE_URL}/static/pdfs/{file_name}#page={page_number}"
                else:
                    pdf_link = f"{config.BASE_URL}/static/pdfs/{file_name}"

            sources.append({
                "file": file_name,
                "page": page_label,
                "score": round(score, 3) if score else None,
                "pdf_link": pdf_link,
                "page_number": page_number
            })

            if page_number and doc_stem:
                if doc_stem not in page_map:
                    page_map[doc_stem] = set()
                page_map[doc_stem].add(page_number)

        found_images = []
        if sources and config.IMAGE_PAGE_MATCH_STRICT:
            for doc_stem, pages in page_map.items():
                doc_images = _get_images_for_pages(doc_stem, pages)
                found_images.extend(doc_images)

        if not sources:
            found_images = []

        total_time = time.time() - start_time

        result = {
            "answer": str(response),
            "sources": sources,
            "images": sorted(list(set(found_images)))[:config.MAX_IMAGES_PER_QUERY] if found_images else [],
            "timing": {
                "total_seconds": round(total_time, 2),
                "retrieval_seconds": round(retrieval_time, 2),
                "llm_seconds": round(total_time - retrieval_time, 2)
            },
            "query_type": "technical"
        }

        if config.VERBOSE:
            print(f"⏱️  Query completed in {total_time:.2f}s")
            print(f"📊 Retrieved {len(sources)} valid chunks, {len(found_images)} images")

        return result

    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ Error after {error_time:.2f}s: {e}")
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "images": [],
            "timing": {"total_seconds": round(error_time, 2)}
        }

def query_piping_data_stream(question: str) -> Generator[str, None, None]:
    """Stream response tokens."""
    try:
        if is_casual_query(question):
            llm = Settings.llm
            response = llm.complete(question)
            yield str(response)
            return

        index = _qdrant_manager.get_index()
        query_engine = index.as_query_engine(
            similarity_top_k=config.SIMILARITY_TOP_K,
            streaming=True
        )

        response = query_engine.query(question)

        for token in response.response_gen:
            yield token

    except Exception as e:
        yield f"\n\n❌ Error: {str(e)}"

def clear_cache():
    global _IMAGE_METADATA_CACHE
    _qdrant_manager.clear()
    _IMAGE_METADATA_CACHE = None
    print("🔄 All caches cleared")

def get_stats() -> Dict:
    try:
        index = _qdrant_manager.get_index()
        client = _qdrant_manager.get_client()
        collection_info = client.get_collection("piping_docs")
        metadata = _load_image_metadata()

        return {
            "status": "ready",
            "vector_count": collection_info.points_count,
            "embedding_model": config.EMBED_MODEL_NAME,
            # UPGRADED: report actual vector dimension (1024 for bge-large)
            "embedding_dim": 1024 if "large" in config.EMBED_MODEL_NAME else 384,
            "llm_model": config.LLM_MODEL,
            "llm_gpu_layers": config.LLM_GPU_LAYERS,
            "context_window": config.LLM_CONTEXT_WINDOW,
            "images_cached": len(metadata),
            "image_page_matching": config.IMAGE_PAGE_MATCH_STRICT,
            "query_classification": config.ENABLE_QUERY_CLASSIFICATION
        }
    except Exception as e:
        return {
            "status": "not_ready",
            "error": str(e)
        }

_load_or_create_index = lambda: _qdrant_manager.get_index()
_get_image_cache = lambda: set(_load_image_metadata().keys())

def get_query_metadata(question: str) -> dict:
    if is_casual_query(question):
        return {"sources": [], "images": []}

    try:
        index = _qdrant_manager.get_index()
        retriever = index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)
        nodes = retriever.retrieve(question)

        sources = []
        page_map = {}

        for node in nodes:
            file_name = node.metadata.get("file_name", "Unknown")
            page_label = node.metadata.get("page_label", None)
            page_number = node.metadata.get("page_number", None)
            doc_stem = node.metadata.get("document_stem", Path(file_name).stem)
            score = node.score if hasattr(node, 'score') else None

            if not _is_valid_page(page_label) or not page_number:
                continue

            if score and score < config.MIN_RELEVANCE_SCORE:
                continue

            pdf_link = None
            if file_name != "Unknown" and hasattr(config, 'PDF_OUTPUT_DIR'):
                if page_number:
                    pdf_link = f"{config.BASE_URL}/static/pdfs/{file_name}#page={page_number}"

            sources.append({
                "file": file_name,
                "page": page_label,
                "pdf_link": pdf_link
            })

            if page_number and doc_stem:
                if doc_stem not in page_map:
                    page_map[doc_stem] = set()
                page_map[doc_stem].add(page_number)

        found_images = []
        if sources:
            for doc_stem, pages in page_map.items():
                doc_images = _get_images_for_pages(doc_stem, pages)
                found_images.extend(doc_images)

        image_urls = [
            f"{config.BASE_URL}/static/images/{img}"
            for img in sorted(set(found_images))[:config.MAX_IMAGES_PER_QUERY]
        ] if found_images else []

        return {
            "sources": sources,
            "images": image_urls
        }

    except Exception as e:
        print(f"❌ Metadata error: {e}")
        return {"sources": [], "images": []}