"""
Configuration for RPMG RAG Assistant
UPGRADED: Higher-capacity models for GPU-capable hardware
- Embedding: bge-large-en-v1.5 (1024-dim, up from 384-dim)
- LLM: llama3.1:8b 4-bit quantized (up from qwen2.5:1.5b)
- Chunk size: 1024 tokens (up from 512)
- Top-K: 4 (up from 2)
- Context window: 8192 tokens (up from 2048)
"""
from pathlib import Path

# ==================== PATHS ====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "piping" / "test_data"
STORAGE_DIR = BASE_DIR / "storage"
STATIC_DIR = BASE_DIR / "static"
IMAGE_OUTPUT_DIR = BASE_DIR / "static" / "images"
PDF_OUTPUT_DIR = BASE_DIR / "static" / "pdfs"
QDRANT_PATH = BASE_DIR / "qdrant_db"

# ==================== MODEL CONFIGURATION ====================

# EMBEDDING MODEL
# UPGRADED: bge-large-en-v1.5 produces 1024-dim vectors (vs 384 in bge-small).
# Much better semantic retrieval quality at the cost of ~3x more RAM.
# Requires re-running ingest_pro.py after this change.
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"  # 1024 dimensions

# GPU CONTROL FOR EMBEDDINGS:
# Set EMBED_USE_GPU = True  → embeddings run on GPU (faster, recommended)
# Set EMBED_USE_GPU = False → embeddings run on CPU (slower but works everywhere)
EMBED_USE_GPU = False  # <-- SET TO False FOR CPU-ONLY MODE

# Batch size for embedding during ingestion.
# Lower this (e.g. 8) if you hit GPU OOM errors during ingest_pro.py.
EMBED_BATCH_SIZE = 32  # <-- LOWER TO 8 IF GPU OOM DURING INGESTION

# LLM CONFIGURATION
# UPGRADED: llama3.1:8b-instruct-q4_k_m.gguf is a strong 8B model at 4-bit
# quantization. Uses ~5-6GB RAM/VRAM. Much better reasoning than 1.5b.
# Download from: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
# File: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
LLM_PROVIDER = "llamacpp"
LLM_MODEL = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
LLM_MODEL_PATH = str(BASE_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

LLM_REQUEST_TIMEOUT = 180.0   # Increased: 8B model takes longer to respond
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024          # UPGRADED: more room for detailed answers (was 512)

# UPGRADED: 8192 token context window (was 2048).
# The 8B model was trained with 128k context but 8k is safe and fast locally.
LLM_CONTEXT_WINDOW = 8192

# GPU CONTROL FOR LLM:
# LLM_GPU_LAYERS controls how many transformer layers run on GPU.
# - Set to a large number (e.g. 999) to put ALL layers on GPU → fastest inference
# - Set to 0 to run entirely on CPU → slower but no VRAM needed
# For llama3.1:8b Q4 (~5GB), a 6GB+ GPU can handle all layers.
LLM_GPU_LAYERS = 0            # 999  # <-- SET TO 0 FOR CPU-ONLY MODE

# ==================== RAG PARAMETERS ====================

# UPGRADED: 1024 token chunks (was 512).
# Larger chunks preserve more context per page section, better for technical docs.
CHUNK_SIZE = 1024

# UPGRADED: 150 token overlap (was 50).
# More overlap reduces information loss at chunk boundaries.
CHUNK_OVERLAP = 150

# UPGRADED: Retrieve top 4 chunks (was 2).
# More context fed to LLM = better answers. Safe since we increased context window.
SIMILARITY_TOP_K = 3

# ==================== PERFORMANCE TUNING ====================
USE_GPU = False          # <-- SET TO False FOR CPU-ONLY MODE (affects general GPU hint)
ENABLE_STREAMING = True
BATCH_SIZE = 8

# ==================== DOCLING SETTINGS ====================
# No changes here — OCR and table extraction remain the same.
DOCLING_OCR = True
DOCLING_TABLE_EXTRACTION = True
DOCLING_IMAGE_SCALE = 1.2
DOCLING_GENERATE_PICTURES = True

# ==================== API SETTINGS ====================
API_HOST = "0.0.0.0"
API_PORT = 8000
BASE_URL = f"http://127.0.0.1:{API_PORT}"

# ==================== IMAGE LINKING SETTINGS ====================
IMAGE_PAGE_MATCH_STRICT = True
IMAGE_ADJACENT_PAGES = 0
MAX_IMAGES_PER_QUERY = 10

# ==================== QUERY CLASSIFICATION ====================
ENABLE_QUERY_CLASSIFICATION = True
MIN_RELEVANCE_SCORE = 0.3

CASUAL_QUERY_PATTERNS = [
    "hello", "hi", "hey", "greetings",
    "good morning", "good afternoon", "good evening", "good night",
    "how are you", "how r u", "sup", "what's up", "whats up",
    "thanks", "thank you", "thx", "ty",
    "bye", "goodbye", "see you", "cya",
    "ok", "okay", "cool", "nice", "great", "ssup", "yo"
]

# ==================== DEBUGGING ====================
VERBOSE = True
LOG_RETRIEVAL = True

# ==================== CPU-ONLY QUICK REFERENCE ====================
# If you want to run this entirely on CPU, change these values:
#
#   EMBED_USE_GPU  = False   (embeddings on CPU)
#   LLM_GPU_LAYERS = 0       (LLM inference on CPU)
#   USE_GPU        = False   (global GPU hint off)
#   EMBED_BATCH_SIZE = 8     (smaller batches, less RAM pressure)
#   LLM_MAX_TOKENS = 512     (shorter responses to keep CPU wait reasonable)
#   LLM_CONTEXT_WINDOW = 4096 (halve context to reduce CPU memory usage)
#
# Everything else stays the same. Expect 3-5x slower responses on CPU.