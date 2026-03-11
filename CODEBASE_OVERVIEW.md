# Piping Knowledge RAG – Codebase Overview

## What This Project Does

This is a **RAG (Retrieval-Augmented Generation)** app for piping knowledge:

1. **Ingest**: PDFs (and optionally PPTX) in `data/piping/test_data` are loaded, chunked, turned into embeddings (vectors), and stored in `backend/storage`.
2. **Query**: A question is sent to the API; the app finds relevant chunks from storage, sends them + the question to an LLM (Llama 3.2 via Ollama), and returns the answer plus source references.

So: **your documents → vector index → search + LLM → answers with citations.**

---

## Project Layout

```
reliance project/
├── backend/           # API + RAG logic + ingestion
│   ├── main.py        # FastAPI server, /ask endpoint
│   ├── engine.py      # Load index, run queries (query_piping_data)
│   ├── ingest_safe.py  # Build index from data/piping/test_data → save to storage
│   ├── ingest.py      # (empty or legacy – use ingest_safe.py)
│   ├── requirements.txt
│   └── storage/       # Created by ingest – vector index (don’t edit by hand)
├── frontend/
│   └── index.html     # Chat UI (calls /ask)
└── data/
    └── piping/
        └── test_data/ # Put PDFs (and optionally PPTX) here for ingestion
```

---

## File-by-File

### 1. `backend/engine.py` – RAG “brain”

- **Role**: Load the saved index and answer questions using it.
- **Config**:
  - **LLM**: `Ollama(model="llama3.2", request_timeout=360.0)` – generates answers.
  - **Embeddings**: `OllamaEmbedding(model_name="nomic-embed-text")` – used for similarity search (same as ingest).
- **Paths**: `PERSIST_DIR = backend/storage` – where the index lives.
- **Main behavior**:
  - `get_query_engine()`: Loads the index from `storage/` once, caches it, returns a LlamaIndex query engine.
  - `query_piping_data(question)`: Uses that engine to run the question, returns `{"answer": "...", "sources": [{file, page}, ...]}`.
- **Important**: Expects `backend/storage` to already exist (created by `ingest_safe.py`). If not, you get `FileNotFoundError`.

So: **engine.py = “load index + run queries”; it does not read PDFs or build the index.**

---

### 2. `backend/main.py` – HTTP API

- **Role**: Expose the RAG as a web API.
- **Endpoints**:
  - `POST /ask`: Body `{"question": "..."}`. Calls `query_piping_data(req.question)` and returns the same dict (answer + sources). Errors → 500 with message.
- **CORS**: Allows all origins so the frontend can call the API.
- **Run**: `uvicorn main:app --reload` (from `backend/`), or `python -m uvicorn main:app --host 0.0.0.0 --port 8000`.

So: **main.py = “HTTP wrapper around engine.query_piping_data”.**

---

### 3. `backend/ingest_safe.py` – Build the index

- **Role**: Turn documents in `data/piping/test_data` into the vector index in `backend/storage`.
- **Flow**:
  1. Set same Settings as engine: Nomic embeddings, Llama 3.2 (for any LLM step during indexing, if used).
  2. **Read**: `SimpleDirectoryReader(input_dir=base_path, recursive=True)` on `data/piping/test_data`. No Docling (avoids Torch/DLL issues on Windows).
  3. **Index**: `VectorStoreIndex.from_documents(documents)` – chunk + embed and build the index.
  4. **Save**: `index.storage_context.persist(persist_dir=...)` → `backend/storage`.
- **Paths**: Works whether you run from project root or from `backend/` (it switches `base_path` and `persist_path` accordingly).

So: **ingest_safe.py = “PDFs in test_data → vector index in backend/storage”.** Run this before the API can answer questions.

---

### 4. `backend/ingest.py`

- Currently **empty** (or legacy). The active ingestion script is **ingest_safe.py**.

---

### 5. `frontend/index.html`

- **Role**: Simple chat UI that talks to the backend.
- **Behavior**: User types a question, frontend sends `POST /ask` with `{question: "..."}`, displays the returned `answer` and `sources`.
- **Assumption**: API at same origin or CORS allowed (backend allows all origins).

So: **index.html = “chat UI for /ask”.**

---

## Data Flow (End-to-End)

```
1. INGEST (one-time or when docs change)
   data/piping/test_data/*.pdf
        → ingest_safe.py (SimpleDirectoryReader → VectorStoreIndex)
        → backend/storage/  (vectors + metadata)

2. QUERY (every user question)
   User question
        → frontend → POST /ask → main.py
        → engine.query_piping_data(question)
        → get_query_engine() loads index from backend/storage
        → similarity search (nomic-embed-text) + LLM (llama3.2)
        → { answer, sources }
        → frontend shows answer + citations
```

---

## What You Need to Run

1. **Ollama** with models: `llama3.2`, `nomic-embed-text`.
2. **Python** with deps from `backend/requirements.txt`.
3. **Data**: PDFs (and optionally PPTX) in `data/piping/test_data`.
4. **Ingest once**: `python backend/ingest_safe.py` (from project root) or `python ingest_safe.py` (from `backend/`).
5. **Start API**: From `backend/`, `python -m uvicorn main:app --host 0.0.0.0 --port 8000`.
6. **Frontend**: Serve `frontend/index.html` (e.g. from same host as API or open file) and point it at the API URL.

---

## Best Next Steps

1. **Stabilize one ingestion path**  
   Use **ingest_safe.py** as the single source of truth. Optionally delete or repurpose `ingest.py` so there’s no confusion.

2. **Serve the frontend with the API**  
   In `main.py`, add a route (e.g. `GET /`) that serves `frontend/index.html`, and optionally mount `frontend/` for static assets. Then “run the backend” gives you both API and chat UI at one URL.

3. **Re-ingest when data changes**  
   Document that after adding/removing/updating files in `data/piping/test_data`, users should run `ingest_safe.py` again so the index and answers stay in sync.

4. **Optional: support more folders**  
   If you want to ingest from multiple dirs (e.g. `test_data` + `production_data`), extend `ingest_safe.py` to accept a list of directories or a config path and merge their documents into one index.

5. **Optional: better chunking**  
   Right now LlamaIndex uses default chunking. For piping docs (tables, specs), consider setting `Settings` with a custom `node_parser` or `text_splitter` (e.g. by token count or section) and use it in both ingest and engine for consistency.

6. **Optional: health check**  
   Add `GET /health` in `main.py` that checks that `backend/storage` exists (and optionally that Ollama is reachable). Helps with deployment and debugging.

7. **Errors and logging**  
   In `main.py`, log request/response and errors; in `engine.py`, log when the index is loaded and when queries fail. That will make “nothing happens” or “500 with no message” easier to debug.

8. **Config**  
   Move model names, timeouts, and paths (e.g. `data/piping/test_data`, `backend/storage`) into a small config (e.g. `config.py` or `.env`) so you can switch environments or models without editing code.

If you tell me your priority (e.g. “serve frontend from backend” or “add health check”), I can outline the exact code changes next.
