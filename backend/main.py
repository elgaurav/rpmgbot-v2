"""
FastAPI Server for RPMG RAG Assistant
Optimized for <30 second response times with streaming support

UPDATED: 
- PDF serving with page anchors and smart image filtering
- Persistent chat history with SQLite database
- Conversation management system
"""

import asyncio
import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from engine import (
    query_piping_data, 
    query_piping_data_stream, 
    clear_cache, 
    get_stats,
    get_query_metadata
)
import config

# ==================== DATABASE SETUP ====================

def init_db():
    """Initialize SQLite database for chat history"""
    db_path = config.BASE_DIR / 'chat_history.db'
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    
    # Conversations table
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
    ''')
    
    # Messages table
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            sources TEXT,
            images TEXT,
            timing TEXT,
            created_at TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {db_path}")

# Initialize database on import
init_db()

# ==================== DATABASE HELPER FUNCTIONS ====================

def get_db_connection():
    """Get database connection"""
    db_path = config.BASE_DIR / 'chat_history.db'
    return sqlite3.connect(str(db_path))

def create_conversation(first_question: str) -> str:
    """Create a new conversation and return its ID"""
    conv_id = str(uuid.uuid4())
    title = first_question[:50] + "..." if len(first_question) > 50 else first_question
    now = datetime.now().isoformat()
    
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        'INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)',
        (conv_id, title, now, now)
    )
    conn.commit()
    conn.close()
    
    return conv_id

def save_message(conversation_id: str, role: str, content: str, 
                sources: List[dict] = None, images: List[str] = None, 
                timing: dict = None):
    """Save a message to the database"""
    msg_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO messages (id, conversation_id, role, content, sources, images, timing, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        msg_id,
        conversation_id,
        role,
        content,
        json.dumps(sources) if sources else None,
        json.dumps(images) if images else None,
        json.dumps(timing) if timing else None,
        now
    ))
    
    # Update conversation timestamp
    c.execute('UPDATE conversations SET updated_at = ? WHERE id = ?', (now, conversation_id))
    
    conn.commit()
    conn.close()

def get_conversation_messages(conversation_id: str) -> List[dict]:
    """Retrieve all messages for a conversation"""
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT id, role, content, sources, images, timing, created_at 
        FROM messages 
        WHERE conversation_id = ? 
        ORDER BY created_at ASC
    ''', (conversation_id,))
    
    messages = []
    for row in c.fetchall():
        messages.append({
            "id": row[0],
            "role": row[1],
            "content": row[2],
            "sources": json.loads(row[3]) if row[3] else None,
            "images": json.loads(row[4]) if row[4] else None,
            "timing": json.loads(row[5]) if row[5] else None,
            "created_at": row[6]
        })
    
    conn.close()
    return messages

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="RPMG Assistant",
    description="High-performance RAG system for piping engineering with source linking and chat history",
    version="3.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = config.BASE_DIR.parent / "frontend"
config.STATIC_DIR.mkdir(parents=True, exist_ok=True)
config.IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
config.PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

# ==================== PYDANTIC MODELS ====================

class AskRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    stream: bool = False

class Message(BaseModel):
    id: str
    role: str
    content: str
    sources: Optional[List[dict]] = None
    images: Optional[List[str]] = None
    timing: Optional[dict] = None
    created_at: str

class Conversation(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    preview: Optional[str] = None

# ==================== ENDPOINTS ====================

@app.get("/")
def serve_chat():
    """Serve the frontend"""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    else:
        raise HTTPException(status_code=404, detail="Frontend not found")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    stats = get_stats()
    return {
        "status": "healthy",
        "service": "RPMG Assistant v3.1",
        "features": [
            "Smart image-page linking",
            "PDF source viewing",
            "Page-level retrieval",
            "Persistent chat history"
        ],
        "config": {
            "llm": config.LLM_MODEL,
            "embedding": config.EMBED_MODEL_NAME,
            "top_k": config.SIMILARITY_TOP_K,
            "strict_image_matching": config.IMAGE_PAGE_MATCH_STRICT,
            "max_images": config.MAX_IMAGES_PER_QUERY
        },
        "system": stats
    }

@app.post("/ask")
async def ask(req: AskRequest):
    """
    Query the RAG system with persistent chat history
    
    Returns answer with:
    - Source citations with PDF page links
    - Images from relevant pages only
    - Performance timing
    - Conversation ID for session continuity
    """
    print(f"\n📥 Query: {req.question}")
    
    try:
        # Create new conversation if not provided
        if not req.conversation_id:
            conversation_id = create_conversation(req.question)
            print(f"🆕 New conversation: {conversation_id}")
        else:
            conversation_id = req.conversation_id
            print(f"💬 Continuing conversation: {conversation_id}")
        
        # Save user message
        save_message(conversation_id, "user", req.question)
        
        # Run RAG query in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            query_piping_data, 
            req.question, 
            False
        )
        
        # Convert image filenames to URLs
        image_urls = [
            f"{config.BASE_URL}/static/images/{img}" 
            for img in result.get("images", [])
        ]
        
        # Save assistant message
        save_message(
            conversation_id,
            "assistant",
            result["answer"],
            result.get("sources"),
            image_urls,
            result.get("timing")
        )
        
        # Prepare response
        response_data = {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "images": image_urls,
            "timing": result.get("timing", {}),
            "conversation_id": conversation_id
        }
        
        # Log performance
        timing = result.get("timing", {})
        total_time = timing.get("total_seconds", 0)
        sources = result.get("sources", [])
        
        print(f"✅ Response: {total_time}s | Sources: {len(sources)} | Images: {len(image_urls)}")
        
        # Log source pages
        pages_cited = [f"{s['file']} p.{s['page']}" for s in sources if s.get('page')]
        if pages_cited:
            print(f"📄 Pages cited: {', '.join(pages_cited)}")
        
        if total_time > 30:
            print(f"⚠️  WARNING: Response took {total_time}s (target: <30s)")
        
        return response_data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-stream")
async def ask_stream(req: AskRequest):
    """
    STREAMING ENDPOINT - Tokens appear as they're generated
    
    Sends metadata (sources + images) first, then streams the answer
    Also saves to conversation history
    """
    print(f"\n📥 Streaming Query: {req.question}")
    
    # Create new conversation if not provided
    if not req.conversation_id:
        conversation_id = create_conversation(req.question)
        print(f"🆕 New conversation: {conversation_id}")
    else:
        conversation_id = req.conversation_id
        print(f"💬 Continuing conversation: {conversation_id}")
    
    # Save user message
    save_message(conversation_id, "user", req.question)
    
    async def stream_generator():
        try:
            # First, get sources and images (fast)
            loop = asyncio.get_event_loop()
            metadata = await loop.run_in_executor(
                None,
                get_query_metadata,
                req.question
            )
            
            # Send conversation ID first
            yield f"data: {json.dumps({'type': 'conversation_id', 'data': conversation_id})}\n\n"
            
            # Send metadata
            yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"
            
            # Stream the answer
            full_answer = ""
            for token in query_piping_data_stream(req.question):
                full_answer += token
                chunk = {"type": "token", "data": token}
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Save assistant message after streaming completes
            save_message(
                conversation_id,
                "assistant",
                full_answer,
                metadata.get("sources"),
                metadata.get("images")
            )
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            error = {"type": "error", "data": str(e)}
            yield f"data: {json.dumps(error)}\n\n"
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/conversations")
async def get_conversations():
    """Get all conversations ordered by most recent"""
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT c.id, c.title, c.created_at, c.updated_at,
               (SELECT content FROM messages WHERE conversation_id = c.id AND role = 'user' LIMIT 1) as preview
        FROM conversations c
        ORDER BY c.updated_at DESC
    ''')
    
    conversations = []
    for row in c.fetchall():
        conversations.append({
            "id": row[0],
            "title": row[1],
            "created_at": row[2],
            "updated_at": row[3],
            "preview": row[4]
        })
    
    conn.close()
    return conversations

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages"""
    messages = get_conversation_messages(conversation_id)
    
    if not messages:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "messages": messages
    }

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages"""
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
    c.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
    
    conn.commit()
    conn.close()
    
    return {"status": "deleted", "conversation_id": conversation_id}

@app.post("/clear-cache")
def clear_system_cache():
    """Clear cached index and images - call after re-ingestion"""
    try:
        clear_cache()
        return {
            "status": "success", 
            "message": "Cache cleared. Restart recommended."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_system_stats():
    """Get system statistics and configuration"""
    stats = get_stats()
    stats["config"] = {
        "strict_image_matching": config.IMAGE_PAGE_MATCH_STRICT,
        "adjacent_pages": config.IMAGE_ADJACENT_PAGES,
        "max_images": config.MAX_IMAGES_PER_QUERY,
        "chunk_size": config.CHUNK_SIZE,
        "top_k": config.SIMILARITY_TOP_K
    }
    
    # Add database stats
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM conversations')
    conversation_count = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM messages')
    message_count = c.fetchone()[0]
    conn.close()
    
    stats["database"] = {
        "conversations": conversation_count,
        "messages": message_count
    }
    
    return stats

@app.get("/list-pdfs")
def list_available_pdfs():
    """List all available source PDFs"""
    try:
        pdf_files = list(config.PDF_OUTPUT_DIR.glob("*.pdf"))
        pdfs = [
            {
                "name": pdf.name,
                "url": f"{config.BASE_URL}/static/pdfs/{pdf.name}",
                "size_mb": round(pdf.stat().st_size / (1024 * 1024), 2)
            }
            for pdf in pdf_files
        ]
        return {"pdfs": pdfs, "count": len(pdfs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== STARTUP ====================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    print("\n" + "="*60)
    print("🚀 RPMG RAG Assistant v3.1 - Starting Up")
    print("="*60)
    print(f"📊 Configuration:")
    print(f"   LLM: {config.LLM_MODEL}")
    print(f"   Embeddings: {config.EMBED_MODEL_NAME}")
    print(f"   Top-K Retrieval: {config.SIMILARITY_TOP_K}")
    print(f"   Max Tokens: {config.LLM_MAX_TOKENS}")
    print(f"   Streaming: {config.ENABLE_STREAMING}")
    print(f"\n🖼️  Image Configuration:")
    print(f"   Strict page matching: {config.IMAGE_PAGE_MATCH_STRICT}")
    print(f"   Adjacent pages: ±{config.IMAGE_ADJACENT_PAGES}")
    print(f"   Max images per query: {config.MAX_IMAGES_PER_QUERY}")
    print(f"\n💾 Database:")
    print(f"   Chat history: ENABLED")
    print(f"   Location: {config.BASE_DIR / 'chat_history.db'}")
    print("="*60)
    
    # Warm up the engine
    try:
        stats = get_stats()
        if stats["status"] == "ready":
            print(f"✅ Index loaded: {stats['vector_count']} vectors")
            print(f"✅ Images cached: {stats['images_cached']}")
            print(f"✅ Page-based matching: {'Enabled' if stats.get('image_page_matching') else 'Disabled'}")
        else:
            print(f"⚠️  Index not ready: {stats.get('error', 'Unknown')}")
    except Exception as e:
        print(f"⚠️  Could not load index: {e}")
        print("   Run ingest_pro.py first!")
    
    # Show database stats
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM conversations')
        conv_count = c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM messages')
        msg_count = c.fetchone()[0]
        conn.close()
        print(f"✅ Database: {conv_count} conversations, {msg_count} messages")
    except Exception as e:
        print(f"⚠️  Database stats unavailable: {e}")
    
    print("="*60)
    print(f"🌐 Server ready at http://localhost:{config.API_PORT}")
    print(f"📋 Available PDFs: /static/pdfs/")
    print(f"🖼️  Images: /static/images/")
    print(f"📊 Stats: /stats")
    print(f"📚 PDF List: /list-pdfs")
    print(f"💬 Conversations: /conversations")
    print("="*60 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info"
    )