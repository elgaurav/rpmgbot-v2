🔷 RPMG AI Assistant (Local RAG Pipeline)
An enterprise-grade, privacy-focused AI chatbot designed for the Reliance Project Management Group (RPMG). This tool uses Retrieval-Augmented Generation (RAG) to allow engineers to query internal technical documents (Piping Material Classes, API 582, Valve Specs) using natural language, without data ever leaving the secure local environment.

🚀 Key Features
🔒 Fully Local & Secure: Runs entirely offline using Ollama and local vector stores. No data is sent to the cloud.

🧠 Optimized for Low-Resource Hardware: Custom "Tiny" architecture capable of running on 4GB RAM laptops using quantized models (Llama 3.2:1b).

🔍 Citation-Backed Answers: Every response includes direct references to the source PDF and page number to ensure engineering accuracy.

🏢 Corporate UI Integration: Custom Frontend designed to match the SharePoint Enterprise aesthetic of the RPMG portal.

🛠️ Tech Stack
Backend: FastAPI, Uvicorn

AI Orchestration: LlamaIndex

LLM & Embeddings: Ollama (Llama 3.2 / DeepSeek-R1), Nomic Embed

Frontend: Vanilla JS, HTML5, CSS3 (SharePoint Styling)

Storage: Local Vector Store (ChromaDB/Flat)

📂 Architecture
The project follows a scalable "Ingest-Query" pipeline:

Ingestion (ingest.py): Reads PDFs from the /data folder, chunks them, and creates vector embeddings.

Current Mode: Text-based extraction (optimized for speed).

Future Mode: Docling integration for complex table/image extraction (ready for 16GB+ RAM workstations).

Storage: Embeddings are persisted locally in backend/storage.

Retrieval (engine.py): Hybrid search locates relevant context.

Generation: The LLM synthesizes the answer using the retrieved engineering context.

⚡ Quick Start
Prerequisites:

Install Ollama

Pull the model: ollama pull llama3.2:1b (for 4GB RAM) or ollama pull deepseek-r1 (for Desktop)

Installation:

Bash
git clone https://github.com/YourUsername/rpmg-ai-assistant.git
cd rpmg-ai-assistant
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
Run the App:

Bash
# 1. Build the Database (First run only)
python backend/ingest_safe.py

# 2. Start the Server
uvicorn backend.main:app --reload
Visit http://127.0.0.1:8000 to access the chat interface.

🗺️ Roadmap
[x] Basic RAG Pipeline (Text)

[x] Low-Memory Optimization (4GB RAM)

[x] Enterprise UI Implementation

[ ] Desktop Upgrade: Switch to DeepSeek-R1 & Enable Docling for Table Analysis.

[ ] Multi-Modal Support: Extract and display technical diagrams/images in chat.

# Reliance PMG AI Assistant

A RAG-based chatbot for Reliance Projects, capable of reading PDFs (API 582, Plot Plans) and extracting technical diagrams.

## 🔧 Prerequisites
- Python 3.10+
- 8GB RAM (Minimum)
- GPU recommended (but works on CPU)

## 🚀 How to Run

### 1. Setup Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

2. Ingest Data (Process PDFs)
Place your PDFs in data/piping/.

PowerShell
python backend/ingest_pro.py
3. Start the Server
PowerShell
cd backend
python -m uvicorn main:app --reload

PS C:\Users\A.Gagrani\Downloads\rpmg-ai-assistant-main\rpmg-ai-assistant-main\backend> cd ..\..
>> .\venv\Scripts\activate
(venv) PS C:\Users\A.Gagrani\Downloads\rpmg-ai-assistant-main> cd rpmg-ai-assistant-main       
(venv) PS C:\Users\A.Gagrani\Downloads\rpmg-ai-assistant-main\rpmg-ai-assistant-main> cd backend