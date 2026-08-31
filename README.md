# 🎓 OmniDoc-RAG

> **Grounded Academic Assistant for 38+ Engineering Disciplines**  
> A high-performance Retrieval-Augmented Generation (RAG) system featuring a production **FastAPI** backend, a responsive **Glassmorphic Web Interface**, and persistent **ChromaDB** retrieval powered by **Groq Cloud** with automatic model cascading and optional local **Ollama** inference.

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-fc5203.svg)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Cloud-f55036.svg)](https://groq.com/)
[![Render](https://img.shields.io/badge/Deployed-Render-46E3B7.svg?logo=render&logoColor=white)](https://omnidoc-rag.onrender.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)

🔗 **Live Production Demo**: [https://omnidoc-rag.onrender.com/](https://omnidoc-rag.onrender.com/)

---

## 📌 Table of Contents
- [Problem & Solution](#-problem--solution)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Project Directory Structure](#-project-directory-structure)
- [Subject Coverage & Dataset](#-subject-coverage--dataset)
- [API Endpoints](#-api-endpoints)
- [Local Quickstart](#-local-quickstart)
- [Ingestion & Vector DB Management](#-ingestion--vector-db-management)
- [Render Cloud Deployment](#-render-cloud-deployment)
- [Configuration Reference](#-configuration-reference)
- [Testing](#-testing)

---

## 🎯 Problem & Solution

* **The Problem:** Engineering students study from fragmented materials — faculty lecture slides (`.pptx`), textbooks (`.pdf`), lab manuals (`.docx`), and scanned handwritten notes across 38+ subjects. Finding accurate, curriculum-specific answers during exam prep is slow and keyword search lacks semantic understanding.
* **The Solution:** **OmniDoc-RAG** indexes 390+ course files (36,000+ chunks) into a subject-partitioned vector store. Using semantic retrieval, anti-hallucination relevance gating, and subject-scoped acronym expansion, it produces structured, exam-grade explanations strictly grounded in course notes.

---

## ✨ Key Features

- **🌐 Production Web Interface:** Dark-mode glassmorphic single-page application (`frontend/index.html`) featuring:
  - Subject selector grouped by academic department (AI, Networks, Core CS, Management).
  - Real-time token streaming using **Server-Sent Events (SSE)**.
  - Interactive sample questions and markdown rendering with code highlighting.
  - Settings modal for custom Groq API key configuration and engine selection.
  - Responsive sidebar with persistent local chat history.
- **⚡ Dual Serving Architecture:**
  - **FastAPI (`api.py`):** Production REST API with SSE streaming, static asset serving, and memory-optimized lifecycle.
  - **Streamlit (`app.py`):** Alternative desktop-friendly UI for local testing.
- **🛡️ Anti-Hallucination Relevance Gate:** Validates retrieved chunks against the active subject before calling the LLM. If a student asks an out-of-scope question (e.g., asking *DevOps* questions under *Java*), the system detects the domain mismatch and suggests the correct subject.
- **🔤 Subject-Scoped Acronym Expansion:** Disambiguates short acronyms based on context (e.g., `CN` in *Computer Networks* → *Computer Networks*, while `CN` in *Software Testing* → *Control Flow Graph*).
- **🔄 Auto-Cascading Model Fallback Pool:** Connects to Groq Cloud and automatically cascades down available models (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `openai/gpt-oss-120b`, `qwen/qwen3.6-27b`, etc.) if rate limits or outages occur, with optional fallback to local **Ollama** (`llama3.2`).
- **📄 Multi-Format Ingestion with OCR:** Extracts content from digital PDFs, DOCX, PPTX slides, and scanned image PDFs using **RapidOCR / Tesseract**.

---

## 🏛️ System Architecture

```text
                                  ┌────────────────────────┐
                                  │   Browser Frontend     │
                                  │  (Glassmorphic HTML/JS)│
                                  └───────────┬────────────┘
                                              │ HTTP / SSE
                                              ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FastAPI Backend (api.py)                               │
│                                                                                        │
│   ┌─────────────────────┐       ┌──────────────────────┐       ┌───────────────────┐   │
│   │  /api/health        │       │  /api/subjects       │       │  /api/chat (SSE)  │   │
│   └─────────────────────┘       └──────────────────────┘       └─────────┬─────────┘   │
└──────────────────────────────────────────────────────────────────────────┼─────────────┘
                                                                           │
               ┌───────────────────────────────────────────────────────────┴─────────────────────────────┐
               ▼                                                                                         ▼
┌───────────────────────────────┐                                                       ┌───────────────────────────────┐
│     Retrieval Pipeline        │                                                       │      LLM Client Pipeline      │
│                               │                                                       │                               │
│  1. Query Expansion           │                                                       │  1. Exam Prompt Formatting    │
│     (Acronym Resolution)      │                                                       │  2. Groq Cloud Inference      │
│  2. ChromaDB Dense Search     │ ────── [ Relevance Gate: Context Checked ] ─────────► │     (Cascading Model Pool)    │
│     (all-MiniLM-L6-v2)        │                                                       │  3. Optional Ollama Fallback  │
│  3. Subject Scope Filtering   │                                                       │  4. Token-by-Token SSE Stream │
└──────────────┬────────────────┘                                                       └───────────────────────────────┘
               │
               ▼
┌───────────────────────────────┐
│     Vector Database           │
│   (ChromaDB: 36K+ Chunks)     │
│   Local / Cloud PDF_db.zip    │
└───────────────────────────────┘
```

---

## 📂 Project Directory Structure

```text
OmniDoc-RAG/
├── api.py                      # Production FastAPI server & SSE endpoints
├── app.py                      # Local Streamlit interface
├── main.py                     # CLI entrypoint for app, ingestion, and tests
├── config.yaml                 # Central application, model & database configuration
├── render.yaml                 # Infrastructure-as-code for Render cloud deployment
├── requirements.txt            # Python dependencies
├── packages.txt                # Linux system dependencies for cloud build
├── PDF_db.zip                  # Compressed vector database archive
├── create_db_zip.py            # Utility script to package ChromaDB into ZIP
├── .env.example                # Template for environment variables
├── .gitignore                  # Git exclusion rules
│
├── frontend/                   # Production Web Client
│   └── index.html              # Modern glassmorphism UI (HTML5, CSS3, Vanilla JS)
│
├── src/                        # Modular Core Python Package
│   ├── __init__.py
│   ├── ingestion/              # Document extraction (PDF, DOCX, PPTX, OCR)
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── chunking/               # Text chunking & subject enrichment
│   │   ├── __init__.py
│   │   └── chunker.py
│   ├── embeddings/             # SentenceTransformers embedding generator
│   │   ├── __init__.py
│   │   └── embedder.py
│   ├── vectordb/               # ChromaDB PersistentClient & cloud ZIP loader
│   │   ├── __init__.py
│   │   └── vector_store.py
│   ├── retrieval/              # Query expansion, search & relevance gate
│   │   ├── __init__.py
│   │   └── retriever.py
│   ├── prompts/                # Structured academic prompt templates
│   │   ├── __init__.py
│   │   └── prompt_templates.py
│   ├── llm/                    # Groq API client with auto-cascading fallback & Ollama
│   │   ├── __init__.py
│   │   └── llm_client.py
│   ├── api/                    # Programmatic query routes
│   │   ├── __init__.py
│   │   └── routes.py
│   └── utils/                  # Subject metadata, sample queries & helpers
│       ├── __init__.py
│       └── helpers.py
│
├── PDF_Data/                   # Raw course materials (38+ subject subfolders)
├── pdf_db/                     # Local ChromaDB persistent storage
├── chat_history_sessions/      # Local chat session storage (.json)
├── tests/                      # Automated unit test suite
│   ├── __init__.py
│   └── test_app.py
└── logs/                       # Application runtime logs
    └── app.log
```

---

## 📚 Subject Coverage & Dataset

The vector store indexes **38+ courses** spanning 4 core academic categories:

| Category | Courses Covered |
|:---|:---|
| **🤖 AI & Data Science** | Artificial Intelligence (AI), Machine Learning (ML), Deep Learning & Neural Networks (NNDL), Natural Language Processing (NLP), Reinforcement Learning (RL), Semantic Web, Data Preparation & Pattern Mining (DPPM), AI-NLP Lab, ML Lab. |
| **🔐 Networks & Security** | Computer Networks (CN), Cryptography & Network Security (CNS), Cloud Computing, Social Network Analysis (SNA), CN Lab, CNS Lab. |
| **💻 Core CS & Systems** | Operating Systems (OS), Database Management Systems (DBMS), Compiler Design (CD), Design & Analysis of Algorithms (DAA), Computer Organization & Architecture (COA), Formal Languages & Automata Theory (FLAT), Data Structures, Discrete Mathematics (DM), DBMS Lab. |
| **🛠️ Software, Management & Electives** | DevOps & DevOps Lab, Software Engineering (SE), Software Testing Methodologies (STM), Java & Java Lab, Python Programming (PP), Principles of Economics (POE), Business Economics & Financial Analysis (BEFA), Management Science & Financial Accounting (MSF), Organizational Behaviour (OB), Total Quality Management (TQM), Web Programming (WP), Advanced Communication Skills (ACS Lab). |

---

## 🔌 API Endpoints

The FastAPI application (`api.py`) exposes the following endpoints:

| Endpoint | Method | Description |
|:---|:---:|:---|
| `/` | `GET` | Serves the single-page frontend application (`frontend/index.html`). |
| `/static/*` | `GET` | Serves static assets from the `frontend/` directory. |
| `/api/health` | `GET` | Health check returning chunk count, subject count, model list, and Groq status. |
| `/api/subjects` | `GET` | Returns all 38+ subjects grouped by category with metadata and sample questions. |
| `/api/chat` | `POST` | Streams token-by-token LLM responses via Server-Sent Events (`text/event-stream`). |

### Example Chat Request:
```bash
curl -X POST "http://localhost:10000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the Bellman-Ford algorithm and what is its time complexity?",
    "subject": "Design and Analysis of Algorithms",
    "chat_history": "",
    "engine": "Auto Cascading Pool"
  }'
```

---

## 🚀 Local Quickstart

### 1. Clone & Setup Environment
```bash
git clone https://github.com/saicharan-r02/OmniDoc-RAG.git
cd OmniDoc-RAG

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and configure your keys:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
OLLAMA_BASE_URL=http://localhost:11434
```

### 3. Launch the Application

* **Option A: Production FastAPI + Web UI (Recommended)**
  ```bash
  python api.py
  ```
  Open **`http://localhost:10000`** in your browser.

* **Option B: Local Streamlit UI**
  ```bash
  streamlit run app.py
  # OR
  python main.py --mode app
  ```
  Open **`http://localhost:8501`** in your browser.

---

## 📥 Ingestion & Vector DB Management

To ingest new documents or rebuild the vector index from raw files in `PDF_Data/`:

```bash
# Incremental ingestion (only processes new files):
python main.py --mode ingest

# Rebuild an entire specific subject:
python main.py --mode ingest --subject "Computer Networks" --rebuild

# Full rebuild of the entire database:
python main.py --mode ingest --rebuild
```

### Packaging Vector DB for Cloud Deployments
To package your indexed ChromaDB directory for cloud deployment without committing gigabytes to Git:
```bash
python create_db_zip.py
```
This generates `PDF_db.zip` (compressed archive of `./pdf_db/chromadb`).

---

## ☁️ Render Cloud Deployment

The repository includes a ready-to-deploy [`render.yaml`](file:///c:/OmniDoc-RAG/render.yaml) blueprint:

1. **Push your repository to GitHub.**
2. **Create a new Web Service on [Render](https://render.com).**
3. Select **Python** runtime and configure:
   - **Build Command:**
     ```bash
     pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     python api.py
     ```
4. **Add Environment Variables in Render Dashboard:**
   - `GROQ_API_KEY`: Your Groq Cloud API key.
   - `VECTOR_DB_URL`: *(Optional)* Direct download link (Google Drive / S3 / GitHub Release) to `PDF_db.zip` if hosting the DB remotely.
   - `OMP_NUM_THREADS`: `1`
   - `MKL_NUM_THREADS`: `1`
   - `TOKENIZERS_PARALLELISM`: `false`
   - `PYTHONUNBUFFERED`: `1`

---

## ⚙️ Configuration Reference

All RAG hyperparameters are centralized in [`config.yaml`](file:///c:/OmniDoc-RAG/config.yaml):

```yaml
app:
  name: "OmniDoc AI"
  version: "2.0.0"

database:
  persist_directory: "./pdf_db/chromadb"
  collection_name: "Document__C"
  cloud_zip_path: "./PDF_db.zip"

embeddings:
  model_name: "all-MiniLM-L6-v2"
  device: "cpu"

chunking:
  chunk_size: 1000
  chunk_overlap: 200

retrieval:
  top_k: 12
  min_context_length: 60
  relevance_threshold: 0.25

llm:
  default_groq_model: "llama-3.3-70b-versatile"
  available_groq_models:
    - "llama-3.3-70b-versatile"
    - "llama-3.1-8b-instant"
    - "openai/gpt-oss-120b"
    - "qwen/qwen3.6-27b"
  default_ollama_model: "llama3.2:latest"
  temperature: 0.0
```

---

## 🧪 Testing

Run the automated test suite to verify retrieval, relevance gating, and model fallback logic:
```bash
python main.py --mode test
# OR
pytest tests/ -v
```

---

## 📜 License & Acknowledgments

- Built for engineering students and educators.
- Powered by [FastAPI](https://fastapi.tiangolo.com/), [ChromaDB](https://www.trychroma.com/), [LangChain](https://www.langchain.com/), [Groq](https://groq.com/), and [SentenceTransformers](https://sbert.net/).

