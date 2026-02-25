# 🤖 Multimodal RAG Chatbot — Production System

Enterprise-grade RAG pipeline with **PDF ingestion**, **image extraction**,
**ChromaDB** vector storage, **Groq** fast inference, **Ollama** local vision,
and full **security layer**.

---

## 🏗️ Architecture

```
PDF Document
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  INGESTION PIPELINE                                      │
│                                                          │
│  PDF Loader → Text Clean → Quality Score → Metadata     │
│       │                                                  │
│       └→ Image Extractor → OCR → Image Embeddings       │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  CHUNKING (Hybrid Strategy)                              │
│                                                          │
│  Parent Chunks (~2000 chars) ─── for LLM context        │
│       │                                                  │
│       └→ Child Chunks (~500 chars) ─── for vector search │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  EMBEDDING & STORAGE                                     │
│                                                          │
│  HuggingFace BGE (text) ──┐                             │
│  CLIP (images)            ├──→ ChromaDB (persistent)    │
│                           │         + Parent Store       │
└──────────────────────────┘──────────────────────────────┘
                           │
                           │  QUERY TIME
                           ▼
┌─────────────────────────────────────────────────────────┐
│  SECURITY LAYER                                          │
│  Jailbreak Detection → Input Sanitization → Audit Log   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  ADVANCED RETRIEVAL                                      │
│                                                          │
│  1. HyDE          → Generate hypothetical answer        │
│  2. Multi-Query   → 3-5 query variations                │
│  3. Vector Search → ChromaDB child chunks               │
│  4. Parent-Child  → Upgrade to parent context           │
│  5. Re-rank       → Confidence scoring + dedup          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  LLM GENERATION (Router)                                 │
│                                                          │
│  Text query  → Groq API  (llama3-70b, ~0.3s)           │
│  Image query → Ollama    (llava:13b, local/private)     │
│  Fallback    → Ollama text model                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
            Output Validation → Safe Response
```

---

## 📁 Project Structure

```
multimodal_rag/
├── config.py               # All configuration (models, chunk sizes, etc.)
├── rag_system.py           # Main production pipeline
├── main.py                 # CLI entry point
├── requirements.txt        # All dependencies
├── .env.template           # Copy to .env and fill keys
│
├── ingest/
│   ├── pdf_loader.py       # PDF loading, OCR, image extraction
│   └── chunker.py          # Fixed / Semantic / Hybrid chunking
│
├── embeddings/
│   └── embedder.py         # HuggingFace BGE + CLIP embeddings
│
├── vectorstore/
│   └── chroma_store.py     # ChromaDB vector storage & search
│
├── retrieval/
│   └── advanced_retriever.py  # HyDE + Multi-Query + Parent-Child
│
├── llm/
│   └── clients.py          # Groq API + Ollama + Router
│
├── security/
│   └── security_manager.py # Jailbreak detection & prevention
│
├── utils/
│   └── logger.py           # Enterprise logging + audit trail
│
└── logs/
    ├── rag_system.log      # JSON structured logs
    └── audit.jsonl         # Security audit trail
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama (local models)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull mistral:latest     # Fast text model
ollama pull llava:13b          # Vision model (for images/charts)

# Start server
ollama serve
```

### 3. Set API Keys
```bash
cp .env.template .env
# Edit .env and add your GROQ_API_KEY
# Get free key from: https://console.groq.com
```

### 4. Run the Chatbot
```bash
# Ingest a PDF and start chatting
python main.py --pdf /path/to/your/document.pdf

# Interactive chat (no PDF)
python main.py

# Demo mode
python main.py --pdf doc.pdf --demo
```

---

## 🔧 Configuration

Edit `config.py` or set environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `text_model` | `BAAI/bge-base-en-v1.5` | HuggingFace embedding model |
| `groq_text_model` | `llama3-70b-8192` | Groq model for text |
| `ollama_vision_model` | `llava:13b` | Ollama model for images |
| `parent_chunk_size` | `2000` | Parent chunk chars |
| `child_chunk_size` | `500` | Child chunk chars |
| `use_hyde` | `True` | Enable HyDE retrieval |
| `use_multi_query` | `True` | Enable Multi-Query expansion |
| `chroma_persist_dir` | `./chroma_db` | ChromaDB storage path |

---

## 💰 Cost

| Component | Cost |
|-----------|------|
| Groq API | **Free** (6,000 req/day) |
| Ollama | **Free** (local) |
| HuggingFace embeddings | **Free** (open models) |
| ChromaDB | **Free** (open-source) |
| **Total** | **₹0** |

---

## 🔒 Security Features

- **Jailbreak Detection**: 15+ regex patterns for common attacks
- **Role-Play Attack Prevention**: Detects "pretend you are..." patterns  
- **Prompt Injection Defense**: XML-delimited secure prompt templates
- **Input Sanitization**: Removes LLM token markers (`<|>`, `[INST]`)
- **Output Validation**: Checks if model was successfully manipulated
- **Audit Logging**: Every query + security event logged to `logs/audit.jsonl`
