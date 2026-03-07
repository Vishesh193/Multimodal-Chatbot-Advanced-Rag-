# ============================================================
# config.py — Central Configuration for Multimodal RAG System
# ============================================================

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
# API Keys (set in .env file)
# ─────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN          = os.getenv("HF_TOKEN", "")

# ─────────────────────────────────────────────
# LLM Model Config
# ─────────────────────────────────────────────
@dataclass
class LLMConfig:
    # Groq models (text)
    groq_text_model: str        = "llama-3.3-70b-versatile"
    groq_fast_model: str        = "llama-3.1-8b-instant"
    groq_mixtral_model: str     = "mixtral-8x7b-32768"

    # Ollama models (local, vision-capable)
    ollama_vision_model: str    = "llava:latest"
    ollama_text_model: str      = "mistral:latest"
    ollama_base_url: str        = "http://localhost:11434"

    # Gemini (fallback)
    gemini_model: str           = "gemini-2.0-flash"

    # Generation params
    max_tokens: int             = 1024
    temperature: float          = 0.1
    top_p: float                = 0.9


# ─────────────────────────────────────────────
# Embedding Config
# ─────────────────────────────────────────────
@dataclass
class EmbeddingConfig:
    # Text embedding model (HuggingFace)
    text_model: str             = "BAAI/bge-base-en-v1.5"
    # Alternative: "sentence-transformers/all-MiniLM-L6-v2"
    # Alternative: "sentence-transformers/all-mpnet-base-v2"

    # Image/Multimodal embedding (CLIP)
    clip_model: str             = "openai/clip-vit-base-patch32"
    # Alternative: "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    # Embedding dimensions
    text_dim: int               = 768
    image_dim: int              = 512

    # Device
    device: str                 = "cpu"   # "cuda" if GPU available
    batch_size: int             = 32


# ─────────────────────────────────────────────
# Chunking Config
# ─────────────────────────────────────────────
@dataclass
class ChunkingConfig:
    # Parent chunks (for context during generation)
    parent_chunk_size: int      = 2000  
    parent_chunk_overlap: int   = 200

    # Child chunks (for retrieval/search)
    child_chunk_size: int       = 300
    child_chunk_overlap: int    = 50

    # General
    min_chunk_size: int         = 50
    max_chunk_size: int         = 2000
    strategy: str               = "hybrid"   # fixed | semantic | hybrid
    tokenizer_model: str        = "gpt-3.5-turbo"


# ─────────────────────────────────────────────
# Vector Store Config
# ─────────────────────────────────────────────
@dataclass
class VectorStoreConfig:
    # ChromaDB
    chroma_persist_dir: str     = "./chroma_db"
    collection_name: str        = "multimodal_rag_docs"
    distance_function: str      = "cosine"

    # FAISS (alternative)
    faiss_index_path: str       = "./faiss_index"

    # Search params
    n_results: int              = 3
    score_threshold: float      = 0.3


# ─────────────────────────────────────────────
# Retrieval Config
# ─────────────────────────────────────────────
@dataclass
class RetrievalConfig:
    # HyDE
    use_hyde: bool              = True
    hyde_max_tokens: int        = 100

    # Multi-Query Expansion
    use_multi_query: bool       = True
    num_query_variations: int   = 2

    # Parent-Child
    use_parent_child: bool      = True

    # Reranking
    use_reranker: bool          = False  # Enable if cross-encoder available
    reranker_model: str         = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Result fusion
    retrieval_count: int        = 6
    use_mmr: bool               = True   # Maximal Marginal Relevance reranking
    mmr_lambda: float           = 0.5    # Diversity vs Relevance balance


# ─────────────────────────────────────────────
# Security Config
# ─────────────────────────────────────────────
@dataclass
class SecurityConfig:
    enable_jailbreak_detection: bool    = True
    enable_input_sanitization: bool     = True
    enable_output_validation: bool      = True
    enable_audit_logging: bool          = True
    max_input_length: int               = 4000
    block_on_high_threat: bool          = True


# ─────────────────────────────────────────────
# Logging Config
# ─────────────────────────────────────────────
@dataclass
class LoggingConfig:
    log_level: str              = "INFO"
    log_dir: str                = "./logs"
    log_file: str               = "rag_system.log"
    max_bytes: int              = 10_000_000   # 10 MB
    backup_count: int           = 5
    enable_console: bool        = True
    enable_file: bool           = True
    format: str                 = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


# ─────────────────────────────────────────────
# Master Config
# ─────────────────────────────────────────────
@dataclass
class RAGConfig:
    llm: LLMConfig              = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig  = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig    = field(default_factory=ChunkingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig  = field(default_factory=RetrievalConfig)
    security: SecurityConfig    = field(default_factory=SecurityConfig)
    logging: LoggingConfig      = field(default_factory=LoggingConfig)

    # Runtime
    groq_api_key: str           = field(default_factory=lambda: GROQ_API_KEY)
    gemini_api_key: str         = field(default_factory=lambda: GEMINI_API_KEY)


# Singleton config instance
CONFIG = RAGConfig()
