# ============================================================
# rag_system.py — Production Multimodal RAG System
# ============================================================
#
# Integrates all components:
#   1. EnterpriseDocumentProcessor  (PDF ingestion)
#   2. IntelligentChunker           (hybrid chunking)
#   3. MultimodalEmbedder           (HuggingFace BGE + CLIP)
#   4. EnterpriseVectorStore        (ChromaDB)
#   5. AdvancedParentChildRetriever (HyDE + multi-query)
#   6. LLMRouter                   (Groq + Ollama)
#   7. EnterpriseSecurityManager    (jailbreak prevention)
#
# Query Flow (6 steps):
#   User Query
#     → [1] Security validation
#     → [2] HyDE + Multi-Query expansion
#     → [3] ChromaDB child chunk search
#     → [4] Parent context retrieval
#     → [5] Secure prompt construction
#     → [6] Groq/Ollama generation + output validation
#     → Safe Response
# ============================================================

import os
import time
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import RAGConfig, CONFIG
from ingest.pdf_loader import EnterpriseDocumentProcessor, ProcessedDocument
from ingest.chunker import IntelligentChunker
from embeddings.embedder import MultimodalEmbedder
from vectorstore.chroma_store import EnterpriseVectorStore
from retrieval.advanced_retriever import AdvancedParentChildRetriever
from security.security_manager import EnterpriseSecurityManager, ThreatLevel
from llm.clients import GroqClient, OllamaClient, LLMRouter
from utils.logger import get_logger, AuditLogger, Timer

logger       = get_logger(__name__)
audit_logger = AuditLogger(log_dir="./logs")


class ProductionMultimodalRAG:
    """
    Complete production-ready Multimodal RAG system.

    Pipeline:
        PDF → Process → Chunk → Embed → ChromaDB
                                             ↓
        Query → Security → Retrieve → LLM → Response

    Supports:
        ✅ PDF documents with text, images, charts
        ✅ Parent-child retrieval (precision + context)
        ✅ HyDE + Multi-Query for maximum recall
        ✅ Groq (fast text) + Ollama (vision/private) routing
        ✅ Enterprise security & jailbreak prevention
        ✅ Comprehensive audit logging
    """

    def __init__(self, config: RAGConfig = CONFIG):
        self.config = config
        logger.info("=" * 60)
        logger.info("🚀 Initializing Production Multimodal RAG System")
        logger.info("=" * 60)

        # ── Document Processor ───────────────────────────────
        self.doc_processor = EnterpriseDocumentProcessor(
            extract_images = True,
            ocr_fallback   = True,
        )

        # ── Chunker ──────────────────────────────────────────
        self.chunker = IntelligentChunker(
            chunk_size    = config.chunking.child_chunk_size,
            chunk_overlap = config.chunking.child_chunk_overlap,
            strategy      = config.chunking.strategy,
        )

        # ── Embedder ─────────────────────────────────────────
        self.embedder = MultimodalEmbedder(
            text_model  = config.embedding.text_model,
            device      = config.embedding.device,
            enable_clip = True,
        )

        # ── Vector Store ─────────────────────────────────────
        self.vector_store = EnterpriseVectorStore(
            persist_directory = config.vectorstore.chroma_persist_dir,
            collection_name   = config.vectorstore.collection_name,
            distance_function = config.vectorstore.distance_function,
        )

        # ── LLM Clients ──────────────────────────────────────
        groq_client   = None
        ollama_client = None

        if config.groq_api_key:
            try:
                groq_client = GroqClient(
                    api_key   = config.groq_api_key,
                    model     = config.llm.groq_text_model,
                )
            except Exception as e:
                logger.warning(f"Groq init failed: {e}")

        try:
            ollama_client = OllamaClient(
                text_model   = config.llm.ollama_text_model,
                vision_model = config.llm.ollama_vision_model,
                base_url     = config.llm.ollama_base_url,
            )
        except Exception as e:
            logger.warning(f"Ollama init failed: {e}")

        # At least one must work
        if not groq_client and not ollama_client:
            logger.warning(
                "⚠️  No LLM configured. Set GROQ_API_KEY in .env "
                "or start Ollama locally."
            )

        self.llm_router = LLMRouter(
            groq_client   = groq_client,
            ollama_client = ollama_client,
        ) if (groq_client or ollama_client) else None

        # ── Retriever ─────────────────────────────────────────
        self.retriever = AdvancedParentChildRetriever(
            vector_store    = self.vector_store,
            embedder        = self.embedder,
            llm_client      = groq_client or ollama_client,
            use_hyde        = config.retrieval.use_hyde,
            use_multi_query = config.retrieval.use_multi_query,
            retrieval_count = config.retrieval.retrieval_count,
        ) if (groq_client or ollama_client) else None

        # ── Security ─────────────────────────────────────────
        self.security = EnterpriseSecurityManager(
            enable_logging       = config.security.enable_audit_logging,
            block_on_high_threat = config.security.block_on_high_threat,
            max_input_length     = config.security.max_input_length,
        )

        # ── State ─────────────────────────────────────────────
        self.indexed_documents:  Dict[str, ProcessedDocument] = {}
        self.performance_metrics: List[Dict] = []

        logger.info("=" * 60)
        logger.info("✅ System components initialized:")
        logger.info(f"   📄 Document Processor : ✅")
        logger.info(f"   ✂️  Intelligent Chunker : ✅")
        logger.info(f"   🔤 Embedder (HuggingFace): ✅")
        logger.info(f"   🗄️  ChromaDB            : ✅")
        logger.info(f"   🔍 Retriever (HyDE+MQ) : ✅")
        logger.info(f"   🔒 Security Manager    : ✅")
        logger.info(f"   ⚡ Groq Client         : {'✅' if groq_client else '⚠️  not configured'}")
        logger.info(f"   🦙 Ollama Client       : {'✅' if ollama_client else '⚠️  not running'}")
        logger.info("=" * 60)

    # ── Document Ingestion ───────────────────────────────────

    def ingest_document(self, pdf_path: str) -> str:
        """
        Full document ingestion pipeline.

        Step 1: Load & process PDF (text + images)
        Step 2: Hybrid intelligent chunking
        Step 3: Parent-child index building
        Step 4: HuggingFace embedding generation
        Step 5: ChromaDB storage with metadata
        Step 6: Parent chunk registration for retrieval

        Args:
            pdf_path: Path to PDF file

        Returns:
            document_id for future reference
        """
        logger.info(f"📥 Ingesting document: {pdf_path}")

        with Timer("full_ingestion") as t:
            # Step 1: Process PDF
            document = self.doc_processor.load_pdf(pdf_path)
            doc_id   = document.metadata.document_id
            logger.info(f"   Step 1 ✅ PDF processed | {document.metadata.total_pages} pages")

            # Step 2: Build parent-child chunks
            parent_chunks, child_chunks = self.chunker.build_parent_child_chunks(
                document     = document,
                parent_size  = self.config.chunking.parent_chunk_size,
                child_size   = self.config.chunking.child_chunk_size,
            )
            logger.info(
                f"   Step 2 ✅ Chunked | "
                f"{len(parent_chunks)} parents, {len(child_chunks)} children"
            )

            # Step 3: Embed child chunks (these go into vector DB)
            child_embeddings = self.embedder.embed_chunks(child_chunks)
            logger.info(
                f"   Step 3 ✅ Embedded | shape={child_embeddings.shape}"
            )

            # Step 4: Store child chunks in ChromaDB
            stats = self.vector_store.add_chunks(
                chunks     = child_chunks,
                embeddings = child_embeddings,
            )
            logger.info(
                f"   Step 4 ✅ Stored in ChromaDB | "
                f"{stats['successful_adds']} chunks added"
            )

            # Step 5: Register parents for retrieval context
            self.vector_store.register_parent_chunks(parent_chunks, child_chunks)
            logger.info(f"   Step 5 ✅ Parent-child index registered")

            # Step 6: Save document reference
            self.indexed_documents[doc_id] = document

        logger.info(f"🎉 Ingestion complete in {t.elapsed:.2f}s | doc_id={doc_id}")

        audit_logger.log_event(
            event_type   = "document_ingestion",
            user_input   = pdf_path,
            threat_level = "safe",
            details      = {
                "doc_id":        doc_id,
                "pages":         document.metadata.total_pages,
                "chunks":        stats["successful_adds"],
                "ingestion_time": t.elapsed,
            },
        )

        return doc_id

    # ── Query Pipeline ───────────────────────────────────────

    def query(
        self,
        user_query:             str,
        include_security_check: bool = True,
        include_images:         bool = True,
        max_tokens:             int  = 1024,
        explicit_image_path:    Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full RAG query pipeline with security.

        Step 1: Security validation & sanitization
        Step 2: Advanced retrieval (HyDE + Multi-Query + Parent-Child)
        Step 3: Context assembly from parent chunks
        Step 4: Secure prompt construction
        Step 5: LLM generation (Groq/Ollama)
        Step 6: Output security validation

        Args:
            user_query:             User's question
            include_security_check: Run jailbreak detection
            include_images:         Include image context if available
            max_tokens:             Max response tokens

        Returns:
            Dict with answer, sources, metadata, performance
        """
        query_start = time.time()
        logger.info(f"❓ Query: '{user_query[:80]}...'")

        # ── Step 1: Security Check ────────────────────────────
        query_text = user_query
        if include_security_check:
            sec_result = self.security.validate_user_input(user_query)

            if sec_result.blocked:
                logger.warning(
                    f"🔴 Query BLOCKED | threat={sec_result.threat_level.value}"
                )
                return {
                    "answer":           "I cannot process this request due to security concerns.",
                    "blocked":          True,
                    "threat_level":     sec_result.threat_level.value,
                    "detected_patterns": sec_result.detected_patterns,
                    "execution_time_s": round(time.time() - query_start, 3),
                }

            query_text = sec_result.sanitized_input
            logger.debug(
                f"✅ Security: {sec_result.threat_level.value} | "
                f"confidence={sec_result.confidence_score:.2f}"
            )

        # ── Step 2: Retrieval ─────────────────────────────────
        if not self.retriever:
            return {
                "answer": "LLM not configured. Please set GROQ_API_KEY or start Ollama.",
                "error":  "no_llm",
            }

        retrieval_results = self.retriever.retrieve(query_text)

        if not retrieval_results:
            return {
                "answer":           "I couldn't find relevant information to answer your question.",
                "retrieval_count":  0,
                "execution_time_s": round(time.time() - query_start, 3),
            }

        logger.info(f"   Retrieved {len(retrieval_results)} results")

        # ── Step 3: Context Assembly ──────────────────────────
        context_parts = []
        source_images: List[str] = []

        for i, result in enumerate(retrieval_results[:5]):   # top-5 contexts
            content = result.best_content
            chunk_id = result.metadata.get("chunk_id", f"chunk_{i}")
            page_str = str(result.metadata.get("page_numbers", "Unknown"))
            context_parts.append(
                f"[Source {i+1} | ID: {chunk_id[:8]} | Page: {page_str} | "
                f"Score: {result.similarity_score:.2f}]\n{content}"
            )

        combined_context = "\n\n---\n\n".join(context_parts)

        # Check for associated images (from indexed documents or explicit input)
        if explicit_image_path and os.path.exists(explicit_image_path):
            source_images.append(explicit_image_path)
        elif include_images:
            # Extract images ONLY from the pages where our retrieved chunks came from
            for r in retrieval_results[:5]:
                doc_id = r.metadata.get("parent_document_id")
                page_str = str(r.metadata.get("page_numbers", ""))
                
                if doc_id and doc_id in self.indexed_documents and page_str:
                    doc = self.indexed_documents[doc_id]
                    # Chunks might span multiple pages (e.g., "3,4")
                    page_nums = [int(p.strip()) for p in page_str.split(",") if p.strip().isdigit()]
                    
                    for page in doc.pages:
                        if page.page_number in page_nums and page.image_paths:
                            for img in page.image_paths:
                                if img not in source_images:
                                    source_images.append(img)
            
            # Limit total images to avoid incredibly slow local vision model processing
            source_images = source_images[:2]

        # ── Step 4: Secure Prompt ─────────────────────────────
        if include_security_check:
            secure_prompt = self.security.create_secure_prompt(
                user_query = query_text,
                context    = combined_context,
            )
        else:
            secure_prompt = f"Context:\n{combined_context}\n\nQuestion: {query_text}\n\nAnswer based solely on the context above:"

        # ── Step 5: LLM Generation ───────────────────────────
        llm_result = self.llm_router.generate(
            prompt     = secure_prompt,
            images     = source_images if source_images and include_images else None,
            max_tokens = max_tokens,
        )

        answer    = llm_result["answer"]
        model_used = llm_result["model_used"]

        # ── Step 6: Output Validation ─────────────────────────
        if include_security_check:
            if not self.security.validate_response(answer):
                logger.warning("⚠️  Response failed security validation")
                answer = "I cannot provide that information."

        execution_time = round(time.time() - query_start, 3)

        # ── Build Response ────────────────────────────────────
        response = {
            "answer":           answer,
            "model_used":       model_used,
            "retrieval_count":  len(retrieval_results),
            "execution_time_s": execution_time,
            "security_checked": include_security_check,
            "sources": [
                {
                    "chunk_id":        r.metadata.get("chunk_id", "")[:12],
                    "similarity_score": round(r.similarity_score, 3),
                    "confidence_score": round(r.confidence_score, 3),
                    "retrieval_method": r.retrieval_method,
                    "content_preview":  r.child_content[:120] + "...",
                }
                for r in retrieval_results[:5]
            ],
        }

        # Audit log the query
        audit_logger.log_query(
            query          = user_query,
            retrieved_count = len(retrieval_results),
            execution_time = execution_time,
            model_used     = model_used,
        )

        logger.info(f"✅ Query answered in {execution_time}s via {model_used}")
        return response

    # ── System Status ─────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Return comprehensive system health status."""
        vector_stats = self.vector_store.get_stats()
        return {
            "system_health":     "healthy",
            "indexed_documents": len(self.indexed_documents),
            "vector_store":      vector_stats,
            "total_queries":     len(self.performance_metrics),
            "components": {
                "document_processor": "active",
                "intelligent_chunker": "active",
                "embedder":           f"active ({self.config.embedding.text_model})",
                "vector_store":       "active (ChromaDB)",
                "security_manager":   "active",
                "llm_router":         "active" if self.llm_router else "not configured",
            },
            "config": {
                "chunking_strategy": self.config.chunking.strategy,
                "parent_chunk_size": self.config.chunking.parent_chunk_size,
                "child_chunk_size":  self.config.chunking.child_chunk_size,
                "hyde_enabled":      self.config.retrieval.use_hyde,
                "multi_query":       self.config.retrieval.use_multi_query,
            },
        }
