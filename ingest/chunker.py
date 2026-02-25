# ============================================================
# ingest/chunker.py — Intelligent Text Chunking Strategies
# ============================================================
#
# Strategies:
#   • Fixed-size    : Simple character/token-aware sliding window
#   • Semantic      : Paragraph & sentence boundary aware
#   • Hybrid        : Semantic structure + fixed-size safety net
#
# Architecture:
#   • Parent chunks  (large  ~2000 chars) → for LLM context
#   • Child chunks   (small  ~500  chars) → for vector search
# ============================================================

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from ingest.pdf_loader import ProcessedDocument, DocumentMetadata
from utils.logger import get_logger, Timer

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────
@dataclass
class ChunkMetadata:
    """Rich metadata attached to every chunk."""
    chunk_id:           str     = field(default_factory=lambda: str(uuid.uuid4()))
    parent_document_id: str     = ""
    parent_chunk_id:    str     = ""      # For child → parent linking
    chunk_index:        int     = 0
    chunk_type:         str     = "content"   # content | header | table | image_caption
    strategy:           str     = "hybrid"
    start_char:         int     = 0
    end_char:           int     = 0
    token_count:        int     = 0
    page_numbers:       List[int] = field(default_factory=list)
    semantic_score:     float   = 0.0
    is_parent:          bool    = False
    child_ids:          List[str] = field(default_factory=list)


@dataclass
class DocumentChunk:
    """A single text chunk with full metadata."""
    content:    str
    metadata:   ChunkMetadata


# ─────────────────────────────────────────────
# Intelligent Chunker
# ─────────────────────────────────────────────
class IntelligentChunker:
    """
    Advanced chunking system with three strategies.

    Optimal Chunk Size Formula:
        optimal_size = min(max_tokens, context_window * overlap_ratio)

    Overlap Strategy:
        chunk[i].end_tokens = chunk[i+1].start_tokens * overlap_percentage
    """

    def __init__(
        self,
        chunk_size:     int  = 1000,
        chunk_overlap:  int  = 200,
        min_chunk_size: int  = 100,
        max_chunk_size: int  = 4000,
        strategy:       str  = "hybrid",
    ):
        self.chunk_size     = chunk_size
        self.chunk_overlap  = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.strategy       = strategy

        # Token counter
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            except Exception:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.tokenizer = None

        logger.info(f"✂️  IntelligentChunker initialized")
        logger.info(f"   Strategy   : {strategy}")
        logger.info(f"   Chunk size : {chunk_size} chars")
        logger.info(f"   Overlap    : {chunk_overlap} chars")
        logger.info(f"   Tiktoken   : {'✅' if TIKTOKEN_AVAILABLE else '⚠️  using char count'}")

    # ── Public API ──────────────────────────────────────────

    def chunk_document(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """
        Chunk a ProcessedDocument using the configured strategy.

        Returns:
            List of DocumentChunk objects with metadata.
        """
        logger.info(f"✂️  Chunking: {document.metadata.filename}")
        logger.info(f"   Strategy : {self.strategy}")

        with Timer("chunking") as t:
            if self.strategy == "fixed":
                chunks = self._fixed_size_chunking(document)
            elif self.strategy == "semantic":
                chunks = self._semantic_chunking(document)
            elif self.strategy == "hybrid":
                chunks = self._hybrid_chunking(document)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            chunks = self._post_process(chunks, document)

        if not chunks:
            logger.warning(f"⚠️ No chunks generated for document {document.metadata.filename}")
            return []

        avg_len = sum(len(c.content) for c in chunks) / max(len(chunks), 1)
        logger.info(f"✅ Chunked in {t.elapsed:.2f}s → {len(chunks)} chunks")
        logger.info(f"   Avg size : {avg_len:.0f} chars")
        logger.info(f"   Range    : {min(len(c.content) for c in chunks)}–"
                    f"{max(len(c.content) for c in chunks)} chars")
        return chunks

    def build_parent_child_chunks(
        self,
        document: ProcessedDocument,
        parent_size: int = 2000,
        child_size:  int = 500,
    ) -> Tuple[List[DocumentChunk], List[DocumentChunk]]:
        """
        Build Parent-Child chunk hierarchy.

        Parent chunks → large, for LLM context (coherence)
        Child chunks  → small, for vector search (precision)

        Each child knows its parent_id so we can retrieve
        the full parent context after finding a child match.
        """
        logger.info(f"🔗 Building parent-child index for: {document.metadata.filename}")

        # ── Parent Chunks ────────────────────────────────────
        parent_chunker = IntelligentChunker(
            chunk_size    = parent_size,
            chunk_overlap = 200,
            strategy      = "hybrid",
        )
        parent_chunks_raw = parent_chunker.chunk_document(document)

        parent_chunks: List[DocumentChunk] = []
        all_child_chunks: List[DocumentChunk] = []

        for p_chunk in parent_chunks_raw:
            # Mark as parent
            p_chunk.metadata.is_parent = True

            # ── Child Chunks from each Parent ─────────────────
            child_chunker = IntelligentChunker(
                chunk_size    = child_size,
                chunk_overlap = 100,
                strategy      = "semantic",
            )

            # Create a temporary mini-document for child chunking
            temp_doc = ProcessedDocument(
                content  = p_chunk.content,
                metadata = DocumentMetadata(
                    document_id = document.metadata.document_id,
                    filename    = document.metadata.filename,
                ),
            )
            children = child_chunker.chunk_document(temp_doc)

            for child in children:
                # Link child → parent
                child.metadata.parent_chunk_id    = p_chunk.metadata.chunk_id
                child.metadata.parent_document_id = document.metadata.document_id
                child.metadata.is_parent           = False

                # Register child in parent's child list
                p_chunk.metadata.child_ids.append(child.metadata.chunk_id)
                all_child_chunks.append(child)

            parent_chunks.append(p_chunk)

        logger.info(f"✅ Parent-child index built:")
        logger.info(f"   Parents  : {len(parent_chunks)}")
        logger.info(f"   Children : {len(all_child_chunks)}")
        logger.info(
            f"   Avg children/parent: "
            f"{len(all_child_chunks)/max(len(parent_chunks),1):.1f}"
        )
        return parent_chunks, all_child_chunks

    # ── Strategy: Fixed Size ─────────────────────────────────

    def _fixed_size_chunking(
        self, document: ProcessedDocument
    ) -> List[DocumentChunk]:
        """Simple sliding window with sentence-boundary snapping."""
        chunks = []
        text   = document.content
        start  = 0
        idx    = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Snap to sentence boundary
            if end < len(text):
                search_start = max(end - 100, start)
                segment      = text[search_start:end + 50]
                sentences    = segment.split(". ")
                if len(sentences) > 1:
                    snapped_end = search_start + len(". ".join(sentences[:-1])) + 1
                    if snapped_end > start:
                        end = snapped_end

            chunk_text = text[start:end].strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(DocumentChunk(
                    content  = chunk_text,
                    metadata = ChunkMetadata(
                        parent_document_id = document.metadata.document_id,
                        chunk_index        = idx,
                        strategy           = "fixed",
                        start_char         = start,
                        end_char           = end,
                        token_count        = self._count_tokens(chunk_text),
                    ),
                ))
                idx += 1

            old_start = start
            start = max(end - self.chunk_overlap, old_start + 1) # Force progression

        return chunks

    # ── Strategy: Semantic ───────────────────────────────────

    def _semantic_chunking(
        self, document: ProcessedDocument
    ) -> List[DocumentChunk]:
        """
        Paragraph and section boundary-aware chunking.
        Splits on double newlines, headers, then merges small paragraphs.
        Uses a robust forward-iterating accumulator.
        """
        text       = document.content
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        chunks     = []
        
        if not paragraphs:
            return chunks
            
        current_paras = []
        current_len   = 0
        idx           = 0
        pos           = 0

        for para in paragraphs:
            para_len = len(para)

            if current_paras and current_len + para_len + 2 > self.chunk_size:
                # Flush the current chunk
                chunk_text = "\n\n".join(current_paras)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(DocumentChunk(
                        content  = chunk_text,
                        metadata = ChunkMetadata(
                            parent_document_id = document.metadata.document_id,
                            chunk_index        = idx,
                            strategy           = "semantic",
                            start_char         = pos - len(chunk_text),
                            end_char           = pos,
                            token_count        = self._count_tokens(chunk_text),
                            semantic_score     = self._score_semantic_completeness(chunk_text),
                        ),
                    ))
                    idx += 1
                
                # Setup next chunk using simple overlap
                if len(current_paras[-1]) <= self.chunk_overlap:
                    current_paras = [current_paras[-1], para]
                    current_len = len(current_paras[0]) + 2 + para_len
                else: 
                    current_paras = [para]
                    current_len = para_len
            else:
                current_paras.append(para)
                current_len += para_len + (2 if len(current_paras) > 1 else 0)
                
            pos += para_len + 2

        # Flush remaining
        if current_paras:
            chunk_text = "\n\n".join(current_paras)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(DocumentChunk(
                    content  = chunk_text,
                    metadata = ChunkMetadata(
                        parent_document_id = document.metadata.document_id,
                        chunk_index        = idx,
                        strategy           = "semantic",
                        start_char         = pos - len(chunk_text),
                        end_char           = pos,
                        token_count        = self._count_tokens(chunk_text),
                        semantic_score     = self._score_semantic_completeness(chunk_text),
                    ),
                ))

        return chunks

    # ── Strategy: Hybrid ─────────────────────────────────────

    def _hybrid_chunking(
        self, document: ProcessedDocument
    ) -> List[DocumentChunk]:
        """
        Hybrid = semantic chunking first, then apply fixed-size
        safety net to any oversized chunks.
        """
        semantic_chunks = self._semantic_chunking(document)
        final_chunks    = []

        for chunk in semantic_chunks:
            if len(chunk.content) > self.max_chunk_size:
                # Split oversized chunk with fixed strategy
                sub_doc = ProcessedDocument(
                    content  = chunk.content,
                    metadata = document.metadata,
                )
                sub_chunks = self._fixed_size_chunking(sub_doc)
                for sc in sub_chunks:
                    sc.metadata.strategy = "hybrid"
                final_chunks.extend(sub_chunks)
            else:
                chunk.metadata.strategy = "hybrid"
                final_chunks.append(chunk)

        return final_chunks

    # ── Post-Processing ──────────────────────────────────────

    def _post_process(
        self, chunks: List[DocumentChunk], document: ProcessedDocument
    ) -> List[DocumentChunk]:
        """Re-index, assign page numbers, filter empties."""
        clean = []
        for i, chunk in enumerate(chunks):
            if len(chunk.content.strip()) < self.min_chunk_size:
                continue
            chunk.metadata.chunk_index        = i
            chunk.metadata.parent_document_id = document.metadata.document_id
            chunk.metadata.page_numbers       = self._find_page_numbers(
                chunk, document
            )
            clean.append(chunk)
        return clean

    def _find_page_numbers(
        self, chunk: DocumentChunk, document: ProcessedDocument
    ) -> List[int]:
        """Determine which pages a chunk spans."""
        pages = []
        pos   = 0
        for page in document.pages:
            page_start = pos
            page_end   = pos + len(page.cleaned_text) + 2
            if (chunk.metadata.start_char < page_end and
                    chunk.metadata.end_char > page_start):
                pages.append(page.page_number)
            pos = page_end
        return pages if pages else [1]

    # ── Helpers ──────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        return len(text) // 4   # ~4 chars/token fallback

    def _get_overlap_text(self, text: str) -> str:
        """Return the last overlap_size chars for context continuity."""
        if len(text) <= self.chunk_overlap:
            return text
        # Snap to word boundary
        overlap = text[-self.chunk_overlap:]
        space   = overlap.find(" ")
        return overlap[space + 1:] if space != -1 else overlap

    def _score_semantic_completeness(self, text: str) -> float:
        """
        Score how semantically complete a chunk is (0–1).
        Rewards: ends with sentence, good length, information density.
        """
        if not text:
            return 0.0

        length = len(text)
        words  = len(text.split())

        # Completeness: ends with proper sentence ending
        completeness = 1.0 if text.rstrip().endswith((".", "!", "?")) else 0.7

        # Length score
        length_score = min(length / self.chunk_size, 1.0)

        # Density: words per 5 chars
        density = min(words / (length / 5), 1.0) if length > 0 else 0.0

        return (completeness + length_score + density) / 3
