# ============================================================
# vectorstore/chroma_store.py — ChromaDB Vector Storage
# ============================================================
#
# Features:
#   • Persistent ChromaDB with cosine similarity
#   • Batch ingestion with error recovery
#   • Rich metadata filtering
#   • Separate collections for text & images
#   • Collection health stats
#   • FAISS fallback option
# ============================================================

import uuid
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger, Timer

logger = get_logger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not installed. Run: pip install chromadb")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


# ─────────────────────────────────────────────
# Search Result
# ─────────────────────────────────────────────
class SearchResult:
    """Single vector search result."""
    def __init__(
        self,
        chunk_id:         str,
        content:          str,
        metadata:         Dict[str, Any],
        similarity_score: float,
        distance:         float,
    ):
        self.chunk_id         = chunk_id
        self.content          = content
        self.metadata         = metadata
        self.similarity_score = similarity_score
        self.distance         = distance

    def __repr__(self):
        return (
            f"SearchResult(score={self.similarity_score:.3f}, "
            f"id={self.chunk_id[:8]}..., "
            f"preview='{self.content[:60]}...')"
        )


# ─────────────────────────────────────────────
# Enterprise ChromaDB Vector Store
# ─────────────────────────────────────────────
class EnterpriseVectorStore:
    """
    Production-grade vector store backed by ChromaDB.

    Architecture:
        • text_collection  : text chunk embeddings (768d BGE)
        • image_collection : image embeddings (512d CLIP) [optional]
        • parent_store     : in-memory parent chunk lookup

    Why ChromaDB:
        ✅ Local persistence (SQLite → PostgreSQL)
        ✅ Rich metadata filtering
        ✅ Python-native API
        ✅ Free & open-source
        ✅ LangChain/LlamaIndex compatible
    """

    def __init__(
        self,
        persist_directory: str  = "./chroma_db",
        collection_name:   str  = "multimodal_rag_docs",
        distance_function: str  = "cosine",
        embedding_model:   str  = "BAAI/bge-base-en-v1.5",
    ):
        if not CHROMA_AVAILABLE:
            raise ImportError("Install chromadb: pip install chromadb")

        self.persist_directory = persist_directory
        self.collection_name   = collection_name
        self.distance_function = distance_function
        self.embedding_model   = embedding_model

        # In-memory parent chunk store (keyed by parent_id)
        self.parent_store:    Dict[str, Any] = {}
        # Child → Parent mapping
        self.child_to_parent: Dict[str, str] = {}

        # Initialize ChromaDB
        logger.info(f"🗄️  Initializing ChromaDB at: {persist_directory}")
        self.client = chromadb.PersistentClient(
            path     = persist_directory,
            settings = Settings(
                anonymized_telemetry = False,
                allow_reset          = True,
            ),
        )
        self.collection = self._get_or_create_collection(collection_name)

        # Initialize BM25 Sparse Index
        self.bm25_docs = []
        self.bm25_store = None
        self._init_bm25()

        logger.info(f"✅ Vector store ready")
        logger.info(f"   Collection : {collection_name}")
        logger.info(f"   Distance   : {distance_function}")
        logger.info(f"   Persist    : {persist_directory}")
        logger.info(f"   Doc count  : {self.collection.count()}")

    # ── Collection Management ────────────────────────────────

    def _init_bm25(self):
        """Initialize in-memory BM25 index from ChromaDB documents."""
        if not BM25_AVAILABLE:
            return
        logger.info("   Initializing BM25 index...")
        all_docs = self.collection.get()
        if all_docs and all_docs.get("ids") and all_docs.get("documents"):
            # Create list of tuples: (id, doc, meta)
            self.bm25_docs = list(zip(all_docs["ids"], all_docs["documents"], all_docs["metadatas"]))
            tokenized = [doc.lower().split() for doc in all_docs["documents"]]
            self.bm25_store = BM25Okapi(tokenized)
            logger.info(f"   ✅ BM25 ready. Docs indexed: {len(self.bm25_docs)}")

    def _get_or_create_collection(self, name: str):
        """Get existing or create new ChromaDB collection."""
        try:
            collection = self.client.get_or_create_collection(
                name     = name,
                metadata = {"hnsw:space": self.distance_function},
            )
            logger.info(f"   📂 Retrieved/Created collection: {name}")
            return collection
        except Exception as e:
            logger.error(f"   ❌ Failed to get/create collection: {e}")
            raise

    # ── Add Chunks ───────────────────────────────────────────

    def add_chunks(
        self,
        chunks,                       # List[DocumentChunk]
        embeddings: np.ndarray,
        batch_size: int = 100,
    ) -> Dict[str, int]:
        """
        Add document chunks with precomputed embeddings to ChromaDB.

        Args:
            chunks:     List of DocumentChunk objects
            embeddings: numpy array (n_chunks, embedding_dim)
            batch_size: Chunks per batch for large documents

        Returns:
            Stats dict: {total, successful, failed, batches}
        """
        assert len(chunks) == len(embeddings), "chunks and embeddings must match"

        logger.info(f"📥 Adding {len(chunks)} chunks (batch={batch_size})")

        stats = {
            "total_chunks":   len(chunks),
            "successful_adds": 0,
            "failed_adds":    0,
            "batches_processed": 0,
        }

        for batch_start in range(0, len(chunks), batch_size):
            batch_end      = min(batch_start + batch_size, len(chunks))
            batch_chunks   = chunks[batch_start:batch_end]
            batch_embeds   = embeddings[batch_start:batch_end]
            batch_num      = batch_start // batch_size + 1
            total_batches  = (len(chunks) + batch_size - 1) // batch_size

            logger.debug(f"   Batch {batch_num}/{total_batches}")

            try:
                documents  = []
                metadatas  = []
                ids        = []
                embed_list = []

                for chunk, emb in zip(batch_chunks, batch_embeds):
                    meta = chunk.metadata
                    # ChromaDB requires all metadata values as strings/int/float/bool
                    chroma_meta = {
                        "chunk_id":           meta.chunk_id,
                        "parent_document_id": meta.parent_document_id,
                        "parent_chunk_id":    meta.parent_chunk_id or "",
                        "chunk_index":        str(meta.chunk_index),
                        "chunk_type":         meta.chunk_type,
                        "strategy":           meta.strategy,
                        "start_char":         str(meta.start_char),
                        "end_char":           str(meta.end_char),
                        "token_count":        str(meta.token_count),
                        "page_numbers":       ",".join(map(str, meta.page_numbers)),
                        "semantic_score":     str(round(meta.semantic_score, 4)),
                        "is_parent":          str(meta.is_parent),
                    }

                    documents.append(chunk.content)
                    metadatas.append(chroma_meta)
                    ids.append(meta.chunk_id)
                    embed_list.append(emb.tolist())

                self.collection.add(
                    documents  = documents,
                    embeddings = embed_list,
                    metadatas  = metadatas,
                    ids        = ids,
                )

                stats["successful_adds"]    += len(batch_chunks)
                stats["batches_processed"]  += 1

            except Exception as e:
                logger.error(f"❌ Batch {batch_num} failed: {e}")
                stats["failed_adds"] += len(batch_chunks)

        logger.info(f"✅ Vector store updated:")
        logger.info(f"   Successful : {stats['successful_adds']}")
        logger.info(f"   Failed     : {stats['failed_adds']}")
        logger.info(f"   Total docs : {self.collection.count()}")
        
        # Re-initialize BM25 index after adds
        self._init_bm25()
        
        return stats

    def register_parent_chunks(self, parent_chunks, child_chunks) -> None:
        """
        Register parent chunks in memory for parent-child retrieval.
        Also builds child → parent ID mapping.
        """
        for parent in parent_chunks:
            self.parent_store[parent.metadata.chunk_id] = parent

        for child in child_chunks:
            if child.metadata.parent_chunk_id:
                self.child_to_parent[child.metadata.chunk_id] = \
                    child.metadata.parent_chunk_id

        logger.info(
            f"🔗 Registered {len(parent_chunks)} parents, "
            f"{len(child_chunks)} children"
        )

    # ── Similarity Search ────────────────────────────────────

    def similarity_search(
        self,
        query_embedding:  np.ndarray,
        n_results:        int                  = 5,
        filter_criteria:  Optional[Dict]       = None,
        include_distances: bool                = True,
    ) -> List[SearchResult]:
        """
        Perform cosine similarity search.

        Args:
            query_embedding:  1D numpy array
            n_results:        Number of results to return
            filter_criteria:  ChromaDB 'where' filter dict
            include_distances: Return distance scores

        Returns:
            List of SearchResult objects, sorted by similarity
        """
        logger.debug(f"🔍 Similarity search (top-{n_results})")

        try:
            search_params: Dict[str, Any] = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results":        min(n_results, self.collection.count()),
                "include":          ["documents", "metadatas", "distances"],
            }
            if filter_criteria:
                search_params["where"] = filter_criteria

            with Timer("vector_search") as t:
                raw = self.collection.query(**search_params)

            results = []
            docs   = raw["documents"][0]
            metas  = raw["metadatas"][0]
            dists  = raw["distances"][0] if include_distances else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                # Convert distance to similarity score (cosine: 1 - distance)
                sim_score = max(0.0, 1.0 - dist)
                results.append(SearchResult(
                    chunk_id         = meta.get("chunk_id", ""),
                    content          = doc,
                    metadata         = meta,
                    similarity_score = sim_score,
                    distance         = dist,
                ))

            logger.debug(
                f"✅ Found {len(results)} results in {t.elapsed*1000:.1f}ms"
            )
            return results

        except Exception as e:
            logger.error(f"❌ Similarity search failed: {e}")
            return []

    def bm25_search(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[SearchResult]:
        """Perform BM25 sparse search."""
        if not getattr(self, "bm25_store", None) or not getattr(self, "bm25_docs", None):
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25_store.get_scores(tokenized_query)
        # Get top-N indices
        top_n_indices = np.argsort(scores)[::-1][:n_results]

        results = []
        for idx in top_n_indices:
            score = scores[idx]
            if score <= 0:
                continue
            
            chunk_id, content, metadata = self.bm25_docs[idx]
            # Normalize BM25 score roughly (0 to 1)
            sim_score = min(1.0, score / 10.0) 
            
            results.append(SearchResult(
                chunk_id         = chunk_id,
                content          = content,
                metadata         = metadata,
                similarity_score = sim_score,
                distance         = 1.0 - sim_score,
            ))
            
        logger.debug(f"✅ Found {len(results)} sparse results")
        return results

    def get_parent_for_child(self, child_id: str) -> Optional[Any]:
        """
        Retrieve parent chunk for a given child chunk ID.
        Core of Parent-Child retrieval architecture.
        """
        parent_id = self.child_to_parent.get(child_id)
        if parent_id:
            return self.parent_store.get(parent_id)
        return None

    # ── Stats & Health ───────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return collection health metrics."""
        try:
            count = self.collection.count()
            sample = self.collection.get(
                limit   = min(100, count),
                include = ["metadatas"],
            )

            stats: Dict[str, Any] = {
                "total_documents":  count,
                "collection_name":  self.collection_name,
                "embedding_model":  self.embedding_model,
                "distance_fn":      self.distance_function,
                "parents_in_memory": len(self.parent_store),
            }

            if sample["metadatas"]:
                doc_ids = set(
                    m.get("parent_document_id", "")
                    for m in sample["metadatas"]
                )
                stats["unique_documents"] = len(doc_ids)

            return stats
        except Exception as e:
            return {"error": str(e)}

    def clear_collection(self) -> bool:
        """Clear all documents from collection."""
        try:
            all_docs = self.collection.get()
            if all_docs["ids"]:
                self.collection.delete(ids=all_docs["ids"])
            self.parent_store.clear()
            self.child_to_parent.clear()
            logger.info(f"🗑️  Collection cleared: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Clear failed: {e}")
            return False
