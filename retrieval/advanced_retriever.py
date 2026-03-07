# ============================================================
# retrieval/advanced_retriever.py — Advanced RAG Retrieval
# ============================================================
#
# Techniques:
#   1. Parent-Child Retrieval  — search children, return parent context
#   2. HyDE                   — Hypothetical Document Embeddings
#   3. Multi-Query Expansion   — generate N query variations
#   4. Result Fusion & Reranking
#
# Flow:
#   User Query
#     → HyDE (generate hypothetical answer → embed)
#     → Multi-Query (3-5 variations)
#     → All queries → FAISS/ChromaDB child search
#     → Child IDs → fetch Parent context
#     → Rank & deduplicate by confidence
#     → Return top-K enriched results
# ============================================================

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger, Timer
from vectorstore.chroma_store import EnterpriseVectorStore, SearchResult
from config import CONFIG

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────
@dataclass
class RetrievalResult:
    """Enriched retrieval result with parent-child context."""
    child_content:    str
    parent_content:   str
    similarity_score: float
    confidence_score: float
    retrieval_method: str           # "vector" | "hyde" | "multi_query"
    metadata:         Dict[str, Any] = field(default_factory=dict)
    query_used:       str            = ""

    @property
    def best_content(self) -> str:
        """Return parent content if available, else child."""
        return self.parent_content if self.parent_content else self.child_content


# ─────────────────────────────────────────────
# Advanced Parent-Child Retriever
# ─────────────────────────────────────────────
class AdvancedParentChildRetriever:
    """
    Enterprise-grade retriever combining:

    1. HyDE (Hypothetical Document Embeddings)
       ─ LLM generates a hypothetical answer
       ─ We embed THAT instead of the raw question
       ─ Better semantic match with stored document chunks
       ─ Formula: HyDE_Query = LLM("Answer: " + original_query)

    2. Multi-Query Expansion
       ─ LLM generates 3-5 rephrased queries
       ─ Each variation captures different aspects
       ─ Results merged + deduplicated
       ─ Maximizes recall

    3. Parent-Child Retrieval
       ─ Small child chunks in vector DB (precision)
       ─ Large parent chunks for LLM context (coherence)
       ─ Child match → retrieve parent → send to LLM
    """

    def __init__(
        self,
        vector_store:      EnterpriseVectorStore,
        embedder,                           # MultimodalEmbedder
        llm_client,                         # GroqClient | OllamaClient
        parent_chunk_size: int  = 2000,
        child_chunk_size:  int  = 500,
        retrieval_count:   int  = 5,
        use_hyde:          bool = True,
        use_multi_query:   bool = True,
    ):
        self.vector_store     = vector_store
        self.embedder         = embedder
        self.llm_client       = llm_client
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size  = child_chunk_size
        self.retrieval_count   = retrieval_count
        self.use_hyde          = use_hyde
        self.use_multi_query   = use_multi_query
        self.use_mmr           = getattr(CONFIG.retrieval, "use_mmr", True)
        self.mmr_lambda        = getattr(CONFIG.retrieval, "mmr_lambda", 0.5)

        logger.info(f"🔍 AdvancedParentChildRetriever initialized")
        logger.info(f"   HyDE         : {'✅' if use_hyde else '❌'}")
        logger.info(f"   Multi-Query  : {'✅' if use_multi_query else '❌'}")
        logger.info(f"   Parent size  : {parent_chunk_size} chars")
        logger.info(f"   Child size   : {child_chunk_size} chars")
        logger.info(f"   Top-K        : {retrieval_count}")

    # ── Main Retrieval ───────────────────────────────────────

    def retrieve(
        self,
        query:           str,
        use_hyde:        Optional[bool] = None,
        use_multi_query: Optional[bool] = None,
    ) -> List[RetrievalResult]:
        """
        Full advanced retrieval pipeline.

        Args:
            query:           User's original question
            use_hyde:        Override instance setting
            use_multi_query: Override instance setting

        Returns:
            Ranked list of RetrievalResult objects
        """
        use_hyde        = use_hyde        if use_hyde is not None        else self.use_hyde
        use_multi_query = use_multi_query if use_multi_query is not None else self.use_multi_query

        logger.info(f"🔍 Advanced retrieval: '{query[:80]}...'")
        start_time = time.time()

        queries_to_process: List[Tuple[str, str]] = []   # (query_text, method_label)

        # ── 1. HyDE ─────────────────────────────────────────
        if use_hyde:
            try:
                hyde_query = self._generate_hyde_query(query)
                queries_to_process.append((hyde_query, "hyde"))
                logger.debug(f"   HyDE query: {hyde_query[:100]}...")
            except Exception as e:
                logger.warning(f"HyDE failed: {e}. Skipping.")

        # ── 2. Multi-Query ───────────────────────────────────
        if use_multi_query:
            try:
                variations = self._generate_multi_queries(query)
                for v in variations:
                    queries_to_process.append((v, "multi_query"))
                logger.debug(f"   Multi-query variations: {len(variations)}")
            except Exception as e:
                logger.warning(f"Multi-query failed: {e}. Skipping.")

        # ── Always include original query ────────────────────
        queries_to_process.append((query, "vector"))

        # ── 3. Retrieve for each query ────────────────────────
        # Collect ALL results from ALL methods in a single pool for RRF
        results_pool:  Dict[str, SearchResult] = {}
        dense_rankings:  List[List[str]] = []   # List of chunk_ids for each query's dense search
        sparse_rankings: List[List[str]] = []   # List of chunk_ids for each query's sparse search

        for query_text, method in queries_to_process:
            try:
                query_embedding = self.embedder.embed_query(query_text)
                
                # Dense Search (Vector)
                dense_results = self.vector_store.similarity_search(
                    query_embedding = query_embedding,
                    n_results       = self.retrieval_count * 2, # Increase pool for fusion
                )
                dense_ids = []
                for res in dense_results:
                    dense_ids.append(res.chunk_id)
                    if res.chunk_id not in results_pool:
                        results_pool[res.chunk_id] = res
                    # Store method metadata for scoring
                    results_pool[res.chunk_id].metadata["method_found"] = method
                dense_rankings.append(dense_ids)
                
                # Sparse Search (BM25) - only on original/variations, not HyDE
                if hasattr(self.vector_store, "bm25_search") and method != "hyde":
                    sparse_results = self.vector_store.bm25_search(
                        query     = query_text,
                        n_results = self.retrieval_count * 2,
                    )
                    sparse_ids = []
                    for res in sparse_results:
                        sparse_ids.append(res.chunk_id)
                        if res.chunk_id not in results_pool:
                            results_pool[res.chunk_id] = res
                    sparse_rankings.append(sparse_ids)
                
            except Exception as e:
                logger.warning(f"Retrieval failed for variation '{query_text[:30]}': {e}")

        # ── 4. Global Reciprocal Rank Fusion (RRF) ───────────
        # This fuses results across Vector vs Sparse AND across different Query Variations
        rrf_scores: Dict[str, float] = {}
        k_rrf = 60
        
        for ranking in dense_rankings + sparse_rankings:
            for rank, chunk_id in enumerate(ranking):
                # Apply boost for original query
                boost = 1.2 if results_pool[chunk_id].metadata.get("method_found") == "vector" else 1.0
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k_rrf + rank + 1)) * boost

        # Sort and select top results
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:self.retrieval_count * 2]
        
        final_pool: List[RetrievalResult] = []
        for chunk_id, rrf_score in sorted_ids:
            res = results_pool[chunk_id]
            method = res.metadata.get("method_found", "unknown")
            
            # Upgrade to Parent Context
            parent_chunk = self.vector_store.get_parent_for_child(chunk_id)
            parent_content = parent_chunk.content if parent_chunk else ""

            final_pool.append(RetrievalResult(
                child_content    = res.content,
                parent_content   = parent_content, # type: ignore
                similarity_score = res.similarity_score,
                confidence_score = self._calc_confidence(res.similarity_score, method),
                retrieval_method = method,
                metadata         = {**res.metadata, "rrf_score": round(rrf_score, 4)},
                query_used       = query,
            ))

        # ── 5. Global Rank & Deduplicate ─────────────────────
        final_results = self._rank_and_deduplicate(final_pool)

        elapsed = time.time() - start_time
        logger.info(
            f"✅ Fused retrieval: {len(final_results)} top results in {elapsed:.2f}s"
        )
        return final_results

    # ── HyDE ─────────────────────────────────────────────────

    def _generate_hyde_query(self, query: str) -> str:
        """
        HyDE: Generate a hypothetical document that would answer the query.

        Instead of embedding the question, we embed an answer-like passage.
        This closes the semantic gap between questions and document chunks.

        Formula: HyDE_Embedding = Embed(LLM("Answer: " + query))
        """
        hyde_prompt = f"""Write a detailed, informative passage that would \
directly answer the following question. Be specific and use terminology \
that would appear in a relevant document.

Question: {query}

Passage (2-3 sentences):"""

        response = self.llm_client.generate(
            prompt     = hyde_prompt,
            max_tokens = 200,
            temperature = 0.1,
        )
        return response.strip()

    # ── Multi-Query ──────────────────────────────────────────

    def _generate_multi_queries(
        self,
        query: str,
        num_queries: int = 3,
    ) -> List[str]:
        """
        Generate multiple semantically equivalent query variations.

        Example:
            Original: "What causes inflation?"
            Variation 1: "Economic factors that lead to price increases"
            Variation 2: "Why do prices rise in the economy?"
            Variation 3: "Monetary policy and inflation relationship"
        """
        prompt = f"""Generate {num_queries} different ways to ask the same \
question. Each variation should:
1. Use different vocabulary while meaning the same thing
2. Focus on different aspects of the question
3. Be suitable for document retrieval

Original question: {query}

Provide {num_queries} variations, one per line (no numbering or bullets):"""

        response = self.llm_client.generate(
            prompt      = prompt,
            max_tokens  = 300,
            temperature = 0.4,
        )

        variations = [
            line.strip().lstrip("0123456789.-) ")
            for line in response.strip().split("\n")
            if len(line.strip()) > 10
        ]

        return variations[:num_queries] if variations else [query]

    # ── Rank & Deduplicate ───────────────────────────────────

    def _rank_and_deduplicate(
        self,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        Group results by parent chunk ID to avoid duplicate context.
        Select best result per parent group.
        Sort by confidence score.
        """
        if not results:
            return []

        # Group by parent ID (or child ID if no parent)
        groups: Dict[str, List[RetrievalResult]] = {}
        for r in results:
            group_key = (
                r.metadata.get("parent_chunk_id")
                or r.metadata.get("chunk_id", str(id(r)))
            )
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(r)

        # Best from each group
        final: List[RetrievalResult] = []
        for group in groups.values():
            best = max(
                group,
                key=lambda x: (x.confidence_score, x.similarity_score),
            )
            final.append(best)

        # Sort descending by confidence
        final.sort(key=lambda x: x.confidence_score, reverse=True)
        
        # ── Optional MMR Reranking ─────────────────────────────
        if self.use_mmr and len(final) > 1:
            final = self._apply_mmr(final)

        return final[:self.retrieval_count]

    def _apply_mmr(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Maximal Marginal Relevance (MMR) for diversity.
        Formula: λ * sim(q, di) - (1-λ) * max(sim(di, dj))
        """
        if not results: return []
        
        # 1. Get embeddings for results to calculate document-document similarity
        contents = [r.child_content for r in results]
        try:
            embeddings = self.embedder.embed_texts(contents)
        except Exception as e:
            logger.warning(f"MMR embedding failed: {e}. Skipping diversity rerank.")
            return results

        selected_indices = [0] # Start with the most relevant document
        remaining_indices = list(range(1, len(results)))
        
        # Normalise relevance scores for MMR
        relevance_scores = np.array([r.confidence_score for r in results])
        if relevance_scores.max() > 0:
            relevance_scores = relevance_scores / relevance_scores.max()

        while remaining_indices and len(selected_indices) < self.retrieval_count:
            best_mmr = -1e9
            best_idx = -1
            
            for i in remaining_indices:
                # Relevance term
                relevance = self.mmr_lambda * relevance_scores[i]
                
                # Diversity term: penalty for similarity to already selected docs
                max_sim_to_selected = 0.0
                for j in selected_indices:
                    # Cosine similarity
                    dot = np.dot(embeddings[i], embeddings[j])
                    norm_i = np.linalg.norm(embeddings[i])
                    norm_j = np.linalg.norm(embeddings[j])
                    sim = dot / (norm_i * norm_j) if (norm_i * norm_j) > 0 else 0
                    max_sim_to_selected = max(max_sim_to_selected, sim)
                
                penalty = (1 - self.mmr_lambda) * max_sim_to_selected
                mmr_score = relevance - penalty
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                # Store MMR score for logging/audit
                results[best_idx].metadata["mmr_score"] = round(float(best_mmr), 4)
            else:
                break
        
        reranked = [results[idx] for idx in selected_indices]
        mmr_str = ", ".join([f"{r.metadata.get('mmr_score', '1.0')}" for r in reranked])
        logger.info(f"📊 MMR Diversified Scores: {mmr_str}")
        return reranked

    # ── Confidence Scoring ───────────────────────────────────

    def _calc_confidence(self, similarity: float, method: str) -> float:
        """
        Compute confidence score from similarity + method weighting.

        Method weights:
          vector     : 1.00  (direct match)
          hyde       : 0.90  (generated content slight penalty)
          multi_query: 0.85  (paraphrase slight penalty)
        """
        method_weights = {
            "vector":      1.00,
            "hyde":        0.90,
            "multi_query": 0.85,
        }
        w = method_weights.get(method, 0.80)

        if similarity > 0.8:
            confidence = similarity * w
        elif similarity > 0.6:
            confidence = similarity * w * 0.95
        else:
            confidence = similarity * w * 0.85

        return min(1.0, confidence)
