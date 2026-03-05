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
        all_results: List[RetrievalResult] = []

        for query_text, method in queries_to_process:
            try:
                query_embedding = self.embedder.embed_query(query_text)
                
                # Dense Search (Chroma)
                dense_results   = self.vector_store.similarity_search(
                    query_embedding = query_embedding,
                    n_results       = self.retrieval_count,
                    include_distances = True,
                )
                
                # Sparse Search (BM25)
                sparse_results = []
                if hasattr(self.vector_store, "bm25_search"):
                    sparse_results = self.vector_store.bm25_search(
                        query     = query_text,
                        n_results = self.retrieval_count,
                    )
                
                # Reciprocal Rank Fusion (RRF)
                rrf_scores = {}
                results_pool = {}
                k_rrf = 60
                
                for rank, res in enumerate(dense_results):
                    rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + 1.0 / (k_rrf + rank + 1)
                    if res.chunk_id not in results_pool or res.similarity_score > results_pool[res.chunk_id].similarity_score:
                        results_pool[res.chunk_id] = res
                    
                for rank, res in enumerate(sparse_results):
                    rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + 1.0 / (k_rrf + rank + 1)
                    if res.chunk_id not in results_pool or res.similarity_score > results_pool[res.chunk_id].similarity_score:
                        results_pool[res.chunk_id] = res
                    
                sorted_fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:self.retrieval_count]
                
                fused_results = []
                for chunk_id, rrf_score in sorted_fused:
                    res = results_pool[chunk_id]
                    # Keep the original similarity_score from the dense or sparse search
                    # instead of overwriting it with an arbitrary RRF formula.
                    fused_results.append(res)

                for result in fused_results:
                    # ── 4. Parent-Child Context Upgrade ─────────
                    child_id       = result.chunk_id
                    parent_chunk   = self.vector_store.get_parent_for_child(child_id)
                    parent_content = parent_chunk.content if parent_chunk else ""

                    retrieval_result = RetrievalResult(
                        child_content    = result.content,
                        parent_content   = parent_content, # type: ignore
                        similarity_score = result.similarity_score,
                        confidence_score = self._calc_confidence(
                            result.similarity_score, method
                        ),
                        retrieval_method = method,
                        metadata         = result.metadata,
                        query_used       = query_text,
                    )
                    all_results.append(retrieval_result)

            except Exception as e:
                logger.warning(f"Retrieval failed for query '{query_text[:50]}': {e}")

        # ── 5. Rank & Deduplicate ─────────────────────────────
        final_results = self._rank_and_deduplicate(all_results)

        elapsed = time.time() - start_time
        logger.info(
            f"✅ Retrieved {len(final_results)} unique results in {elapsed:.2f}s"
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
