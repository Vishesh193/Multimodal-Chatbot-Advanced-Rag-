"""
relevance.py — Answer Relevance Evaluation
============================================
Measures how well the generated answer addresses the user query.

Two strategies:
  1. LLM scoring (1–5 scale)  — primary
  2. Semantic similarity fallback using sentence-transformers cosine similarity
"""

import re
from typing import Dict, Optional, Union

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Relevance scoring prompt ──────────────────────────────────────────────
RELEVANCE_PROMPT = """\
You are an expert evaluator for a question-answering system.

Rate the relevance of the generated answer to the user query on a scale from 1 to 5,
where:
  1 = completely irrelevant
  2 = mostly irrelevant, minor relation to the query
  3 = partially relevant, addresses some aspects
  4 = mostly relevant, addresses the main question
  5 = fully relevant, directly and completely addresses the query

User Query:
{query}

Generated Answer:
{answer}

Instructions:
- Reply with ONLY the integer score (1, 2, 3, 4, or 5). No explanation.

Score:"""


class RelevanceEvaluator:
    """
    Evaluates how relevant the generated answer is to the user's query.

    Args:
        llm_client: a Groq or Ollama client (optional).
                    Falls back to semantic cosine similarity if None.
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self._embedder  = None   # lazy init

    # ------------------------------------------------------------------
    # LLM scoring (1–5)
    # ------------------------------------------------------------------
    def _llm_score(self, query: str, answer: str) -> float:
        """
        Ask LLM to rate relevance 1–5 and return normalised value ∈ [0, 1].
        """
        prompt = RELEVANCE_PROMPT.format(query=query, answer=answer)
        try:
            result = self.llm_client.generate(prompt, max_tokens=8)
            raw = result.get("answer", "").strip()

            # Extract the first integer in the response
            numbers = re.findall(r"\b[1-5]\b", raw)
            if numbers:
                raw_score = int(numbers[0])
                normalised = (raw_score - 1) / 4.0   # maps [1,5] → [0,1]
                logger.debug(f"LLM relevance score: {raw_score}/5  → {normalised:.2f}")
                return round(normalised, 4)

            logger.warning(f"Could not parse LLM relevance score from: '{raw}'")
            return 0.5   # neutral default

        except Exception as e:
            logger.error(f"LLM relevance call failed: {e}")
            return 0.5

    # ------------------------------------------------------------------
    # Semantic similarity fallback
    # ------------------------------------------------------------------
    def _semantic_score(self, query: str, answer: str) -> float:
        """
        Cosine similarity between sentence embeddings of query and answer.
        Provides a relevance proxy without an LLM.
        """
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading sentence-transformer for relevance scoring …")
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

            import numpy as np
            q_emb = self._embedder.encode(query,  convert_to_numpy=True)
            a_emb = self._embedder.encode(answer, convert_to_numpy=True)

            cos_sim = float(
                np.dot(q_emb, a_emb) /
                (np.linalg.norm(q_emb) * np.linalg.norm(a_emb) + 1e-9)
            )
            # cosine similarity ∈ [-1, 1], clip to [0, 1]
            score = round(max(0.0, min(1.0, cos_sim)), 4)
            logger.debug(f"Semantic relevance score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"Semantic relevance fallback failed: {e}. Returning 0.5.")
            return 0.5

    # ------------------------------------------------------------------
    # Public evaluate()
    # ------------------------------------------------------------------
    def evaluate(
        self,
        query: str,
        generated_answer: str,
    ) -> Dict[str, Union[float, str, int]]:
        """
        Evaluate relevance of the generated answer to the query.

        Args:
            query            : original user question
            generated_answer : answer produced by the RAG pipeline

        Returns:
            {
              "score":        float ∈ [0, 1],
              "raw_score":    1–5  (only for LLM method),
              "method":       "llm_scoring" | "semantic_similarity",
            }
        """
        if self.llm_client is not None:
            normalised = self._llm_score(query, generated_answer)
            raw_score  = round(normalised * 4 + 1)   # back to 1-5 for readability
            result = {
                "score":     normalised,
                "raw_score": raw_score,
                "method":    "llm_scoring",
            }
        else:
            score = self._semantic_score(query, generated_answer)
            result = {
                "score":     score,
                "raw_score": round(score * 4 + 1),
                "method":    "semantic_similarity",
            }

        logger.info(
            f"Relevance [{result['method']}]: "
            f"score={result['score']:.4f} ({result['raw_score']}/5)"
        )
        return result
