"""
faithfulness.py — Faithfulness / Hallucination Detection
==========================================================
Checks whether the generated answer is supported by the retrieved context.

Two strategies:
  1. LLM-as-Judge  (default) — sends a structured prompt to the configured LLM.
  2. Keyword overlap fallback — used when no LLM is available.
"""

import re
from typing import Dict, List, Optional, Union

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Faithfulness labels ───────────────────────────────────────────────────
FAITHFUL            = "faithful"
PARTIALLY_FAITHFUL  = "partially_faithful"
HALLUCINATED        = "hallucinated"

# ── Judge prompt template ─────────────────────────────────────────────────
FAITHFULNESS_PROMPT = """\
You are an expert fact-checker for a Retrieval-Augmented Generation (RAG) system.

Given the user query, the retrieved context documents, and the generated answer,
determine whether the generated answer is fully supported by the retrieved context.

User Query:
{query}

Retrieved Context:
{context}

Generated Answer:
{answer}

Instructions:
- Reply with EXACTLY ONE of the following labels (no extra text):
    faithful            — every claim in the answer is supported by the context
    partially_faithful  — most claims are supported but some are not
    hallucinated        — the answer contains claims not supported by the context

Label:"""


class FaithfulnessEvaluator:
    """
    Evaluates whether the generated answer is grounded in retrieved context.

    Args:
        llm_client: a Groq or Ollama client instance (optional).
                    If None, falls back to keyword-overlap heuristic.
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # LLM-as-Judge
    # ------------------------------------------------------------------
    def _llm_judge(
        self,
        query: str,
        context: str,
        answer: str,
    ) -> str:
        """
        Ask the LLM to classify faithfulness.
        Returns one of: faithful | partially_faithful | hallucinated
        """
        prompt = FAITHFULNESS_PROMPT.format(
            query=query,
            context=context[:4000],   # truncate to avoid token overflow
            answer=answer,
        )
        try:
            result = self.llm_client.generate(prompt, max_tokens=16)
            raw = result.get("answer", "").strip().lower()

            for label in (FAITHFUL, PARTIALLY_FAITHFUL, HALLUCINATED):
                if label in raw:
                    return label

            # Fuzzy fallback for partial matches
            if "partial" in raw:
                return PARTIALLY_FAITHFUL
            if "hallucinat" in raw or "not support" in raw:
                return HALLUCINATED
            if "faithful" in raw:
                return FAITHFUL

            logger.warning(f"Unexpected LLM faithfulness response: '{raw}' — defaulting to partially_faithful")
            return PARTIALLY_FAITHFUL

        except Exception as e:
            logger.error(f"LLM judge call failed: {e}")
            return PARTIALLY_FAITHFUL

    # ------------------------------------------------------------------
    # Keyword-overlap heuristic (no-LLM fallback)
    # ------------------------------------------------------------------
    @staticmethod
    def _keyword_overlap_judge(
        context: str,
        answer: str,
        faithful_threshold: float = 0.6,
        hallucinated_threshold: float = 0.3,
    ) -> str:
        """
        Heuristic: fraction of answer n-grams found in context.

        - ≥ faithful_threshold  → faithful
        - ≥ hallucinated_threshold → partially_faithful
        - < hallucinated_threshold → hallucinated
        """
        def tokens(text: str):
            return set(re.findall(r"\b\w+\b", text.lower()))

        ctx_tokens = tokens(context)
        ans_tokens = tokens(answer)

        if not ans_tokens:
            return FAITHFUL   # empty answer can't hallucinate

        overlap = ans_tokens & ctx_tokens
        ratio   = len(overlap) / len(ans_tokens)

        if ratio >= faithful_threshold:
            return FAITHFUL
        elif ratio >= hallucinated_threshold:
            return PARTIALLY_FAITHFUL
        else:
            return HALLUCINATED

    # ------------------------------------------------------------------
    # Public evaluate()
    # ------------------------------------------------------------------
    def evaluate(
        self,
        query: str,
        retrieved_documents: List[str],
        generated_answer: str,
    ) -> Dict[str, Union[str, float]]:
        """
        Assess faithfulness of the generated answer w.r.t. retrieved context.

        Args:
            query               : original user question
            retrieved_documents : list of retrieved document text chunks
            generated_answer    : answer produced by the RAG pipeline

        Returns:
            {
              "label":  "faithful" | "partially_faithful" | "hallucinated",
              "method": "llm_judge" | "keyword_overlap",
              "score":  1.0 | 0.5 | 0.0   (numerical proxy for downstream use)
            }
        """
        combined_context = "\n\n---\n\n".join(retrieved_documents)

        if self.llm_client is not None:
            label  = self._llm_judge(query, combined_context, generated_answer)
            method = "llm_judge"
        else:
            label  = self._keyword_overlap_judge(combined_context, generated_answer)
            method = "keyword_overlap"

        score_map = {
            FAITHFUL:           1.0,
            PARTIALLY_FAITHFUL: 0.5,
            HALLUCINATED:       0.0,
        }
        score = score_map.get(label, 0.5)

        logger.info(f"Faithfulness [{method}]: {label} (score={score})")
        return {"label": label, "method": method, "score": score}
