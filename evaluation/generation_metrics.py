"""
generation_metrics.py — Generation Quality Evaluation
=======================================================
Metrics:
  - BLEU Score   (n-gram overlap)
  - ROUGE-1/2/L  (recall-based overlap)
  - BERTScore    (semantic similarity via contextual embeddings)
"""

import re
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Internal helpers
# ───────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase + split on whitespace/punctuation."""
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _count_ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(_ngrams(tokens, n))


# ───────────────────────────────────────────────────────────────────────────
# BLEU Score
# ───────────────────────────────────────────────────────────────────────────

class BLEUScorer:
    """
    Corpus BLEU (1–4 gram) with brevity penalty.
    Uses modified precision (clips candidate counts to reference max).
    """

    @staticmethod
    def sentence_bleu(
        hypothesis: str,
        reference: str,
        max_n: int = 4,
        weights: Optional[List[float]] = None,
    ) -> float:
        """
        Compute BLEU score between a single hypothesis and reference.

        Args:
            hypothesis : generated answer
            reference  : reference / ground-truth answer
            max_n      : maximum n-gram order (default 4 → BLEU-4)
            weights    : per n-gram weights (default uniform)

        Returns:
            BLEU score ∈ [0, 1]
        """
        if weights is None:
            weights = [1.0 / max_n] * max_n

        hyp_tokens = _tokenize(hypothesis)
        ref_tokens = _tokenize(reference)

        if not hyp_tokens:
            return 0.0

        # Brevity penalty
        bp = 1.0
        if len(hyp_tokens) < len(ref_tokens):
            bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens))

        log_sum = 0.0
        for n, w in enumerate(weights, start=1):
            hyp_ngrams = _count_ngrams(hyp_tokens, n)
            ref_ngrams = _count_ngrams(ref_tokens, n)

            if not hyp_ngrams:
                return 0.0

            clipped = sum(
                min(count, ref_ngrams[gram])
                for gram, count in hyp_ngrams.items()
            )
            total = sum(hyp_ngrams.values())
            precision = clipped / total if total > 0 else 0.0

            if precision == 0:
                return 0.0

            log_sum += w * math.log(precision)

        bleu = bp * math.exp(log_sum)
        logger.debug(f"BLEU: {bleu:.4f}")
        return round(bleu, 4)


# ───────────────────────────────────────────────────────────────────────────
# ROUGE Score
# ───────────────────────────────────────────────────────────────────────────

class ROUGEScorer:
    """
    ROUGE-1, ROUGE-2, ROUGE-L.
    Each returns precision, recall, and F1.
    """

    @staticmethod
    def _rouge_n(
        hypothesis: str,
        reference: str,
        n: int,
    ) -> Dict[str, float]:
        hyp_tokens = _tokenize(hypothesis)
        ref_tokens = _tokenize(reference)

        hyp_ngrams = _count_ngrams(hyp_tokens, n)
        ref_ngrams = _count_ngrams(ref_tokens, n)

        matches = sum(
            min(count, ref_ngrams[gram])
            for gram, count in hyp_ngrams.items()
        )

        hyp_total = sum(hyp_ngrams.values())
        ref_total = sum(ref_ngrams.values())

        precision = matches / hyp_total if hyp_total > 0 else 0.0
        recall    = matches / ref_total  if ref_total  > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        return {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
        }

    @staticmethod
    def _lcs_length(x: List[str], y: List[str]) -> int:
        """Standard LCS via dynamic programming."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    def rouge_l(
        self,
        hypothesis: str,
        reference: str,
    ) -> Dict[str, float]:
        """ROUGE-L based on Longest Common Subsequence."""
        hyp_tokens = _tokenize(hypothesis)
        ref_tokens = _tokenize(reference)

        lcs = self._lcs_length(hyp_tokens, ref_tokens)
        precision = lcs / len(hyp_tokens) if hyp_tokens else 0.0
        recall    = lcs / len(ref_tokens)  if ref_tokens  else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        return {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
        }

    def score(
        self,
        hypothesis: str,
        reference: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute ROUGE-1, ROUGE-2, and ROUGE-L.

        Returns:
            {
              "rouge_1": {"precision": ..., "recall": ..., "f1": ...},
              "rouge_2": {...},
              "rouge_l": {...},
            }
        """
        r1 = self._rouge_n(hypothesis, reference, n=1)
        r2 = self._rouge_n(hypothesis, reference, n=2)
        rl = self.rouge_l(hypothesis, reference)
        logger.debug(f"ROUGE-1 F1: {r1['f1']:.4f} | ROUGE-2 F1: {r2['f1']:.4f} | ROUGE-L F1: {rl['f1']:.4f}")
        return {"rouge_1": r1, "rouge_2": r2, "rouge_l": rl}


# ───────────────────────────────────────────────────────────────────────────
# BERTScore
# ───────────────────────────────────────────────────────────────────────────

class BERTScorer:
    """
    Semantic similarity via cosine similarity of contextual BERT embeddings.

    Uses sentence-transformers' all-MiniLM-L6-v2 (fast, already in project).
    Falls back to ROUGE-L if the model cannot be loaded.
    """

    _model = None   # lazy-loaded singleton

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading sentence-transformer model for BERTScore …")
                cls._model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Sentence-transformer loaded ✅")
            except Exception as e:
                logger.warning(f"Could not load sentence-transformer: {e}. BERTScore will use ROUGE-L fallback.")
                cls._model = "fallback"
        return cls._model

    @staticmethod
    def _cosine(a, b) -> float:
        import numpy as np
        a, b = np.array(a), np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    def score(
        self,
        hypothesis: str,
        reference: str,
    ) -> Dict[str, float]:
        """
        Compute BERTScore (precision, recall, F1) between hypothesis and reference.

        Concept:
          - Embed each token of hypothesis and reference with BERT.
          - Precision = average max cosine similarity for each hypothesis token.
          - Recall    = average max cosine similarity for each reference token.
          - F1        = harmonic mean of P and R.

        In practice we use sentence-level embeddings for efficiency and
        compute a single cosine similarity value (reported as F1).

        Returns:
            {"precision": ..., "recall": ..., "f1": ...}
        """
        model = self._get_model()

        if model == "fallback":
            # Graceful fallback to ROUGE-L F1
            rouge_l = ROUGEScorer().rouge_l(hypothesis, reference)
            logger.warning("BERTScore falling back to ROUGE-L values.")
            return {
                "precision": rouge_l["precision"],
                "recall":    rouge_l["recall"],
                "f1":        rouge_l["f1"],
                "method":    "rouge_l_fallback",
            }

        import numpy as np

        # Sentence-level embeddings
        hyp_emb = model.encode(hypothesis, convert_to_numpy=True)
        ref_emb = model.encode(reference,  convert_to_numpy=True)
        sim = self._cosine(hyp_emb, ref_emb)

        # Token-level BERTScore approximation
        hyp_tokens = _tokenize(hypothesis)
        ref_tokens = _tokenize(reference)

        if hyp_tokens and ref_tokens:
            hyp_token_embs = model.encode(hyp_tokens, convert_to_numpy=True, batch_size=64)
            ref_token_embs = model.encode(ref_tokens,  convert_to_numpy=True, batch_size=64)

            # Precision: for each hyp token, find max similarity with any ref token
            p_scores = [
                max(self._cosine(h, r) for r in ref_token_embs)
                for h in hyp_token_embs
            ]
            # Recall: for each ref token, find max similarity with any hyp token
            r_scores = [
                max(self._cosine(r, h) for h in hyp_token_embs)
                for r in ref_token_embs
            ]

            precision = float(np.mean(p_scores))
            recall    = float(np.mean(r_scores))
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )
        else:
            precision = recall = f1 = sim

        logger.debug(f"BERTScore — P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")
        return {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "method":    "bertscore_token_level",
        }


# ───────────────────────────────────────────────────────────────────────────
# Combined Generation Evaluator
# ───────────────────────────────────────────────────────────────────────────

class GenerationEvaluator:
    """
    Wraps BLEU, ROUGE, and BERTScore into a single evaluate() call.
    """

    def __init__(self):
        self._bleu  = BLEUScorer()
        self._rouge = ROUGEScorer()
        self._bert  = BERTScorer()

    def evaluate(
        self,
        generated_answer: str,
        reference_answer: str,
    ) -> Dict:
        """
        Compute all generation quality metrics.

        Args:
            generated_answer: answer produced by the RAG pipeline
            reference_answer: ground-truth / human reference answer

        Returns:
            {
              "bleu":      float,
              "rouge":     {"rouge_1": {...}, "rouge_2": {...}, "rouge_l": {...}},
              "bertscore": {"precision": ..., "recall": ..., "f1": ...},
            }
        """
        bleu  = self._bleu.sentence_bleu(generated_answer, reference_answer)
        rouge = self._rouge.score(generated_answer, reference_answer)
        bert  = self._bert.score(generated_answer, reference_answer)

        return {
            "bleu":      bleu,
            "rouge":     rouge,
            "bertscore": bert,
        }
