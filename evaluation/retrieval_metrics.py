"""
retrieval_metrics.py — Retrieval Evaluation Metrics
=======================================================
Metrics:
  - Precision@K
  - Recall@K
  - Mean Reciprocal Rank (MRR)
  - F1 Score
"""

from typing import Dict, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalEvaluator:
    """
    Evaluates the quality of retrieved documents against ground truth.

    All methods accept:
        retrieved_docs  : list of doc IDs / content strings in ranked order
        ground_truth_docs: list of relevant doc IDs / content strings
    """

    # ------------------------------------------------------------------
    # Precision@K
    # ------------------------------------------------------------------
    @staticmethod
    def precision_at_k(
        retrieved_docs: List[str],
        ground_truth_docs: List[str],
        k: int,
    ) -> float:
        """
        P@K = (# relevant docs in top-K) / K

        Args:
            retrieved_docs   : ranked list of retrieved document ids/texts
            ground_truth_docs: list of relevant document ids/texts
            k                : cut-off rank

        Returns:
            Precision@K  ∈ [0, 1]
        """
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        top_k = retrieved_docs[:k]
        gt_set = set(ground_truth_docs)
        relevant_in_top_k = sum(1 for doc in top_k if doc in gt_set)
        precision = relevant_in_top_k / k
        logger.debug(f"Precision@{k}: {precision:.4f} ({relevant_in_top_k}/{k} relevant)")
        return round(precision, 4)

    # ------------------------------------------------------------------
    # Recall@K
    # ------------------------------------------------------------------
    @staticmethod
    def recall_at_k(
        retrieved_docs: List[str],
        ground_truth_docs: List[str],
        k: int,
    ) -> float:
        """
        R@K = (# relevant docs in top-K) / (total relevant docs)

        Args:
            retrieved_docs   : ranked list of retrieved document ids/texts
            ground_truth_docs: list of relevant document ids/texts
            k                : cut-off rank

        Returns:
            Recall@K  ∈ [0, 1]
        """
        if not ground_truth_docs:
            logger.warning("ground_truth_docs is empty — Recall@K is undefined, returning 0.")
            return 0.0

        top_k = retrieved_docs[:k]
        gt_set = set(ground_truth_docs)
        relevant_in_top_k = sum(1 for doc in top_k if doc in gt_set)
        recall = relevant_in_top_k / len(gt_set)
        logger.debug(f"Recall@{k}: {recall:.4f} ({relevant_in_top_k}/{len(gt_set)} relevant)")
        return round(recall, 4)

    # ------------------------------------------------------------------
    # MRR (Mean Reciprocal Rank)
    # ------------------------------------------------------------------
    @staticmethod
    def mean_reciprocal_rank(
        all_retrieved: List[List[str]],
        all_ground_truth: List[List[str]],
    ) -> float:
        """
        MRR = (1/N) * Σ (1 / rank_i)

        rank_i = position (1-indexed) of first relevant doc for query i.
        If no relevant doc is found, reciprocal rank = 0.

        Args:
            all_retrieved   : list of ranked retrieved lists, one per query
            all_ground_truth: list of ground-truth lists, one per query

        Returns:
            MRR  ∈ [0, 1]
        """
        if len(all_retrieved) != len(all_ground_truth):
            raise ValueError(
                "all_retrieved and all_ground_truth must have the same length."
            )

        n = len(all_retrieved)
        if n == 0:
            return 0.0

        rr_sum = 0.0
        for retrieved, gt in zip(all_retrieved, all_ground_truth):
            gt_set = set(gt)
            rr = 0.0
            for rank, doc in enumerate(retrieved, start=1):
                if doc in gt_set:
                    rr = 1.0 / rank
                    break
            rr_sum += rr

        mrr = rr_sum / n
        logger.debug(f"MRR: {mrr:.4f} over {n} queries")
        return round(mrr, 4)

    # ------------------------------------------------------------------
    # F1 Score
    # ------------------------------------------------------------------
    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        """
        F1 = 2 × (P × R) / (P + R)

        Args:
            precision: Precision@K value
            recall   : Recall@K value

        Returns:
            F1  ∈ [0, 1]
        """
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        logger.debug(f"F1: {f1:.4f} (P={precision}, R={recall})")
        return round(f1, 4)

    # ------------------------------------------------------------------
    # Combined single-query evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        retrieved_docs: List[str],
        ground_truth_docs: List[str],
        k: int = 5,
    ) -> Dict[str, float]:
        """
        Run all retrieval metrics for a single query.

        Args:
            retrieved_docs   : ranked list of retrieved document ids/texts
            ground_truth_docs: list of relevant document ids/texts
            k                : cut-off rank

        Returns:
            Dict with keys: precision_at_k, recall_at_k, f1, mrr
        """
        p = self.precision_at_k(retrieved_docs, ground_truth_docs, k)
        r = self.recall_at_k(retrieved_docs, ground_truth_docs, k)
        f1 = self.f1_score(p, r)

        # MRR for a single query
        mrr = self.mean_reciprocal_rank(
            all_retrieved=[retrieved_docs],
            all_ground_truth=[ground_truth_docs],
        )

        return {
            f"precision_at_{k}": p,
            f"recall_at_{k}":    r,
            "f1":                f1,
            "mrr":               mrr,
        }
