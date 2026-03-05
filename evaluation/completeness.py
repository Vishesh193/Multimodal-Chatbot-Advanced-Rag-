"""
completeness.py — Completeness Evaluation
==========================================
Checks whether the generated answer covers all important information
from the reference answer.

Two sub-metrics:
  1. Topic Coverage   — key topic overlap via TF-IDF keywords or spaCy noun chunks
  2. Entity Coverage  — named entity (NER) overlap via regex heuristics
                        (no spaCy/NLTK required; uses a lightweight pattern set)
"""

import re
from typing import Dict, List, Optional, Set, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Lightweight keyword / topic extraction (no heavy NLP deps) ────────────

# Common English stop-words to filter out
_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "this", "that", "these", "those", "it", "its", "as", "so", "if",
    "then", "than", "when", "where", "which", "who", "whom", "what",
    "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "up", "about", "against", "while",
}

# Named-entity patterns (people, orgs, locations, dates, numbers)
_NE_PATTERNS = [
    # Dates: "March 2024", "2024-03-05", "05/03/2024"
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",
    r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
    # Years alone
    r"\b(?:19|20)\d{2}\b",
    # Capitalised multi-word phrases (likely proper nouns / organisations)
    r"\b(?:[A-Z][a-z]+\s){1,4}[A-Z][a-z]+\b",
    # All-caps acronyms
    r"\b[A-Z]{2,}\b",
    # Percentages & monetary values
    r"\b\d+(?:\.\d+)?%\b",
    r"\$\d+(?:[,\d]*)?(?:\.\d+)?(?:\s*(?:million|billion|trillion))?\b",
    # Cardinal numbers
    r"\b\d+(?:,\d{3})*(?:\.\d+)?\b",
]

_NE_COMPILED = [re.compile(p) for p in _NE_PATTERNS]


def _extract_keywords(text: str, top_n: int = 30) -> Set[str]:
    """
    Extract top-N content words as a proxy for key topics.
    Uses token frequency weighted against stop-word list (lightweight TF-IDF proxy).
    """
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    freq: Dict[str, int] = {}
    for t in tokens:
        if t not in _STOP_WORDS:
            freq[t] = freq.get(t, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: -x[1])
    return {w for w, _ in sorted_words[:top_n]}


def _extract_entities(text: str) -> Set[str]:
    """
    Extract named entities / key facts using regex patterns.
    Returns a set of matched strings (lowercased for comparison).
    """
    entities: Set[str] = set()
    for pattern in _NE_COMPILED:
        for match in pattern.finditer(text):
            entities.add(match.group().strip().lower())
    return entities


# ───────────────────────────────────────────────────────────────────────────
# Completeness Evaluator
# ───────────────────────────────────────────────────────────────────────────

class CompletenessEvaluator:
    """
    Measures how completely the generated answer covers the reference answer.

    Metrics
    -------
    topic_coverage  : fraction of reference key-topics present in the generated answer
    entity_coverage : fraction of reference named entities present in the generated answer
    combined        : weighted average (0.5 topic + 0.5 entity)
    """

    def __init__(
        self,
        topic_top_n: int = 30,
        topic_weight: float = 0.5,
        entity_weight: float = 0.5,
    ):
        self.topic_top_n     = topic_top_n
        self.topic_weight    = topic_weight
        self.entity_weight   = entity_weight

    # ------------------------------------------------------------------
    # Topic Coverage
    # ------------------------------------------------------------------
    def topic_coverage(
        self,
        generated_answer: str,
        reference_answer: str,
    ) -> Dict[str, float]:
        """
        Coverage = matched_topics / total_reference_topics

        Args:
            generated_answer: answer produced by the RAG pipeline
            reference_answer: ground-truth / human reference answer

        Returns:
            {
              "coverage":         float ∈ [0, 1],
              "matched_topics":   int,
              "reference_topics": int,
              "missing_topics":   list[str],
            }
        """
        ref_keywords = _extract_keywords(reference_answer, self.topic_top_n)
        gen_keywords = _extract_keywords(generated_answer, self.topic_top_n * 2)

        matched  = ref_keywords & gen_keywords
        coverage = len(matched) / len(ref_keywords) if ref_keywords else 1.0
        missing  = sorted(ref_keywords - gen_keywords)

        logger.debug(
            f"Topic Coverage: {coverage:.4f} "
            f"({len(matched)}/{len(ref_keywords)} topics matched)"
        )
        return {
            "coverage":         round(coverage, 4),
            "matched_topics":   len(matched),
            "reference_topics": len(ref_keywords),
            "missing_topics":   missing[:10],   # show up to 10 missing
        }

    # ------------------------------------------------------------------
    # Entity Coverage
    # ------------------------------------------------------------------
    def entity_coverage(
        self,
        generated_answer: str,
        reference_answer: str,
    ) -> Dict[str, float]:
        """
        Entity Coverage = matched_entities / total_reference_entities

        Extracts people, dates, organisations, locations, numbers, etc.
        using lightweight regex patterns.

        Args:
            generated_answer: answer produced by the RAG pipeline
            reference_answer: ground-truth / human reference answer

        Returns:
            {
              "coverage":           float ∈ [0, 1],
              "matched_entities":   int,
              "reference_entities": int,
              "missing_entities":   list[str],
            }
        """
        ref_entities = _extract_entities(reference_answer)
        gen_entities = _extract_entities(generated_answer)

        if not ref_entities:
            logger.debug("No entities found in reference — entity coverage = 1.0 (vacuously true)")
            return {
                "coverage":           1.0,
                "matched_entities":   0,
                "reference_entities": 0,
                "missing_entities":   [],
            }

        matched  = ref_entities & gen_entities
        coverage = len(matched) / len(ref_entities)
        missing  = sorted(ref_entities - gen_entities)

        logger.debug(
            f"Entity Coverage: {coverage:.4f} "
            f"({len(matched)}/{len(ref_entities)} entities matched)"
        )
        return {
            "coverage":           round(coverage, 4),
            "matched_entities":   len(matched),
            "reference_entities": len(ref_entities),
            "missing_entities":   missing[:10],
        }

    # ------------------------------------------------------------------
    # Combined evaluate()
    # ------------------------------------------------------------------
    def evaluate(
        self,
        generated_answer: str,
        reference_answer: str,
    ) -> Dict:
        """
        Compute topic and entity coverage + combined completeness score.

        Args:
            generated_answer: answer produced by the RAG pipeline
            reference_answer: ground-truth / human reference answer

        Returns:
            {
              "topic_coverage":    {...},
              "entity_coverage":   {...},
              "combined_score":    float ∈ [0, 1],
            }
        """
        tc = self.topic_coverage(generated_answer, reference_answer)
        ec = self.entity_coverage(generated_answer, reference_answer)

        combined = round(
            self.topic_weight  * tc["coverage"] +
            self.entity_weight * ec["coverage"],
            4,
        )

        logger.info(
            f"Completeness — Topic: {tc['coverage']:.4f} | "
            f"Entity: {ec['coverage']:.4f} | Combined: {combined:.4f}"
        )
        return {
            "topic_coverage":  tc,
            "entity_coverage": ec,
            "combined_score":  combined,
        }
