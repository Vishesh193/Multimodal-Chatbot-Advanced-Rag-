"""
policy_comparator.py — Multi-Policy Comparator (Feature 1)
============================================================
Compares two insurance policies side-by-side for a given query.

Example:
    query = "Does knee surgery get covered?"
    result = comparator.compare("Star Health", "HDFC ERGO", query)
    → Structured comparison table + recommendation

How it works:
    1. Look up both policies in PolicyManager to get their document_ids
    2. Run retrieval filtered by each document_id separately
    3. Ask LLM to compare the retrieved clauses and produce a table
"""

from typing import Dict, List, Optional, Any

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Comparison prompt ─────────────────────────────────────────────────────
COMPARISON_PROMPT = """\
You are an expert insurance advisor. Compare the two insurance policies \
below for the user's specific query. Be objective and factual.

User Query: {query}

Policy A — {name_a} ({insurer_a}):
{context_a}

Policy B — {name_b} ({insurer_b}):
{context_b}

Provide your comparison in the following format:

## Comparison: {name_a} vs {name_b}

| Aspect | {name_a} | {name_b} |
|--------|----------|----------|
| Coverage | ... | ... |
| Exclusions | ... | ... |
| Claim Process | ... | ... |
| Waiting Period | ... | ... |
| Sub-limits / Caps | ... | ... |
| Sum Insured | ... | ... |

## Summary
[2-3 sentences comparing the key differences]

## Recommendation
[Which policy is better for this specific query and why]

Note: Base your answer ONLY on the policy documents provided above. \
If information is not available, write "Not specified in document".
"""


class PolicyComparator:
    """
    Side-by-side multi-policy comparator.

    Args:
        rag_system    : ProductionMultimodalRAG instance (or InsuranceRAG)
        policy_manager: PolicyManager instance
        llm_client    : Groq or Ollama client for comparison generation
    """

    def __init__(self, rag_system, policy_manager, llm_client):
        self.rag       = rag_system
        self.pm        = policy_manager
        self.llm       = llm_client

    # ------------------------------------------------------------------
    # Internal: retrieve context for one policy filtered by document_id
    # ------------------------------------------------------------------
    def _retrieve_for_policy(
        self,
        query:       str,
        document_id: str,
        top_k:       int = 4,
    ) -> str:
        """
        Retrieve relevant chunks from ChromaDB filtered to a single policy document.

        Uses ChromaDB's 'where' filter on parent_document_id metadata.
        """
        try:
            query_embedding = self.rag.embedder.embed_query(query)

            results = self.rag.vector_store.similarity_search(
                query_embedding  = query_embedding,
                n_results        = top_k,
                filter_criteria  = {"parent_document_id": document_id},
            )

            if not results:
                return "No relevant clauses found in this policy document."

            parts = []
            for i, r in enumerate(results):
                # Try to get parent context for richer content
                parent = self.rag.vector_store.get_parent_for_child(r.chunk_id)
                content = parent.content if parent else r.content
                parts.append(f"[Clause {i+1} | Score: {r.similarity_score:.2f}]\n{content}")

            return "\n\n---\n\n".join(parts)

        except Exception as e:
            logger.error(f"Retrieval failed for doc {document_id}: {e}")
            return "Could not retrieve policy clauses."

    # ------------------------------------------------------------------
    # Main comparison
    # ------------------------------------------------------------------
    def compare(
        self,
        policy_name_a: str,
        policy_name_b: str,
        query:         str,
        language:      str = "english",
        max_tokens:    int = 1500,
    ) -> Dict[str, Any]:
        """
        Compare two policies for a given user query.

        Args:
            policy_name_a : Name or insurer of policy A (fuzzy matched)
            policy_name_b : Name or insurer of policy B (fuzzy matched)
            query         : User's comparison question
            language      : Response language (see LanguageSupport)
            max_tokens    : Max LLM response tokens

        Returns:
            {
              "comparison_table": str  (markdown formatted),
              "policy_a":         PolicyRecord,
              "policy_b":         PolicyRecord,
              "context_a":        str,
              "context_b":        str,
              "query":            str,
            }
        """
        logger.info(f"⚖️  Comparing '{policy_name_a}' vs '{policy_name_b}'")
        logger.info(f"   Query: {query[:80]}")

        # ── 1. Resolve policy records ─────────────────────────
        rec_a = self.pm.get_by_name(policy_name_a)
        rec_b = self.pm.get_by_name(policy_name_b)

        if not rec_a:
            return {"error": f"Policy '{policy_name_a}' not found in registry. Please ingest it first."}
        if not rec_b:
            return {"error": f"Policy '{policy_name_b}' not found in registry. Please ingest it first."}
        if rec_a.policy_id == rec_b.policy_id:
            return {"error": "Both policy names resolve to the same document."}

        # ── 2. Retrieve relevant context per policy ───────────
        logger.info(f"   Retrieving context for '{rec_a.policy_name}' …")
        context_a = self._retrieve_for_policy(query, rec_a.document_id)

        logger.info(f"   Retrieving context for '{rec_b.policy_name}' …")
        context_b = self._retrieve_for_policy(query, rec_b.document_id)

        # ── 3. Build comparison prompt ────────────────────────
        prompt = COMPARISON_PROMPT.format(
            query     = query,
            name_a    = rec_a.policy_name,
            insurer_a = rec_a.insurer,
            context_a = context_a[:3000],
            name_b    = rec_b.policy_name,
            insurer_b = rec_b.insurer,
            context_b = context_b[:3000],
        )

        # ── 4. Language prefix if needed ──────────────────────
        if language.lower() != "english":
            from insurance.language_support import LanguageSupport, SUPPORTED_LANGUAGES
            lang_display = SUPPORTED_LANGUAGES.get(language.lower(), "Hindi (हिन्दी)")
            prompt = (
                f"IMPORTANT: Respond ONLY in {lang_display}.\n\n" + prompt
            )

        # ── 5. Generate comparison ────────────────────────────
        logger.info("   Generating comparison …")
        result = self.llm.generate(prompt=prompt, max_tokens=max_tokens)
        comparison_table = result.get("answer", "Comparison could not be generated.")

        logger.info("✅ Comparison complete")

        return {
            "comparison_table": comparison_table,
            "policy_a":         rec_a,
            "policy_b":         rec_b,
            "context_a":        context_a,
            "context_b":        context_b,
            "query":            query,
            "model_used":       result.get("model_used", "unknown"),
        }

    # ------------------------------------------------------------------
    # Quick helper: compare all policies of the same type
    # ------------------------------------------------------------------
    def compare_all_health(self, query: str) -> List[Dict]:
        """
        Compare ALL registered health policies for the query.
        Returns a list of pairwise comparison dicts.
        """
        health_policies = self.pm.list_by_type("health")
        if len(health_policies) < 2:
            return [{"error": "Need at least 2 health policies ingested to compare."}]

        results = []
        for i in range(len(health_policies)):
            for j in range(i + 1, len(health_policies)):
                results.append(
                    self.compare(
                        health_policies[i].policy_name,
                        health_policies[j].policy_name,
                        query,
                    )
                )
        return results
