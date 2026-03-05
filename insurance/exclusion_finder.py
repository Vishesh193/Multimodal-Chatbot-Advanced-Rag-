"""
exclusion_finder.py — Exclusion Highlighter (Feature 3)
=========================================================
Retrieves and surfaces all exclusion clauses from a policy document.

Answers queries like:
  - "What is NOT covered in my policy?"
  - "Are pre-existing diseases covered?"
  - "Is cosmetic surgery excluded?"

Approach:
  1. Use exclusion-focused retrieval queries against ChromaDB
  2. Ask LLM to extract and categorise all exclusion clauses
  3. Highlight commonly misunderstood exclusions
"""

from typing import Dict, List, Optional, Any
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Exclusion-focused retrieval queries ──────────────────────────────────
EXCLUSION_RETRIEVAL_QUERIES = [
    "exclusions not covered",
    "what is excluded from coverage",
    "pre-existing disease exclusion",
    "waiting period exclusion clause",
    "permanent exclusions",
    "specific disease exclusions",
    "maternity exclusion",
    "cosmetic treatment exclusion",
    "self-inflicted injury exclusion",
    "war terrorism exclusion",
    "experimental treatment not covered",
    "dental optical exclusion",
    "mental illness exclusion",
    "alcohol drug exclusion",
    "adventure sports exclusion",
]

# ── Commonly misunderstood coverage items ─────────────────────────────────
COMMON_MISCONCEPTIONS = [
    "pre-existing diseases",
    "maternity and childbirth",
    "dental and optical treatment",
    "cosmetic and plastic surgery",
    "mental health treatment",
    "obesity treatment / weight loss",
    "alternative medicine (Ayurveda, Homeopathy)",
    "adventure or hazardous sports injuries",
    "self-inflicted injuries",
    "war, terrorism, nuclear risks",
    "alcohol or drug-related conditions",
    "infertility and IVF treatment",
    "experimental or unproven treatments",
    "HIV/AIDS (in some policies)",
    "congenital diseases",
]

# ── Exclusion extraction prompt ───────────────────────────────────────────
EXCLUSION_PROMPT = """\
You are a meticulous insurance policy analyst. Extract and present ALL \
exclusion clauses from the policy document context below.

User Query: {user_query}

Policy: {policy_name} ({insurer})

Policy Document Context (Exclusion Relevant Sections):
{exclusion_context}

Commonly Misunderstood Items to Check:
{misconceptions}

Present your findings in this exact format:

## 🚫 What Is NOT Covered — {policy_name}

### Permanent Exclusions
(Things that will NEVER be covered under any circumstances)
- ...

### Waiting Period Exclusions
(Things covered ONLY after a waiting period)
| Condition | Waiting Period |
|-----------|---------------|
| Pre-existing diseases | ... |
| Specific diseases | ... |
| Maternity | ... |

### Specific Disease / Procedure Exclusions
- ...

### Commonly Assumed But NOT Covered ⚠️
(People often assume these are covered — they are NOT)
- **[Item]**: [Explanation from policy]

### Partial / Sub-limit Coverage (Not Full Exclusion)
(These ARE covered but with caps or sub-limits)
- ...

### What IS Covered (for context)
[Brief 2-3 line summary of main coverage]

---
⚠️ Disclaimer: This is based on the retrieved policy document sections. \
Always refer to your complete policy document or contact your insurer for \
official confirmation.

Base your answer ONLY on the provided document context. \
Mark items as "Not specified in retrieved sections" if not found.
"""


class ExclusionFinder:
    """
    Retrieves and highlights all exclusion clauses from insurance policies.

    Args:
        rag_system    : ProductionMultimodalRAG instance
        policy_manager: PolicyManager instance
        llm_client    : Groq or Ollama client
    """

    def __init__(self, rag_system, policy_manager, llm_client):
        self.rag = rag_system
        self.pm  = policy_manager
        self.llm = llm_client

    # ------------------------------------------------------------------
    # Internal: retrieve exclusion-focused context
    # ------------------------------------------------------------------
    def _retrieve_exclusion_context(
        self,
        user_query:  str,
        document_id: Optional[str] = None,
        top_k:       int = 4,
        max_queries: int = 6,
    ) -> str:
        """
        Run multiple exclusion-focused queries and aggregate unique chunks.
        """
        queries = [user_query] + EXCLUSION_RETRIEVAL_QUERIES[:max_queries - 1]
        all_content = []
        seen_ids    = set()

        filter_criteria = {"parent_document_id": document_id} if document_id else None

        for query in queries:
            try:
                query_emb = self.rag.embedder.embed_query(query)
                results   = self.rag.vector_store.similarity_search(
                    query_embedding  = query_emb,
                    n_results        = top_k,
                    filter_criteria  = filter_criteria,
                )
                for r in results:
                    if r.chunk_id not in seen_ids and r.similarity_score > 0.3:
                        seen_ids.add(r.chunk_id)
                        parent  = self.rag.vector_store.get_parent_for_child(r.chunk_id)
                        content = parent.content if parent else r.content
                        all_content.append(f"[Score: {r.similarity_score:.2f}]\n{content}")
            except Exception as e:
                logger.warning(f"Exclusion retrieval failed for '{query[:40]}': {e}")

        logger.info(f"   Retrieved {len(all_content)} unique exclusion-related chunks")
        return "\n\n---\n\n".join(all_content[:8]) if all_content else \
               "No exclusion clauses found in retrieved sections."

    # ------------------------------------------------------------------
    # Main find_exclusions()
    # ------------------------------------------------------------------
    def find_exclusions(
        self,
        user_query:  str = "What is not covered in my policy?",
        policy_name: Optional[str] = None,
        language:    str = "english",
        max_tokens:  int = 1500,
    ) -> Dict[str, Any]:
        """
        Find and surface all exclusion clauses for a policy.

        Args:
            user_query  : User's exclusion-related question
            policy_name : (Optional) specific policy name
            language    : Response language key
            max_tokens  : Max generation tokens

        Returns:
            {
              "exclusions":  str  (markdown formatted exclusion list),
              "policy_used": PolicyRecord | None,
              "model_used":  str,
              "chunks_used": int,
            }
        """
        logger.info(f"🔍 Finding exclusions")
        logger.info(f"   Query: {user_query[:80]}")

        # ── 1. Resolve policy ─────────────────────────────────
        policy_record = None
        document_id   = None
        if policy_name:
            policy_record = self.pm.get_by_name(policy_name)
            if policy_record:
                document_id = policy_record.document_id
                logger.info(f"   Policy: {policy_record.policy_name}")
            else:
                logger.warning(f"   Policy '{policy_name}' not found, searching all")

        policy_name_display = (
            policy_record.policy_name if policy_record else "Your Policy"
        )
        insurer_display = (
            policy_record.insurer if policy_record else "Your Insurer"
        )

        # ── 2. Retrieve exclusion context ─────────────────────
        exclusion_context = self._retrieve_exclusion_context(
            user_query  = user_query,
            document_id = document_id,
            top_k       = 4,
            max_queries = 7,
        )

        # ── 3. Build prompt ───────────────────────────────────
        misconceptions_text = "\n".join(
            f"- {item}" for item in COMMON_MISCONCEPTIONS
        )

        prompt = EXCLUSION_PROMPT.format(
            user_query        = user_query,
            policy_name       = policy_name_display,
            insurer           = insurer_display,
            exclusion_context = exclusion_context[:4500],
            misconceptions    = misconceptions_text,
        )

        # ── 4. Language prefix ────────────────────────────────
        if language.lower() != "english":
            from insurance.language_support import SUPPORTED_LANGUAGES
            lang_display = SUPPORTED_LANGUAGES.get(language.lower(), "Hindi (हिन्दी)")
            prompt = f"IMPORTANT: Respond ONLY in {lang_display}.\n\n" + prompt

        # ── 5. Generate exclusion report ──────────────────────
        logger.info("   Generating exclusion report …")
        result = self.llm.generate(prompt=prompt, max_tokens=max_tokens)
        exclusions = result.get("answer", "Exclusion analysis could not be generated.")

        logger.info("✅ Exclusion report generated")

        return {
            "exclusions":  exclusions,
            "policy_used": policy_record,
            "model_used":  result.get("model_used", "unknown"),
        }

    # ------------------------------------------------------------------
    # Quick check: is a specific item excluded?
    # ------------------------------------------------------------------
    def is_excluded(
        self,
        item:        str,
        policy_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Quick yes/no check: is a specific treatment/condition excluded?

        Args:
            item        : e.g. "knee replacement", "dental treatment", "IVF"
            policy_name : (Optional) specific policy name

        Returns:
            {"item": str, "verdict": "excluded"|"covered"|"partial"|"not_found",
             "explanation": str, "policy": str}
        """
        query = f"Is {item} covered or excluded in my insurance policy?"

        policy_record = self.pm.get_by_name(policy_name) if policy_name else None
        document_id   = policy_record.document_id if policy_record else None

        context = self._retrieve_exclusion_context(
            user_query  = query,
            document_id = document_id,
            top_k       = 3,
            max_queries = 3,
        )

        quick_prompt = (
            f"Based on the policy document below, answer in ONE sentence: "
            f"Is '{item}' covered, excluded, or partially covered?\n\n"
            f"Policy Context:\n{context[:2000]}\n\n"
            f"Answer with EXACTLY: "
            f"'EXCLUDED', 'COVERED', 'PARTIALLY COVERED', or 'NOT SPECIFIED', "
            f"followed by a brief explanation."
        )

        result = self.llm.generate(quick_prompt, max_tokens=150)
        answer = result.get("answer", "NOT SPECIFIED").strip()

        verdict = "not_found"
        if "EXCLUDED" in answer.upper():
            verdict = "excluded"
        elif "PARTIALLY" in answer.upper():
            verdict = "partial"
        elif "COVERED" in answer.upper():
            verdict = "covered"

        return {
            "item":        item,
            "verdict":     verdict,
            "explanation": answer,
            "policy":      policy_record.policy_name if policy_record else "All policies",
        }
