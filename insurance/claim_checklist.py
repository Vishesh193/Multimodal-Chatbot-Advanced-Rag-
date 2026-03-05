"""
claim_checklist.py — Claim Checklist Generator (Feature 2)
============================================================
Generates a personalised, step-by-step claim checklist
for a given claim type and policy.

Example:
    "I need to claim for a road accident hospitalization"
    → Step-by-step checklist with documents, timeline, hospital check
"""

from typing import Dict, List, Optional, Any
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Claim type categories with keyword hints for retrieval ────────────────
CLAIM_TYPES: Dict[str, List[str]] = {
    "hospitalisation": [
        "hospitalisation claim", "hospital admission", "inpatient claim",
        "discharge summary", "room charges", "medical bills",
    ],
    "surgery": [
        "surgical procedure claim", "operation claim", "pre-authorization surgery",
        "surgeon fees", "OT charges", "anaesthesia",
    ],
    "accident": [
        "accident claim", "road accident", "emergency hospitalisation",
        "accidental injury", "personal accident", "medico-legal",
    ],
    "maternity": [
        "maternity claim", "delivery charges", "ante-natal care",
        "newborn cover", "caesarean", "normal delivery",
    ],
    "critical_illness": [
        "critical illness claim", "cancer claim", "heart attack claim",
        "kidney failure", "stroke claim", "lump sum benefit",
    ],
    "reimbursement": [
        "reimbursement claim", "cashless not available", "non-network hospital",
        "out-of-pocket expenses", "reimbursement documents",
    ],
    "cashless": [
        "cashless claim", "pre-authorization", "network hospital",
        "TPA approval", "cashless facility",
    ],
    "motor": [
        "motor claim", "car accident claim", "vehicle damage",
        "third party claim", "own damage", "FIR", "surveyor",
    ],
    "general": [
        "claim process", "required documents", "how to file a claim",
        "claim settlement", "claim timeline",
    ],
}

# ── Checklist generation prompt ───────────────────────────────────────────
CHECKLIST_PROMPT = """\
You are an expert insurance claim advisor. Generate a clear, personalised \
claim checklist for the user based on their policy details and claim type.

User's Situation: {user_query}

Policy Details:
{policy_context}

Claim Process Guidelines from Policy Document:
{process_context}

Generate a step-by-step claim checklist in the following format:

## Claim Checklist: {claim_type_label}

### ✅ Required Documents
List all documents the user must collect (numbered list)

### 📋 Step-by-Step Claim Process
Detailed numbered steps from intimation to settlement

### ⏰ Important Timelines
- Intimation deadline: ...
- Document submission deadline: ...
- Settlement timeline: ...

### 🏥 Before You Go to Hospital (if applicable)
- Network hospital check: ...
- Pre-authorization requirement: ...
- Emergency vs planned: ...

### ⚠️ Common Reasons for Claim Rejection (Avoid These)
List 3-5 most common mistakes

### 📞 Contact Information
- Claim helpline: (from policy document if available, else "Contact your insurer")
- TPA name: (if mentioned in document)

Base your checklist ONLY on the retrieved policy documents. \
Mark items as "Not specified in document" if not found.
"""


class ClaimChecklistGenerator:
    """
    Generates personalised step-by-step claim checklists.

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
    # Internal: detect claim type from query
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_claim_type(query: str) -> str:
        """Detect the most relevant claim type from the user's query."""
        query_lower = query.lower()
        scores: Dict[str, int] = {}

        for claim_type, keywords in CLAIM_TYPES.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[claim_type] = score

        if scores:
            return max(scores, key=scores.__getitem__)
        return "general"

    # ------------------------------------------------------------------
    # Internal: build retrieval queries for a claim type
    # ------------------------------------------------------------------    
    @staticmethod
    def _get_retrieval_queries(claim_type: str, user_query: str) -> List[str]:
        """Build a list of retrieval queries to maximise relevant context."""
        base_queries = CLAIM_TYPES.get(claim_type, CLAIM_TYPES["general"])
        return [user_query] + base_queries[:3]  # user query + top 3 keyword hints

    # ------------------------------------------------------------------
    # Internal: retrieve and merge context for a policy
    # ------------------------------------------------------------------
    def _retrieve_claim_context(
        self,
        queries:     List[str],
        document_id: Optional[str] = None,
        top_k:       int = 4,
    ) -> str:
        """Retrieve claim-relevant chunks via multiple queries."""
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
                    if r.chunk_id not in seen_ids:
                        seen_ids.add(r.chunk_id)
                        parent = self.rag.vector_store.get_parent_for_child(r.chunk_id)
                        content = parent.content if parent else r.content
                        all_content.append(content)
            except Exception as e:
                logger.warning(f"Retrieval failed for query '{query[:50]}': {e}")

        return "\n\n---\n\n".join(all_content[:6]) if all_content else "No relevant claim information found."

    # ------------------------------------------------------------------
    # Main generate_checklist()
    # ------------------------------------------------------------------
    def generate_checklist(
        self,
        user_query:  str,
        policy_name: Optional[str] = None,
        language:    str = "english",
        max_tokens:  int = 1500,
    ) -> Dict[str, Any]:
        """
        Generate a personalised claim checklist.

        Args:
            user_query  : User's natural language description of their claim
            policy_name : (Optional) specific policy name to filter retrieval
            language    : Response language key
            max_tokens  : Max generation tokens

        Returns:
            {
              "checklist":      str  (markdown formatted checklist),
              "claim_type":     str,
              "policy_used":    PolicyRecord | None,
              "model_used":     str,
            }
        """
        logger.info(f"📋 Generating claim checklist")
        logger.info(f"   Query: {user_query[:80]}")

        # ── 1. Detect claim type ──────────────────────────────
        claim_type = self._detect_claim_type(user_query)
        claim_type_label = claim_type.replace("_", " ").title()
        logger.info(f"   Detected claim type: {claim_type}")

        # ── 2. Resolve policy if provided ─────────────────────
        policy_record = None
        document_id   = None
        if policy_name:
            policy_record = self.pm.get_by_name(policy_name)
            if policy_record:
                document_id = policy_record.document_id
                logger.info(f"   Policy: {policy_record.policy_name}")
            else:
                logger.warning(f"   Policy '{policy_name}' not found, searching all docs")

        # ── 3. Build retrieval queries ────────────────────────
        retrieval_queries = self._get_retrieval_queries(claim_type, user_query)

        # ── 4. Retrieve policy context ────────────────────────
        policy_context = (
            f"Policy: {policy_record.policy_name}\n"
            f"Insurer: {policy_record.insurer}\n"
            f"Type: {policy_record.policy_type}\n"
            f"Sum Insured: {policy_record.sum_insured or 'N/A'}\n"
        ) if policy_record else "No specific policy details provided."

        process_context = self._retrieve_claim_context(
            queries     = retrieval_queries,
            document_id = document_id,
            top_k       = 4,
        )

        # ── 5. Build prompt ───────────────────────────────────
        prompt = CHECKLIST_PROMPT.format(
            user_query       = user_query,
            policy_context   = policy_context,
            process_context  = process_context[:4000],
            claim_type_label = claim_type_label,
        )

        # ── 6. Language prefix ────────────────────────────────
        if language.lower() != "english":
            from insurance.language_support import SUPPORTED_LANGUAGES
            lang_display = SUPPORTED_LANGUAGES.get(language.lower(), "Hindi (हिन्दी)")
            prompt = f"IMPORTANT: Respond ONLY in {lang_display}.\n\n" + prompt

        # ── 7. Generate checklist ─────────────────────────────
        logger.info("   Generating checklist …")
        result = self.llm.generate(prompt=prompt, max_tokens=max_tokens)
        checklist = result.get("answer", "Checklist could not be generated.")

        logger.info("✅ Claim checklist generated")

        return {
            "checklist":   checklist,
            "claim_type":  claim_type,
            "policy_used": policy_record,
            "model_used":  result.get("model_used", "unknown"),
        }
