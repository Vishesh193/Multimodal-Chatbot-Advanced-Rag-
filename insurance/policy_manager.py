"""
policy_manager.py — Multi-Policy Manager
==========================================
Manages multiple insurance policy PDFs per session.

Responsibilities:
  - Store policy metadata (insurer, type, policy number, document_id)
  - Persist registry to JSON so it survives restarts
  - Provide lookup helpers used by comparator, checklist, exclusion finder
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

POLICY_REGISTRY_FILE = "./insurance_policies.json"

# ── Supported policy types ────────────────────────────────────────────────
POLICY_TYPES = ["health", "motor", "life", "travel", "home", "other"]


@dataclass
class PolicyRecord:
    """Metadata record for one ingested insurance policy."""
    policy_id:    str          # unique identifier (auto-generated)
    policy_name:  str          # human label e.g. "Star Health Family Floater"
    insurer:      str          # company e.g. "Star Health", "HDFC ERGO"
    policy_type:  str          # health | motor | life | travel | home | other
    policy_number: str         # actual policy number from document (if known)
    document_id:  str          # RAG system doc_id from ingest_document()
    pdf_path:     str          # original PDF file path
    sum_insured:  str = ""     # e.g. "₹5,00,000"
    premium:      str = ""     # e.g. "₹12,000/year"
    holder_name:  str = ""     # policy holder name
    tags:         List[str] = field(default_factory=list)  # custom tags


class PolicyManager:
    """
    Registry of all ingested insurance policy documents.

    Persists metadata to a local JSON file so policies survive
    application restarts (the actual vectors stay in ChromaDB).

    Usage:
        pm = PolicyManager()
        pm.register(PolicyRecord(...))
        record = pm.get_by_name("Star Health")
        all_health = pm.list_by_type("health")
    """

    def __init__(self, registry_path: str = POLICY_REGISTRY_FILE):
        self.registry_path = registry_path
        self._policies: Dict[str, PolicyRecord] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self):
        """Load registry from JSON file if it exists."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for pid, data in raw.items():
                    self._policies[pid] = PolicyRecord(**data)
                logger.info(f"📂 Loaded {len(self._policies)} policies from registry")
            except Exception as e:
                logger.warning(f"Could not load policy registry: {e}")

    def _save(self):
        """Persist registry to JSON file."""
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(
                    {pid: asdict(rec) for pid, rec in self._policies.items()},
                    f, indent=2, ensure_ascii=False,
                )
        except Exception as e:
            logger.error(f"Could not save policy registry: {e}")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def register(self, record: PolicyRecord) -> PolicyRecord:
        """Add or overwrite a policy record and persist."""
        self._policies[record.policy_id] = record
        self._save()
        logger.info(
            f"✅ Registered policy: '{record.policy_name}' "
            f"({record.insurer} | {record.policy_type}) — id={record.policy_id}"
        )
        return record

    def get(self, policy_id: str) -> Optional[PolicyRecord]:
        """Fetch by policy_id."""
        return self._policies.get(policy_id)

    def get_by_name(self, name: str) -> Optional[PolicyRecord]:
        """
        Fuzzy name lookup (case-insensitive substring match).
        Returns the first match.
        """
        name_lower = name.lower()
        for rec in self._policies.values():
            if name_lower in rec.policy_name.lower() or name_lower in rec.insurer.lower():
                return rec
        return None

    def get_by_document_id(self, document_id: str) -> Optional[PolicyRecord]:
        """Reverse lookup by RAG document_id."""
        for rec in self._policies.values():
            if rec.document_id == document_id:
                return rec
        return None

    def list_all(self) -> List[PolicyRecord]:
        """Return all registered policies."""
        return list(self._policies.values())

    def list_by_type(self, policy_type: str) -> List[PolicyRecord]:
        """Filter by policy type (health / motor / life …)."""
        return [r for r in self._policies.values()
                if r.policy_type.lower() == policy_type.lower()]

    def list_by_insurer(self, insurer: str) -> List[PolicyRecord]:
        """Filter by insurer name (case-insensitive)."""
        insurer_lower = insurer.lower()
        return [r for r in self._policies.values()
                if insurer_lower in r.insurer.lower()]

    def remove(self, policy_id: str) -> bool:
        """Remove a policy record (does NOT delete from ChromaDB)."""
        if policy_id in self._policies:
            name = self._policies[policy_id].policy_name
            del self._policies[policy_id]
            self._save()
            logger.info(f"🗑️  Removed policy '{name}' (id={policy_id})")
            return True
        return False

    def summary(self) -> str:
        """Human-readable summary table of all policies."""
        if not self._policies:
            return "No policies registered yet."
        lines = [
            f"{'Policy Name':<35} {'Insurer':<20} {'Type':<10} {'Sum Insured':<15}",
            "-" * 85,
        ]
        for rec in self._policies.values():
            lines.append(
                f"{rec.policy_name:<35} {rec.insurer:<20} "
                f"{rec.policy_type:<10} {rec.sum_insured or 'N/A':<15}"
            )
        return "\n".join(lines)
