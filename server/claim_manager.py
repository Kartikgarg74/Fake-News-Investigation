"""ClaimManager — thin facade over the segregated database layer.

This module existed in the original monolithic architecture as a 400+ line
file that owned the claims table, the migration logic, and the sample data.
All of that has been moved into `server.databases.ClaimsDB` (and siblings).

This facade is kept for backward compatibility: `environment.py` imports
`BUDGET_MAP`, `SOURCE_CATEGORIES`, and `ClaimManager` from here. Breaking
those imports would break the env server. So this file re-exports them.

New code should import directly from `server.databases`.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

from .databases import ClaimsDB
from .databases.claims import BUDGET_MAP, DIFFICULTY_MAP

# Re-exported for backward compatibility with environment.py
__all__ = ["ClaimManager", "BUDGET_MAP", "DIFFICULTY_MAP", "SOURCE_CATEGORIES"]

# Source categories available to the agent. Kept here so existing imports
# still resolve; the canonical list lives in environment.py's action dispatch.
SOURCE_CATEGORIES = [
    "government_data",
    "academic_papers",
    "news_articles",
    "fact_checks",
    "medical_journals",
    "statistical_reports",
    "international_organizations",
    "industry_reports",
    "image_analysis",        # Visual evidence (legacy)
    "wikipedia",             # NEW: live Wikipedia REST retrieval
    "fact_check_api",        # NEW: Google Fact Check Tools API
]


class ClaimManager:
    # NOTE: This facade exists purely for backward compatibility with environment.py
    # and tests that import BUDGET_MAP/SOURCE_CATEGORIES from this module. New code
    # should import directly from server.databases. Removing this file would require
    # updating all import sites, which risks breaking the hackathon validator.
    """Backward-compat wrapper around ClaimsDB.

    The original ClaimManager exposed get_random_claim() and a handful of
    internal methods. environment.py still calls get_random_claim directly,
    so we preserve that signature and delegate to ClaimsDB underneath.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db = ClaimsDB(db_path=db_path)
        # Expose db_path for any code that reads it (tests, debugging)
        self.db_path = self._db.db_path

    def get_random_claim(self, difficulty: str = "easy") -> Dict[str, Any]:
        """Pick a random claim. Returns a dict shape matching the old API.

        Raises ValueError only if the DB is completely empty for that
        difficulty — matches the original behavior so existing error
        handling in environment.py still triggers correctly.
        """
        claim = self._db.get_random(difficulty)
        if claim is None:
            raise ValueError(f"No claims found for difficulty: {difficulty}")

        # environment.py expects `evidence_passages` to be present (it's
        # used by _handle_request_source). If the underlying DB has the
        # legacy column, ClaimsDB._row_to_dict passes it through. If it
        # doesn't, we return an empty dict so the old code falls back
        # cleanly to its "no evidence found" branch.
        if "evidence_passages" not in claim:
            claim["evidence_passages"] = {}
        return claim

    def get_claim_count(self, difficulty: Optional[str] = None) -> int:
        """Total claims, optionally filtered by difficulty."""
        return self._db.count(difficulty)

    # -- Legacy internal methods kept as no-ops for import compatibility --
    # These were called from tests or introspection. They're now handled
    # inside ClaimsDB but we keep the names so nothing breaks.

    def _ensure_db(self) -> None:
        """No-op: ClaimsDB handles schema creation on init."""
        pass

    def _migrate_db(self) -> None:
        """No-op: ClaimsDB handles migrations on init."""
        pass
