"""CredibilityChecker — thin facade over SourcesDB.

Originally this owned a hardcoded 30-entry dict. Now it delegates to
SourcesDB which is seeded from MBFC (thousands of publishers) and supports
fuzzy domain matching + bulk updates.

The facade keeps the old API (check(source_id) -> dict) so environment.py
doesn't need to change how it calls it.
"""

from __future__ import annotations

from typing import Any, Dict

from .databases import SourcesDB


class CredibilityChecker:
    """Facade over SourcesDB. Keeps the old check() API."""

    def __init__(self):
        self._db = SourcesDB()

    def check(self, source_id: str) -> Dict[str, Any]:
        """Look up credibility for a source.

        Returns a dict with keys matching the legacy API:
            {source, bias, factual_reporting, credibility_score, found}

        Never raises. Unknown sources get a neutral 0.5 score so the agent
        can still reason about them.
        """
        result = self._db.lookup(source_id)
        # Legacy API used key 'source' not 'domain'; SourcesDB returns both
        return {
            "source": result.get("source", source_id),
            "bias": result.get("bias", "Unknown"),
            "factual_reporting": result.get("factual_reporting", "Unknown"),
            "credibility_score": float(result.get("credibility_score", 0.5)),
            "found": bool(result.get("found", False)),
            # Extra fields for agents that can use them
            "name": result.get("name", ""),
            "country": result.get("country", ""),
            "media_type": result.get("media_type", ""),
        }
