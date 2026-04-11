"""TemporalDB — claim and evidence timelines for temporal verification.

Meta's real fact-checking pipeline cares deeply about WHEN claims were made
vs. WHEN contradicting evidence emerged. A claim that was defensible in 2019
may be indefensible in 2024 after new evidence. This DB tracks that.

Schema:
- claim_timeline: when a claim was first made, when it was last updated
- evidence_timeline: when evidence pieces were published, when they changed
- contradictions: (claim_id, evidence_id, first_seen_at) — when contradicting
  evidence first emerged for a claim
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import DatabaseManager


class TemporalDB(DatabaseManager):
    filename = "temporal.db"

    schema = """
    CREATE TABLE IF NOT EXISTS claim_timeline (
        claim_id TEXT PRIMARY KEY,
        first_seen_date TEXT,
        last_seen_date TEXT,
        peak_spread_date TEXT,
        status_at_creation TEXT DEFAULT 'unknown',
        current_status TEXT DEFAULT 'unknown',
        updated_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS evidence_timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        evidence_id TEXT NOT NULL,
        claim_id TEXT NOT NULL,
        published_date TEXT,
        supports_or_contradicts TEXT DEFAULT 'contradicts',
        source_domain TEXT DEFAULT '',
        title TEXT DEFAULT '',
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_evtimeline_claim ON evidence_timeline(claim_id);
    CREATE INDEX IF NOT EXISTS idx_evtimeline_published ON evidence_timeline(published_date);

    CREATE TABLE IF NOT EXISTS contradictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim_id TEXT NOT NULL,
        evidence_id TEXT NOT NULL,
        first_seen_at TEXT NOT NULL,
        strength REAL DEFAULT 1.0,
        source_domain TEXT DEFAULT '',
        notes TEXT DEFAULT ''
    );

    CREATE INDEX IF NOT EXISTS idx_contradictions_claim ON contradictions(claim_id);
    """

    def record_claim(
        self,
        claim_id: str,
        first_seen_date: Optional[str] = None,
        current_status: str = "unknown",
    ) -> bool:
        return self.write(
            """INSERT OR REPLACE INTO claim_timeline
               (claim_id, first_seen_date, current_status)
               VALUES (?, ?, ?)""",
            (claim_id, first_seen_date, current_status),
        )

    def record_evidence(
        self,
        evidence_id: str,
        claim_id: str,
        published_date: Optional[str] = None,
        supports_or_contradicts: str = "contradicts",
        source_domain: str = "",
        title: str = "",
    ) -> bool:
        return self.write(
            """INSERT INTO evidence_timeline
               (evidence_id, claim_id, published_date,
                supports_or_contradicts, source_domain, title)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (evidence_id, claim_id, published_date,
             supports_or_contradicts, source_domain, title),
        )

    def get_timeline(self, claim_id: str) -> Dict[str, Any]:
        """Return the full timeline for a claim: when made, when contradicted."""
        claim_row = self.execute_one(
            "SELECT * FROM claim_timeline WHERE claim_id = ?", (claim_id,)
        )
        evidence_rows = self.execute(
            """SELECT * FROM evidence_timeline
               WHERE claim_id = ?
               ORDER BY published_date ASC""",
            (claim_id,),
        )

        claim = dict(claim_row) if claim_row else {}
        evidence = [dict(r) for r in evidence_rows]

        # Compute the delta: years between claim first made and first contradiction
        delta_analysis = self._compute_delta(claim, evidence)

        return {
            "claim_id": claim_id,
            "claim_timeline": claim,
            "evidence_timeline": evidence,
            "delta_analysis": delta_analysis,
        }

    def _compute_delta(
        self, claim: Dict[str, Any], evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze temporal gap between claim and contradicting evidence."""
        first_seen = claim.get("first_seen_date")
        if not first_seen:
            return {"status": "no_claim_date", "message": "Claim has no recorded first-seen date."}

        contradicting = [
            e for e in evidence
            if e.get("supports_or_contradicts") == "contradicts"
            and e.get("published_date")
        ]
        if not contradicting:
            return {
                "status": "no_contradictions",
                "message": f"No contradicting evidence found. Claim first seen {first_seen}.",
            }

        first_contradiction = contradicting[0].get("published_date")
        try:
            claim_dt = datetime.fromisoformat(first_seen[:10])
            ev_dt = datetime.fromisoformat(first_contradiction[:10])
            delta_days = (ev_dt - claim_dt).days
            return {
                "status": "analyzed",
                "claim_first_seen": first_seen,
                "first_contradiction": first_contradiction,
                "delta_days": delta_days,
                "message": (
                    f"Claim first seen {first_seen}, first contradicted "
                    f"{first_contradiction} ({delta_days} days later). "
                    f"{'Contradiction predates claim.' if delta_days < 0 else 'Evidence post-dates claim.'}"
                ),
            }
        except Exception:
            return {"status": "parse_error", "message": "Could not parse dates."}

    def count_claims(self) -> int:
        row = self.execute_one("SELECT COUNT(*) FROM claim_timeline")
        return int(row[0]) if row else 0
