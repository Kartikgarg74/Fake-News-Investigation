"""EvidenceDB — FTS5-indexed corpus of retrieved evidence.

Evidence comes from multiple sources:
- Wikipedia REST API (live, cached)
- Google Fact Check Tools API (live, cached)
- PolitiFact article text (one-time scrape)
- Snopes article text (one-time scrape)
- Templated fallback (existing synthetic evidence, used as last resort)

Each evidence row is keyed by (source_type, query_hash) so identical queries
hit the cache. TTL is 7 days for live sources, infinite for static scrapes.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

from .base import DatabaseManager


class EvidenceDB(DatabaseManager):
    filename = "evidence.db"

    schema = """
    CREATE TABLE IF NOT EXISTS evidence (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim_id TEXT NOT NULL,
        source_type TEXT NOT NULL,
        source_url TEXT DEFAULT '',
        source_domain TEXT DEFAULT '',
        query TEXT NOT NULL,
        query_hash TEXT NOT NULL,
        content TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        fetched_at INTEGER NOT NULL,
        ttl_seconds INTEGER DEFAULT 604800,
        is_synthetic INTEGER DEFAULT 0
    );

    CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_cache_key
        ON evidence(source_type, query_hash);
    CREATE INDEX IF NOT EXISTS idx_evidence_claim ON evidence(claim_id);
    CREATE INDEX IF NOT EXISTS idx_evidence_source ON evidence(source_type);

    CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
        content,
        query,
        source_type UNINDEXED,
        source_domain UNINDEXED,
        content='evidence',
        content_rowid='id'
    );

    CREATE TRIGGER IF NOT EXISTS evidence_ai AFTER INSERT ON evidence BEGIN
        INSERT INTO evidence_fts(rowid, content, query, source_type, source_domain)
        VALUES (new.id, new.content, new.query, new.source_type, new.source_domain);
    END;

    CREATE TRIGGER IF NOT EXISTS evidence_ad AFTER DELETE ON evidence BEGIN
        INSERT INTO evidence_fts(evidence_fts, rowid, content, query, source_type, source_domain)
        VALUES('delete', old.id, old.content, old.query, old.source_type, old.source_domain);
    END;
    """

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def get_cached(
        self, source_type: str, query: str, max_age: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Look up cached evidence by (source_type, query). Respects TTL."""
        qhash = self._hash(f"{source_type}|{query}")
        row = self.execute_one(
            """SELECT * FROM evidence
               WHERE source_type = ? AND query_hash = ?
               ORDER BY fetched_at DESC LIMIT 1""",
            (source_type, qhash),
        )
        if row is None:
            return None
        d = dict(row)
        if max_age is not None:
            age = int(time.time()) - int(d.get("fetched_at", 0))
            # Use >= so max_age=0 always invalidates (useful for tests)
            if age >= max_age:
                return None
        return d

    def store(
        self,
        claim_id: str,
        source_type: str,
        query: str,
        content: str,
        source_url: str = "",
        source_domain: str = "",
        ttl_seconds: int = 604800,
        is_synthetic: bool = False,
    ) -> bool:
        """Insert or replace cached evidence."""
        qhash = self._hash(f"{source_type}|{query}")
        chash = self._hash(content)
        fetched_at = int(time.time())
        # Delete stale entry first so FTS triggers stay consistent
        self.write(
            "DELETE FROM evidence WHERE source_type = ? AND query_hash = ?",
            (source_type, qhash),
        )
        return self.write(
            """INSERT INTO evidence
               (claim_id, source_type, source_url, source_domain,
                query, query_hash, content, content_hash,
                fetched_at, ttl_seconds, is_synthetic)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                claim_id, source_type, source_url, source_domain,
                query, qhash, content, chash,
                fetched_at, ttl_seconds, 1 if is_synthetic else 0,
            ),
        )

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Full-text search across the evidence corpus.

        Returns ranked results by BM25 relevance. Quotes are stripped and
        special FTS5 characters are escaped to prevent syntax errors from
        user input.
        """
        safe_query = query.replace('"', '').replace("'", "")
        # FTS5 needs terms wrapped for multi-word queries
        try:
            rows = self.execute(
                """SELECT e.id, e.claim_id, e.source_type, e.source_url,
                          e.source_domain, e.content, e.query, bm25(evidence_fts) as rank
                   FROM evidence_fts
                   JOIN evidence e ON e.id = evidence_fts.rowid
                   WHERE evidence_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (safe_query, limit),
            )
        except Exception:
            rows = []
        return [dict(r) for r in rows]

    def get_for_claim(self, claim_id: str) -> List[Dict[str, Any]]:
        """All evidence rows for a specific claim."""
        rows = self.execute(
            "SELECT * FROM evidence WHERE claim_id = ? ORDER BY fetched_at DESC",
            (claim_id,),
        )
        return [dict(r) for r in rows]

    def count(self) -> int:
        row = self.execute_one("SELECT COUNT(*) FROM evidence")
        return int(row[0]) if row else 0
