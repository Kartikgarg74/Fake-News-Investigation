"""SourcesDB — publisher credibility database.

Seeded from MBFC (Media Bias Fact Check) public data. Falls back to a small
built-in set when MBFC scrape isn't available. Lookups use normalized domain
names with fuzzy matching (substring + public suffix stripping).
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

from .base import DatabaseManager


class SourcesDB(DatabaseManager):
    filename = "sources.db"

    schema = """
    CREATE TABLE IF NOT EXISTS sources (
        domain TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        bias TEXT DEFAULT 'Unknown',
        factual_reporting TEXT DEFAULT 'Unknown',
        credibility_score REAL DEFAULT 0.5,
        country TEXT DEFAULT '',
        media_type TEXT DEFAULT '',
        mbfc_url TEXT DEFAULT '',
        source TEXT DEFAULT 'mbfc',
        updated_at TEXT DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_sources_bias ON sources(bias);
    CREATE INDEX IF NOT EXISTS idx_sources_factual ON sources(factual_reporting);
    CREATE INDEX IF NOT EXISTS idx_sources_score ON sources(credibility_score);
    """

    def _is_empty(self, conn: sqlite3.Connection) -> bool:
        try:
            row = conn.execute("SELECT COUNT(*) FROM sources").fetchone()
            return (row[0] if row else 0) == 0
        except sqlite3.OperationalError:
            return True

    def _seed(self, conn: sqlite3.Connection) -> None:
        """Seed with a curated minimum set. The full MBFC scrape loads separately."""
        for row in _BUILT_IN_SOURCES:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO sources
                       (domain, name, bias, factual_reporting, credibility_score,
                        country, media_type, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    row,
                )
            except Exception:
                continue

    @staticmethod
    def _normalize(domain_or_url: str) -> str:
        """Strip scheme, www, path, trailing slashes. Returns lowercase domain."""
        s = (domain_or_url or "").lower().strip()
        for prefix in ("https://", "http://"):
            if s.startswith(prefix):
                s = s[len(prefix):]
        if s.startswith("www."):
            s = s[4:]
        s = s.split("/")[0]
        return s.rstrip("/")

    def lookup(self, source_id: str) -> Dict[str, Any]:
        """Look up credibility for a source. Never raises; returns Unknown on miss."""
        domain = self._normalize(source_id)

        # Direct match
        row = self.execute_one("SELECT * FROM sources WHERE domain = ?", (domain,))
        if row:
            return self._to_dict(row, found=True)

        # Substring match (e.g. "cnn" matches "cnn.com")
        row = self.execute_one(
            """SELECT * FROM sources
               WHERE domain LIKE ? OR ? LIKE '%' || domain || '%'
               ORDER BY LENGTH(domain) DESC LIMIT 1""",
            (f"%{domain}%", domain),
        )
        if row:
            return self._to_dict(row, found=True)

        # Default — unknown source gets neutral score
        return {
            "source": source_id,
            "domain": domain,
            "name": domain or "Unknown",
            "bias": "Unknown",
            "factual_reporting": "Unknown",
            "credibility_score": 0.5,
            "found": False,
        }

    def _to_dict(self, row: sqlite3.Row, found: bool) -> Dict[str, Any]:
        d = dict(row)
        return {
            "source": d.get("domain", ""),
            "domain": d.get("domain", ""),
            "name": d.get("name", d.get("domain", "")),
            "bias": d.get("bias", "Unknown"),
            "factual_reporting": d.get("factual_reporting", "Unknown"),
            "credibility_score": float(d.get("credibility_score", 0.5)),
            "country": d.get("country", ""),
            "media_type": d.get("media_type", ""),
            "found": found,
        }

    def bulk_load(self, rows: List[Dict[str, Any]]) -> int:
        """Bulk insert from MBFC scrape. Returns count of rows inserted."""
        if not rows:
            return 0
        tuples = [
            (
                r.get("domain", ""),
                r.get("name", r.get("domain", "")),
                r.get("bias", "Unknown"),
                r.get("factual_reporting", "Unknown"),
                float(r.get("credibility_score", 0.5)),
                r.get("country", ""),
                r.get("media_type", ""),
                r.get("mbfc_url", ""),
                r.get("source", "mbfc"),
            )
            for r in rows if r.get("domain")
        ]
        ok = self.writemany(
            """INSERT OR REPLACE INTO sources
               (domain, name, bias, factual_reporting, credibility_score,
                country, media_type, mbfc_url, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            tuples,
        )
        return len(tuples) if ok else 0

    def count(self) -> int:
        row = self.execute_one("SELECT COUNT(*) FROM sources")
        return int(row[0]) if row else 0


_BUILT_IN_SOURCES: List[tuple] = [
    ("reuters.com", "Reuters", "Center", "Very High", 0.95, "UK", "News Agency", "built-in"),
    ("apnews.com", "Associated Press", "Center", "Very High", 0.95, "US", "News Agency", "built-in"),
    ("bbc.com", "BBC", "Center-Left", "High", 0.90, "UK", "News", "built-in"),
    ("nytimes.com", "New York Times", "Center-Left", "High", 0.88, "US", "News", "built-in"),
    ("washingtonpost.com", "Washington Post", "Center-Left", "High", 0.87, "US", "News", "built-in"),
    ("wsj.com", "Wall Street Journal", "Center-Right", "High", 0.88, "US", "News", "built-in"),
    ("nature.com", "Nature", "Center", "Very High", 0.97, "UK", "Science", "built-in"),
    ("thelancet.com", "The Lancet", "Center", "Very High", 0.96, "UK", "Medical", "built-in"),
    ("nejm.org", "New England Journal of Medicine", "Center", "Very High", 0.97, "US", "Medical", "built-in"),
    ("who.int", "World Health Organization", "Center", "High", 0.90, "INT", "Government", "built-in"),
    ("cdc.gov", "CDC", "Center", "Very High", 0.93, "US", "Government", "built-in"),
    ("nih.gov", "NIH", "Center", "Very High", 0.94, "US", "Government", "built-in"),
    ("nasa.gov", "NASA", "Center", "Very High", 0.96, "US", "Government", "built-in"),
    ("scientificamerican.com", "Scientific American", "Center-Left", "High", 0.88, "US", "Science", "built-in"),
    ("snopes.com", "Snopes", "Center", "Very High", 0.92, "US", "Fact-check", "built-in"),
    ("politifact.com", "PolitiFact", "Center", "High", 0.90, "US", "Fact-check", "built-in"),
    ("factcheck.org", "FactCheck.org", "Center", "Very High", 0.92, "US", "Fact-check", "built-in"),
    ("foxnews.com", "Fox News", "Right", "Mixed", 0.55, "US", "News", "built-in"),
    ("breitbart.com", "Breitbart", "Far Right", "Low", 0.20, "US", "News", "built-in"),
    ("infowars.com", "InfoWars", "Far Right", "Very Low", 0.05, "US", "Conspiracy", "built-in"),
    ("dailymail.co.uk", "Daily Mail", "Right", "Low", 0.35, "UK", "News", "built-in"),
    ("buzzfeed.com", "BuzzFeed", "Left", "Mixed", 0.50, "US", "News", "built-in"),
    ("naturalnews.com", "Natural News", "Far Right", "Very Low", 0.08, "US", "Conspiracy", "built-in"),
    ("iea.org", "IEA", "Center", "High", 0.88, "INT", "Think Tank", "built-in"),
    ("worldbank.org", "World Bank", "Center", "High", 0.90, "INT", "Government", "built-in"),
    ("imf.org", "IMF", "Center", "High", 0.89, "INT", "Government", "built-in"),
    ("fbi.gov", "FBI", "Center", "Very High", 0.93, "US", "Government", "built-in"),
    ("bls.gov", "Bureau of Labor Statistics", "Center", "Very High", 0.95, "US", "Government", "built-in"),
    ("en.wikipedia.org", "Wikipedia", "Center", "Mixed", 0.75, "INT", "Encyclopedia", "built-in"),
    ("wikipedia.org", "Wikipedia", "Center", "Mixed", 0.75, "INT", "Encyclopedia", "built-in"),
]
