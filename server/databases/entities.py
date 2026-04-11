"""EntitiesDB — cached entity metadata from Wikidata.

The agent needs to resolve entities mentioned in claims (people, orgs, places,
events) to canonical knowledge. Wikidata is the public-domain source of truth
for this. We cache lookups here so we don't hammer their SPARQL endpoint.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

from .base import DatabaseManager


class EntitiesDB(DatabaseManager):
    filename = "entities.db"

    schema = """
    CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name_normalized TEXT NOT NULL,
        display_name TEXT NOT NULL,
        wikidata_id TEXT DEFAULT '',
        entity_type TEXT DEFAULT 'unknown',
        description TEXT DEFAULT '',
        aliases_json TEXT DEFAULT '[]',
        properties_json TEXT DEFAULT '{}',
        fetched_at INTEGER NOT NULL,
        ttl_seconds INTEGER DEFAULT 2592000
    );

    CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_name ON entities(name_normalized);
    CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type);
    CREATE INDEX IF NOT EXISTS idx_entity_wikidata ON entities(wikidata_id);
    """

    @staticmethod
    def _normalize(name: str) -> str:
        return (name or "").lower().strip()

    def lookup(self, name: str) -> Optional[Dict[str, Any]]:
        """Look up a cached entity by name. Returns None on miss."""
        key = self._normalize(name)
        if not key:
            return None

        row = self.execute_one(
            "SELECT * FROM entities WHERE name_normalized = ?", (key,)
        )
        if row is None:
            return None

        d = dict(row)
        # Respect TTL
        age = int(time.time()) - int(d.get("fetched_at", 0))
        if age > int(d.get("ttl_seconds", 2592000)):
            return None

        try:
            aliases = json.loads(d.get("aliases_json") or "[]")
        except Exception:
            aliases = []
        try:
            properties = json.loads(d.get("properties_json") or "{}")
        except Exception:
            properties = {}

        return {
            "name": d.get("display_name", name),
            "wikidata_id": d.get("wikidata_id", ""),
            "type": d.get("entity_type", "unknown"),
            "description": d.get("description", ""),
            "aliases": aliases,
            "properties": properties,
        }

    def store(
        self,
        name: str,
        display_name: str,
        wikidata_id: str = "",
        entity_type: str = "unknown",
        description: str = "",
        aliases: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        key = self._normalize(name)
        if not key:
            return False
        self.write("DELETE FROM entities WHERE name_normalized = ?", (key,))
        return self.write(
            """INSERT INTO entities
               (name_normalized, display_name, wikidata_id, entity_type,
                description, aliases_json, properties_json, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                key,
                display_name,
                wikidata_id,
                entity_type,
                description,
                json.dumps(aliases or []),
                json.dumps(properties or {}),
                int(time.time()),
            ),
        )

    def count(self) -> int:
        row = self.execute_one("SELECT COUNT(*) FROM entities")
        return int(row[0]) if row else 0
