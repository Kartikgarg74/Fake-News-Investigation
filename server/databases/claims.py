"""ClaimsDB — claim metadata only.

This DB is the canonical source of claims. Evidence and other artifacts live
in other DBs, joined by claim_id. That separation is the point.
"""

from __future__ import annotations

import json
import random
import sqlite3
from typing import Any, Dict, List, Optional

from .base import DatabaseManager


# Difficulty tier mapping from LIAR labels (single source of truth)
DIFFICULTY_MAP = {
    "true": "easy",
    "false": "easy",
    "mostly-true": "medium",
    "barely-true": "medium",
    "half-true": "hard",
    "pants-fire": "hard",
}

BUDGET_MAP = {"easy": 10, "medium": 8, "hard": 6}


class ClaimsDB(DatabaseManager):
    filename = "claims.db"

    schema = """
    CREATE TABLE IF NOT EXISTS claims (
        id TEXT PRIMARY KEY,
        claim TEXT NOT NULL,
        label TEXT NOT NULL,
        speaker TEXT DEFAULT '',
        topic TEXT DEFAULT '',
        difficulty TEXT NOT NULL,
        claim_date TEXT DEFAULT NULL,
        has_image INTEGER DEFAULT 0,
        image_url TEXT DEFAULT NULL,
        gold_evidence TEXT DEFAULT '[]',
        gold_reasoning TEXT DEFAULT '',
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_claims_difficulty ON claims(difficulty);
    CREATE INDEX IF NOT EXISTS idx_claims_label ON claims(label);
    CREATE INDEX IF NOT EXISTS idx_claims_has_image ON claims(has_image);
    """

    # Columns that may be missing in DBs created before the multimedia /
    # temporal feature was added. _migrate() attempts to ALTER TABLE add each
    # one; duplicate-column errors are swallowed so this is idempotent.
    _MIGRATIONS = [
        ("image_url", "TEXT DEFAULT NULL"),
        ("has_image", "INTEGER DEFAULT 0"),
        ("claim_date", "TEXT DEFAULT NULL"),
        ("created_at", "TEXT"),  # SQLite can't add DEFAULT (datetime('now')) on ALTER
    ]

    def _migrate(self) -> None:
        """Add any missing columns to legacy claims.db files.

        Safe to run repeatedly — duplicate-column errors are swallowed.
        Also silently succeeds on read-only paths (the outer try/except
        catches sqlite3.OperationalError from the connection itself).
        """
        try:
            with self.connect() as conn:
                for col_name, col_def in self._MIGRATIONS:
                    try:
                        conn.execute(
                            f"ALTER TABLE claims ADD COLUMN {col_name} {col_def}"
                        )
                        conn.commit()
                    except sqlite3.OperationalError:
                        # Column already exists or table missing — both fine
                        pass
                # Backfill created_at for rows that don't have it (best effort)
                try:
                    conn.execute(
                        "UPDATE claims SET created_at = datetime('now') "
                        "WHERE created_at IS NULL OR created_at = ''"
                    )
                    conn.commit()
                except sqlite3.OperationalError:
                    pass
        except Exception:
            # Read-only DB or corrupt file — nothing we can do. The
            # environment will still function because _row_to_dict uses
            # dict(row).get() to tolerate missing columns.
            pass

    def _is_empty(self, conn: sqlite3.Connection) -> bool:
        try:
            row = conn.execute("SELECT COUNT(*) FROM claims").fetchone()
            return (row[0] if row else 0) == 0
        except sqlite3.OperationalError:
            return True

    def _seed(self, conn: sqlite3.Connection) -> None:
        """Load built-in sample claims so the env is functional on first boot."""
        for row in _BUILT_IN_SAMPLES:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO claims
                       (id, claim, label, speaker, topic, difficulty,
                        claim_date, has_image, image_url,
                        gold_evidence, gold_reasoning)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    row,
                )
            except Exception:
                continue

    def get_random(self, difficulty: str = "easy") -> Optional[Dict[str, Any]]:
        """Pick a random claim from a difficulty tier.

        Returns None if the DB is unavailable (not an exception — the env
        should degrade gracefully, not crash).
        """
        row = self.execute_one(
            "SELECT * FROM claims WHERE difficulty = ? ORDER BY RANDOM() LIMIT 1",
            (difficulty,),
        )
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_by_id(self, claim_id: str) -> Optional[Dict[str, Any]]:
        row = self.execute_one("SELECT * FROM claims WHERE id = ?", (claim_id,))
        return self._row_to_dict(row) if row else None

    def count(self, difficulty: Optional[str] = None) -> int:
        if difficulty:
            row = self.execute_one(
                "SELECT COUNT(*) FROM claims WHERE difficulty = ?", (difficulty,)
            )
        else:
            row = self.execute_one("SELECT COUNT(*) FROM claims")
        return int(row[0]) if row else 0

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a sqlite3.Row to a plain dict with JSON fields decoded.

        Uses dict(row).get() throughout — critical because older migrated
        DBs may be missing columns (image_url, claim_date, etc.) and we
        must never raise IndexError here.

        Legacy columns like `evidence_passages` (from the pre-split schema)
        are passed through transparently so `environment.py` can keep reading
        them during the transition period. Once all evidence retrieval goes
        through EvidenceDB this pass-through can be removed.
        """
        d = dict(row)

        def safe_json(val, default):
            if val is None:
                return default
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return default

        out = {
            "id": d.get("id", ""),
            "claim": d.get("claim", ""),
            "label": d.get("label", "false"),
            "speaker": d.get("speaker", ""),
            "topic": d.get("topic", ""),
            "difficulty": d.get("difficulty", "easy"),
            "claim_date": d.get("claim_date"),
            "has_image": bool(d.get("has_image", 0)),
            "image_url": d.get("image_url"),
            "gold_evidence": safe_json(d.get("gold_evidence"), []),
            "gold_reasoning": d.get("gold_reasoning", ""),
        }
        # Legacy pass-through: evidence_passages was embedded in claims.db
        # in the old monolithic schema. New code should use EvidenceDB.
        if "evidence_passages" in d:
            out["evidence_passages"] = safe_json(d.get("evidence_passages"), {})
        return out


# =========================================================================
# Built-in sample claims (minimal, just to keep the env bootable without
# any external data). Full LIAR dataset is loaded via setup_data.py.
# =========================================================================

_BUILT_IN_SAMPLES: List[tuple] = [
    (
        "easy_001",
        "The Great Wall of China is visible from space with the naked eye.",
        "false", "Common myth", "science", "easy",
        "2010-01-01", 0, None,
        json.dumps(["nasa_statement", "astronaut_testimonies"]),
        "Multiple astronauts have confirmed the Great Wall is not visible from low Earth orbit with the naked eye. The wall is only about 15 feet wide, too narrow to be seen from orbital altitude.",
    ),
    (
        "easy_002",
        "Humans use only 10 percent of their brains.",
        "false", "Popular culture", "science", "easy",
        "2005-01-01", 0, None,
        json.dumps(["neuroscience_studies", "brain_imaging_research"]),
        "Brain imaging studies show virtually all brain regions are active. The 10% myth has been debunked by neuroscientists.",
    ),
    (
        "easy_003",
        "Water boils at 100 degrees Celsius at sea level.",
        "true", "Science textbook", "science", "easy",
        "1900-01-01", 0, None,
        json.dumps(["physics_reference", "measurement_standards"]),
        "Water boils at 100°C at standard atmospheric pressure (1 atm). This defines the Celsius scale.",
    ),
    (
        "easy_004",
        "Lightning never strikes the same place twice.",
        "false", "Common saying", "science", "easy",
        "1990-01-01", 0, None,
        json.dumps(["weather_data", "lightning_research"]),
        "The Empire State Building is struck by lightning 20-25 times per year. Tall structures attract repeated strikes.",
    ),
    (
        "easy_005",
        "The Earth revolves around the Sun.",
        "true", "Science", "science", "easy",
        "1543-01-01", 0, None,
        json.dumps(["astronomy_data", "space_observation"]),
        "The heliocentric model has been confirmed by centuries of observation since Copernicus (1543).",
    ),
    (
        "medium_001",
        "A Harvard study found that drinking coffee reduces cancer risk by 50 percent.",
        "barely-true", "Health blog", "health", "medium",
        "2015-06-01", 0, None,
        json.dumps(["harvard_coffee_study", "cancer_meta_analyses"]),
        "Harvard researchers found modest risk reduction (~15%) for specific cancers, not 50% across the board. The claim exaggerates actual findings.",
    ),
    (
        "medium_002",
        "Crime rates have doubled under the current administration.",
        "barely-true", "Political commentator", "crime", "medium",
        "2023-01-01", 0, None,
        json.dumps(["fbi_crime_stats", "bjs_reports"]),
        "FBI UCR data does not support a doubling of national crime rates. Specific categories increased while others decreased.",
    ),
    (
        "medium_003",
        "Renewable energy is now cheaper than fossil fuels in every country.",
        "barely-true", "Clean energy advocate", "energy", "medium",
        "2023-06-01", 0, None,
        json.dumps(["irena_cost_reports", "iea_analysis"]),
        "IRENA 2023 data shows 86% of new renewables are cheaper than fossils, but not in every country — developing nations face higher costs.",
    ),
    (
        "hard_001",
        "The economy grew 12 percent last quarter according to official government statistics.",
        "half-true", "Government official", "economy", "hard",
        "2021-06-01", 0, None,
        json.dumps(["gdp_raw_data", "economic_context", "base_effect_analysis"]),
        "The 12% figure is a base-effect artifact from pandemic-quarter comparison. Adjusted quarter-over-quarter growth was ~2%.",
    ),
    (
        "hard_002",
        "More people died from the flu than from COVID-19 last year.",
        "pants-fire", "Social media post", "health", "hard",
        "2021-01-01", 0, None,
        json.dumps(["cdc_death_data", "who_mortality_data"]),
        "COVID-19 deaths exceeded flu deaths by 5-20x. CDC and WHO data consistently show this.",
    ),
    (
        "hard_003",
        "Electric vehicles produce more lifetime carbon emissions than gasoline cars when you account for battery manufacturing.",
        "half-true", "Auto industry analyst", "environment", "hard",
        "2020-01-01", 0, None,
        json.dumps(["lifecycle_analyses", "battery_manufacturing_data"]),
        "EVs produce 50-70% fewer lifecycle emissions than gasoline cars in most countries. The claim cherry-picks coal-heavy grid scenarios.",
    ),
    # Visual claims — has_image=1, image_url populated
    (
        "visual_001",
        "This photograph shows flooding in New York City caused by Hurricane Sandy in 2012.",
        "false", "Social media", "disaster", "easy",
        "2012-10-29", 1,
        "https://upload.wikimedia.org/wikipedia/commons/a/a8/2011_Thailand_flooding_Nakhon_Ratchasima.jpg",
        json.dumps(["image_analysis", "fact_checks"]),
        "The image is from the 2011 Thailand floods, not Hurricane Sandy. Misattributed and reposted with false caption.",
    ),
    (
        "visual_002",
        "This chart proves crime has skyrocketed 400% in the last two years.",
        "half-true", "Political commentator", "crime", "hard",
        "2023-01-01", 1, None,
        json.dumps(["image_analysis", "statistical_reports"]),
        "Chart uses truncated Y-axis starting at 950 instead of 0, making a 4% increase look like 400%. Data is real but presentation is misleading.",
    ),
    (
        "visual_003",
        "Leaked photo shows a government official signing a secret executive order banning protests.",
        "pants-fire", "Anonymous social media", "politics", "hard",
        "2024-01-01", 1, None,
        json.dumps(["image_analysis", "fact_checks"]),
        "Image is AI-generated. Visual forensics show inconsistent lighting, illegible background text, skin texture artifacts. No such executive order exists.",
    ),
]
