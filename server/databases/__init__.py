"""Segregated database layer for Veritas.

Seven specialized SQLite databases, each with a single responsibility:

- claims.db       — claim metadata (text, label, difficulty, date, speaker, topic)
- evidence.db     — FTS5-indexed evidence corpus (Wikipedia, PolitiFact, Snopes)
- sources.db      — publisher credibility (MBFC-sourced, ~4k publishers)
- images.db       — perceptual hashes of known misattributed images
- temporal.db     — claim + evidence timelines for temporal verification
- entities.db     — Wikidata-backed entity cache
- trajectories.db — RL trajectory logs + chain-of-custody audit

Separation rationale:
1. Each DB can evolve its schema independently (no coupling).
2. Different read/write patterns can be tuned per DB (e.g. FTS5 on evidence).
3. A corrupted or stale DB fails gracefully without taking down the rest.
4. Clear mental model: one question → one database.
"""

from .base import DatabaseManager
from .claims import ClaimsDB
from .evidence import EvidenceDB
from .sources import SourcesDB
from .images import ImagesDB
from .temporal import TemporalDB
from .entities import EntitiesDB
from .trajectories import TrajectoriesDB

__all__ = [
    "DatabaseManager",
    "ClaimsDB",
    "EvidenceDB",
    "SourcesDB",
    "ImagesDB",
    "TemporalDB",
    "EntitiesDB",
    "TrajectoriesDB",
]
