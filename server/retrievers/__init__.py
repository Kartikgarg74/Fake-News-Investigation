"""Retrieval layer — live APIs that populate EvidenceDB.

Each retriever implements a common interface:

    class Retriever:
        SOURCE_TYPE: str  # stable identifier used as cache key

        def retrieve(self, claim: dict, query: str) -> dict:
            '''Return {"content": str, "source_url": str, "source_domain": str,
                       "published_date": str | None, "ok": bool}'''

Retrievers NEVER raise. On any failure (network error, rate limit, empty
response) they return {"ok": False, "content": "", ...}. The caller decides
how to handle misses.
"""

from .wikipedia import WikipediaRetriever
from .factcheck_api import FactCheckAPIRetriever
from .wikidata import WikidataRetriever
from .orchestrator import RetrievalOrchestrator

__all__ = [
    "WikipediaRetriever",
    "FactCheckAPIRetriever",
    "WikidataRetriever",
    "RetrievalOrchestrator",
]
