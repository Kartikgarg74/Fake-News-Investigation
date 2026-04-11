"""RetrievalOrchestrator — the glue between retrievers, cache, and audit log.

Contract:

    orchestrator.fetch(claim, source_type, query, episode_id) -> dict

    1. Check EvidenceDB cache first. If hit (within TTL), return it + mark
       `cache_hit=True`. Also log an audit row with the cached content hash.
    2. On miss, invoke the appropriate retriever, cache the result, and log
       the audit row with the fresh content hash.
    3. If the retriever fails, return a synthetic fallback from the legacy
       evidence_passages dict (if available on the claim) so the env never
       hands back an empty observation.

This is the ONE place where live network calls happen during an episode.
Everything else reads from EvidenceDB.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

from ..databases import EvidenceDB, TrajectoriesDB
from .factcheck_api import FactCheckAPIRetriever
from .wikidata import WikidataRetriever
from .wikipedia import WikipediaRetriever


class RetrievalOrchestrator:
    def __init__(
        self,
        evidence_db: Optional[EvidenceDB] = None,
        trajectories_db: Optional[TrajectoriesDB] = None,
    ):
        self.evidence = evidence_db or EvidenceDB()
        self.trajectories = trajectories_db or TrajectoriesDB()
        self.wikipedia = WikipediaRetriever()
        self.factcheck = FactCheckAPIRetriever()
        self.wikidata = WikidataRetriever()

        # Map source_type aliases to actual retrievers. The agent uses the
        # existing source category names (wikipedia, fact_check_api, etc.)
        # but we also map the legacy names (fact_checks, government_data)
        # to the closest real source so live retrieval works for old actions.
        self.retriever_map = {
            "wikipedia": self.wikipedia,
            "fact_check_api": self.factcheck,
            # Legacy aliases: map old category names to real retrievers
            "fact_checks": self.factcheck,
            "academic_papers": self.wikipedia,  # Wikipedia is reasonable fallback
            "news_articles": self.wikipedia,
            "government_data": self.wikipedia,
            "medical_journals": self.wikipedia,
            "statistical_reports": self.wikipedia,
            "international_organizations": self.wikipedia,
            "industry_reports": self.wikipedia,
        }

    def fetch(
        self,
        claim: Dict[str, Any],
        source_type: str,
        query: Optional[str] = None,
        episode_id: str = "",
        max_cache_age: int = 604800,
    ) -> Dict[str, Any]:
        """Fetch evidence for a claim from the given source type.

        Returns a dict with keys:
            content: str
            source_url: str
            source_domain: str
            source_type: str
            cache_hit: bool
            ok: bool
            is_synthetic: bool  (True if we fell back to templated evidence)
        """
        claim_id = claim.get("id", "")
        query_text = query or claim.get("claim", "")

        # 1. Cache check
        cached = self.evidence.get_cached(source_type, query_text, max_age=max_cache_age)
        if cached:
            self._log_audit(
                episode_id=episode_id,
                claim_id=claim_id,
                source_url=cached.get("source_url", ""),
                source_type=source_type,
                content_hash=cached.get("content_hash", ""),
                status="cache_hit",
            )
            return {
                "content": cached.get("content", ""),
                "source_url": cached.get("source_url", ""),
                "source_domain": cached.get("source_domain", ""),
                "source_type": source_type,
                "cache_hit": True,
                "ok": True,
                "is_synthetic": bool(cached.get("is_synthetic", 0)),
            }

        # 2. Live retrieval
        retriever = self.retriever_map.get(source_type)
        if retriever is not None:
            result = retriever.retrieve(claim, query=query_text)
            if result.get("ok"):
                # Cache the fresh result
                self.evidence.store(
                    claim_id=claim_id,
                    source_type=source_type,
                    query=query_text,
                    content=result["content"],
                    source_url=result.get("source_url", ""),
                    source_domain=result.get("source_domain", ""),
                    is_synthetic=False,
                )
                content_hash = hashlib.sha256(
                    result["content"].encode("utf-8")
                ).hexdigest()[:16]
                self._log_audit(
                    episode_id=episode_id,
                    claim_id=claim_id,
                    source_url=result.get("source_url", ""),
                    source_type=source_type,
                    content_hash=content_hash,
                    status="live",
                )
                return {
                    "content": result["content"],
                    "source_url": result.get("source_url", ""),
                    "source_domain": result.get("source_domain", ""),
                    "source_type": source_type,
                    "cache_hit": False,
                    "ok": True,
                    "is_synthetic": False,
                }

        # 3. Fallback to legacy templated evidence (if attached to the claim)
        legacy = self._legacy_fallback(claim, source_type)
        if legacy:
            # Cache the synthetic fallback too so we don't keep re-checking
            self.evidence.store(
                claim_id=claim_id,
                source_type=source_type,
                query=query_text,
                content=legacy,
                source_url="",
                source_domain="legacy",
                is_synthetic=True,
            )
            content_hash = hashlib.sha256(legacy.encode("utf-8")).hexdigest()[:16]
            self._log_audit(
                episode_id=episode_id,
                claim_id=claim_id,
                source_url="",
                source_type=source_type,
                content_hash=content_hash,
                status="synthetic",
            )
            return {
                "content": legacy,
                "source_url": "",
                "source_domain": "legacy",
                "source_type": source_type,
                "cache_hit": False,
                "ok": True,
                "is_synthetic": True,
            }

        # 4. Complete miss
        return {
            "content": "",
            "source_url": "",
            "source_domain": "",
            "source_type": source_type,
            "cache_hit": False,
            "ok": False,
            "is_synthetic": False,
        }

    def _legacy_fallback(self, claim: Dict[str, Any], source_type: str) -> str:
        """Return templated evidence from the legacy evidence_passages dict."""
        passages = claim.get("evidence_passages") or {}
        if not isinstance(passages, dict):
            return ""
        # Direct hit
        if source_type in passages:
            return str(passages[source_type])
        # Substring match (e.g. "wikipedia" matches "academic_papers" nope,
        # but "fact_checks" might match "fact_check_api")
        for key, val in passages.items():
            if source_type in key or key in source_type:
                return str(val)
        return ""

    def _log_audit(
        self,
        episode_id: str,
        claim_id: str,
        source_url: str,
        source_type: str,
        content_hash: str,
        status: str,
    ) -> None:
        """Write a chain-of-custody audit row. Failures swallowed — audit
        must never block a retrieval."""
        try:
            self.trajectories.log_audit(
                episode_id=episode_id,
                claim_id=claim_id,
                source_url=source_url,
                source_type=source_type,
                content_hash=content_hash,
                status=status,
            )
        except Exception:
            pass
