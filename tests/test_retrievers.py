"""Unit tests for the retrieval layer.

All network-dependent tests are skipped by default and only run when the
VERITAS_LIVE_TESTS env var is set. The default test suite uses mocked
retrievers so CI is fast and deterministic.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fake_news_investigator.server.databases import EvidenceDB, TrajectoriesDB
from fake_news_investigator.server.retrievers import (
    RetrievalOrchestrator,
    WikipediaRetriever,
    FactCheckAPIRetriever,
    WikidataRetriever,
)

LIVE = os.environ.get("VERITAS_LIVE_TESTS") == "1"


@pytest.fixture
def tmp_dbs():
    with tempfile.TemporaryDirectory() as td:
        evidence = EvidenceDB(db_path=str(Path(td) / "evidence.db"))
        trajectories = TrajectoriesDB(db_path=str(Path(td) / "trajectories.db"))
        yield evidence, trajectories


# ---------- Orchestrator: cache, audit, fallback ----------

def test_orchestrator_cache_hit_on_second_call(tmp_dbs):
    evidence, trajectories = tmp_dbs
    orch = RetrievalOrchestrator(evidence_db=evidence, trajectories_db=trajectories)

    # Mock the Wikipedia retriever to return a deterministic result
    orch.wikipedia = MagicMock()
    orch.wikipedia.retrieve.return_value = {
        "ok": True,
        "content": "Mocked Wikipedia content about a test claim.",
        "source_url": "https://en.wikipedia.org/wiki/Test",
        "source_domain": "en.wikipedia.org",
    }
    orch.retriever_map["wikipedia"] = orch.wikipedia

    claim = {"id": "c1", "claim": "Test claim"}

    # First call: should be live
    r1 = orch.fetch(claim, source_type="wikipedia", episode_id="ep1")
    assert r1["ok"]
    assert r1["cache_hit"] is False
    assert orch.wikipedia.retrieve.call_count == 1

    # Second call: should be cached, no additional retriever call
    r2 = orch.fetch(claim, source_type="wikipedia", episode_id="ep1")
    assert r2["ok"]
    assert r2["cache_hit"] is True
    assert orch.wikipedia.retrieve.call_count == 1  # still 1


def test_orchestrator_legacy_fallback_when_no_retriever(tmp_dbs):
    evidence, trajectories = tmp_dbs
    orch = RetrievalOrchestrator(evidence_db=evidence, trajectories_db=trajectories)

    claim = {
        "id": "c2",
        "claim": "Claim with legacy evidence",
        "evidence_passages": {"fact_checks": "Legacy templated evidence."},
    }
    # medical_journals has a retriever mapped to wikipedia; test a truly
    # unmapped category... actually all source_types have a mapping in v2.
    # Force a miss by mocking wikipedia to return ok=False
    orch.wikipedia.retrieve = MagicMock(return_value={"ok": False, "content": ""})

    r = orch.fetch(claim, source_type="fact_checks", episode_id="ep2")
    # fact_checks maps to factcheck retriever which has no API key -> ok=False
    # Then falls through to legacy_fallback
    assert r["ok"] is True
    assert r["is_synthetic"] is True
    assert "Legacy templated" in r["content"]


def test_orchestrator_complete_miss(tmp_dbs):
    evidence, trajectories = tmp_dbs
    orch = RetrievalOrchestrator(evidence_db=evidence, trajectories_db=trajectories)

    # Mock all retrievers to fail
    for retriever in (orch.wikipedia, orch.factcheck):
        retriever.retrieve = MagicMock(return_value={"ok": False, "content": ""})

    claim = {"id": "c3", "claim": "Unknown claim", "evidence_passages": {}}
    r = orch.fetch(claim, source_type="wikipedia", episode_id="ep3")
    assert r["ok"] is False


def test_orchestrator_logs_audit_rows(tmp_dbs):
    evidence, trajectories = tmp_dbs
    orch = RetrievalOrchestrator(evidence_db=evidence, trajectories_db=trajectories)

    orch.wikipedia = MagicMock()
    orch.wikipedia.retrieve.return_value = {
        "ok": True,
        "content": "Audit test content.",
        "source_url": "https://en.wikipedia.org/wiki/Audit",
        "source_domain": "en.wikipedia.org",
    }
    orch.retriever_map["wikipedia"] = orch.wikipedia

    claim = {"id": "c_audit", "claim": "Audit claim"}
    orch.fetch(claim, source_type="wikipedia", episode_id="ep_audit")

    # Verify audit row was written
    with trajectories.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM audit WHERE episode_id = 'ep_audit'"
        ).fetchall()
    assert len(rows) == 1


# ---------- WikipediaRetriever unit tests (no network) ----------

def test_wikipedia_retriever_extracts_capitalized_entities():
    r = WikipediaRetriever()
    q = r._extract_search_terms("The Great Wall of China was built during the Ming Dynasty.")
    # Should find a multi-word capitalized phrase
    assert any(term in q for term in ("Great Wall", "China", "Ming Dynasty"))


def test_wikipedia_retriever_empty_claim():
    r = WikipediaRetriever()
    result = r.retrieve({"claim": ""})
    assert result["ok"] is False


# ---------- FactCheckAPIRetriever ----------

def test_factcheck_retriever_degrades_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_FACTCHECK_API_KEY", raising=False)
    r = FactCheckAPIRetriever()
    result = r.retrieve({"claim": "test claim"})
    assert result["ok"] is False
    assert result["error"] == "api_key_missing"


# ---------- Live tests (opt-in via env var) ----------

@pytest.mark.skipif(not LIVE, reason="Set VERITAS_LIVE_TESTS=1 to enable")
def test_wikipedia_live_retrieval():
    r = WikipediaRetriever()
    result = r.retrieve({"claim": "The Great Wall of China is visible from space."})
    assert result["ok"] is True
    assert "Great Wall" in result["content"] or "wall" in result["content"].lower()


@pytest.mark.skipif(not LIVE, reason="Set VERITAS_LIVE_TESTS=1 to enable")
def test_wikidata_live_retrieval():
    r = WikidataRetriever()
    result = r.retrieve("Great Wall of China")
    assert result["ok"] is True
    assert result["wikidata_id"].startswith("Q")
