"""Unit tests for the segregated database layer.

Scope: schema creation, basic CRUD, safe-query behavior on missing data.
Does NOT hit any network — all tests are offline and fast.
"""

import tempfile
from pathlib import Path

import pytest

from fake_news_investigator.server.databases import (
    ClaimsDB,
    EvidenceDB,
    SourcesDB,
    ImagesDB,
    TemporalDB,
    EntitiesDB,
    TrajectoriesDB,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


# ---------- ClaimsDB ----------

def test_claims_db_boots_with_samples(tmp_dir):
    db = ClaimsDB(db_path=str(tmp_dir / "claims.db"))
    assert db.count() >= 10, "Built-in samples should seed the DB"
    for diff in ("easy", "medium", "hard"):
        assert db.count(diff) > 0, f"Difficulty {diff} should have at least one claim"


def test_claims_db_get_random(tmp_dir):
    db = ClaimsDB(db_path=str(tmp_dir / "claims.db"))
    c = db.get_random("easy")
    assert c is not None
    assert "id" in c
    assert "claim" in c
    assert "label" in c
    assert "image_url" in c or c.get("image_url") is None


def test_claims_db_get_random_unknown_difficulty_returns_none(tmp_dir):
    db = ClaimsDB(db_path=str(tmp_dir / "claims.db"))
    # Should return None, not raise
    assert db.get_random("nonexistent") is None


def test_claims_db_missing_columns_safe(tmp_dir):
    """ClaimsDB._row_to_dict must not raise even if columns are missing."""
    import sqlite3
    db_path = tmp_dir / "legacy.db"

    # Create a legacy-schema DB (no image_url / has_image / claim_date)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE claims (
            id TEXT PRIMARY KEY, claim TEXT, label TEXT,
            speaker TEXT, topic TEXT, difficulty TEXT,
            gold_evidence TEXT, gold_reasoning TEXT, evidence_passages TEXT
        )
    """)
    conn.execute(
        "INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("legacy_1", "test claim", "true", "speaker", "topic",
         "easy", "[]", "reason", "{}")
    )
    conn.commit()
    conn.close()

    db = ClaimsDB(db_path=str(db_path))
    c = db.get_random("easy")
    assert c is not None
    assert c["id"] == "legacy_1"
    # Missing columns should default to None / False
    assert c.get("image_url") is None
    assert c.get("has_image") in (False, 0)


# ---------- EvidenceDB ----------

def test_evidence_db_store_and_retrieve(tmp_dir):
    db = EvidenceDB(db_path=str(tmp_dir / "evidence.db"))
    assert db.count() == 0

    ok = db.store(
        claim_id="c1",
        source_type="wikipedia",
        query="test query",
        content="Test content about the claim.",
        source_url="https://en.wikipedia.org/wiki/Test",
        source_domain="en.wikipedia.org",
    )
    assert ok
    assert db.count() == 1

    cached = db.get_cached("wikipedia", "test query")
    assert cached is not None
    assert cached["content"] == "Test content about the claim."


def test_evidence_db_cache_ttl(tmp_dir):
    db = EvidenceDB(db_path=str(tmp_dir / "evidence.db"))
    db.store(claim_id="c1", source_type="wikipedia", query="q", content="x")
    # max_age=0 should invalidate everything
    assert db.get_cached("wikipedia", "q", max_age=0) is None


def test_evidence_db_fts_search(tmp_dir):
    db = EvidenceDB(db_path=str(tmp_dir / "evidence.db"))
    db.store(claim_id="c1", source_type="wiki", query="moon", content="The moon is Earth's satellite")
    db.store(claim_id="c2", source_type="wiki", query="sun", content="The sun is a star at the solar system center")
    hits = db.search("moon")
    assert len(hits) >= 1
    assert any("moon" in h["content"].lower() for h in hits)


def test_evidence_db_search_empty_query_doesnt_crash(tmp_dir):
    db = EvidenceDB(db_path=str(tmp_dir / "evidence.db"))
    # Should return empty list, not raise
    hits = db.search("'nonexistent\"")  # quotes should be stripped
    assert isinstance(hits, list)


# ---------- SourcesDB ----------

def test_sources_db_bootstrapped_with_builtin(tmp_dir):
    db = SourcesDB(db_path=str(tmp_dir / "sources.db"))
    assert db.count() >= 20, "Should have built-in publishers"


def test_sources_db_lookup_exact(tmp_dir):
    db = SourcesDB(db_path=str(tmp_dir / "sources.db"))
    r = db.lookup("reuters.com")
    assert r["found"]
    assert r["credibility_score"] > 0.9


def test_sources_db_lookup_with_scheme(tmp_dir):
    db = SourcesDB(db_path=str(tmp_dir / "sources.db"))
    r = db.lookup("https://www.bbc.com/news/world")
    assert r["found"]
    assert "bbc" in r["domain"]


def test_sources_db_lookup_unknown_returns_neutral(tmp_dir):
    db = SourcesDB(db_path=str(tmp_dir / "sources.db"))
    r = db.lookup("some-random-unknown-domain.xyz")
    assert not r["found"]
    assert r["credibility_score"] == 0.5


def test_sources_db_bulk_load(tmp_dir):
    db = SourcesDB(db_path=str(tmp_dir / "sources.db"))
    before = db.count()
    n = db.bulk_load([
        {"domain": "testsite.com", "name": "Test Site", "bias": "Center",
         "factual_reporting": "High", "credibility_score": 0.85},
    ])
    assert n == 1
    assert db.count() == before + 1
    assert db.lookup("testsite.com")["found"]


# ---------- ImagesDB ----------

def test_images_db_phash_match(tmp_dir):
    db = ImagesDB(db_path=str(tmp_dir / "images.db"))
    db.add(
        phash="ff00ff00ff00ff00",
        image_url="https://example.com/test.jpg",
        verdict="misattributed",
        description="Test image",
    )
    # Exact match
    match = db.find_similar("ff00ff00ff00ff00", threshold=0)
    assert match is not None
    assert match["verdict"] == "misattributed"


def test_images_db_hamming_distance():
    from fake_news_investigator.server.databases.images import ImagesDB
    assert ImagesDB._hamming("ff00ff00ff00ff00", "ff00ff00ff00ff00") == 0
    assert ImagesDB._hamming("ff00ff00ff00ff00", "ff00ff00ff00ff01") == 1
    assert ImagesDB._hamming("00000000", "ffffffff") == 32
    # Mismatched length returns max
    assert ImagesDB._hamming("ff", "ffff") == 64


def test_images_db_no_match_over_threshold(tmp_dir):
    db = ImagesDB(db_path=str(tmp_dir / "images.db"))
    db.add(phash="ff00ff00ff00ff00", image_url="u", verdict="x", description="")
    # 64 bits flipped -> 64 hamming distance, way above threshold
    assert db.find_similar("00ff00ff00ff00ff", threshold=5) is None


# ---------- TemporalDB ----------

def test_temporal_db_records_and_reads(tmp_dir):
    db = TemporalDB(db_path=str(tmp_dir / "temporal.db"))
    db.record_claim("c1", first_seen_date="2020-01-15")
    db.record_evidence(
        evidence_id="e1",
        claim_id="c1",
        published_date="2022-03-10",
        supports_or_contradicts="contradicts",
        source_domain="wikipedia.org",
        title="Contradiction",
    )
    timeline = db.get_timeline("c1")
    assert timeline["claim_id"] == "c1"
    assert timeline["claim_timeline"]["first_seen_date"] == "2020-01-15"
    assert len(timeline["evidence_timeline"]) == 1
    delta = timeline["delta_analysis"]
    assert delta["status"] == "analyzed"
    assert delta["delta_days"] > 0


def test_temporal_db_no_claim_date(tmp_dir):
    db = TemporalDB(db_path=str(tmp_dir / "temporal.db"))
    timeline = db.get_timeline("nonexistent")
    assert timeline["delta_analysis"]["status"] == "no_claim_date"


# ---------- EntitiesDB ----------

def test_entities_db_store_and_lookup(tmp_dir):
    db = EntitiesDB(db_path=str(tmp_dir / "entities.db"))
    ok = db.store(
        name="NASA",
        display_name="NASA",
        wikidata_id="Q23548",
        entity_type="space_agency",
        description="National Aeronautics and Space Administration",
        aliases=["National Aeronautics and Space Administration"],
        properties={"country": "US"},
    )
    assert ok
    r = db.lookup("NASA")
    assert r is not None
    assert r["wikidata_id"] == "Q23548"
    assert r["type"] == "space_agency"
    assert "National" in r["description"]


def test_entities_db_lookup_miss_returns_none(tmp_dir):
    db = EntitiesDB(db_path=str(tmp_dir / "entities.db"))
    assert db.lookup("nonexistent_entity_xyz") is None


# ---------- TrajectoriesDB ----------

def test_trajectories_db_log_step(tmp_dir):
    db = TrajectoriesDB(db_path=str(tmp_dir / "trajectories.db"))
    ok = db.log_step(
        episode_id="ep1",
        step_index=0,
        claim_id="c1",
        difficulty="easy",
        state={"budget": 10},
        action={"action_type": "request_source", "source_id": "wikipedia"},
        reward=0.5,
        done=False,
    )
    assert ok
    assert db.count_steps() == 1
    assert db.count_episodes() == 1

    steps = db.get_episode("ep1")
    assert len(steps) == 1
    assert steps[0]["reward"] == 0.5


def test_trajectories_db_audit_log(tmp_dir):
    db = TrajectoriesDB(db_path=str(tmp_dir / "trajectories.db"))
    ok = db.log_audit(
        episode_id="ep1",
        claim_id="c1",
        source_url="https://en.wikipedia.org/wiki/Test",
        source_type="wikipedia",
        content_hash="abc123",
        status="live",
    )
    assert ok


def test_trajectories_db_export_jsonl(tmp_dir):
    db = TrajectoriesDB(db_path=str(tmp_dir / "trajectories.db"))
    db.log_step("ep1", 0, "c1", "easy", {"s": 1}, {"a": "x"}, 0.0, False)
    db.log_step("ep1", 1, "c1", "easy", {"s": 2}, {"a": "y"}, 0.3, True)
    exported = db.export_jsonl()
    assert len(exported) == 2
    assert exported[0]["state"] == {"s": 1}
    assert exported[1]["reward"] == 0.3
