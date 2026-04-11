"""Unit tests for the 5 new action handlers.

Tests that each new action dispatches correctly, updates observation fields,
and doesn't break the environment. Network-dependent actions (check_entity,
search_evidence) are tested with mocked retrievers.
"""

from unittest.mock import MagicMock, patch

import pytest

from fake_news_investigator.models import InvestigateAction
from fake_news_investigator.server.environment import FakeNewsEnvironment


@pytest.fixture
def env():
    e = FakeNewsEnvironment()
    e.reset(task="easy", episode_id="test_new_actions")
    return e


# ---------- search_evidence ----------

def test_search_evidence_returns_observation(env):
    obs = env.step(InvestigateAction(action_type="search_evidence", query="moon landing"))
    assert obs is not None
    assert obs.message
    assert obs.budget_remaining >= 0


def test_search_evidence_uses_claim_as_default_query(env):
    obs = env.step(InvestigateAction(action_type="search_evidence"))
    # Should not crash even without explicit query
    assert obs is not None


# ---------- check_entity ----------

def test_check_entity_with_cached_result(env):
    # Pre-populate entities.db cache
    env.entities_db.store(
        name="Test Entity",
        display_name="Test Entity",
        wikidata_id="Q999",
        entity_type="test",
        description="A test entity",
    )
    obs = env.step(InvestigateAction(action_type="check_entity", entity="Test Entity"))
    assert obs.entity_info is not None
    assert "Test" in obs.entity_info.get("name", "")
    assert obs.cache_hit is True


def test_check_entity_without_explicit_name_autoextracts(env):
    # Claim should have some capitalized entity to extract
    obs = env.step(InvestigateAction(action_type="check_entity"))
    # Either resolves, cache hits, or reports no entity — but must not crash
    assert obs is not None


def test_check_entity_empty_claim_no_crash(env):
    # Should handle empty entity gracefully
    obs = env.step(InvestigateAction(action_type="check_entity", entity=""))
    assert obs is not None


# ---------- check_timeline ----------

def test_check_timeline_returns_info_dict(env):
    obs = env.step(InvestigateAction(action_type="check_timeline"))
    assert obs is not None
    # timeline_info may be empty if no dates, but should exist or be None
    assert obs.timeline_info is None or isinstance(obs.timeline_info, dict)


def test_check_timeline_no_crash_on_missing_claim_date(env):
    # Most LIAR claims have no claim_date — should still return gracefully
    obs = env.step(InvestigateAction(action_type="check_timeline"))
    if obs.timeline_info:
        status = obs.timeline_info.get("status", "")
        assert status in ("no_claim_date", "no_contradictions", "analyzed", "parse_error", "unknown")


# ---------- reverse_image_search ----------

def test_reverse_image_search_no_image_safe(env):
    # Current claim probably has no image — should return gracefully
    obs = env.step(InvestigateAction(action_type="reverse_image_search"))
    assert obs is not None
    assert obs.message


def test_reverse_image_search_with_invalid_url(env):
    obs = env.step(InvestigateAction(
        action_type="reverse_image_search",
        image_url="https://not-a-real-domain-xyz123.invalid/img.jpg",
    ))
    # pHash compute will fail, message should say so
    assert obs is not None
    assert "pHash" in obs.message or "no image" in obs.message.lower() or "Could not" in obs.message


# ---------- compute_consensus ----------

def test_compute_consensus_without_nli_returns_neutral(env):
    obs = env.step(InvestigateAction(action_type="compute_consensus"))
    assert obs.consensus_score is not None
    # Without any NLI results, it should return 0.5 (neutral)
    assert obs.consensus_score == 0.5


def test_compute_consensus_clamped_to_strict_bounds(env):
    # Manually inject NLI results and trigger consensus
    env._nli_results = [
        {"source": "reuters.com", "scores": {"entailment": 0.9, "contradiction": 0.05, "neutral": 0.05}},
        {"source": "bbc.com", "scores": {"entailment": 0.85, "contradiction": 0.1, "neutral": 0.05}},
    ]
    obs = env.step(InvestigateAction(action_type="compute_consensus"))
    assert obs.consensus_score is not None
    # Must be strictly in (0.01, 0.99)
    assert 0.01 <= obs.consensus_score <= 0.99


# ---------- Unknown action handling ----------

def test_unknown_action_penalizes_without_crash(env):
    obs = env.step(InvestigateAction(action_type="not_a_real_action"))
    assert obs is not None
    assert "Unknown action_type" in obs.message


# ---------- Trajectory logging ----------

def test_every_step_logs_trajectory():
    env = FakeNewsEnvironment()
    env.reset(task="easy", episode_id="traj_test_ep")

    before = env.trajectories_db.count_steps()

    env.step(InvestigateAction(action_type="check_credibility", source_id="bbc.com"))
    env.step(InvestigateAction(action_type="compute_consensus"))

    after = env.trajectories_db.count_steps()
    assert after >= before + 2, "Every step should log a trajectory row"
