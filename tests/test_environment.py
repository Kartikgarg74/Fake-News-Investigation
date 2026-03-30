"""Smoke tests for the Fake News Investigator environment."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fake_news_investigator.models import InvestigateAction
from fake_news_investigator.server.environment import FakeNewsEnvironment
from fake_news_investigator.server.grading_engine import (
    compute_reward,
    score_confidence,
    score_efficiency,
    score_evidence,
    score_verdict,
)


def test_reset_returns_valid_observation():
    env = FakeNewsEnvironment()
    for tier in ["easy", "medium", "hard"]:
        obs = env.reset(task=tier)
        assert obs.claim, f"Claim should not be empty for tier={tier}"
        assert len(obs.available_sources) > 0
        assert obs.budget_remaining > 0
        assert obs.done is False
        assert obs.reward is None


def test_request_source():
    env = FakeNewsEnvironment()
    env.reset(task="easy")
    obs = env.step(InvestigateAction(action_type="request_source", source_id="fact_checks"))
    assert obs.budget_remaining < 10
    assert obs.done is False


def test_cross_reference():
    env = FakeNewsEnvironment()
    env.reset(task="easy")
    obs = env.step(InvestigateAction(action_type="cross_reference", source_id="fact_checks"))
    assert obs.cross_ref_result is not None
    assert "entailment" in obs.cross_ref_result
    assert "contradiction" in obs.cross_ref_result
    assert "neutral" in obs.cross_ref_result


def test_check_credibility():
    env = FakeNewsEnvironment()
    env.reset(task="easy")
    obs = env.step(InvestigateAction(action_type="check_credibility", source_id="bbc.com"))
    assert obs.credibility_score is not None
    assert 0.0 <= obs.credibility_score <= 1.0


def test_submit_verdict_ends_episode():
    env = FakeNewsEnvironment()
    env.reset(task="easy")
    obs = env.step(InvestigateAction(
        action_type="submit_verdict",
        verdict="FALSE",
        evidence=["fact_checks"],
        confidence=0.8,
        reasoning="Test reasoning.",
    ))
    assert obs.done is True
    assert obs.reward is not None
    assert 0.0 <= obs.reward <= 1.0


def test_empty_source_id_handled():
    env = FakeNewsEnvironment()
    env.reset(task="easy")
    obs = env.step(InvestigateAction(action_type="request_source", source_id=""))
    assert "No source_id" in obs.message


def test_unknown_action_type():
    env = FakeNewsEnvironment()
    env.reset(task="easy")
    obs = env.step(InvestigateAction(action_type="invalid_action", source_id="test"))
    assert "Unknown action_type" in obs.message


def test_budget_exhaustion():
    env = FakeNewsEnvironment()
    env.reset(task="hard")  # budget = 6
    for _ in range(6):
        env.step(InvestigateAction(action_type="request_source", source_id="fact_checks"))
    obs = env.step(InvestigateAction(action_type="request_source", source_id="news"))
    assert "budget exhausted" in obs.message.lower() or "must submit" in obs.message.lower()


def test_post_done_step():
    env = FakeNewsEnvironment()
    env.reset(task="easy")
    env.step(InvestigateAction(
        action_type="submit_verdict", verdict="TRUE",
        evidence=[], confidence=0.5, reasoning="test",
    ))
    obs = env.step(InvestigateAction(action_type="request_source", source_id="test"))
    assert "already complete" in obs.message.lower()


def test_state_property():
    env = FakeNewsEnvironment()
    env.reset(task="medium")
    s = env.state
    assert s.difficulty == "medium"
    assert s.episode_id != ""
    assert s.ground_truth_verdict != ""


def test_grading_determinism():
    """Same inputs should always produce same reward."""
    r1 = compute_reward("FALSE", "false", ["fc"], ["fc", "gd"], 3, 10, 0.8, "test", "test")
    r2 = compute_reward("FALSE", "false", ["fc"], ["fc", "gd"], 3, 10, 0.8, "test", "test")
    assert r1 == r2, "Grading must be deterministic"


def test_reward_in_range():
    """Reward should always be 0.0-1.0."""
    for verdict in ["TRUE", "FALSE", "HALF_TRUE", "PANTS_ON_FIRE"]:
        for gt in ["true", "false", "half-true", "pants-fire"]:
            r = compute_reward(verdict, gt, ["fc"], ["fc"], 5, 10, 0.5, "r", "r")
            assert 0.0 <= r["total"] <= 1.0, f"Reward out of range: {r['total']}"


def test_score_verdict_labels():
    assert score_verdict("FALSE", "false") == 1.0
    assert score_verdict("TRUE", "true") == 1.0
    assert score_verdict("FALSE", "mostly-false") == 0.5  # adjacent
    assert score_verdict("TRUE", "false") == 0.0  # opposite
    # Critical: test LIAR-specific labels normalize correctly
    assert score_verdict("PANTS_ON_FIRE", "pants-fire") == 1.0
    assert score_verdict("MOSTLY_FALSE", "barely-true") == 1.0
    assert score_verdict("FALSE", "pants-fire") == 0.5  # adjacent to PANTS_ON_FIRE


def test_score_evidence_f1():
    assert score_evidence(["a", "b"], ["a", "b"]) == 1.0
    assert score_evidence([], ["a"]) == 0.0
    assert score_evidence(["a"], ["a", "b"]) > 0.0


def test_score_efficiency():
    assert score_efficiency(0, 10) == 1.0
    assert score_efficiency(10, 10) == 0.0
    assert score_efficiency(5, 10) == 0.5


def test_score_confidence():
    assert score_confidence(0.9, True) == 0.9
    assert score_confidence(0.9, False) == 0.1


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")
    sys.exit(1 if failed else 0)
