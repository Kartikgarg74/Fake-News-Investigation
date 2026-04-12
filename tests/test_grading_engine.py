"""Fix #16 — Grading engine edge case tests."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from fake_news_investigator.server.grading_engine import (
    compute_reward,
    score_confidence,
    score_efficiency,
    score_evidence,
    score_reasoning,
    score_verdict,
)


class TestScoreVerdict:
    def test_exact_match(self):
        assert score_verdict("TRUE", "TRUE") == 1.0

    def test_adjacent_label(self):
        assert score_verdict("TRUE", "MOSTLY_TRUE") == 0.5

    def test_two_away(self):
        assert score_verdict("TRUE", "HALF_TRUE") == 0.25

    def test_opposite(self):
        assert score_verdict("TRUE", "PANTS_ON_FIRE") == 0.0

    def test_case_insensitive(self):
        assert score_verdict("true", "TRUE") == 1.0

    def test_alias_pants_fire(self):
        assert score_verdict("PANTS_FIRE", "PANTS_ON_FIRE") == 1.0

    def test_alias_barely_true(self):
        assert score_verdict("BARELY_TRUE", "MOSTLY_FALSE") == 1.0

    def test_unknown_label(self):
        assert score_verdict("UNKNOWN", "TRUE") == 0.0

    def test_dash_normalization(self):
        assert score_verdict("pants-fire", "pants-fire") == 1.0


class TestScoreEvidence:
    def test_perfect_match(self):
        assert score_evidence(["a", "b"], ["a", "b"]) == 1.0

    def test_no_cited(self):
        assert score_evidence([], ["a"]) == 0.0

    def test_no_gold(self):
        assert score_evidence(["a"], []) == 0.5

    def test_both_empty(self):
        assert score_evidence([], []) == 1.0

    def test_partial_overlap(self):
        s = score_evidence(["a", "c"], ["a", "b"])
        assert 0.0 < s < 1.0


class TestScoreEfficiency:
    def test_zero_steps(self):
        assert score_efficiency(0, 10) == 1.0

    def test_full_budget(self):
        assert score_efficiency(10, 10) == 0.0

    def test_half_budget(self):
        assert score_efficiency(5, 10) == 0.5

    def test_zero_budget(self):
        assert score_efficiency(0, 0) == 0.0


class TestScoreConfidence:
    def test_high_conf_correct(self):
        assert score_confidence(1.0, True) == 1.0

    def test_high_conf_wrong(self):
        assert score_confidence(1.0, False) == 0.0

    def test_low_conf_wrong(self):
        assert score_confidence(0.0, False) == 1.0

    def test_mid_conf(self):
        assert score_confidence(0.5, True) == 0.5


class TestScoreReasoning:
    def test_empty_reasoning(self):
        assert score_reasoning("", "some gold") == 0.0

    def test_empty_gold(self):
        assert score_reasoning("some reasoning", "") == 0.5

    def test_perfect_overlap(self):
        s = score_reasoning(
            "the wall china visible space",
            "the wall china visible space",
        )
        assert s > 0.8


class TestComputeReward:
    def test_total_clamped_above_zero(self):
        r = compute_reward(
            "TRUE", "PANTS_ON_FIRE", [], ["x"], 10, 10, 1.0, "", "gold", penalties=5.0
        )
        assert r["total"] >= 0.01

    def test_total_clamped_below_one(self):
        r = compute_reward(
            "TRUE", "TRUE", ["a"], ["a"], 0, 10, 1.0,
            "perfect reasoning match here", "perfect reasoning match here",
        )
        assert r["total"] <= 0.99

    def test_penalties_reduce_score(self):
        r_no_pen = compute_reward(
            "TRUE", "TRUE", ["a"], ["a"], 2, 10, 0.8, "r", "r", penalties=0.0
        )
        r_pen = compute_reward(
            "TRUE", "TRUE", ["a"], ["a"], 2, 10, 0.8, "r", "r", penalties=0.5
        )
        assert r_pen["total"] < r_no_pen["total"]

    def test_breakdown_has_all_keys(self):
        r = compute_reward("FALSE", "FALSE", [], [], 1, 10, 0.5, "", "")
        for key in (
            "total",
            "verdict_accuracy",
            "evidence_quality",
            "efficiency",
            "confidence_calibration",
            "reasoning_quality",
            "penalties",
        ):
            assert key in r
