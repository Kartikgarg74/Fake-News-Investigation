"""Regression tests for inference.py output format.

These exist to catch any regression in the 5 validator patterns. If any of
these fail, the submission will fail Phase 2 validation.
"""

import io
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path


INFERENCE_PATH = Path(__file__).parent.parent / "inference.py"


def test_inference_file_exists():
    assert INFERENCE_PATH.exists()


def test_inference_reads_api_key_from_environ():
    content = INFERENCE_PATH.read_text()
    # Must read API_KEY from environment (not hardcoded, not HF_TOKEN-only)
    assert 'os.environ.get("API_KEY"' in content or 'os.environ["API_KEY"]' in content


def test_inference_reads_api_base_url_from_environ():
    content = INFERENCE_PATH.read_text()
    assert 'os.environ.get("API_BASE_URL"' in content or 'os.environ["API_BASE_URL"]' in content


def test_inference_no_bare_zero_score_literals():
    """No reward=0.0 or score=0.0 in output paths."""
    content = INFERENCE_PATH.read_text()
    # Match standalone 0.0 not followed by more digits (so 0.01 is allowed)
    bad = re.findall(r'(?:reward|score)=0\.0[^0-9]', content)
    assert not bad, f"Found bare 0.0 score literals: {bad}"


def test_inference_no_bare_return_zero():
    content = INFERENCE_PATH.read_text()
    # Match return 0.0 at end of line
    bad = re.findall(r'return 0\.0$', content, re.MULTILINE)
    assert not bad, f"Found return 0.0 statements: {bad}"


def test_inference_has_min_max_score_constants():
    content = INFERENCE_PATH.read_text()
    assert "MIN_SCORE" in content
    assert "MAX_SCORE" in content
    # MIN_SCORE should be > 0 and < 0.1
    # MAX_SCORE should be > 0.9 and < 1
    assert "0.01" in content
    assert "0.99" in content


def test_inference_flush_true_count_is_high():
    """Every print to stdout should use flush=True."""
    content = INFERENCE_PATH.read_text()
    flush_count = content.count("flush=True")
    assert flush_count >= 10, f"Expected at least 10 flush=True, got {flush_count}"


def test_inference_wraps_fakeenv_in_try_except():
    content = INFERENCE_PATH.read_text()
    # Look for the pattern: try: ... FakeNewsEnvironment() ... except
    # (liberal regex)
    pattern = r'try:[\s\S]{0,200}FakeNewsEnvironment\(\)[\s\S]{0,200}except'
    assert re.search(pattern, content), "FakeNewsEnvironment() should be wrapped in try/except"


def test_inference_wraps_env_reset_in_try_except():
    content = INFERENCE_PATH.read_text()
    pattern = r'try:[\s\S]{0,100}env\.reset\(task=task\)[\s\S]{0,100}except'
    assert re.search(pattern, content), "env.reset(task=task) should be wrapped in try/except"


def test_inference_has_start_step_end_output_blocks():
    content = INFERENCE_PATH.read_text()
    assert '"[START]' in content or "'[START]" in content
    assert '"[STEP]' in content or "'[STEP]" in content
    assert '"[END]' in content or "'[END]" in content


def test_inference_start_uses_task_format():
    content = INFERENCE_PATH.read_text()
    # [START] task={task} pattern
    assert re.search(r'\[START\] task=\{task\}', content), "Missing [START] task={task} format"


def test_inference_step_uses_reward_format():
    content = INFERENCE_PATH.read_text()
    # [STEP] step=N reward=R pattern
    assert re.search(r'\[STEP\] step=\{.*?\} reward=', content), "Missing [STEP] step=N reward=R format"


def test_inference_end_uses_score_format():
    content = INFERENCE_PATH.read_text()
    # [END] task={task} score=S steps=N pattern
    assert re.search(r'\[END\] task=\{task\} score=', content), "Missing [END] task score format"


def test_grading_engine_clamps_to_strict_bounds():
    """The grading engine must clamp total to (0.01, 0.99) not [0, 1]."""
    from fake_news_investigator.server.grading_engine import compute_reward
    # Max reward case
    r = compute_reward(
        predicted_verdict="TRUE", ground_truth_verdict="TRUE",
        cited_evidence=["a", "b"], gold_evidence=["a", "b"],
        steps_used=1, max_budget=10,
        confidence=1.0, agent_reasoning="perfect reasoning",
        gold_reasoning="perfect reasoning",
    )
    assert r["total"] < 1.0
    assert r["total"] >= 0.01

    # Min reward case
    r = compute_reward(
        predicted_verdict="TRUE", ground_truth_verdict="PANTS_ON_FIRE",
        cited_evidence=[], gold_evidence=["x"],
        steps_used=10, max_budget=10,
        confidence=1.0, agent_reasoning="",
        gold_reasoning="something different",
        penalties=5.0,
    )
    assert r["total"] > 0.0
    assert r["total"] <= 0.99


def test_inference_clamp_score_helper():
    """The clamp_score helper must clamp any input into (0.01, 0.99)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("inference", str(INFERENCE_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.clamp_score(0.0) == 0.01
    assert mod.clamp_score(1.0) == 0.99
    assert mod.clamp_score(0.5) == 0.5
    assert mod.clamp_score(None) == 0.01
    assert mod.clamp_score("not a number") == 0.01
    assert mod.clamp_score(-5) == 0.01
    assert mod.clamp_score(100) == 0.99
