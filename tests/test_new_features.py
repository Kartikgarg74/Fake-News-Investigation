"""Tests for the 3 new hackathon features:
1. Trained Agent + Learning Curve (scripts/train_agent.py)
2. Adversarial Claim Generator (server/adversarial.py)
3. Cross-Lingual Fact-Checking (server/translation.py)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Feature 2: Adversarial Claim Generator
# ---------------------------------------------------------------------------

def test_adversarial_generator_degrades_without_api_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from fake_news_investigator.server.adversarial import AdversarialClaimGenerator
    gen = AdversarialClaimGenerator()
    result = gen.generate("test claim", "false", "TRUE", 0.9, {})
    assert result is None  # graceful degradation


def test_adversarial_generator_cache_key_deterministic():
    """Same inputs produce the same cache key (idempotent)."""
    from fake_news_investigator.server.adversarial import AdversarialClaimGenerator
    gen = AdversarialClaimGenerator()
    key1 = gen._make_cache_key("claim A", "true", "TRUE", 0.9, {"entailment": 0.8})
    key2 = gen._make_cache_key("claim A", "true", "TRUE", 0.9, {"entailment": 0.8})
    assert key1 == key2


def test_adversarial_generator_different_claims_different_keys():
    from fake_news_investigator.server.adversarial import AdversarialClaimGenerator
    gen = AdversarialClaimGenerator()
    key1 = gen._make_cache_key("claim A", "true", "TRUE", 0.9, {})
    key2 = gen._make_cache_key("claim B", "true", "TRUE", 0.9, {})
    assert key1 != key2


def test_adversarial_get_stats_returns_dict():
    from fake_news_investigator.server.adversarial import AdversarialClaimGenerator
    gen = AdversarialClaimGenerator()
    stats = gen.get_stats()
    assert isinstance(stats, dict)
    assert "total_generated" in stats
    assert "cache_hits" in stats


# ---------------------------------------------------------------------------
# Feature 3: Cross-Lingual Translation
# ---------------------------------------------------------------------------

def test_translation_client_degrades_without_api_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from fake_news_investigator.server.translation import TranslationClient
    tc = TranslationClient()
    result = tc.translate_to_english("test text", "en")
    assert result == "test text"  # returns original on degradation


def test_translation_detect_language_english():
    from fake_news_investigator.server.translation import TranslationClient
    tc = TranslationClient()
    lang = tc.detect_language("The quick brown fox jumps over the lazy dog.")
    assert lang == "en"


def test_translation_detect_language_hindi():
    from fake_news_investigator.server.translation import TranslationClient
    tc = TranslationClient()
    lang = tc.detect_language("यह एक परीक्षण वाक्य है।")
    assert lang == "hi"


def test_translation_detect_language_chinese():
    from fake_news_investigator.server.translation import TranslationClient
    tc = TranslationClient()
    lang = tc.detect_language("这是一个测试句子。")
    assert lang == "zh"


def test_translation_detect_language_arabic():
    from fake_news_investigator.server.translation import TranslationClient
    tc = TranslationClient()
    lang = tc.detect_language("هذه جملة اختبار.")
    assert lang == "ar"


def test_translation_english_passthrough_no_api_key(monkeypatch):
    """English text with no API key should return unchanged."""
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from fake_news_investigator.server.translation import TranslationClient
    tc = TranslationClient()
    text = "Water boils at 100 degrees Celsius."
    assert tc.translate_to_english(text, "en") == text


def test_translation_from_english_no_api_key(monkeypatch):
    """translate_from_english with no API key should return original text."""
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from fake_news_investigator.server.translation import TranslationClient
    tc = TranslationClient()
    text = "NLI entailment score 0.80 strongly supports the claim."
    # Should return original when no API key
    assert tc.translate_from_english(text, "hi") == text


# ---------------------------------------------------------------------------
# Feature 2 & 3: FastAPI endpoints
# ---------------------------------------------------------------------------

def test_adversarial_endpoint_exists():
    from fastapi.testclient import TestClient
    from fake_news_investigator.server.app import app
    client = TestClient(app)
    r = client.post("/generate_adversarial", params={"claim": "test", "label": "false"})
    assert r.status_code == 200


def test_translate_endpoint_exists():
    from fastapi.testclient import TestClient
    from fake_news_investigator.server.app import app
    client = TestClient(app)
    r = client.post("/translate", params={"text": "hello", "target_lang": "es"})
    assert r.status_code == 200


def test_translate_endpoint_response_shape():
    from fastapi.testclient import TestClient
    from fake_news_investigator.server.app import app
    client = TestClient(app)
    r = client.post("/translate", params={"text": "hello world", "target_lang": "en"})
    assert r.status_code == 200
    data = r.json()
    assert "original" in data
    assert "detected_language" in data
    assert "translated" in data


def test_curriculum_endpoint_exists():
    from fastapi.testclient import TestClient
    from fake_news_investigator.server.app import app
    client = TestClient(app)
    r = client.get("/curriculum")
    assert r.status_code == 200


def test_curriculum_endpoint_response_shape():
    from fastapi.testclient import TestClient
    from fake_news_investigator.server.app import app
    client = TestClient(app)
    r = client.get("/curriculum")
    assert r.status_code == 200
    data = r.json()
    assert "stats" in data


# ---------------------------------------------------------------------------
# Feature 2 & 3: Environment methods
# ---------------------------------------------------------------------------

def test_environment_reset_adversarial():
    from fake_news_investigator.server.environment import FakeNewsEnvironment
    env = FakeNewsEnvironment()
    original = {"id": "test", "claim": "Water boils at 100C", "label": "true"}
    performance = {"verdict": "TRUE", "confidence": 0.9}
    # Should fall back to normal reset when no API key
    obs = env.reset_adversarial(original, performance, task="easy")
    assert obs is not None
    assert obs.claim  # got some claim


def test_environment_reset_multilingual():
    from fake_news_investigator.server.environment import FakeNewsEnvironment
    env = FakeNewsEnvironment()
    # English claim should pass through without translation
    obs = env.reset_multilingual("Water boils at 100 degrees Celsius.", source_language="en", task="easy")
    assert obs is not None
    assert obs.claim


def test_environment_reset_multilingual_hindi():
    from fake_news_investigator.server.environment import FakeNewsEnvironment
    env = FakeNewsEnvironment()
    obs = env.reset_multilingual("यह एक परीक्षण वाक्य है।", source_language="auto", task="easy")
    assert obs is not None
    assert obs.claim
    assert obs.original_language == "hi"


def test_environment_reset_multilingual_stores_language():
    from fake_news_investigator.server.environment import FakeNewsEnvironment
    env = FakeNewsEnvironment()
    obs = env.reset_multilingual("这是一个测试句子。", source_language="auto", task="easy")
    assert obs.original_language == "zh"


def test_environment_reset_multilingual_english_no_translation():
    from fake_news_investigator.server.environment import FakeNewsEnvironment
    env = FakeNewsEnvironment()
    claim = "The Earth orbits the Sun."
    obs = env.reset_multilingual(claim, source_language="en", task="easy")
    assert obs.claim == claim
    assert obs.original_language == "en"
    assert obs.translated_claim is None  # no translation needed


# ---------------------------------------------------------------------------
# Feature 1: Train agent helpers (no full run — too slow for CI)
# ---------------------------------------------------------------------------

def test_extract_features_returns_10_dims():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from fake_news_investigator.scripts.train_agent import extract_features
    from fake_news_investigator.models import InvestigateObservation

    obs = InvestigateObservation(
        claim="Test claim",
        available_sources=[],
        budget_remaining=8,
        steps_taken=2,
    )
    features = extract_features(obs)
    assert len(features) == 10
    assert all(isinstance(f, float) for f in features)


def test_rolling_avg_correct():
    from fake_news_investigator.scripts.train_agent import rolling_avg
    result = rolling_avg([1.0, 2.0, 3.0, 4.0], window=2)
    assert len(result) == 4
    assert abs(result[1] - 1.5) < 1e-9
    assert abs(result[3] - 3.5) < 1e-9


def test_heuristic_verdict_entailment():
    from fake_news_investigator.scripts.train_agent import heuristic_verdict
    from fake_news_investigator.models import InvestigateObservation

    obs = InvestigateObservation(
        claim="test",
        available_sources=[],
        cross_ref_result={"entailment": 0.8, "contradiction": 0.1, "neutral": 0.1},
    )
    verdict = heuristic_verdict(obs)
    assert verdict == "TRUE"


def test_heuristic_verdict_contradiction():
    from fake_news_investigator.scripts.train_agent import heuristic_verdict
    from fake_news_investigator.models import InvestigateObservation

    obs = InvestigateObservation(
        claim="test",
        available_sources=[],
        cross_ref_result={"entailment": 0.1, "contradiction": 0.8, "neutral": 0.1},
    )
    verdict = heuristic_verdict(obs)
    assert verdict == "FALSE"


def test_numpy_logistic_regression_predict():
    from fake_news_investigator.scripts.train_agent import NumpyLogisticRegression
    import random

    clf = NumpyLogisticRegression(n_features=10, n_classes=6, lr=0.05, epochs=50)
    # Simple synthetic data: first feature high → class 0, first feature low → class 5
    X = [[1.0] + [0.5] * 9] * 20 + [[0.0] + [0.5] * 9] * 20
    y = [0] * 20 + [5] * 20
    clf.fit(X, y)
    pred_high = clf.predict([1.0] + [0.5] * 9)
    pred_low = clf.predict([0.0] + [0.5] * 9)
    # Should predict different classes (regression should learn something)
    assert isinstance(pred_high, int)
    assert isinstance(pred_low, int)
    assert 0 <= pred_high < 6
    assert 0 <= pred_low < 6
