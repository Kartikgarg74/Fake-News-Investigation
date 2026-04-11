"""Unit tests for the cloud ML signal layer.

Tests the degradation paths (no HF_TOKEN, no API_KEY) and the local pHash
computation. Network-dependent paths are mocked or skipped by default.
"""

import hashlib
import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from fake_news_investigator.server.ml import (
    NLIClient,
    CLIPClient,
    compute_phash,
    hamming_distance,
)


# ---------- NLIClient ----------

def test_nli_neutral_when_no_tokens(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    nli = NLIClient(hf_token="")  # Force empty
    r = nli.classify("The sky is blue.", "The sky appears blue due to Rayleigh scattering.")
    # Should degrade to neutral fallback
    assert nli.last_tier == "neutral_fallback"
    assert abs(r["entailment"] - 0.33) < 0.01
    assert abs(r["contradiction"] - 0.34) < 0.01


def test_nli_cache_hit_on_second_call(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    nli = NLIClient(hf_token="")
    nli.classify("claim", "evidence")
    assert nli.last_tier == "neutral_fallback"
    nli.classify("claim", "evidence")
    assert nli.last_tier == "cache"


def test_nli_empty_inputs_return_neutral():
    nli = NLIClient()
    r1 = nli.classify("", "evidence")
    assert abs(sum(r1.values()) - 1.0) < 0.01
    r2 = nli.classify("claim", "")
    assert abs(sum(r2.values()) - 1.0) < 0.01


def test_nli_parse_hf_response_valid():
    nli = NLIClient()
    data = [
        {"label": "ENTAILMENT", "score": 0.8},
        {"label": "CONTRADICTION", "score": 0.1},
        {"label": "NEUTRAL", "score": 0.1},
    ]
    scores = nli._parse_hf_response(data)
    assert scores is not None
    assert scores["entailment"] > scores["contradiction"]
    assert abs(sum(scores.values()) - 1.0) < 0.01


def test_nli_parse_hf_response_nested_list():
    nli = NLIClient()
    data = [[
        {"label": "entailment", "score": 0.5},
        {"label": "contradiction", "score": 0.4},
        {"label": "neutral", "score": 0.1},
    ]]
    scores = nli._parse_hf_response(data)
    assert scores is not None
    assert abs(sum(scores.values()) - 1.0) < 0.01


def test_nli_parse_json_scores_with_fences():
    nli = NLIClient()
    content = '```json\n{"entailment": 0.2, "contradiction": 0.5, "neutral": 0.3}\n```'
    scores = nli._parse_json_scores(content)
    assert scores is not None
    assert scores["contradiction"] > scores["entailment"]


def test_nli_parse_json_scores_malformed_returns_none():
    nli = NLIClient()
    assert nli._parse_json_scores("not json at all") is None
    assert nli._parse_json_scores("") is None


def test_nli_clear_cache():
    nli = NLIClient(hf_token="")
    nli.classify("a", "b")
    assert len(nli._cache) == 1
    nli.clear_cache()
    assert len(nli._cache) == 0


# ---------- CLIPClient ----------

def test_clip_degrades_without_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    clip = CLIPClient(hf_token="")
    r = clip.align(image_url="https://example.com/test.jpg", claim="A test claim")
    assert r["ok"] is False
    assert r["error"] == "no_hf_token"


def test_clip_empty_inputs():
    clip = CLIPClient()
    r1 = clip.align(image_url="", claim="test")
    assert r1["ok"] is False
    r2 = clip.align(image_url="https://x", claim="")
    assert r2["ok"] is False


def test_clip_default_labels_includes_claim():
    clip = CLIPClient()
    labels = clip._default_labels("Photo of a cat sitting on a mat.")
    assert len(labels) >= 3
    # First label should be the claim itself
    assert "cat" in labels[0]


# ---------- pHash ----------

def test_phash_computes_16_char_hex_on_real_image():
    # Requires Pillow, which is a required dependency now
    from PIL import Image
    img = Image.new("RGB", (64, 64), (100, 50, 200))
    buf = BytesIO()
    img.save(buf, format="PNG")
    h = compute_phash(buf.getvalue())
    assert h is not None
    assert isinstance(h, str)
    assert len(h) == 16
    # Must be valid hex
    int(h, 16)


def test_phash_returns_none_on_invalid_bytes():
    assert compute_phash(b"not an image") is None


def test_phash_returns_none_on_non_str_non_bytes():
    assert compute_phash(12345) is None  # type: ignore[arg-type]


def test_phash_same_image_produces_same_hash():
    from PIL import Image
    img = Image.new("L", (64, 64), 128)
    buf = BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    h1 = compute_phash(data)
    h2 = compute_phash(data)
    assert h1 == h2


def test_phash_different_images_produce_different_hashes():
    from PIL import Image
    img1 = Image.new("RGB", (64, 64), (255, 0, 0))
    img2 = Image.new("RGB", (64, 64), (0, 0, 255))
    b1 = BytesIO(); img1.save(b1, format="PNG")
    b2 = BytesIO(); img2.save(b2, format="PNG")
    h1 = compute_phash(b1.getvalue())
    h2 = compute_phash(b2.getvalue())
    assert h1 != h2


# ---------- hamming_distance ----------

def test_hamming_distance_basic():
    assert hamming_distance("ff", "ff") == 0
    assert hamming_distance("ff", "fe") == 1
    assert hamming_distance("ff", "00") == 8
    assert hamming_distance("ffff", "0000") == 16


def test_hamming_distance_mismatched_lengths():
    # Mismatched lengths return max distance (64)
    assert hamming_distance("ff", "ffff") == 64


def test_hamming_distance_invalid_hex():
    assert hamming_distance("xxyy", "zzww") == 64
