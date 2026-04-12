"""CLIP multimodal client for image-text alignment.

Uses HuggingFace Inference API zero-shot image classification. We send an
image URL + candidate labels (phrases derived from the claim) and get back
similarity scores. This lets us verify whether an image actually depicts
what the claim says it depicts.

Without HF_TOKEN, the client degrades gracefully to None, same pattern
as NLIClient.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from .url_validator import validate_url


class BoundedCache(OrderedDict):
    """LRU cache with a fixed maximum size.

    Evicts the least-recently-used entry when the limit is exceeded.
    """

    def __init__(self, max_size: int = 512):
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)

HF_INFERENCE_URL = "https://api-inference.huggingface.co/models"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
TIMEOUT = 15.0  # CLIP is slower than NLI


class CLIPClient:
    """Image-text alignment via CLIP on HF Inference API."""

    def __init__(
        self,
        model: str = DEFAULT_CLIP_MODEL,
        hf_token: Optional[str] = None,
    ):
        self.model = model
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self._cache: BoundedCache = BoundedCache(max_size=512)

    def align(
        self, image_url: str, claim: str, candidate_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Check if an image aligns with a textual claim.

        Returns:
            {
                "ok": bool,
                "claim_score": float,          # similarity to the claim itself
                "contradiction_score": float,  # similarity to claim negation
                "top_label": str,
                "scores": {label: score},
                "verdict": "supports" | "contradicts" | "neutral",
            }
        """
        if not image_url or not claim:
            return self._empty()

        cache_key = f"{image_url}|{claim[:200]}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.hf_token:
            result = self._empty(reason="no_hf_token")
            self._cache[cache_key] = result
            return result

        labels = candidate_labels or self._default_labels(claim)

        try:
            image_bytes = self._fetch_image(image_url)
            if not image_bytes:
                result = self._empty(reason="image_fetch_failed")
                self._cache[cache_key] = result
                return result

            scores = self._call_hf_inference(image_bytes, labels)
            if not scores:
                result = self._empty(reason="inference_failed")
                self._cache[cache_key] = result
                return result

            claim_score = scores.get(labels[0], 0.0)
            contradiction_score = scores.get(labels[1] if len(labels) > 1 else "", 0.0)
            top_label = max(scores, key=scores.get)

            if claim_score > contradiction_score + 0.1:
                verdict = "supports"
            elif contradiction_score > claim_score + 0.1:
                verdict = "contradicts"
            else:
                verdict = "neutral"

            result = {
                "ok": True,
                "claim_score": round(claim_score, 4),
                "contradiction_score": round(contradiction_score, 4),
                "top_label": top_label,
                "scores": {k: round(v, 4) for k, v in scores.items()},
                "verdict": verdict,
            }
            self._cache[cache_key] = result
            return result
        except Exception:
            result = self._empty(reason="exception")
            self._cache[cache_key] = result
            return result

    def _fetch_image(self, url: str) -> Optional[bytes]:
        """Download image bytes for upload to HF Inference API.

        SSRF protection: the URL is validated before connecting — private IP
        ranges, non-http(s) schemes, and unresolvable hostnames are rejected.
        """
        err = validate_url(url)
        if err is not None:
            return None
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Veritas-Vision/1.0"},
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # nosec B310 — URL pre-validated above
                if resp.length and resp.length > 10_000_000:  # 10MB cap
                    return None
                return resp.read()
        except Exception:
            return None

    def _call_hf_inference(self, image_bytes: bytes, labels: List[str]) -> Optional[Dict[str, float]]:
        """POST image bytes + candidate labels to HF zero-shot image classification."""
        url = f"{HF_INFERENCE_URL}/{self.model}"
        # Zero-shot image classification takes labels in the parameters
        import base64
        b64_image = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "inputs": b64_image,
            "parameters": {"candidate_labels": labels},
            "options": {"wait_for_model": True},
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.hf_token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # nosec B310
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

        if not isinstance(data, list):
            return None
        scores = {}
        for item in data:
            if isinstance(item, dict) and "label" in item and "score" in item:
                scores[item["label"]] = float(item["score"])
        return scores if scores else None

    def _default_labels(self, claim: str) -> List[str]:
        """Build default candidate labels from a claim.

        We create a claim/negation pair plus a few distractors so CLIP has
        something to rank against.
        """
        return [
            claim[:200],
            f"This image does not show: {claim[:150]}",
            "a photograph",
            "a diagram or chart",
            "an AI-generated image",
            "a screenshot",
        ]

    def _empty(self, reason: str = "") -> Dict[str, Any]:
        return {
            "ok": False,
            "claim_score": 0.0,
            "contradiction_score": 0.0,
            "top_label": "",
            "scores": {},
            "verdict": "unknown",
            "error": reason,
        }
