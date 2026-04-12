"""Real NLI client using HuggingFace Inference API.

Replaces the _simulate_nli function in environment.py that was returning
label-based noise. This version actually sends (claim, evidence) pairs to
a DeBERTa NLI model and parses the entailment / contradiction / neutral
probabilities from the response.

Two-tier strategy:
1. HuggingFace Inference API (preferred) — real model inference, no local weights.
2. LiteLLM-compatible chat completions fallback (via the validator-provided
   API_BASE_URL + API_KEY) — uses a structured prompt to extract NLI scores.

The second tier matters because the validator container injects API_KEY +
API_BASE_URL for the LiteLLM proxy. If HF Inference API isn't reachable
(rate limits, no HF_TOKEN), we can still get NLI via the proxy.

Results are cached in-memory for the duration of an episode because NLI
calls are the hottest path — an agent may cross-reference 5-10 times per
episode against the same claim.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from collections import OrderedDict
from typing import Any, Dict, Optional


class BoundedCache(OrderedDict):
    """LRU cache with a fixed maximum size.

    Evicts the least-recently-used entry when the limit is exceeded.
    Thread-safety is not required because each FakeNewsEnvironment instance
    owns its own NLIClient and is never shared across threads.
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
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
TIMEOUT = 10.0


class NLIClient:
    """Client for (claim, evidence) NLI classification."""

    def __init__(
        self,
        model: str = DEFAULT_NLI_MODEL,
        hf_token: Optional[str] = None,
        use_proxy_fallback: bool = True,
    ):
        self.model = model
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.use_proxy_fallback = use_proxy_fallback
        # Episode-lifetime cache. Keyed by (claim_hash, evidence_hash).
        self._cache: BoundedCache = BoundedCache(max_size=512)
        # Track which tier served the last call (for debugging/metrics)
        self.last_tier: str = ""

    def classify(self, claim: str, evidence: str) -> Dict[str, float]:
        """Return {entailment, contradiction, neutral} probabilities.

        Never raises. On any failure, returns a neutral distribution
        (0.33/0.34/0.33) so the env can degrade gracefully.
        """
        if not claim or not evidence:
            return self._neutral()

        cache_key = self._cache_key(claim, evidence)
        if cache_key in self._cache:
            self.last_tier = "cache"
            return self._cache[cache_key]

        # Tier 1: HF Inference API
        result = self._call_hf_inference(claim, evidence)
        if result:
            self.last_tier = "hf_inference"
            self._cache[cache_key] = result
            return result

        # Tier 2: LiteLLM proxy fallback
        if self.use_proxy_fallback:
            result = self._call_proxy_fallback(claim, evidence)
            if result:
                self.last_tier = "proxy_fallback"
                self._cache[cache_key] = result
                return result

        # Tier 3: neutral distribution (graceful degradation)
        self.last_tier = "neutral_fallback"
        result = self._neutral()
        self._cache[cache_key] = result
        return result

    def _call_hf_inference(self, claim: str, evidence: str) -> Optional[Dict[str, float]]:
        """Call the HF Inference API for a cross-encoder NLI model."""
        if not self.hf_token:
            return None

        url = f"{HF_INFERENCE_URL}/{self.model}"
        # HF Inference API for text-classification cross-encoders expects
        # a single "text" field with the pair separated by [SEP], but some
        # models want {"text": claim, "text_pair": evidence}. Try both.
        payload = {
            "inputs": {
                "text": claim,
                "text_pair": evidence,
            },
            "options": {"wait_for_model": True},
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.hf_token}",
                    "Content-Type": "application/json",
                    "User-Agent": "Veritas-NLI/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # nosec B310 — fixed HF host
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

        return self._parse_hf_response(data)

    def _parse_hf_response(self, data: Any) -> Optional[Dict[str, float]]:
        """HF Inference API returns a list of {label, score} dicts."""
        if isinstance(data, list) and data and isinstance(data[0], list):
            data = data[0]
        if not isinstance(data, list):
            return None

        scores = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
        for item in data:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).lower()
            score = float(item.get("score", 0.0))
            if "entail" in label:
                scores["entailment"] = score
            elif "contradict" in label:
                scores["contradiction"] = score
            elif "neutral" in label:
                scores["neutral"] = score

        total = sum(scores.values())
        if total <= 0:
            return None
        return {k: round(v / total, 4) for k, v in scores.items()}

    def _call_proxy_fallback(self, claim: str, evidence: str) -> Optional[Dict[str, float]]:
        """Use the LiteLLM proxy to get NLI scores via prompt engineering.

        This is a last resort when HF Inference isn't available. We ask the
        LLM to output JSON with entailment/contradiction/neutral scores.
        """
        api_key = os.environ.get("API_KEY", "")
        api_base_url = os.environ.get("API_BASE_URL", "")
        model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
        if not api_key or not api_base_url:
            return None

        prompt = (
            "Classify the relationship between a CLAIM and EVIDENCE. "
            "Output ONLY valid JSON with three float fields that sum to 1.0: "
            "entailment, contradiction, neutral.\n\n"
            f"CLAIM: {claim[:500]}\n\n"
            f"EVIDENCE: {evidence[:1500]}\n\n"
            'Output JSON only, example: {"entailment": 0.1, "contradiction": 0.7, "neutral": 0.2}'
        )

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 80,
        }
        url = api_base_url.rstrip("/") + "/chat/completions"
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # nosec B310 — validator-provided URL
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None

        return self._parse_json_scores(content)

    def _parse_json_scores(self, content: str) -> Optional[Dict[str, float]]:
        """Extract NLI scores from an LLM JSON response. Tolerant to fences."""
        if not content:
            return None
        stripped = content.strip()
        for fence in ("```json", "```"):
            if stripped.startswith(fence):
                stripped = stripped[len(fence):].strip()
            if stripped.endswith("```"):
                stripped = stripped[:-3].strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            # Try to extract the first {...} block
            match = re.search(r"\{[^{}]*\}", stripped)
            if not match:
                return None
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

        if not isinstance(parsed, dict):
            return None

        try:
            scores = {
                "entailment": float(parsed.get("entailment", 0.0)),
                "contradiction": float(parsed.get("contradiction", 0.0)),
                "neutral": float(parsed.get("neutral", 0.0)),
            }
        except (TypeError, ValueError):
            return None

        total = sum(scores.values())
        if total <= 0:
            return None
        return {k: round(v / total, 4) for k, v in scores.items()}

    @staticmethod
    def _cache_key(claim: str, evidence: str) -> str:
        return hashlib.sha256(f"{claim}||{evidence}".encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _neutral() -> Dict[str, float]:
        return {"entailment": 0.33, "contradiction": 0.34, "neutral": 0.33}

    def clear_cache(self) -> None:
        self._cache.clear()
