"""Adversarial claim generator for curriculum learning.

Uses the LLM (via the validator-provided proxy or HF Inference API) to
generate harder variants of claims that the agent solved easily. This
creates a self-play dynamic: as the agent improves, the environment
generates harder challenges.

The generator takes:
- An original claim + its label
- The agent's verdict + confidence + NLI scores
- A difficulty target (e.g., "make this harder to classify")

And produces a new claim that is semantically related but harder to
fact-check (more ambiguous, better-crafted misinformation, more
subtle distortion).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional


class AdversarialClaimGenerator:
    """Generate adversarially harder variants of fact-checking claims.

    Uses the LLM proxy to craft claims that are semantically similar to
    the original but more difficult for the agent to classify correctly.
    Degrades gracefully when no API key is present.
    """

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._api_base_url = api_base_url or os.environ.get(
            "API_BASE_URL", "https://router.huggingface.co/v1"
        )
        self._api_key = api_key or os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
        # In-memory cache: hash(input) -> result dict
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}
        # Stats tracking for /curriculum endpoint
        self._stats: Dict[str, int] = {
            "total_generated": 0,
            "easy": 0,
            "medium": 0,
            "hard": 0,
            "cache_hits": 0,
            "api_failures": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        claim_text: str,
        original_label: str,
        agent_verdict: str,
        agent_confidence: float,
        nli_scores: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Generate an adversarially harder variant of *claim_text*.

        Returns a dict with keys:
            claim           : the harder claim text
            expected_label  : the correct label for the new claim
            difficulty      : "easy" | "medium" | "hard"
            reasoning       : why this variant is harder

        Returns None if no API key is configured or the LLM call fails.
        """
        if not self._api_key:
            return None

        cache_key = self._make_cache_key(
            claim_text, original_label, agent_verdict, agent_confidence, nli_scores
        )
        if cache_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]

        result = self._call_llm(
            claim_text, original_label, agent_verdict, agent_confidence, nli_scores
        )
        self._cache[cache_key] = result

        if result is not None:
            self._stats["total_generated"] += 1
            difficulty = result.get("difficulty", "medium")
            if difficulty in self._stats:
                self._stats[difficulty] += 1
        else:
            self._stats["api_failures"] += 1

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Return stats about generated adversarial claims."""
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_cache_key(
        self,
        claim_text: str,
        original_label: str,
        agent_verdict: str,
        agent_confidence: float,
        nli_scores: Dict[str, float],
    ) -> str:
        raw = json.dumps(
            {
                "claim": claim_text,
                "label": original_label,
                "verdict": agent_verdict,
                "conf": round(agent_confidence, 2),
                "nli": {k: round(v, 2) for k, v in (nli_scores or {}).items()},
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def _call_llm(
        self,
        claim_text: str,
        original_label: str,
        agent_verdict: str,
        agent_confidence: float,
        nli_scores: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Call the LLM to produce a harder claim variant."""
        try:
            import urllib.request

            entailment = nli_scores.get("entailment", 0.33) if nli_scores else 0.33
            contradiction = nli_scores.get("contradiction", 0.33) if nli_scores else 0.33

            prompt = (
                "You are an adversarial claim generator for a fact-checking RL environment.\n\n"
                f"ORIGINAL CLAIM: {claim_text}\n"
                f"CORRECT LABEL: {original_label}\n"
                f"AGENT VERDICT: {agent_verdict} (confidence={agent_confidence:.2f})\n"
                f"NLI SCORES: entailment={entailment:.2f}, contradiction={contradiction:.2f}\n\n"
                "Generate a harder variant that is:\n"
                "- Semantically related to the original claim\n"
                "- More ambiguous or subtly misleading\n"
                "- Harder for an AI agent to classify correctly\n\n"
                "Respond with ONLY valid JSON (no markdown, no explanation outside JSON):\n"
                '{"claim": "...", "expected_label": "TRUE|MOSTLY_TRUE|HALF_TRUE|MOSTLY_FALSE|FALSE|PANTS_ON_FIRE", '
                '"difficulty": "easy|medium|hard", "reasoning": "..."}'
            )

            payload = json.dumps(
                {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.7,
                }
            ).encode("utf-8")

            req = urllib.request.Request(
                f"{self._api_base_url.rstrip('/')}/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            content = data["choices"][0]["message"]["content"].strip()
            # Strip markdown fences if the model wrapped the JSON
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content.strip())

            # Validate required keys
            for key in ("claim", "expected_label", "difficulty", "reasoning"):
                if key not in result:
                    return None

            return result

        except Exception:
            return None
