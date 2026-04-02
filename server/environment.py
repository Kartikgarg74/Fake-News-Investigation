"""Fake News Investigator — the core OpenEnv environment."""

import uuid
from typing import Any, Optional

from openenv.core.env_server import Environment

from ..models import InvestigateAction, InvestigateObservation, InvestigateState
from .claim_manager import BUDGET_MAP, SOURCE_CATEGORIES, ClaimManager
from .credibility_checker import CredibilityChecker
from .grading_engine import compute_reward


class FakeNewsEnvironment(
    Environment[InvestigateAction, InvestigateObservation, InvestigateState]
):
    """Interactive fact-checking environment.

    Agents learn to investigate claims by:
    1. Requesting evidence from source categories
    2. Cross-referencing claims against evidence
    3. Checking source credibility
    4. Submitting a reasoned verdict with evidence and confidence
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    # Shared storage for completed episode scores (class-level for /grader access)
    _completed_episodes: dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.claim_manager = ClaimManager()
        self.credibility_checker = CredibilityChecker()
        self._reset_episode_state()

    def _reset_episode_state(self):
        """Clear all episode-level state."""
        self._current_claim = None
        self._difficulty = "easy"
        self._budget = 10
        self._steps_used = 0
        self._episode_id = ""
        self._done = False
        self._reward = 0.0
        self._accessed_sources = []
        self._penalties = 0.0
        self._last_source_content = None
        self._last_cross_ref = None
        self._last_credibility = None
        self._last_credibility_details = None
        self._last_message = ""
        self._grading_breakdown = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> InvestigateObservation:
        """Start a new investigation episode.

        Pass task="easy"|"medium"|"hard" in kwargs to select difficulty.
        """
        self._reset_episode_state()

        self._difficulty = kwargs.get("task", "easy")
        if self._difficulty not in BUDGET_MAP:
            self._difficulty = "easy"

        self._budget = BUDGET_MAP[self._difficulty]
        self._episode_id = episode_id or str(uuid.uuid4())

        # Pick a random claim from the difficulty tier
        self._current_claim = self.claim_manager.get_random_claim(self._difficulty)

        return InvestigateObservation(
            claim=self._current_claim["claim"],
            available_sources=SOURCE_CATEGORIES,
            source_content=None,
            cross_ref_result=None,
            credibility_score=None,
            credibility_details=None,
            budget_remaining=self._budget,
            steps_taken=0,
            message=f"New investigation started. Difficulty: {self._difficulty}. "
            f"You have {self._budget} investigation steps. "
            f"Investigate the claim and submit your verdict.",
            done=False,
            reward=None,
        )

    def step(
        self,
        action: InvestigateAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> InvestigateObservation:
        """Process one investigation action."""
        if self._done:
            return self._make_observation(
                message="Episode is already complete. Call reset() to start a new one."
            )

        if self._current_claim is None:
            return self._make_observation(
                message="No active episode. Call reset() first."
            )

        # Check budget (except for submit_verdict which is always allowed)
        if action.action_type != "submit_verdict" and self._steps_used >= self._budget:
            return self._make_observation(
                message="Investigation budget exhausted. You must submit_verdict now.",
            )

        # Clear previous action results
        self._last_source_content = None
        self._last_cross_ref = None
        self._last_credibility = None
        self._last_credibility_details = None

        # Dispatch by action type
        if action.action_type == "request_source":
            return self._handle_request_source(action)
        elif action.action_type == "cross_reference":
            return self._handle_cross_reference(action)
        elif action.action_type == "check_credibility":
            return self._handle_check_credibility(action)
        elif action.action_type == "submit_verdict":
            return self._handle_submit_verdict(action)
        else:
            self._penalties += 0.02
            return self._make_observation(
                message=f"Unknown action_type: '{action.action_type}'. "
                f"Valid types: request_source, cross_reference, "
                f"check_credibility, submit_verdict",
            )

    @property
    def state(self) -> InvestigateState:
        """Return episode metadata."""
        claim = self._current_claim or {}
        return InvestigateState(
            episode_id=self._episode_id,
            step_count=self._steps_used,
            difficulty=self._difficulty,
            ground_truth_verdict=claim.get("label", ""),
            gold_evidence=claim.get("gold_evidence", []),
            claim_topic=claim.get("topic", ""),
            claim_id=claim.get("id", ""),
        )

    # =========================================================================
    # Action Handlers
    # =========================================================================

    def _handle_request_source(
        self, action: InvestigateAction
    ) -> InvestigateObservation:
        """Handle request_source action — return evidence from a source category."""
        self._steps_used += 1
        source_id = (action.source_id or "").lower().strip()

        if not source_id:
            self._last_message = "No source_id specified. Provide a source category."
            return self._make_observation(message=self._last_message)

        # Validate source_id against known categories
        if source_id not in SOURCE_CATEGORIES:
            self._penalties += 0.02
            self._last_message = (
                f"Unknown source_id '{source_id}'. "
                f"Available categories: {', '.join(SOURCE_CATEGORIES)}"
            )
            return self._make_observation(message=self._last_message)

        evidence_passages = self._current_claim.get("evidence_passages", {})

        # Exact match first, then substring fallback
        matching_key = None
        if source_id in evidence_passages:
            matching_key = source_id
        else:
            for key in evidence_passages:
                if source_id in key or key in source_id:
                    matching_key = key
                    break

        if matching_key:
            self._last_source_content = evidence_passages[matching_key]
            self._accessed_sources.append(source_id)
            self._last_message = (
                f"Found evidence from '{source_id}'. "
                f"Budget remaining: {self._budget - self._steps_used}"
            )
        else:
            # No evidence found for this category — slight penalty
            self._penalties += 0.03
            self._last_source_content = (
                f"No specific evidence found for '{source_id}' related to this claim. "
                f"Try a different source category."
            )
            self._last_message = (
                f"No relevant evidence found for '{source_id}'. "
                f"Budget remaining: {self._budget - self._steps_used}"
            )

        return self._make_observation(message=self._last_message)

    def _handle_cross_reference(
        self, action: InvestigateAction
    ) -> InvestigateObservation:
        """Handle cross_reference action — check claim against evidence using NLI."""
        self._steps_used += 1
        source_id = (action.source_id or "").lower().strip()

        if not source_id:
            self._last_cross_ref = {"entailment": 0.33, "contradiction": 0.33, "neutral": 0.34}
            self._last_message = "No source_id specified for cross-reference."
            return self._make_observation(message=self._last_message)

        evidence_passages = self._current_claim.get("evidence_passages", {})

        # Find matching evidence (exact match first)
        evidence_text = evidence_passages.get(source_id)
        if evidence_text is None:
            for key in evidence_passages:
                if source_id in key or key in source_id:
                    evidence_text = evidence_passages[key]
                    break

        if not evidence_text:
            self._last_cross_ref = {
                "entailment": 0.33,
                "contradiction": 0.33,
                "neutral": 0.34,
            }
            self._last_message = (
                f"No evidence found for '{source_id}' to cross-reference against."
            )
        else:
            # Deterministic NLI simulation based on claim label
            label = self._current_claim["label"].lower()
            self._last_cross_ref = self._simulate_nli(label, source_id)
            self._last_message = (
                f"Cross-referenced claim against '{source_id}'. "
                f"Budget remaining: {self._budget - self._steps_used}"
            )

        return self._make_observation(message=self._last_message)

    def _handle_check_credibility(
        self, action: InvestigateAction
    ) -> InvestigateObservation:
        """Handle check_credibility action — look up source reliability."""
        self._steps_used += 1
        source_id = action.source_id or ""

        result = self.credibility_checker.check(source_id)
        self._last_credibility = result["credibility_score"]
        self._last_credibility_details = {
            "bias": result["bias"],
            "factual_reporting": result["factual_reporting"],
            "found": str(result["found"]),
        }

        if result["found"]:
            self._last_message = (
                f"Credibility check for '{source_id}': "
                f"Bias={result['bias']}, "
                f"Factual={result['factual_reporting']}, "
                f"Score={result['credibility_score']:.2f}"
            )
        else:
            self._last_message = (
                f"Source '{source_id}' not found in credibility database. "
                f"Default score: 0.5"
            )

        return self._make_observation(message=self._last_message)

    def _handle_submit_verdict(
        self, action: InvestigateAction
    ) -> InvestigateObservation:
        """Handle submit_verdict action — grade and end episode."""
        self._done = True

        verdict = action.verdict or "UNKNOWN"
        evidence = action.evidence or []
        confidence = action.confidence if action.confidence is not None else 0.5
        reasoning = action.reasoning or ""

        # Compute reward
        self._grading_breakdown = compute_reward(
            predicted_verdict=verdict,
            ground_truth_verdict=self._current_claim["label"],
            cited_evidence=evidence,
            gold_evidence=self._current_claim["gold_evidence"],
            steps_used=self._steps_used,
            max_budget=self._budget,
            confidence=confidence,
            agent_reasoning=reasoning,
            gold_reasoning=self._current_claim["gold_reasoning"],
            penalties=self._penalties,
        )

        self._reward = self._grading_breakdown["total"]

        # Store for /grader endpoint access
        FakeNewsEnvironment._completed_episodes[self._episode_id] = self._grading_breakdown

        return InvestigateObservation(
            claim=self._current_claim["claim"],
            available_sources=SOURCE_CATEGORIES,
            source_content=None,
            cross_ref_result=None,
            credibility_score=None,
            credibility_details=None,
            budget_remaining=max(0, self._budget - self._steps_used),
            steps_taken=self._steps_used,
            message=(
                f"Investigation complete. Your verdict: {verdict}. "
                f"Ground truth: {self._current_claim['label']}. "
                f"Score: {self._reward:.4f}"
            ),
            done=True,
            reward=self._reward,
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _make_observation(self, message: str) -> InvestigateObservation:
        """Build an observation from current state."""
        return InvestigateObservation(
            claim=self._current_claim["claim"] if self._current_claim else "",
            available_sources=SOURCE_CATEGORIES,
            source_content=self._last_source_content,
            cross_ref_result=self._last_cross_ref,
            credibility_score=self._last_credibility,
            credibility_details=self._last_credibility_details,
            budget_remaining=max(0, self._budget - self._steps_used),
            steps_taken=self._steps_used,
            message=message,
            done=self._done,
            reward=self._reward if self._done else None,
        )

    def _simulate_nli(self, label: str, source_id: str) -> dict:
        """Simulate NLI scores based on claim label with added noise.

        For the hackathon, this provides cross-reference results with slight
        randomness to prevent agents from gaming deterministic patterns.
        In production, this would run DeBERTa inference.
        """
        import random
        # Evidence from fact-checks and government data tends to contradict false claims
        authoritative_sources = {
            "government_data", "fact_checks", "medical_journals",
            "academic_papers", "international_organizations",
        }
        is_authoritative = any(s in source_id for s in authoritative_sources)

        if label == "false":
            if is_authoritative:
                return self._add_nli_noise({"entailment": 0.05, "contradiction": 0.88, "neutral": 0.07})
            return self._add_nli_noise({"entailment": 0.15, "contradiction": 0.65, "neutral": 0.20})
        elif label == "pants-fire":
            # Pants-on-fire is HARD tier — make NLI more ambiguous
            # so heuristic agents can't easily distinguish from half-true
            if is_authoritative:
                return self._add_nli_noise({"entailment": 0.15, "contradiction": 0.50, "neutral": 0.35})
            return self._add_nli_noise({"entailment": 0.25, "contradiction": 0.40, "neutral": 0.35})
        elif label in ("barely-true", "mostly-false"):
            if is_authoritative:
                return self._add_nli_noise({"entailment": 0.12, "contradiction": 0.68, "neutral": 0.20})
            return self._add_nli_noise({"entailment": 0.25, "contradiction": 0.45, "neutral": 0.30})
        elif label == "half-true":
            # Half-true is HARD tier — highly ambiguous NLI
            return self._add_nli_noise({"entailment": 0.33, "contradiction": 0.34, "neutral": 0.33})
        elif label in ("mostly-true",):
            if is_authoritative:
                return self._add_nli_noise({"entailment": 0.70, "contradiction": 0.10, "neutral": 0.20})
            return self._add_nli_noise({"entailment": 0.50, "contradiction": 0.20, "neutral": 0.30})
        elif label == "true":
            if is_authoritative:
                return self._add_nli_noise({"entailment": 0.88, "contradiction": 0.05, "neutral": 0.07})
            return self._add_nli_noise({"entailment": 0.65, "contradiction": 0.15, "neutral": 0.20})
        else:
            base = {"entailment": 0.33, "contradiction": 0.33, "neutral": 0.34}
            return self._add_nli_noise(base)

    @staticmethod
    def _add_nli_noise(scores: dict, noise_range: float = 0.08) -> dict:
        """Add small random noise to NLI scores so agents can't memorize exact values."""
        import random
        noisy = {}
        for key, val in scores.items():
            noisy[key] = max(0.0, min(1.0, val + random.uniform(-noise_range, noise_range)))
        # Re-normalize to sum to 1.0
        total = sum(noisy.values())
        if total > 0:
            noisy = {k: round(v / total, 4) for k, v in noisy.items()}
        return noisy
