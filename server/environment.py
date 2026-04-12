"""Veritas (formerly Fake News Investigator) — the core OpenEnv environment.

This is the heart of the system. An episode:

1. reset() picks a claim from ClaimsDB and starts an investigation budget.
2. The agent calls step(action) repeatedly. Ten action types are supported:
     - request_source        : real live retrieval via RetrievalOrchestrator
     - cross_reference       : real NLI via NLIClient (DeBERTa on HF Inference)
     - check_credibility     : real publisher lookup in SourcesDB (MBFC-seeded)
     - analyze_image         : real CLIP alignment + pHash matching
     - search_evidence       : FTS5 search across cached evidence corpus
     - check_entity          : Wikidata entity resolution
     - check_timeline        : temporal analysis via TemporalDB
     - reverse_image_search  : pHash match against ImagesDB
     - compute_consensus     : aggregate agreement across all retrieved evidence
     - submit_verdict        : final verdict, grades the episode, ends it
3. Every step is logged to trajectories.db for RL training.
4. Every retrieval is logged to the audit table for chain-of-custody.

All heavy lifting (retrieval, NLI, CLIP) is cloud-hosted via HF Inference API
and the validator-provided LiteLLM proxy. No local ML models — Docker stays
small and the laptop doesn't need a GPU.
"""

from __future__ import annotations

import hashlib
import re
import statistics
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Fix #24 — precompile entity-extraction regex at module level for efficiency
_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")


class _BoundedDict(OrderedDict):
    """OrderedDict that evicts the oldest entry once max_size is exceeded."""

    def __init__(self, max_size: int = 1000):
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)

from openenv.core.env_server import Environment

from ..models import InvestigateAction, InvestigateObservation, InvestigateState
from .claim_manager import BUDGET_MAP, SOURCE_CATEGORIES, ClaimManager
from .credibility_checker import CredibilityChecker
from .databases import (
    EntitiesDB,
    EvidenceDB,
    ImagesDB,
    SourcesDB,
    TemporalDB,
    TrajectoriesDB,
)
from .grading_engine import compute_reward
from .ml import CLIPClient, NLIClient, compute_phash
from .retrievers import RetrievalOrchestrator, WikidataRetriever


# Fix #12 — EpisodeContext dataclass holding all mutable per-episode state
@dataclass
class EpisodeContext:
    """All mutable state for a single investigation episode."""
    current_claim: Optional[Dict[str, Any]] = None
    difficulty: str = "easy"
    budget: int = 10
    steps_used: int = 0
    episode_id: str = ""
    done: bool = False
    reward: float = 0.0
    accessed_sources: List[str] = field(default_factory=list)
    retrieved_evidence: List[Dict[str, Any]] = field(default_factory=list)
    nli_results: List[Dict[str, Any]] = field(default_factory=list)
    penalties: float = 0.0
    # Fix #23 — O(1) evidence lookup index: source_type -> content
    evidence_index: Dict[str, str] = field(default_factory=dict)
    # Last-step observation fields
    last_source_content: Optional[str] = None
    last_cross_ref: Optional[Dict[str, float]] = None
    last_credibility: Optional[float] = None
    last_credibility_details: Optional[Dict[str, str]] = None
    last_entity_info: Optional[Dict[str, Any]] = None
    last_timeline_info: Optional[Dict[str, Any]] = None
    last_image_match: Optional[Dict[str, Any]] = None
    last_consensus_score: Optional[float] = None
    last_cache_hit: Optional[bool] = None
    last_message: str = ""
    grading_breakdown: Optional[Dict[str, Any]] = None


class FakeNewsEnvironment(
    Environment[InvestigateAction, InvestigateObservation, InvestigateState]
):
    """Interactive fact-checking environment with real retrieval + ML signals."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    # Shared storage for completed episode scores (class-level for /grader access).
    # Bounded to 1000 entries to prevent unbounded memory growth.
    _completed_episodes: _BoundedDict = _BoundedDict(max_size=1000)

    # Canonical list of valid action types — used by step() dispatch
    _VALID_ACTIONS = (
        "request_source",
        "cross_reference",
        "check_credibility",
        "analyze_image",
        "search_evidence",
        "check_entity",
        "check_timeline",
        "reverse_image_search",
        "compute_consensus",
        "submit_verdict",
    )

    def __init__(
        self,
        claim_manager=None,
        credibility_checker=None,
        evidence_db=None,
        sources_db=None,
        images_db=None,
        temporal_db=None,
        entities_db=None,
        trajectories_db=None,
        nli_client=None,
        clip_client=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Fix #13 — dependency injection: accept collaborators as constructor params
        # so tests can pass mocks without monkey-patching.
        self.claim_manager = claim_manager or ClaimManager()
        self.credibility_checker = credibility_checker or CredibilityChecker()

        # Segregated databases (each owns a single concern)
        self.evidence_db = evidence_db or EvidenceDB()
        self.sources_db = sources_db or SourcesDB()
        self.images_db = images_db or ImagesDB()
        self.temporal_db = temporal_db or TemporalDB()
        self.entities_db = entities_db or EntitiesDB()
        self.trajectories_db = trajectories_db or TrajectoriesDB()

        # Live retrieval + cloud ML
        self.retrieval = RetrievalOrchestrator(
            evidence_db=self.evidence_db,
            trajectories_db=self.trajectories_db,
        )
        self.nli = nli_client or NLIClient()
        self.clip = clip_client or CLIPClient()
        self.wikidata = WikidataRetriever()

        self._reset_episode_state()

    # =====================================================================
    # Episode lifecycle
    # =====================================================================

    def _reset_episode_state(self):
        """Clear all episode-level state via a fresh EpisodeContext."""
        # Fix #12 — all mutable episode fields live in a single dataclass
        self._ctx = EpisodeContext()

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

        self._ctx.difficulty = kwargs.get("task", "easy")
        if self._ctx.difficulty not in BUDGET_MAP:
            self._ctx.difficulty = "easy"

        self._ctx.budget = BUDGET_MAP[self._ctx.difficulty]
        self._ctx.episode_id = episode_id or str(uuid.uuid4())

        # Clear NLI cache at the start of each episode — otherwise stale
        # scores leak between episodes for the same claim text.
        self.nli.clear_cache()

        # Pick a random claim. If the DB is completely unavailable
        # (should never happen in practice), fall back to a hardcoded
        # minimal claim so the env doesn't crash the validator.
        try:
            self._ctx.current_claim = self.claim_manager.get_random_claim(self._ctx.difficulty)
        except Exception:
            self._ctx.current_claim = _EMERGENCY_CLAIM.copy()
            self._ctx.current_claim["difficulty"] = self._ctx.difficulty

        image_url = self._ctx.current_claim.get("image_url")
        visual_note = (
            " This claim has an associated image — use analyze_image or "
            "reverse_image_search to examine it." if image_url else ""
        )
        return InvestigateObservation(
            claim=self._ctx.current_claim["claim"],
            available_sources=SOURCE_CATEGORIES,
            source_content=None,
            cross_ref_result=None,
            credibility_score=None,
            credibility_details=None,
            budget_remaining=self._ctx.budget,
            steps_taken=0,
            message=(
                f"New investigation started. Difficulty: {self._ctx.difficulty}. "
                f"You have {self._ctx.budget} investigation steps."
                f"{visual_note} Investigate the claim and submit your verdict."
            ),
            done=False,
            reward=None,
            image_url=image_url,
        )

    def reset_with_custom_claim(  # noqa: D102
        self,
        claim_dict: dict,
        difficulty: str = "easy",
        episode_id: str = "",
    ) -> InvestigateObservation:
        """Reset with a user-provided claim instead of pulling from ClaimsDB.

        Use this instead of directly mutating private fields from external
        callers such as the /demo/stream endpoint. The claim_dict must
        contain at minimum a 'claim' key with the claim text.
        """
        self._reset_episode_state()
        self._ctx.difficulty = difficulty if difficulty in BUDGET_MAP else "easy"
        self._ctx.budget = BUDGET_MAP[self._ctx.difficulty]
        self._ctx.episode_id = episode_id or str(uuid.uuid4())
        self.nli.clear_cache()
        self._ctx.current_claim = claim_dict

        image_url = self._ctx.current_claim.get("image_url")
        visual_note = (
            " This claim has an associated image — use analyze_image or "
            "reverse_image_search to examine it." if image_url else ""
        )
        return InvestigateObservation(
            claim=self._ctx.current_claim["claim"],
            available_sources=SOURCE_CATEGORIES,
            source_content=None,
            cross_ref_result=None,
            credibility_score=None,
            credibility_details=None,
            budget_remaining=self._ctx.budget,
            steps_taken=0,
            message=(
                f"New investigation started. Difficulty: {self._ctx.difficulty}. "
                f"You have {self._ctx.budget} investigation steps."
                f"{visual_note} Investigate the claim and submit your verdict."
            ),
            done=False,
            reward=None,
            image_url=image_url,
        )

    def step(
        self,
        action: InvestigateAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> InvestigateObservation:
        """Process one investigation action."""
        if self._ctx.done:
            return self._make_observation(
                message="Episode is already complete. Call reset() to start a new one."
            )
        if self._ctx.current_claim is None:
            return self._make_observation(
                message="No active episode. Call reset() first."
            )

        # Check budget (except for submit_verdict which is always allowed)
        if action.action_type != "submit_verdict" and self._ctx.steps_used >= self._ctx.budget:
            return self._make_observation(
                message="Investigation budget exhausted. You must submit_verdict now.",
            )

        # Clear previous action results
        self._ctx.last_source_content = None
        self._ctx.last_cross_ref = None
        self._ctx.last_credibility = None
        self._ctx.last_credibility_details = None
        self._ctx.last_entity_info = None
        self._ctx.last_timeline_info = None
        self._ctx.last_image_match = None
        self._ctx.last_consensus_score = None
        self._ctx.last_cache_hit = None

        # Dispatch by action type
        handler = {
            "request_source": self._handle_request_source,
            "cross_reference": self._handle_cross_reference,
            "check_credibility": self._handle_check_credibility,
            "analyze_image": self._handle_analyze_image,
            "search_evidence": self._handle_search_evidence,
            "check_entity": self._handle_check_entity,
            "check_timeline": self._handle_check_timeline,
            "reverse_image_search": self._handle_reverse_image_search,
            "compute_consensus": self._handle_compute_consensus,
            "submit_verdict": self._handle_submit_verdict,
        }.get(action.action_type)

        if handler is None:
            self._ctx.penalties += 0.02
            obs = self._make_observation(
                message=(
                    f"Unknown action_type: '{action.action_type}'. "
                    f"Valid types: {', '.join(self._VALID_ACTIONS)}"
                )
            )
            self._log_trajectory(action, obs)
            return obs

        # Every handler updates self._ctx.steps_used (for non-submit actions) and
        # returns an observation. We wrap in a try/except to guarantee the
        # env never raises — the validator test fails if any step blows up.
        try:
            obs = handler(action)
        except Exception as exc:
            self._ctx.penalties += 0.03
            obs = self._make_observation(
                message=f"Handler error in {action.action_type}: {str(exc)[:120]}"
            )

        self._log_trajectory(action, obs)
        return obs

    @property
    def state(self) -> InvestigateState:
        """Return episode metadata."""
        claim = self._ctx.current_claim or {}
        return InvestigateState(
            episode_id=self._ctx.episode_id,
            step_count=self._ctx.steps_used,
            difficulty=self._ctx.difficulty,
            ground_truth_verdict=claim.get("label", ""),
            gold_evidence=claim.get("gold_evidence", []),
            claim_topic=claim.get("topic", ""),
            claim_id=claim.get("id", ""),
        )

    # =====================================================================
    # Action handlers — existing 5 (upgraded to real backends)
    # =====================================================================

    def _handle_request_source(self, action: InvestigateAction) -> InvestigateObservation:
        """Fetch evidence from a source category via RetrievalOrchestrator.

        Before: returned a templated string from evidence_passages.
        After: hits Wikipedia / Fact Check API / cached corpus depending on
        the source category. Result is cached in evidence.db for the episode.
        """
        self._ctx.steps_used += 1
        source_id = (action.source_id or "").lower().strip()

        if not source_id:
            self._ctx.last_message = "No source_id specified. Provide a source category."
            return self._make_observation(message=self._ctx.last_message)

        if source_id not in SOURCE_CATEGORIES:
            self._ctx.penalties += 0.02
            self._ctx.last_message = (
                f"Unknown source_id '{source_id}'. "
                f"Available categories: {', '.join(SOURCE_CATEGORIES)}"
            )
            return self._make_observation(message=self._ctx.last_message)

        result = self.retrieval.fetch(
            claim=self._ctx.current_claim,
            source_type=source_id,
            query=action.query,
            episode_id=self._ctx.episode_id,
        )

        if result["ok"] and result["content"]:
            self._ctx.last_source_content = result["content"]
            self._ctx.last_cache_hit = result["cache_hit"]
            self._ctx.accessed_sources.append(source_id)
            self._ctx.retrieved_evidence.append({
                "source_type": source_id,
                "content": result["content"],
                "url": result["source_url"],
                "domain": result["source_domain"],
                "is_synthetic": result["is_synthetic"],
            })
            # Fix #23 — also populate the O(1) evidence index
            self._ctx.evidence_index[source_id] = result["content"]
            tag = "[cached]" if result["cache_hit"] else ("[synthetic]" if result["is_synthetic"] else "[live]")
            self._ctx.last_message = (
                f"Retrieved evidence from '{source_id}' {tag}. "
                f"Budget remaining: {self._ctx.budget - self._ctx.steps_used}"
            )
        else:
            self._ctx.penalties += 0.03
            self._ctx.last_source_content = (
                f"No evidence retrieved for '{source_id}'. Try a different source category."
            )
            self._ctx.last_message = (
                f"No evidence for '{source_id}'. "
                f"Budget remaining: {self._ctx.budget - self._ctx.steps_used}"
            )

        return self._make_observation(message=self._ctx.last_message)

    def _handle_cross_reference(self, action: InvestigateAction) -> InvestigateObservation:
        """Real NLI via NLIClient — no more label-leaking simulation.

        The agent specifies which retrieved source to cross-reference against.
        We pull the cached evidence text for that source and classify
        (claim, evidence) with DeBERTa on HF Inference API.
        """
        self._ctx.steps_used += 1
        source_id = (action.source_id or "").lower().strip()

        if not source_id:
            self._ctx.last_cross_ref = {"entailment": 0.33, "contradiction": 0.33, "neutral": 0.34}
            self._ctx.last_message = "No source_id specified for cross-reference."
            return self._make_observation(message=self._ctx.last_message)

        # Find evidence text for this source. Preference order:
        # 1. Evidence retrieved this episode (O(1) via evidence_index)
        # 2. EvidenceDB cache
        # 3. Legacy evidence_passages on the claim
        evidence_text = self._find_evidence_text(source_id)

        if not evidence_text:
            self._ctx.last_cross_ref = {"entailment": 0.33, "contradiction": 0.33, "neutral": 0.34}
            self._ctx.last_message = (
                f"No evidence found for '{source_id}'. "
                f"Call request_source or search_evidence first."
            )
            return self._make_observation(message=self._ctx.last_message)

        claim_text = self._ctx.current_claim.get("claim", "")
        scores = self.nli.classify(claim=claim_text, evidence=evidence_text)
        self._ctx.last_cross_ref = scores
        self._ctx.nli_results.append({"source": source_id, "scores": scores, "tier": self.nli.last_tier})

        self._ctx.last_message = (
            f"Cross-referenced against '{source_id}' [{self.nli.last_tier}]. "
            f"E={scores['entailment']:.2f} C={scores['contradiction']:.2f} N={scores['neutral']:.2f}. "
            f"Budget remaining: {self._ctx.budget - self._ctx.steps_used}"
        )
        return self._make_observation(message=self._ctx.last_message)

    def _handle_check_credibility(self, action: InvestigateAction) -> InvestigateObservation:
        """Look up publisher reputation in SourcesDB (thousands of entries)."""
        self._ctx.steps_used += 1
        source_id = action.source_id or ""

        result = self.credibility_checker.check(source_id)
        self._ctx.last_credibility = result["credibility_score"]
        self._ctx.last_credibility_details = {
            "bias": result["bias"],
            "factual_reporting": result["factual_reporting"],
            "found": str(result["found"]),
            "name": result.get("name", ""),
            "country": result.get("country", ""),
        }

        if result["found"]:
            self._ctx.last_message = (
                f"Credibility for '{source_id}' -> {result.get('name','')}: "
                f"bias={result['bias']}, factual={result['factual_reporting']}, "
                f"score={result['credibility_score']:.2f}"
            )
        else:
            self._ctx.last_message = (
                f"Source '{source_id}' not found in credibility DB. "
                f"Default score: 0.5"
            )

        return self._make_observation(message=self._ctx.last_message)

    def _handle_analyze_image(self, action: InvestigateAction) -> InvestigateObservation:
        """Real CLIP alignment + pHash matching for visual claims."""
        self._ctx.steps_used += 1
        image_url = action.image_url or self._ctx.current_claim.get("image_url")

        if not image_url:
            self._ctx.last_source_content = (
                "No image is associated with this claim. "
                "This is a text-only claim — use request_source or search_evidence instead."
            )
            self._ctx.penalties += 0.02
            self._ctx.last_message = (
                f"No image to analyze. Budget remaining: {self._ctx.budget - self._ctx.steps_used}"
            )
            return self._make_observation(message=self._ctx.last_message)

        # CLIP alignment (returns neutral if HF_TOKEN is missing)
        claim_text = self._ctx.current_claim.get("claim", "")
        clip_result = self.clip.align(image_url=image_url, claim=claim_text)

        # pHash match against known misattributed images
        phash = compute_phash(image_url)
        image_match = None
        if phash:
            image_match = self.images_db.find_similar(phash, threshold=12)
            if image_match:
                self._ctx.last_image_match = {
                    "original_source": image_match.get("original_source", ""),
                    "verdict": image_match.get("verdict", ""),
                    "description": image_match.get("description", ""),
                    "hamming_distance": str(image_match.get("hamming_distance", "")),
                }

        parts = [f"CLIP verdict: {clip_result.get('verdict', 'unknown')}"]
        if clip_result.get("ok"):
            parts.append(
                f"claim_score={clip_result.get('claim_score', 0.0):.3f} "
                f"contradiction={clip_result.get('contradiction_score', 0.0):.3f}"
            )
        if image_match:
            parts.append(
                f"pHash match: {image_match.get('verdict')} "
                f"(originally from {image_match.get('original_source')}, "
                f"hamming={image_match.get('hamming_distance')})"
            )
        else:
            parts.append("no pHash match in DB")

        self._ctx.last_source_content = " | ".join(parts)
        self._ctx.accessed_sources.append("image_analysis")
        self._ctx.last_message = (
            f"Visual analysis complete. {parts[0]}. "
            f"Budget remaining: {self._ctx.budget - self._ctx.steps_used}"
        )
        return self._make_observation(message=self._ctx.last_message)

    # =====================================================================
    # Action handlers — new 5 actions
    # =====================================================================

    def _handle_search_evidence(self, action: InvestigateAction) -> InvestigateObservation:
        """FTS5 search across the evidence corpus + live Wikipedia fallback."""
        self._ctx.steps_used += 1
        query = (action.query or self._ctx.current_claim.get("claim", ""))[:500]

        # 1. Full-text search in evidence.db
        fts_hits = self.evidence_db.search(query, limit=3)

        # 2. If no hits, fall back to live Wikipedia retrieval
        if not fts_hits:
            retrieval_result = self.retrieval.fetch(
                claim=self._ctx.current_claim,
                source_type="wikipedia",
                query=query,
                episode_id=self._ctx.episode_id,
            )
            if retrieval_result["ok"] and retrieval_result["content"]:
                self._ctx.last_source_content = retrieval_result["content"]
                self._ctx.last_cache_hit = retrieval_result["cache_hit"]
                self._ctx.retrieved_evidence.append({
                    "source_type": "wikipedia_search",
                    "content": retrieval_result["content"],
                    "url": retrieval_result["source_url"],
                    "domain": retrieval_result["source_domain"],
                    "is_synthetic": retrieval_result["is_synthetic"],
                })
                # Fix #23 — populate evidence index for O(1) cross-reference
                self._ctx.evidence_index["wikipedia_search"] = retrieval_result["content"]
                self._ctx.last_message = (
                    f"Searched evidence for '{query[:50]}' -> 1 live result from Wikipedia. "
                    f"Budget remaining: {self._ctx.budget - self._ctx.steps_used}"
                )
                return self._make_observation(message=self._ctx.last_message)

            self._ctx.last_source_content = f"No evidence found for query: {query[:100]}"
            self._ctx.last_message = (
                f"No evidence found. Budget remaining: {self._ctx.budget - self._ctx.steps_used}"
            )
            return self._make_observation(message=self._ctx.last_message)

        # Format FTS hits
        summary_lines = []
        for i, hit in enumerate(fts_hits[:3], 1):
            summary_lines.append(
                f"[{i}] {hit.get('source_type', '?')}: {hit.get('content', '')[:200]}"
            )
            ev_entry = {
                "source_type": hit.get("source_type", "evidence_search"),
                "content": hit.get("content", ""),
                "url": hit.get("source_url", ""),
                "domain": hit.get("source_domain", ""),
                "is_synthetic": False,
            }
            self._ctx.retrieved_evidence.append(ev_entry)
            # Fix #23 — populate evidence index for O(1) cross-reference
            self._ctx.evidence_index[ev_entry["source_type"]] = ev_entry["content"]

        self._ctx.last_source_content = "\n".join(summary_lines)
        self._ctx.last_message = (
            f"Search returned {len(fts_hits)} hits. "
            f"Budget remaining: {self._ctx.budget - self._ctx.steps_used}"
        )
        return self._make_observation(message=self._ctx.last_message)

    def _handle_check_entity(self, action: InvestigateAction) -> InvestigateObservation:
        """Resolve a named entity via Wikidata, cached in entities.db."""
        self._ctx.steps_used += 1
        entity_name = (action.entity or "").strip()

        if not entity_name:
            # Try to auto-extract an entity from the claim
            claim_text = self._ctx.current_claim.get("claim", "")
            entity_name = self._extract_first_entity(claim_text)

        if not entity_name:
            self._ctx.last_message = "No entity to resolve. Provide action.entity."
            return self._make_observation(message=self._ctx.last_message)

        # Check cache first
        cached = self.entities_db.lookup(entity_name)
        if cached:
            self._ctx.last_entity_info = {
                "name": cached.get("name", ""),
                "type": cached.get("type", "unknown"),
                "description": cached.get("description", ""),
                "wikidata_id": cached.get("wikidata_id", ""),
            }
            self._ctx.last_cache_hit = True
            self._ctx.last_message = (
                f"Entity '{entity_name}' -> {cached.get('name')} "
                f"(cached, type={cached.get('type')}). "
                f"Budget remaining: {self._ctx.budget - self._ctx.steps_used}"
            )
            return self._make_observation(message=self._ctx.last_message)

        # Live Wikidata fetch
        result = self.wikidata.retrieve(entity_name)
        if result.get("ok"):
            self.entities_db.store(
                name=entity_name,
                display_name=result.get("name", entity_name),
                wikidata_id=result.get("wikidata_id", ""),
                entity_type=result.get("type", "unknown"),
                description=result.get("description", ""),
                properties=result.get("properties", {}),
            )
            self._ctx.last_entity_info = {
                "name": result.get("name", ""),
                "type": result.get("type", "unknown"),
                "description": result.get("description", ""),
                "wikidata_id": result.get("wikidata_id", ""),
            }
            self._ctx.last_cache_hit = False
            self._ctx.last_message = (
                f"Resolved '{entity_name}' -> {result.get('name')} "
                f"[{result.get('wikidata_id')}]. "
                f"Budget remaining: {self._ctx.budget - self._ctx.steps_used}"
            )
        else:
            self._ctx.last_entity_info = {"name": entity_name, "type": "not_found"}
            self._ctx.last_message = (
                f"Could not resolve entity '{entity_name}'. "
                f"Budget remaining: {self._ctx.budget - self._ctx.steps_used}"
            )

        return self._make_observation(message=self._ctx.last_message)

    def _handle_check_timeline(self, action: InvestigateAction) -> InvestigateObservation:
        """Temporal analysis: when was the claim made vs when was it contradicted?"""
        self._ctx.steps_used += 1
        claim_id = self._ctx.current_claim.get("id", "")
        claim_date = self._ctx.current_claim.get("claim_date")

        if claim_date:
            self.temporal_db.record_claim(claim_id=claim_id, first_seen_date=claim_date)

        # Record any retrieved evidence dates we've seen this episode
        for ev in self._ctx.retrieved_evidence:
            if ev.get("is_synthetic"):
                continue
            self.temporal_db.record_evidence(
                evidence_id=hashlib.sha256(ev.get("content", "").encode()).hexdigest()[:16],
                claim_id=claim_id,
                published_date=None,
                supports_or_contradicts="contradicts",
                source_domain=ev.get("domain", ""),
                title=ev.get("source_type", ""),
            )

        timeline = self.temporal_db.get_timeline(claim_id)
        delta = timeline.get("delta_analysis", {})
        self._ctx.last_timeline_info = {
            "status": delta.get("status", "unknown"),
            "message": delta.get("message", ""),
            "delta_days": str(delta.get("delta_days", "")),
            "claim_first_seen": delta.get("claim_first_seen", claim_date or ""),
        }
        self._ctx.last_message = (
            f"Timeline: {delta.get('message', 'no data')}. "
            f"Budget remaining: {self._ctx.budget - self._ctx.steps_used}"
        )
        return self._make_observation(message=self._ctx.last_message)

    def _handle_reverse_image_search(self, action: InvestigateAction) -> InvestigateObservation:
        """pHash lookup against ImagesDB of known misattributed/AI-generated images."""
        self._ctx.steps_used += 1
        image_url = action.image_url or self._ctx.current_claim.get("image_url", "")
        if not image_url:
            self._ctx.last_message = "No image URL provided and claim has no image."
            self._ctx.penalties += 0.02
            return self._make_observation(message=self._ctx.last_message)

        phash = compute_phash(image_url)
        if not phash:
            self._ctx.last_message = "Could not compute pHash (fetch or decode failed)."
            return self._make_observation(message=self._ctx.last_message)

        match = self.images_db.find_similar(phash, threshold=12)
        if match:
            self._ctx.last_image_match = {
                "verdict": match.get("verdict", "unknown"),
                "original_source": match.get("original_source", ""),
                "description": match.get("description", ""),
                "hamming_distance": str(match.get("hamming_distance", "")),
                "fact_check_url": match.get("fact_check_url", ""),
            }
            self._ctx.last_message = (
                f"Image match: {match.get('verdict')} "
                f"(originally from {match.get('original_source')}). "
                f"Budget remaining: {self._ctx.budget - self._ctx.steps_used}"
            )
        else:
            self._ctx.last_image_match = {"verdict": "no_match"}
            self._ctx.last_message = (
                f"No pHash match in database. "
                f"Budget remaining: {self._ctx.budget - self._ctx.steps_used}"
            )
        return self._make_observation(message=self._ctx.last_message)

    def _handle_compute_consensus(self, action: InvestigateAction) -> InvestigateObservation:
        """Aggregate multi-source agreement across all retrieved evidence.

        Combines:
        - NLI entailment - contradiction scores collected this episode
        - Credibility of the sources those evidences came from
        - Number of distinct sources

        Returns a single [0, 1] consensus score where higher = more sources
        agree the claim is supported.
        """
        self._ctx.steps_used += 1

        if not self._ctx.nli_results:
            self._ctx.last_consensus_score = 0.5
            self._ctx.last_message = (
                "No NLI results yet. Call cross_reference first. "
                f"Budget remaining: {self._ctx.budget - self._ctx.steps_used}"
            )
            return self._make_observation(message=self._ctx.last_message)

        # Per-source net score: entailment - contradiction, weighted by
        # source credibility when known.
        weighted_scores: List[float] = []
        for r in self._ctx.nli_results:
            scores = r.get("scores", {})
            net = scores.get("entailment", 0.0) - scores.get("contradiction", 0.0)
            source_name = r.get("source", "")
            cred = self.sources_db.lookup(source_name)
            weight = float(cred.get("credibility_score", 0.5))
            weighted_scores.append(net * weight)

        mean_net = statistics.mean(weighted_scores) if weighted_scores else 0.0
        # Remap from [-1, 1] to [0, 1]
        consensus = (mean_net + 1.0) / 2.0
        # Strict bounds (validator will reject exact 0 or 1)
        consensus = max(0.01, min(0.99, consensus))
        self._ctx.last_consensus_score = round(consensus, 4)

        n = len(weighted_scores)
        self._ctx.last_message = (
            f"Consensus across {n} sources: {consensus:.3f} "
            f"(1.0 = strongly supports, 0.0 = strongly contradicts). "
            f"Budget remaining: {self._ctx.budget - self._ctx.steps_used}"
        )
        return self._make_observation(message=self._ctx.last_message)

    # =====================================================================
    # Submit verdict — grade and finalize
    # =====================================================================

    def _handle_submit_verdict(self, action: InvestigateAction) -> InvestigateObservation:
        """Grade the episode and return the final observation."""
        self._ctx.done = True

        verdict = action.verdict or "UNKNOWN"
        evidence = action.evidence or []
        confidence = action.confidence if action.confidence is not None else 0.5
        reasoning = action.reasoning or ""

        self._ctx.grading_breakdown = compute_reward(
            predicted_verdict=verdict,
            ground_truth_verdict=self._ctx.current_claim.get("label", "false"),
            cited_evidence=evidence,
            gold_evidence=self._ctx.current_claim.get("gold_evidence", []),
            steps_used=self._ctx.steps_used,
            max_budget=self._ctx.budget,
            confidence=confidence,
            agent_reasoning=reasoning,
            gold_reasoning=self._ctx.current_claim.get("gold_reasoning", ""),
            penalties=self._ctx.penalties,
        )
        self._ctx.reward = self._ctx.grading_breakdown["total"]
        FakeNewsEnvironment._completed_episodes[self._ctx.episode_id] = self._ctx.grading_breakdown

        return InvestigateObservation(
            claim=self._ctx.current_claim.get("claim", ""),
            available_sources=SOURCE_CATEGORIES,
            source_content=None,
            cross_ref_result=None,
            credibility_score=None,
            credibility_details=None,
            budget_remaining=max(0, self._ctx.budget - self._ctx.steps_used),
            steps_taken=self._ctx.steps_used,
            message=(
                f"Investigation complete. Verdict: {verdict}. "
                f"Ground truth: {self._ctx.current_claim.get('label', '?')}. "
                f"Score: {self._ctx.reward:.4f}"
            ),
            done=True,
            reward=self._ctx.reward,
        )

    # =====================================================================
    # Advanced feature: adversarial reset
    # =====================================================================

    def reset_adversarial(
        self,
        original_claim: dict,
        agent_performance: dict,
        **kwargs,
    ) -> InvestigateObservation:
        """Reset with an adversarially-generated harder claim.

        Takes the original claim and the agent's performance metrics,
        uses AdversarialClaimGenerator to produce a harder variant, then
        resets with that variant. Falls back to a normal reset() if
        generation fails or no API key is configured.
        """
        try:
            from .adversarial import AdversarialClaimGenerator

            gen = AdversarialClaimGenerator()
            nli_scores = agent_performance.get("nli_scores", {})
            result = gen.generate(
                claim_text=original_claim.get("claim", ""),
                original_label=original_claim.get("label", ""),
                agent_verdict=agent_performance.get("verdict", "HALF_TRUE"),
                agent_confidence=float(agent_performance.get("confidence", 0.5)),
                nli_scores=nli_scores if isinstance(nli_scores, dict) else {},
            )
            if result and result.get("claim"):
                adversarial_dict = {
                    "id": original_claim.get("id", "adv_001") + "_adv",
                    "claim": result["claim"],
                    "label": result.get("expected_label", original_claim.get("label", "unknown")),
                    "speaker": original_claim.get("speaker", "adversarial"),
                    "topic": original_claim.get("topic", "adversarial"),
                    "difficulty": result.get("difficulty", kwargs.get("task", "hard")),
                    "claim_date": None,
                    "has_image": False,
                    "image_url": None,
                    "gold_evidence": [],
                    "gold_reasoning": result.get("reasoning", ""),
                    "evidence_passages": {},
                }
                difficulty = adversarial_dict["difficulty"]
                if difficulty not in ("easy", "medium", "hard"):
                    difficulty = kwargs.get("task", "hard")
                return self.reset_with_custom_claim(adversarial_dict, difficulty=difficulty)
        except Exception:
            pass

        # Fallback to normal reset
        return self.reset(**kwargs)

    # =====================================================================
    # Advanced feature: multilingual reset
    # =====================================================================

    def reset_multilingual(
        self,
        claim_text: str,
        source_language: str = "auto",
        **kwargs,
    ) -> InvestigateObservation:
        """Reset with a claim in any language. Translates to English internally.

        Detects the language of *claim_text* (or uses *source_language* if
        provided), translates to English for investigation, and stores the
        original language in the observation's original_language field.
        Falls back gracefully when translation is unavailable.
        """
        from .translation import TranslationClient

        tc = TranslationClient()

        # Detect language
        if source_language == "auto" or not source_language:
            detected_lang = tc.detect_language(claim_text)
        else:
            detected_lang = source_language

        # Translate to English if needed
        if detected_lang != "en":
            english_claim = tc.translate_to_english(claim_text, detected_lang)
        else:
            english_claim = claim_text

        difficulty = kwargs.get("task", "easy")
        claim_dict = {
            "id": "multilingual_claim",
            "claim": english_claim,
            "label": "unknown",
            "speaker": "user",
            "topic": "multilingual",
            "difficulty": difficulty,
            "claim_date": None,
            "has_image": False,
            "image_url": None,
            "gold_evidence": [],
            "gold_reasoning": "",
            "evidence_passages": {},
            "_original_language": detected_lang,
            "_original_claim": claim_text,
        }

        obs = self.reset_with_custom_claim(claim_dict, difficulty=difficulty)

        # Annotate the observation with language metadata (rebuild with extra fields)
        return InvestigateObservation(
            claim=obs.claim,
            available_sources=obs.available_sources,
            source_content=obs.source_content,
            cross_ref_result=obs.cross_ref_result,
            credibility_score=obs.credibility_score,
            credibility_details=obs.credibility_details,
            budget_remaining=obs.budget_remaining,
            steps_taken=obs.steps_taken,
            message=obs.message,
            done=obs.done,
            reward=obs.reward,
            image_url=obs.image_url,
            original_language=detected_lang,
            translated_claim=english_claim if detected_lang != "en" else None,
        )

    # =====================================================================
    # Helpers
    # =====================================================================

    def _make_observation(self, message: str) -> InvestigateObservation:
        """Build an observation from current episode state."""
        claim_text = self._ctx.current_claim.get("claim", "") if self._ctx.current_claim else ""
        image_url = self._ctx.current_claim.get("image_url") if self._ctx.current_claim else None
        return InvestigateObservation(
            claim=claim_text,
            available_sources=SOURCE_CATEGORIES,
            source_content=self._ctx.last_source_content,
            cross_ref_result=self._ctx.last_cross_ref,
            credibility_score=self._ctx.last_credibility,
            credibility_details=self._ctx.last_credibility_details,
            budget_remaining=max(0, self._ctx.budget - self._ctx.steps_used),
            steps_taken=self._ctx.steps_used,
            message=message,
            done=self._ctx.done,
            reward=self._ctx.reward if self._ctx.done else None,
            image_url=image_url,
            entity_info=self._serialize_dict(self._ctx.last_entity_info),
            timeline_info=self._serialize_dict(self._ctx.last_timeline_info),
            image_match=self._serialize_dict(self._ctx.last_image_match),
            consensus_score=self._ctx.last_consensus_score,
            cache_hit=self._ctx.last_cache_hit,
        )

    @staticmethod
    def _serialize_dict(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        """Pydantic observation fields want Dict[str, str] — stringify values."""
        if d is None:
            return None
        return {k: str(v) for k, v in d.items()}

    def _find_evidence_text(self, source_id: str) -> str:
        """Find cached or retrieved evidence text for a source.

        Priority:
        1. Fix #23 — O(1) evidence_index check (exact source_type match)
        2. Episode-local cache (by source_type substring match, for partial keys)
        3. EvidenceDB cache (any row for this claim with this source_type)
        4. Legacy evidence_passages dict on the claim
        """
        # 1. O(1) exact-match in episode evidence index
        if source_id in self._ctx.evidence_index:
            return self._ctx.evidence_index[source_id]

        # 2. Episode-local fallback (substring match for partial source_id)
        for ev in self._ctx.retrieved_evidence:
            ev_type = ev.get("source_type", "")
            if source_id in ev_type or ev_type in source_id:
                return ev.get("content", "")

        # 3. EvidenceDB
        claim_id = self._ctx.current_claim.get("id", "")
        if claim_id:
            rows = self.evidence_db.get_for_claim(claim_id)
            for row in rows:
                if source_id in row.get("source_type", "") or row.get("source_type", "") in source_id:
                    return row.get("content", "")

        # 4. Legacy
        passages = self._ctx.current_claim.get("evidence_passages", {}) or {}
        if source_id in passages:
            return passages[source_id]
        for key, val in passages.items():
            if source_id in key or key in source_id:
                return val
        return ""

    @staticmethod
    def _extract_first_entity(text: str) -> str:
        """Naive entity extraction — longest capitalized multi-word phrase."""
        # Fix #24 — uses module-level precompiled _ENTITY_PATTERN (no re.findall overhead)
        phrases = _ENTITY_PATTERN.findall(text)
        return max(phrases, key=len) if phrases else ""

    def _log_trajectory(self, action: InvestigateAction, obs: InvestigateObservation) -> None:
        """Log every step to trajectories.db for RL training. Failures are silent."""
        if not self._ctx.episode_id or self._ctx.current_claim is None:
            return
        try:
            state = {
                "budget_remaining": obs.budget_remaining,
                "steps_taken": obs.steps_taken,
                "accessed_sources": list(self._ctx.accessed_sources),
                "nli_count": len(self._ctx.nli_results),
                "evidence_count": len(self._ctx.retrieved_evidence),
            }
            action_dict = {
                "action_type": action.action_type,
                "source_id": action.source_id,
                "query": getattr(action, "query", None),
                "entity": getattr(action, "entity", None),
                "verdict": action.verdict,
                "confidence": action.confidence,
            }
            self.trajectories_db.log_step(
                episode_id=self._ctx.episode_id,
                step_index=self._ctx.steps_used,
                claim_id=self._ctx.current_claim.get("id", ""),
                difficulty=self._ctx.difficulty,
                state=state,
                action=action_dict,
                reward=self._ctx.reward if self._ctx.done else 0.0,
                done=self._ctx.done,
            )
        except Exception:
            pass


# Emergency fallback claim if ClaimsDB is completely unavailable. Never
# used in normal operation, but keeps the env bootable in degraded cases.
_EMERGENCY_CLAIM: Dict[str, Any] = {
    "id": "emergency_001",
    "claim": "Water boils at 100 degrees Celsius at sea level.",
    "label": "true",
    "speaker": "Built-in",
    "topic": "science",
    "difficulty": "easy",
    "claim_date": None,
    "has_image": False,
    "image_url": None,
    "gold_evidence": ["physics_reference"],
    "gold_reasoning": "This is a well-established physical fact at 1 atm pressure.",
    "evidence_passages": {},
}
