from typing import Dict, List, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class InvestigateAction(Action):
    """What the agent sends to the environment each step.

    Action space (10 total):
    - request_source       : fetch evidence from a source category (real retrieval)
    - cross_reference      : run NLI on claim vs retrieved evidence (real DeBERTa)
    - check_credibility    : look up publisher in sources.db (4k+ publishers)
    - analyze_image        : run CLIP + pHash on the claim's image
    - search_evidence      : full-text search across evidence corpus + live Wikipedia
    - check_entity         : resolve a named entity via Wikidata
    - check_timeline       : temporal analysis (when made vs when contradicted)
    - reverse_image_search : pHash lookup against known-misattributed images
    - compute_consensus    : aggregate agreement across all retrieved evidence
    - submit_verdict       : final verdict with evidence and reasoning
    """

    action_type: str = Field(
        description=(
            "One of: request_source, cross_reference, check_credibility, "
            "analyze_image, search_evidence, check_entity, check_timeline, "
            "reverse_image_search, compute_consensus, submit_verdict"
        )
    )
    source_id: Optional[str] = Field(
        default=None,
        description="Source category or specific source ID to investigate",
    )
    query: Optional[str] = Field(
        default=None,
        description="Free-text query for search_evidence (optional; defaults to claim text)",
    )
    entity: Optional[str] = Field(
        default=None,
        description="Named entity to resolve via Wikidata (for check_entity)",
    )
    image_url: Optional[str] = Field(
        default=None,
        description="Image URL for reverse_image_search (defaults to the claim's image)",
    )
    verdict: Optional[str] = Field(
        default=None,
        description=(
            "Final verdict: TRUE, MOSTLY_TRUE, HALF_TRUE, "
            "MOSTLY_FALSE, FALSE, PANTS_ON_FIRE"
        ),
    )
    evidence: Optional[List[str]] = Field(
        default=None,
        description="List of source IDs supporting the verdict",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in its verdict (0.0-1.0)",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's explanation for its verdict",
    )


class InvestigateObservation(Observation):
    """What the environment returns after each step."""

    claim: str = Field(description="The claim being investigated")
    available_sources: List[str] = Field(
        default_factory=list,
        description="Source categories available for investigation",
    )
    source_content: Optional[str] = Field(
        default=None,
        description="Content returned from request_source action",
    )
    cross_ref_result: Optional[Dict[str, float]] = Field(
        default=None,
        description="NLI scores: {entailment, contradiction, neutral}",
    )
    credibility_score: Optional[float] = Field(
        default=None,
        description="Source credibility rating (0.0-1.0)",
    )
    credibility_details: Optional[Dict[str, str]] = Field(
        default=None,
        description="Bias rating, factual reporting level, etc.",
    )
    budget_remaining: int = Field(
        default=0, description="Investigation steps remaining"
    )
    steps_taken: int = Field(default=0, description="Steps used so far")
    message: str = Field(default="", description="Feedback from the environment")
    image_url: Optional[str] = Field(
        default=None,
        description="URL of associated image if this is a visual claim",
    )
    # New observation fields for expanded action space
    entity_info: Optional[Dict[str, str]] = Field(
        default=None,
        description="Resolved entity metadata from Wikidata (check_entity)",
    )
    timeline_info: Optional[Dict[str, str]] = Field(
        default=None,
        description="Claim + evidence temporal analysis (check_timeline)",
    )
    image_match: Optional[Dict[str, str]] = Field(
        default=None,
        description="Reverse image search result (reverse_image_search)",
    )
    consensus_score: Optional[float] = Field(
        default=None,
        description="Aggregated multi-source agreement score in [0,1] (compute_consensus)",
    )
    cache_hit: Optional[bool] = Field(
        default=None,
        description="Whether the last retrieval came from cache (debug/metrics)",
    )


class InvestigateState(State):
    """Episode metadata (accessible via state property)."""

    difficulty: str = Field(default="easy", description="easy, medium, or hard")
    ground_truth_verdict: str = Field(
        default="", description="The correct verdict"
    )
    gold_evidence: List[str] = Field(
        default_factory=list, description="The ideal evidence set"
    )
    claim_topic: str = Field(default="", description="Topic category of the claim")
    claim_id: str = Field(default="", description="ID of the current claim")
