from typing import Dict, List, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class InvestigateAction(Action):
    """What the agent sends to the environment each step."""

    action_type: str = Field(
        description=(
            "One of: request_source, cross_reference, "
            "check_credibility, submit_verdict, analyze_image"
        )
    )
    source_id: Optional[str] = Field(
        default=None,
        description="Source category or specific source ID to investigate",
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
