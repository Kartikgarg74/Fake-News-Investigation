"""FastAPI server for the Fake News Investigator environment."""

from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from openenv.core.env_server import create_fastapi_app

from ..models import InvestigateAction, InvestigateObservation
from .environment import FakeNewsEnvironment

# Create the standard OpenEnv app (auto-generates /ws, /reset, /step, /state, /health, /docs)
app = create_fastapi_app(
    FakeNewsEnvironment,
    InvestigateAction,
    InvestigateObservation,
    max_concurrent_envs=10,
)

# CORS for frontend testing (allows browser to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/frontend", response_class=HTMLResponse)
def serve_frontend():
    """Serve the testing frontend (only available on frontend-testing branch)."""
    html_path = FRONTEND_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Frontend not found</h1><p>This endpoint is only available on the frontend-testing branch.</p>", status_code=404)


# =========================================================================
# Hackathon-required custom endpoints
# =========================================================================


@app.get("/tasks")
def get_tasks():
    """Return list of tasks and action schema."""
    return {
        "tasks": [
            {
                "id": "easy",
                "description": (
                    "Clear-cut factual claims (TRUE/FALSE) with abundant "
                    "evidence. Single factual assertion, all sources agree."
                ),
                "budget": 10,
                "target_score_range": "0.7-1.0",
            },
            {
                "id": "medium",
                "description": (
                    "Distorted or exaggerated claims (MOSTLY_TRUE/MOSTLY_FALSE) "
                    "requiring source analysis and nuance detection."
                ),
                "budget": 8,
                "target_score_range": "0.3-0.6",
            },
            {
                "id": "hard",
                "description": (
                    "Sophisticated misinformation (HALF_TRUE/PANTS_ON_FIRE) with "
                    "misleading framing, cherry-picked statistics, and conflicting sources."
                ),
                "budget": 6,
                "target_score_range": "0.0-0.3",
            },
        ],
        "action_schema": {
            "action_type": {
                "type": "string",
                "enum": [
                    "request_source",
                    "cross_reference",
                    "check_credibility",
                    "submit_verdict",
                ],
                "description": "The type of investigation action to perform",
            },
            "source_id": {
                "type": "string",
                "description": (
                    "Source category (for request_source) or specific source ID "
                    "(for cross_reference/check_credibility). Categories: "
                    "government_data, academic_papers, news_articles, fact_checks, "
                    "medical_journals, statistical_reports, international_organizations, "
                    "industry_reports"
                ),
            },
            "verdict": {
                "type": "string",
                "enum": [
                    "TRUE",
                    "MOSTLY_TRUE",
                    "HALF_TRUE",
                    "MOSTLY_FALSE",
                    "FALSE",
                    "PANTS_ON_FIRE",
                ],
                "description": "Final verdict (only for submit_verdict)",
            },
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of source IDs supporting the verdict",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Agent's confidence in the verdict (0.0-1.0)",
            },
            "reasoning": {
                "type": "string",
                "description": "Agent's explanation for the verdict",
            },
        },
    }


@app.get("/grader")
def get_grader(episode_id: str = ""):
    """Return grader score for a completed episode."""
    if episode_id and episode_id in FakeNewsEnvironment._completed_episodes:
        breakdown = FakeNewsEnvironment._completed_episodes[episode_id]
        return {
            "episode_id": episode_id,
            "score": breakdown["total"],
            "breakdown": breakdown,
        }

    return {
        "message": (
            "No episode_id provided or episode not found. "
            "Provide ?episode_id=<id> for a completed episode's score."
        ),
        "completed_episodes": len(FakeNewsEnvironment._completed_episodes),
        "scoring_weights": {
            "verdict_accuracy": 0.30,
            "evidence_quality": 0.25,
            "efficiency": 0.15,
            "confidence_calibration": 0.15,
            "reasoning_quality": 0.15,
        },
    }


@app.get("/baseline")
def run_baseline():
    """Run a simple baseline agent and return scores.

    This uses a deterministic heuristic baseline (no LLM API needed).
    The full LLM baseline is in baseline.py.
    """
    from ..models import InvestigateAction

    env = FakeNewsEnvironment()
    results = {}

    for task in ["easy", "medium", "hard"]:
        scores = []
        for _ in range(5):
            obs = env.reset(task=task)

            obs = env.step(InvestigateAction(
                action_type="request_source",
                source_id="fact_checks",
            ))
            obs = env.step(InvestigateAction(
                action_type="request_source",
                source_id="government_data",
            ))
            obs = env.step(InvestigateAction(
                action_type="submit_verdict",
                verdict="HALF_TRUE",
                evidence=["fact_checks", "government_data"],
                confidence=0.5,
                reasoning="Based on available evidence from fact-checking sources.",
            ))
            if obs.reward is not None:
                scores.append(obs.reward)

        avg = sum(scores) / len(scores) if scores else 0.0
        results[task] = {
            "average_score": round(avg, 4),
            "episodes": len(scores),
            "scores": [round(s, 4) for s in scores],
        }

    return {"baseline_results": results, "method": "heuristic (non-LLM)"}
