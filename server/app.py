"""FastAPI server for the Veritas fact-checking environment."""
import json
import os

from fastapi.responses import HTMLResponse, StreamingResponse

from openenv.core.env_server import create_fastapi_app

from ..models import InvestigateAction, InvestigateObservation
from .databases import TrajectoriesDB
from .environment import FakeNewsEnvironment

# Create the standard OpenEnv app (auto-generates /ws, /reset, /step, /state, /health, /docs)
app = create_fastapi_app(
    FakeNewsEnvironment,
    InvestigateAction,
    InvestigateObservation,
    max_concurrent_envs=10,
)


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


# =========================================================================
# Live demo dashboard — /demo HTML + /demo/stream SSE endpoint
# =========================================================================

_DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Veritas — Live Fact-Checking Demo</title>
<style>
  :root {
    --bg: #0e1117; --panel: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #8b949e;
    --green: #3fb950; --red: #f85149; --yellow: #d29922; --blue: #58a6ff;
  }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 2rem; }
  .container { max-width: 960px; margin: 0 auto; }
  header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 2rem; }
  h1 { font-size: 1.6rem; margin: 0; }
  h1 .subtitle { color: var(--muted); font-weight: 400; font-size: 1rem; margin-left: 0.5rem; }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem; }
  textarea { width: 100%; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 0.75rem; font-family: inherit; font-size: 0.95rem; resize: vertical; }
  select, button { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 0.55rem 1rem; font-size: 0.95rem; cursor: pointer; }
  button { background: var(--blue); color: white; border: none; font-weight: 600; }
  button:disabled { background: #22272e; color: var(--muted); cursor: not-allowed; }
  button:hover:not(:disabled) { background: #4493f8; }
  .controls { display: flex; gap: 0.75rem; align-items: center; margin-top: 0.75rem; }
  .step { border-left: 3px solid var(--border); padding: 0.6rem 0.9rem; margin: 0.5rem 0; background: rgba(255,255,255,0.02); border-radius: 4px; }
  .step.action { border-left-color: var(--blue); }
  .step.success { border-left-color: var(--green); }
  .step.warn { border-left-color: var(--yellow); }
  .step.error { border-left-color: var(--red); }
  .step-head { display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--muted); margin-bottom: 0.3rem; }
  .step-body { font-size: 0.92rem; white-space: pre-wrap; word-break: break-word; }
  .tag { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 10px; font-size: 0.75rem; font-weight: 600; }
  .tag-live { background: var(--green); color: #0e1117; }
  .tag-cached { background: var(--blue); color: #0e1117; }
  .tag-synthetic { background: var(--yellow); color: #0e1117; }
  #status { color: var(--muted); font-size: 0.9rem; }
  footer { margin-top: 2rem; color: var(--muted); font-size: 0.85rem; text-align: center; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Veritas <span class="subtitle">live fact-checking demo</span></h1>
    <span id="status">idle</span>
  </header>

  <div class="panel">
    <div style="color: var(--muted); font-size: 0.85rem; margin-bottom: 0.5rem;">CLAIM TO INVESTIGATE</div>
    <textarea id="claim" rows="3" placeholder="e.g. The Great Wall of China is visible from space with the naked eye.">The Great Wall of China is visible from space with the naked eye.</textarea>
    <div class="controls">
      <select id="difficulty">
        <option value="easy">easy (10 step budget)</option>
        <option value="medium">medium (8 step budget)</option>
        <option value="hard">hard (6 step budget)</option>
      </select>
      <button id="run">Investigate</button>
      <span style="color: var(--muted); font-size: 0.85rem;">Heuristic strategy — runs a real investigation with live retrieval</span>
    </div>
  </div>

  <div id="output"></div>

  <footer>
    7 segregated databases · 10 actions · real Wikipedia / Wikidata / NLI retrieval ·
    <a href="/docs" style="color: var(--blue);">/docs</a> · <a href="/tasks" style="color: var(--blue);">/tasks</a> ·
    <a href="/grader" style="color: var(--blue);">/grader</a>
  </footer>
</div>

<script>
const $ = (id) => document.getElementById(id);
const output = $('output');
const runBtn = $('run');
const statusEl = $('status');

function appendStep(stepType, title, body) {
  const div = document.createElement('div');
  div.className = 'step ' + stepType;
  div.innerHTML = `<div class="step-head"><span>${title}</span><span>${new Date().toLocaleTimeString()}</span></div><div class="step-body">${body}</div>`;
  output.appendChild(div);
  div.scrollIntoView({behavior: 'smooth', block: 'end'});
}

function runInvestigation() {
  const claim = $('claim').value.trim();
  const difficulty = $('difficulty').value;
  if (!claim) return;
  output.innerHTML = '';
  runBtn.disabled = true;
  statusEl.textContent = 'investigating...';

  const params = new URLSearchParams({claim, difficulty});
  const es = new EventSource('/demo/stream?' + params.toString());

  es.addEventListener('step', (e) => {
    const d = JSON.parse(e.data);
    let tag = '';
    if (d.cache_hit === true) tag = ' <span class="tag tag-cached">cached</span>';
    else if (d.is_synthetic === true) tag = ' <span class="tag tag-synthetic">synthetic</span>';
    else if (d.ok) tag = ' <span class="tag tag-live">live</span>';
    appendStep('action', `[${d.step_num}] ${d.action_type}${tag}`, d.message + (d.content ? '\\n\\n' + d.content : ''));
  });

  es.addEventListener('verdict', (e) => {
    const d = JSON.parse(e.data);
    appendStep('success', 'FINAL VERDICT: ' + d.verdict, `Ground truth: ${d.ground_truth}\\nScore: ${d.score.toFixed(4)}\\nReasoning: ${d.reasoning}`);
  });

  es.addEventListener('error', (e) => {
    try { const d = JSON.parse(e.data); appendStep('error', 'error', d.message); } catch(_) {}
    es.close();
    runBtn.disabled = false;
    statusEl.textContent = 'idle';
  });

  es.addEventListener('done', (_) => {
    es.close();
    runBtn.disabled = false;
    statusEl.textContent = 'done';
  });
}

runBtn.addEventListener('click', runInvestigation);
</script>
</body>
</html>
"""


@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    """Live demo dashboard — paste a claim, watch the agent investigate."""
    return _DEMO_HTML


@app.get("/demo/stream")
def demo_stream(claim: str = "", difficulty: str = "easy"):
    """SSE stream: run a heuristic investigation and emit events step by step.

    This is the same agent loop as run_heuristic_fallback in inference.py but
    streams each step over Server-Sent Events so the UI can render progress.
    """
    def event_stream():
        if not claim:
            yield f"event: error\ndata: {json.dumps({'message': 'No claim provided'})}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        try:
            env = FakeNewsEnvironment()
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'message': f'Env init failed: {exc}'})}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        # Inject the user's claim as the current_claim, bypassing LIAR dataset
        env._reset_episode_state()
        env._difficulty = difficulty if difficulty in ("easy", "medium", "hard") else "easy"
        env._budget = {"easy": 10, "medium": 8, "hard": 6}[env._difficulty]
        env._episode_id = "demo_" + str(int(__import__("time").time()))
        env._current_claim = {
            "id": "demo_live",
            "claim": claim,
            "label": "unknown",
            "speaker": "user",
            "topic": "user_submitted",
            "difficulty": env._difficulty,
            "claim_date": None,
            "has_image": False,
            "image_url": None,
            "gold_evidence": [],
            "gold_reasoning": "",
            "evidence_passages": {},
        }

        # Scripted investigation: search_evidence -> request_source wikipedia ->
        # cross_reference -> check_credibility -> compute_consensus -> submit
        actions_sequence = [
            {"action_type": "search_evidence", "query": claim[:200]},
            {"action_type": "request_source", "source_id": "wikipedia"},
            {"action_type": "cross_reference", "source_id": "wikipedia"},
            {"action_type": "check_credibility", "source_id": "en.wikipedia.org"},
            {"action_type": "compute_consensus"},
        ]

        step_num = 0
        last_cross_ref = None
        for action_dict in actions_sequence:
            step_num += 1
            try:
                action = InvestigateAction(**action_dict)
                obs = env.step(action)
            except Exception as exc:
                yield f"event: step\ndata: {json.dumps({'step_num': step_num, 'action_type': action_dict['action_type'], 'message': f'failed: {exc}', 'content': '', 'ok': False})}\n\n"
                continue

            if obs.cross_ref_result:
                last_cross_ref = obs.cross_ref_result

            payload = {
                "step_num": step_num,
                "action_type": action_dict["action_type"],
                "message": obs.message,
                "content": (obs.source_content or "")[:600],
                "cache_hit": obs.cache_hit,
                "is_synthetic": False,
                "ok": True,
                "cross_ref": obs.cross_ref_result,
                "credibility_score": obs.credibility_score,
                "consensus_score": obs.consensus_score,
            }
            yield f"event: step\ndata: {json.dumps(payload)}\n\n"

        # Decide final verdict based on consensus + NLI
        verdict, confidence, reasoning = "HALF_TRUE", 0.5, "Insufficient evidence"
        if last_cross_ref:
            e = last_cross_ref.get("entailment", 0.33)
            c = last_cross_ref.get("contradiction", 0.33)
            if c > 0.6:
                verdict, confidence = "FALSE", 0.75
                reasoning = f"NLI contradiction score {c:.2f} strongly contradicts the claim."
            elif e > 0.6:
                verdict, confidence = "TRUE", 0.75
                reasoning = f"NLI entailment score {e:.2f} strongly supports the claim."
            else:
                verdict, confidence = "HALF_TRUE", 0.5
                reasoning = f"NLI scores inconclusive (E={e:.2f}, C={c:.2f})."

        try:
            final_obs = env.step(InvestigateAction(
                action_type="submit_verdict",
                verdict=verdict,
                evidence=["wikipedia"],
                confidence=confidence,
                reasoning=reasoning,
            ))
            score = float(final_obs.reward or 0.0)
        except Exception:
            score = 0.01

        yield f"event: verdict\ndata: {json.dumps({'verdict': verdict, 'ground_truth': 'user-submitted (unknown)', 'score': score, 'reasoning': reasoning})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/trajectories")
def list_trajectories(limit: int = 50):
    """Export recent RL trajectories from the audit/training log."""
    db = TrajectoriesDB()
    steps = db.export_jsonl(limit=limit)
    return {
        "total_episodes": db.count_episodes(),
        "total_steps": db.count_steps(),
        "returned": len(steps),
        "steps": steps,
    }


def main():
    """Entry point for the OpenEnv server (required by [project.scripts])."""
    import uvicorn
    uvicorn.run(
        "fake_news_investigator.server.app:app",
        host=os.environ.get("HOST", "127.0.0.1"),  # Security: Configurable host, defaults to localhost
        port=int(os.environ.get("PORT", 8000)),
        workers=1,
    )


if __name__ == "__main__":
    main()
