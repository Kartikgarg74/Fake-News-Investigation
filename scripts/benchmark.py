#!/usr/bin/env python3
"""Veritas benchmark CLI.

Runs the agent across easy/medium/hard tiers and prints a markdown results
table. Two modes:

  --mode heuristic   : no LLM calls (fast, offline-safe)
  --mode llm         : uses inference.run_episode with API_KEY + API_BASE_URL

Example:
  python scripts/benchmark.py --episodes 3 --mode heuristic
  python scripts/benchmark.py --episodes 5 --mode llm

Outputs a markdown table, writes full JSON to benchmark_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Allow running from any directory
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT.parent))

from fake_news_investigator.models import InvestigateAction
from fake_news_investigator.server.environment import FakeNewsEnvironment

MIN_SCORE = 0.01
MAX_SCORE = 0.99


def clamp(s: float) -> float:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return MIN_SCORE
    return max(MIN_SCORE, min(MAX_SCORE, v))


def run_heuristic_episode(env: FakeNewsEnvironment, task: str) -> Dict[str, Any]:
    """One heuristic episode: retrieve from wikipedia, cross-reference, submit verdict."""
    t0 = time.time()
    try:
        obs = env.reset(task=task)
    except Exception as exc:
        return {"score": MIN_SCORE, "steps": 0, "error": str(exc), "duration_s": 0.0}

    steps = 0
    try:
        # Real Wikipedia retrieval
        obs = env.step(InvestigateAction(
            action_type="request_source", source_id="wikipedia"
        ))
        steps += 1

        # Real NLI cross-reference against what we just retrieved
        obs = env.step(InvestigateAction(
            action_type="cross_reference", source_id="wikipedia"
        ))
        steps += 1

        # Decide verdict from NLI scores
        cross = obs.cross_ref_result or {}
        ent = cross.get("entailment", 0.33)
        con = cross.get("contradiction", 0.34)

        if con > 0.55:
            verdict, conf = "FALSE", 0.7
        elif ent > 0.55:
            verdict, conf = "TRUE", 0.7
        else:
            verdict, conf = "HALF_TRUE", 0.5

        obs = env.step(InvestigateAction(
            action_type="submit_verdict",
            verdict=verdict,
            evidence=["wikipedia"],
            confidence=conf,
            reasoning=f"NLI: entailment={ent:.2f}, contradiction={con:.2f}",
        ))
        steps += 1
    except Exception as exc:
        return {"score": MIN_SCORE, "steps": steps, "error": str(exc), "duration_s": time.time() - t0}

    return {
        "score": clamp(obs.reward if obs and obs.reward is not None else MIN_SCORE),
        "steps": steps,
        "duration_s": round(time.time() - t0, 3),
        "error": None,
    }


def run_benchmark(mode: str, episodes_per_task: int) -> Dict[str, Any]:
    """Run the full benchmark and return results dict."""
    env = FakeNewsEnvironment()

    tiers: Dict[str, Dict[str, Any]] = {}
    for task in ("easy", "medium", "hard"):
        print(f"\n[{task}] running {episodes_per_task} episodes...", flush=True)
        scores: List[float] = []
        steps_list: List[int] = []
        errors: List[str] = []
        durations: List[float] = []

        for i in range(episodes_per_task):
            result = run_heuristic_episode(env, task)
            scores.append(result["score"])
            steps_list.append(result["steps"])
            durations.append(result["duration_s"])
            if result["error"]:
                errors.append(result["error"])
            print(
                f"  ep{i+1}: score={result['score']:.4f} "
                f"steps={result['steps']} time={result['duration_s']:.2f}s",
                flush=True,
            )

        tiers[task] = {
            "episodes": len(scores),
            "avg_score": round(statistics.mean(scores) if scores else MIN_SCORE, 4),
            "min_score": round(min(scores) if scores else MIN_SCORE, 4),
            "max_score": round(max(scores) if scores else MIN_SCORE, 4),
            "stdev": round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 4),
            "avg_steps": round(statistics.mean(steps_list) if steps_list else 0.0, 2),
            "avg_duration_s": round(statistics.mean(durations) if durations else 0.0, 3),
            "error_count": len(errors),
            "errors": errors[:3],
        }

    return {
        "mode": mode,
        "episodes_per_task": episodes_per_task,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tiers": tiers,
    }


def format_markdown(results: Dict[str, Any]) -> str:
    """Format results as a markdown table."""
    lines = []
    lines.append(f"# Veritas Benchmark Results")
    lines.append("")
    lines.append(f"- **Mode**: `{results['mode']}`")
    lines.append(f"- **Episodes per task**: {results['episodes_per_task']}")
    lines.append(f"- **Timestamp**: {results['timestamp']}")
    lines.append("")
    lines.append("| Task | Episodes | Avg Score | Min | Max | Stdev | Avg Steps | Avg Time (s) | Errors |")
    lines.append("|------|----------|-----------|-----|-----|-------|-----------|--------------|--------|")
    for task in ("easy", "medium", "hard"):
        t = results["tiers"][task]
        lines.append(
            f"| {task} | {t['episodes']} | "
            f"{t['avg_score']:.4f} | {t['min_score']:.4f} | {t['max_score']:.4f} | "
            f"{t['stdev']:.4f} | {t['avg_steps']:.2f} | {t['avg_duration_s']:.3f} | "
            f"{t['error_count']} |"
        )
    lines.append("")
    overall = statistics.mean([results["tiers"][t]["avg_score"] for t in ("easy", "medium", "hard")])
    lines.append(f"**Overall average score**: {overall:.4f}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Veritas benchmark")
    parser.add_argument("--mode", choices=["heuristic"], default="heuristic",
                        help="Benchmark mode (llm mode requires API_KEY)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes per difficulty tier")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"),
                        help="Where to write JSON results")
    args = parser.parse_args(argv)

    print(f"Running Veritas benchmark: mode={args.mode}, episodes={args.episodes}")
    results = run_benchmark(args.mode, args.episodes)

    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nFull results written to: {args.output}")

    md = format_markdown(results)
    print()
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
