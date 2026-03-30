"""
Baseline inference script for Fake News Investigator environment.

Uses OpenAI GPT-4o-mini to investigate claims across 3 difficulty tiers.
Reads OPENAI_API_KEY from environment variables.

Usage:
    export OPENAI_API_KEY="sk-..."
    python baseline.py [--url http://localhost:8000] [--episodes 5]
"""

import argparse
import json
import os
import sys
import time

# Add parent dir to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fake_news_investigator.models import InvestigateAction, InvestigateObservation
from fake_news_investigator.server.environment import FakeNewsEnvironment

# Supported LLM providers (all use OpenAI-compatible API)
PROVIDERS = {
    "openai": {
        "base_url": None,  # Uses default OpenAI URL
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "env_key": "OPENAI_API_KEY",  # Hackathon requires this env var name
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "env_key": "OPENAI_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2",
        "env_key": None,  # No key needed
    },
}

SYSTEM_PROMPT = """You are a professional fact-checking investigator. You are given a claim to verify.
You have a limited investigation budget. Use your steps wisely.

Available actions (respond with valid JSON only, no markdown):

1. Request evidence from a source category:
   {"action_type": "request_source", "source_id": "<category>"}
   Categories: government_data, academic_papers, news_articles, fact_checks,
   medical_journals, statistical_reports, international_organizations, industry_reports

2. Cross-reference the claim against a source:
   {"action_type": "cross_reference", "source_id": "<category>"}

3. Check source credibility:
   {"action_type": "check_credibility", "source_id": "<source_name>"}

4. Submit your final verdict:
   {"action_type": "submit_verdict", "verdict": "<LABEL>", "evidence": ["src1", "src2"],
    "confidence": 0.0-1.0, "reasoning": "Your explanation"}
   Labels: TRUE, MOSTLY_TRUE, HALF_TRUE, MOSTLY_FALSE, FALSE, PANTS_ON_FIRE

Strategy:
- Start by requesting evidence from fact_checks or government_data
- Cross-reference key claims against authoritative sources
- Check credibility of any suspicious sources
- Submit verdict with evidence, confidence, and reasoning
- Use fewer steps for higher efficiency score"""


def run_with_openai(env, task: str, episodes: int = 5, provider: str = "groq"):
    """Run baseline using OpenAI-compatible API.

    Supports: openai, groq (free), together (free tier), ollama (local).
    All use the OpenAI Python client as required by the hackathon.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI package not installed. Install with: pip install openai")
        return None

    prov = PROVIDERS.get(provider, PROVIDERS["groq"])
    api_key = os.getenv(prov["env_key"] or "OPENAI_API_KEY", "")

    if prov["env_key"] and not api_key:
        print(f"API key not set ({prov['env_key']}). Running heuristic baseline instead.")
        return None

    client_kwargs = {"api_key": api_key or "not-needed"}
    if prov["base_url"]:
        client_kwargs["base_url"] = prov["base_url"]

    client = OpenAI(**client_kwargs)
    model = prov["model"]
    print(f"  Using provider: {provider} | model: {model}")
    scores = []

    for ep in range(episodes):
        obs = env.reset(task=task)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"CLAIM TO INVESTIGATE: \"{obs.claim}\"\n\n"
                f"Available source categories: {', '.join(obs.available_sources)}\n"
                f"Investigation budget: {obs.budget_remaining} steps\n\n"
                f"Begin your investigation. Respond with a JSON action."
            )},
        ]

        step_count = 0
        initial_budget = obs.budget_remaining
        while not obs.done and step_count < initial_budget + 2:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=500,
                )
                content = response.choices[0].message.content.strip()

                # Try to parse JSON from response
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()

                action_data = json.loads(content)
                action = InvestigateAction(**action_data)
                obs = env.step(action)
                step_count += 1

                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": (
                    f"Result: {obs.message}\n"
                    + (f"Source content: {obs.source_content[:400]}...\n"
                       if obs.source_content and len(obs.source_content) > 0 else "")
                    + (f"Cross-reference: {json.dumps(obs.cross_ref_result)}\n"
                       if obs.cross_ref_result else "")
                    + (f"Credibility: {obs.credibility_score} - {obs.credibility_details}\n"
                       if obs.credibility_score is not None else "")
                    + f"Budget remaining: {obs.budget_remaining}\n"
                    + ("Submit your verdict now." if obs.budget_remaining <= 1 else
                       "Continue investigating or submit verdict.")
                )})

            except (json.JSONDecodeError, Exception) as e:
                # If LLM returns invalid JSON, force a verdict submission
                obs = env.step(InvestigateAction(
                    action_type="submit_verdict",
                    verdict="HALF_TRUE",
                    evidence=[],
                    confidence=0.3,
                    reasoning=f"Unable to complete investigation. Error: {str(e)[:100]}",
                ))
                break

        if obs.reward is not None:
            scores.append(obs.reward)

    return scores


def run_heuristic(env, task: str, episodes: int = 5):
    """Run a simple heuristic baseline (no LLM needed).

    Uses a two-phase strategy:
    1. Gather evidence from 2 sources + cross-reference
    2. Use NLI contradiction/entailment signal to pick verdict
    """
    scores = []

    for _ in range(episodes):
        obs = env.reset(task=task)

        # Step 1: Request fact_checks
        obs = env.step(InvestigateAction(
            action_type="request_source",
            source_id="fact_checks",
        ))
        evidence_text = obs.source_content or ""

        # Step 2: Cross-reference claim against fact_checks
        obs = env.step(InvestigateAction(
            action_type="cross_reference",
            source_id="fact_checks",
        ))

        # Step 3: Request government_data
        obs = env.step(InvestigateAction(
            action_type="request_source",
            source_id="government_data",
        ))

        # Step 4: Check credibility of a source
        obs = env.step(InvestigateAction(
            action_type="check_credibility",
            source_id="politifact.com",
        ))

        # Step 5: Decide verdict based on NLI + evidence keywords
        verdict = "HALF_TRUE"
        confidence = 0.4

        # Use keywords from evidence to improve the guess
        evidence_lower = evidence_text.lower()
        has_contradiction = any(w in evidence_lower for w in [
            "false", "debunked", "incorrect", "misleading", "not true",
            "no evidence", "contradicts", "refuted", "inaccurate",
        ])
        has_support = any(w in evidence_lower for w in [
            "confirmed", "accurate", "correct", "true", "verified",
            "supports", "consistent",
        ])

        if has_contradiction and not has_support:
            verdict = "FALSE"
            confidence = 0.65
        elif has_support and not has_contradiction:
            verdict = "TRUE"
            confidence = 0.65
        elif has_contradiction and has_support:
            verdict = "MOSTLY_FALSE"
            confidence = 0.45
        else:
            verdict = "HALF_TRUE"
            confidence = 0.35

        obs = env.step(InvestigateAction(
            action_type="submit_verdict",
            verdict=verdict,
            evidence=["fact_checks", "government_data"],
            confidence=confidence,
            reasoning=(
                f"Based on evidence from fact-checking sources and government data. "
                f"Key evidence indicators: "
                f"{'contradiction detected' if has_contradiction else 'no contradiction'}, "
                f"{'support detected' if has_support else 'no support'}. "
                f"Verdict: {verdict.lower().replace('_', ' ')}."
            ),
        ))

        if obs.reward is not None:
            scores.append(obs.reward)

    return scores


def main():
    parser = argparse.ArgumentParser(description="Fake News Investigator Baseline")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per tier")
    parser.add_argument("--method", choices=["llm", "heuristic", "auto"],
                        default="auto", help="Baseline method")
    parser.add_argument("--provider", choices=list(PROVIDERS.keys()),
                        default="groq",
                        help="LLM provider (default: groq, free)")
    args = parser.parse_args()

    env = FakeNewsEnvironment()
    all_results = {}

    print("=" * 60)
    print("Fake News Investigator — Baseline Evaluation")
    print("=" * 60)

    for task in ["easy", "medium", "hard"]:
        print(f"\n--- Task: {task} ({args.episodes} episodes) ---")

        scores = None
        method_used = "heuristic"

        if args.method in ("llm", "auto"):
            scores = run_with_openai(env, task, args.episodes, provider=args.provider)
            if scores:
                method_used = f"llm ({args.provider})"

        if scores is None:
            scores = run_heuristic(env, task, args.episodes)
            method_used = "heuristic"

        avg = sum(scores) / len(scores) if scores else 0.0
        all_results[task] = {
            "method": method_used,
            "average_score": round(avg, 4),
            "min_score": round(min(scores), 4) if scores else 0,
            "max_score": round(max(scores), 4) if scores else 0,
            "episodes": len(scores),
            "scores": [round(s, 4) for s in scores],
        }

        print(f"  Method: {method_used}")
        print(f"  Avg Score: {avg:.4f}")
        print(f"  Min/Max: {min(scores):.4f} / {max(scores):.4f}")
        print(f"  Scores: {[round(s, 4) for s in scores]}")

    print("\n" + "=" * 60)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 60)
    print(json.dumps(all_results, indent=2))

    return all_results


if __name__ == "__main__":
    main()
