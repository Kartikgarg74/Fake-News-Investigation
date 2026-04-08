"""
Inference Script — Fake News Investigator Environment
=====================================================
MANDATORY ENVIRONMENT VARIABLES (injected by validator):
    API_BASE_URL   The LiteLLM proxy endpoint
    API_KEY        The API key for the proxy
    MODEL_NAME     The model identifier to use for inference

Uses OpenAI Client for all LLM calls as required by the hackathon.
"""

import json
import os
import re
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from fake_news_investigator.models import InvestigateAction
from fake_news_investigator.server.environment import FakeNewsEnvironment

# =========================================================================
# MANDATORY environment variables — validator injects API_KEY + API_BASE_URL
# =========================================================================
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

MAX_STEPS = 8
TEMPERATURE = 0.0
MAX_TOKENS = 400

SYSTEM_PROMPT = textwrap.dedent("""
You are a professional fact-checking investigator. You investigate claims
to determine their veracity. You have a limited investigation budget.

Available actions (respond with VALID JSON only, no markdown, no explanation):

1. Request evidence from a source category:
   {"action_type": "request_source", "source_id": "<category>"}
   Categories: government_data, academic_papers, news_articles, fact_checks,
   medical_journals, statistical_reports, international_organizations, industry_reports

2. Cross-reference the claim against a source:
   {"action_type": "cross_reference", "source_id": "<category>"}

3. Check source credibility:
   {"action_type": "check_credibility", "source_id": "<source_name_or_url>"}

4. Analyze associated image (if the claim has an image_url):
   {"action_type": "analyze_image"}
   Use this when the observation includes an image_url — returns visual forensic analysis.

5. Submit your final verdict:
   {"action_type": "submit_verdict", "verdict": "<LABEL>",
    "evidence": ["source1", "source2"],
    "confidence": 0.0-1.0,
    "reasoning": "Your explanation"}
   Labels: TRUE, MOSTLY_TRUE, HALF_TRUE, MOSTLY_FALSE, FALSE, PANTS_ON_FIRE

Strategy:
- If the claim has an image_url, call analyze_image as your FIRST action
- Start by requesting fact_checks or government_data
- Cross-reference the claim against authoritative sources
- Check credibility of any suspicious sources
- Submit verdict with evidence and reasoning
- Be efficient: fewer steps = higher score

RESPOND WITH JSON ONLY. NO MARKDOWN. NO EXPLANATION OUTSIDE THE JSON.
""").strip()

ACTION_PATTERN = re.compile(r'\{[^{}]*"action_type"[^{}]*\}', re.DOTALL)


def extract_json_action(text: str) -> dict:
    """Extract a JSON action from the model's response."""
    # Try direct parse
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try regex extraction
    match = ACTION_PATTERN.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def run_episode(client: OpenAI, env: FakeNewsEnvironment, task: str) -> float:
    """Run a single investigation episode."""
    print(f"[START] task={task}", flush=True)
    try:
        obs = env.reset(task=task)
    except Exception as exc:
        print(f"  env.reset() failed: {exc}")
        print(f"[END] task={task} score=0.0 steps=0", flush=True)
        return 0.0
    initial_budget = obs.budget_remaining

    image_note = f"\nAssociated image: {obs.image_url}" if obs.image_url else ""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"CLAIM TO INVESTIGATE: \"{obs.claim}\"{image_note}\n\n"
            f"Available source categories: {', '.join(obs.available_sources)}\n"
            f"Investigation budget: {obs.budget_remaining} steps\n\n"
            f"Begin your investigation. Respond with a JSON action."
        )},
    ]

    step_count = 0
    while not obs.done and step_count < initial_budget + 2:
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            error_msg = str(exc)[:200]
            # Redact any API keys that might appear in error messages
            _key = os.environ.get("API_KEY", "")
            if _key and _key in error_msg:
                error_msg = error_msg.replace(_key, "***REDACTED***")
            print(f"  LLM request failed: {error_msg}. Submitting fallback verdict.")
            obs = env.step(InvestigateAction(
                action_type="submit_verdict",
                verdict="HALF_TRUE",
                evidence=[],
                confidence=0.3,
                reasoning="Investigation incomplete due to LLM error.",
            ))
            step_count += 1
            print(f"[STEP] step={step_count} reward={obs.reward if obs.reward is not None else 0.0:.4f}", flush=True)
            break

        action_data = extract_json_action(response_text)
        if action_data is None:
            # Force verdict submission on parse failure
            obs = env.step(InvestigateAction(
                action_type="submit_verdict",
                verdict="HALF_TRUE",
                evidence=[],
                confidence=0.3,
                reasoning="Unable to parse investigation action.",
            ))
            step_count += 1
            print(f"[STEP] step={step_count} reward={obs.reward if obs.reward is not None else 0.0:.4f}", flush=True)
            break

        try:
            action = InvestigateAction(**action_data)
            obs = env.step(action)
            step_count += 1
            print(f"[STEP] step={step_count} reward={obs.reward if obs.reward is not None else 0.0:.4f}", flush=True)
        except Exception as exc:
            print(f"  Invalid action: {exc}. Submitting fallback.")
            obs = env.step(InvestigateAction(
                action_type="submit_verdict",
                verdict="HALF_TRUE",
                evidence=[],
                confidence=0.3,
                reasoning=f"Invalid action: {str(exc)[:100]}",
            ))
            step_count += 1
            print(f"[STEP] step={step_count} reward={obs.reward if obs.reward is not None else 0.0:.4f}", flush=True)
            break

        # Add to conversation
        messages.append({"role": "assistant", "content": response_text})

        feedback = f"Result: {obs.message}\n"
        if obs.source_content:
            feedback += f"Source content: {obs.source_content[:500]}\n"
        if obs.cross_ref_result:
            feedback += f"Cross-reference NLI: {json.dumps(obs.cross_ref_result)}\n"
        if obs.credibility_score is not None:
            feedback += f"Credibility: {obs.credibility_score} ({obs.credibility_details})\n"
        feedback += f"Budget remaining: {obs.budget_remaining}\n"

        if obs.budget_remaining <= 1 and not obs.done:
            feedback += "WARNING: Budget almost exhausted. Submit your verdict NOW."

        messages.append({"role": "user", "content": feedback})

    final_score = obs.reward if obs.reward is not None else 0.0
    print(f"[END] task={task} score={final_score:.4f} steps={step_count}", flush=True)
    return final_score


def main():
    # Debug: dump all relevant env vars so we can see what the validator injects
    print("=" * 60, flush=True)
    print("Fake News Investigator — Inference Script", flush=True)
    print("=" * 60, flush=True)
    print("Environment variables:", flush=True)
    for k, v in sorted(os.environ.items()):
        kup = k.upper()
        if any(x in kup for x in ["API", "KEY", "TOKEN", "URL", "MODEL", "HF_", "OPENAI"]):
            # Show first 20 chars of value for debugging (redact the rest)
            safe_v = (v[:20] + "...") if len(v) > 20 else v
            print(f"  {k}={safe_v}", flush=True)
    print(flush=True)

    # Read validator-injected variables (they inject API_KEY + API_BASE_URL)
    api_key = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or ""
    api_base_url = os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1"

    print(f"Using API_BASE_URL: {api_base_url}", flush=True)
    print(f"Using API_KEY: {'yes (' + str(len(api_key)) + ' chars)' if api_key else 'NO'}", flush=True)
    print(f"Using MODEL_NAME: {MODEL_NAME}", flush=True)
    print(flush=True)

    if not api_key:
        print("WARNING: No API_KEY or HF_TOKEN found. Running heuristic.", flush=True)
        return run_heuristic_fallback()

    # Initialize OpenAI client with validator-provided credentials
    client = OpenAI(base_url=api_base_url, api_key=api_key)

    # Verify proxy connectivity with a minimal test call
    print("Testing proxy connection...", flush=True)
    try:
        test = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        print(f"Proxy OK: {test.choices[0].message.content}", flush=True)
    except Exception as e:
        print(f"Proxy test failed: {e}", flush=True)
        # Continue anyway — individual episode errors are handled

    try:
        env = FakeNewsEnvironment()
    except Exception as exc:
        print(f"FakeNewsEnvironment() failed: {exc}. Running heuristic fallback.")
        return run_heuristic_fallback()
    all_results = {}
    episodes_per_task = 5

    for task in ["easy", "medium", "hard"]:
        print(f"\n--- Task: {task} ({episodes_per_task} episodes) ---")
        scores = []

        for ep in range(episodes_per_task):
            try:
                score = run_episode(client, env, task)
            except Exception as exc:
                print(f"  Episode {ep+1} failed: {exc}. Score: 0.0")
                score = 0.0
            scores.append(score)
            print(f"  Episode {ep+1}: score={score:.4f}")

        avg = sum(scores) / len(scores) if scores else 0.0
        all_results[task] = {
            "average_score": round(avg, 4),
            "min_score": round(min(scores), 4) if scores else 0,
            "max_score": round(max(scores), 4) if scores else 0,
            "episodes": len(scores),
            "scores": [round(s, 4) for s in scores],
        }
        print(f"  Average: {avg:.4f}")

    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(json.dumps(all_results, indent=2))
    return all_results


def run_heuristic_fallback():
    """Heuristic baseline when no API key is available."""
    env = FakeNewsEnvironment()
    all_results = {}

    for task in ["easy", "medium", "hard"]:
        scores = []
        for _ in range(5):
            print(f"[START] task={task}", flush=True)
            try:
                obs = env.reset(task=task)
            except Exception as exc:
                print(f"  env.reset() failed: {exc}")
                print(f"[END] task={task} score=0.0 steps=0", flush=True)
                continue
            obs = env.step(InvestigateAction(
                action_type="request_source", source_id="fact_checks"))
            print(f"[STEP] step=1 reward=0.0", flush=True)

            evidence_text = (obs.source_content or "").lower()
            has_contradiction = any(w in evidence_text for w in [
                "false", "debunked", "incorrect", "misleading",
                "contradicts", "refuted", "inaccurate",
            ])
            has_support = any(w in evidence_text for w in [
                "confirmed", "accurate", "correct", "true",
                "verified", "supports",
            ])

            verdict = "HALF_TRUE"
            conf = 0.4
            if has_contradiction and not has_support:
                verdict, conf = "FALSE", 0.65
            elif has_support and not has_contradiction:
                verdict, conf = "TRUE", 0.65

            obs = env.step(InvestigateAction(
                action_type="submit_verdict",
                verdict=verdict,
                evidence=["fact_checks"],
                confidence=conf,
                reasoning=f"Heuristic: {'contradiction' if has_contradiction else 'support' if has_support else 'ambiguous'} detected.",
            ))
            ep_score = obs.reward if obs.reward is not None else 0.0
            print(f"[STEP] step=2 reward={ep_score:.4f}", flush=True)
            print(f"[END] task={task} score={ep_score:.4f} steps=2", flush=True)
            if obs.reward is not None:
                scores.append(obs.reward)

        avg = sum(scores) / len(scores) if scores else 0.0
        all_results[task] = {
            "average_score": round(avg, 4),
            "episodes": len(scores),
            "scores": [round(s, 4) for s in scores],
        }
        print(f"{task:8s} | avg={avg:.4f}")

    print("\n" + json.dumps(all_results, indent=2))
    return all_results


if __name__ == "__main__":
    main()
