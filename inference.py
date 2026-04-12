"""
Veritas — Fact-Checking Investigator (inference script)
=======================================================

Runs an LLM agent across the Veritas fact-checking environment. The agent
has 10 actions available (request_source, cross_reference, check_credibility,
analyze_image, search_evidence, check_entity, check_timeline,
reverse_image_search, compute_consensus, submit_verdict) and investigates
claims from the LIAR dataset across easy/medium/hard difficulty tiers.

MANDATORY ENVIRONMENT VARIABLES (injected by the validator):
    API_BASE_URL   The LiteLLM proxy endpoint
    API_KEY        The API key for the proxy
    MODEL_NAME     The model identifier (default: meta-llama/Llama-3.3-70B-Instruct)

OUTPUT FORMAT (required by the validator):
    [START] task=NAME
    [STEP] step=N reward=R
    [END] task=NAME score=S steps=N

All scores are clamped to (0.01, 0.99) per validator requirement — strictly
between 0 and 1, never exactly 0 or 1.

All prints use flush=True. FakeNewsEnvironment() and env.reset() are wrapped
in try/except so a single failure never crashes the script. See
feedback_openenv_validator_patterns.md for the full rules.
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
# Model config — read from env at main() time, not module load time
# =========================================================================
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

MAX_STEPS = 10
TEMPERATURE = 0.0
MAX_TOKENS = 500
MIN_SCORE = 0.01  # validator requires scores strictly > 0
MAX_SCORE = 0.99  # validator requires scores strictly < 1

# =========================================================================
# System prompt — describes the 10-action space
# =========================================================================
SYSTEM_PROMPT = textwrap.dedent("""
You are a professional fact-checking investigator with access to real retrieval
and verification tools. You investigate claims to determine their veracity.
You have a limited investigation budget.

Available actions (respond with VALID JSON only, no markdown, no explanation):

1. Retrieve evidence from a source category (real live retrieval):
   {"action_type": "request_source", "source_id": "<category>"}
   Categories: wikipedia, fact_check_api, government_data, academic_papers,
   news_articles, fact_checks, medical_journals, statistical_reports,
   international_organizations, industry_reports

2. Cross-reference the claim against retrieved evidence (runs real NLI):
   {"action_type": "cross_reference", "source_id": "<category>"}

3. Check publisher credibility (4000+ publishers indexed):
   {"action_type": "check_credibility", "source_id": "<domain_or_name>"}

4. Analyze the claim's associated image (CLIP + pHash):
   {"action_type": "analyze_image"}

5. Full-text search across the evidence corpus (FTS5 + live Wikipedia):
   {"action_type": "search_evidence", "query": "<search_query>"}

6. Resolve a named entity via Wikidata:
   {"action_type": "check_entity", "entity": "<name>"}

7. Temporal analysis — when was the claim made vs contradicted:
   {"action_type": "check_timeline"}

8. Reverse image search against known-misattributed photos:
   {"action_type": "reverse_image_search"}

9. Compute multi-source consensus across retrieved evidence:
   {"action_type": "compute_consensus"}

10. Submit your final verdict (ends episode):
    {"action_type": "submit_verdict", "verdict": "<LABEL>",
     "evidence": ["source1", "source2"],
     "confidence": 0.0-1.0,
     "reasoning": "Your explanation"}
    Labels: TRUE, MOSTLY_TRUE, HALF_TRUE, MOSTLY_FALSE, FALSE, PANTS_ON_FIRE

Strategy:
- Start with search_evidence or request_source to retrieve live evidence
- Use check_credibility to weight sources by reputation
- Use cross_reference to run NLI (entailment/contradiction/neutral)
- For visual claims, use reverse_image_search first, then analyze_image
- Use compute_consensus near the end to aggregate your findings
- Submit a well-reasoned verdict with cited evidence

RESPOND WITH JSON ONLY. NO MARKDOWN. NO EXPLANATION OUTSIDE THE JSON.
""").strip()

ACTION_PATTERN = re.compile(r'\{[^{}]*"action_type"[^{}]*\}', re.DOTALL)


def extract_json_action(text: str) -> dict | None:
    """Extract a JSON action from the model's response.

    Tolerant to markdown fences, leading/trailing whitespace, and the
    occasional preamble. Returns None if no valid JSON found.
    """
    if not text:
        return None
    text = text.strip()
    # Strip code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Regex fallback — find first JSON-like block with action_type
    match = ACTION_PATTERN.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def clamp_score(score) -> float:
    """Clamp any score into the validator-safe (0.01, 0.99) range.

    Validator rejects exact 0.0 or 1.0 scores. Any non-numeric input
    (None, NaN, string) collapses to MIN_SCORE.
    """
    try:
        s = float(score)
    except (TypeError, ValueError):
        return MIN_SCORE
    if s != s:  # NaN check
        return MIN_SCORE
    return max(MIN_SCORE, min(MAX_SCORE, s))


def run_episode(client: OpenAI, env: FakeNewsEnvironment, task: str) -> float:
    """Run a single investigation episode and return its final score.

    Prints [START]/[STEP]/[END] output blocks to stdout with flush=True.
    Every exit path writes [END] before returning. env.reset() and every
    env.step() call are wrapped in try/except so this function can never
    propagate an unhandled exception up to main().
    """
    print(f"[START] task={task}", flush=True)

    # Safely start the episode
    try:
        obs = env.reset(task=task)
    except Exception as exc:
        print(f"  env.reset() failed: {exc}", flush=True)
        print(f"[END] task={task} score={MIN_SCORE:.4f} steps=0", flush=True)
        return MIN_SCORE

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
    final_score = MIN_SCORE

    while not obs.done and step_count < initial_budget + 3:
        # ---- LLM call (must go through the proxy; this is the LLM Criteria Check) ----
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = (completion.choices[0].message.content or "") if completion.choices else ""
        except Exception as exc:
            error_msg = str(exc)[:200]
            # Redact any API key that leaked into the error message
            key = os.environ.get("API_KEY", "")
            if key and key in error_msg:
                error_msg = error_msg.replace(key, "***REDACTED***")
            print(f"  LLM request failed: {error_msg}. Submitting fallback verdict.", flush=True)
            try:
                obs = env.step(InvestigateAction(
                    action_type="submit_verdict",
                    verdict="HALF_TRUE",
                    evidence=[],
                    confidence=0.3,
                    reasoning="Investigation incomplete due to LLM error.",
                ))
            except Exception as e:
                print(f"  fallback env.step failed: {e}", flush=True)
            step_count += 1
            reward = clamp_score(obs.reward if obs and obs.reward is not None else MIN_SCORE)
            print(f"[STEP] step={step_count} reward={reward:.4f}", flush=True)
            break

        action_data = extract_json_action(response_text)
        if action_data is None:
            # Force a verdict submission on parse failure
            try:
                obs = env.step(InvestigateAction(
                    action_type="submit_verdict",
                    verdict="HALF_TRUE",
                    evidence=[],
                    confidence=0.3,
                    reasoning="Unable to parse investigation action.",
                ))
            except Exception as e:
                print(f"  parse-failure env.step failed: {e}", flush=True)
            step_count += 1
            reward = clamp_score(obs.reward if obs and obs.reward is not None else MIN_SCORE)
            print(f"[STEP] step={step_count} reward={reward:.4f}", flush=True)
            break

        # ---- Apply the action ----
        try:
            action = InvestigateAction(**action_data)
            obs = env.step(action)
            step_count += 1
            reward = clamp_score(obs.reward if obs and obs.reward is not None else MIN_SCORE)
            print(f"[STEP] step={step_count} reward={reward:.4f}", flush=True)
        except Exception as exc:
            print(f"  Invalid action: {str(exc)[:150]}. Submitting fallback.", flush=True)
            try:
                obs = env.step(InvestigateAction(
                    action_type="submit_verdict",
                    verdict="HALF_TRUE",
                    evidence=[],
                    confidence=0.3,
                    reasoning=f"Invalid action: {str(exc)[:100]}",
                ))
            except Exception as e:
                print(f"  invalid-action fallback env.step failed: {e}", flush=True)
            step_count += 1
            reward = clamp_score(obs.reward if obs and obs.reward is not None else MIN_SCORE)
            print(f"[STEP] step={step_count} reward={reward:.4f}", flush=True)
            break

        # ---- Build feedback for the next turn ----
        messages.append({"role": "assistant", "content": response_text})

        feedback = f"Result: {obs.message}\n"
        if obs.source_content:
            feedback += f"Source content: {obs.source_content[:500]}\n"
        if obs.cross_ref_result:
            feedback += f"NLI scores: {json.dumps(obs.cross_ref_result)}\n"
        if obs.credibility_score is not None:
            feedback += f"Credibility: {obs.credibility_score:.3f} {obs.credibility_details}\n"
        if obs.entity_info:
            feedback += f"Entity: {obs.entity_info}\n"
        if obs.timeline_info:
            feedback += f"Timeline: {obs.timeline_info}\n"
        if obs.image_match:
            feedback += f"Image match: {obs.image_match}\n"
        if obs.consensus_score is not None:
            feedback += f"Consensus: {obs.consensus_score:.3f}\n"
        feedback += f"Budget remaining: {obs.budget_remaining}\n"

        if obs.budget_remaining <= 1 and not obs.done:
            feedback += "WARNING: Budget almost exhausted. Submit your verdict NOW."

        messages.append({"role": "user", "content": feedback})

    final_score = clamp_score(obs.reward if obs and obs.reward is not None else MIN_SCORE)
    print(f"[END] task={task} score={final_score:.4f} steps={step_count}", flush=True)
    return final_score


def main():
    """Main entry point — reads validator-injected env vars and runs episodes."""
    # Read validator-injected variables (they guarantee these exist during grading)
    api_base_url = os.environ.get("API_BASE_URL", "")
    api_key = os.environ.get("API_KEY", "")

    # Debug dump — helps diagnose env var issues in the validator log
    print("=" * 60, flush=True)
    print("Veritas — Fact-Checking Investigator", flush=True)
    print("=" * 60, flush=True)
    print("Environment variables:", flush=True)
    for k, v in sorted(os.environ.items()):
        ku = k.upper()
        if any(x in ku for x in ("API", "KEY", "TOKEN", "URL", "MODEL", "HF_", "OPENAI")):
            safe_v = f"SET ({len(v)} chars)" if v else "EMPTY"
            print(f"  {k}={safe_v}", flush=True)
    print(flush=True)

    # Fallback for local dev: if validator didn't inject API_KEY, try HF_TOKEN
    if not api_key:
        api_key = os.environ.get("HF_TOKEN", "")
        if not api_base_url:
            api_base_url = "https://router.huggingface.co/v1"

    print(f"Using API_BASE_URL: {api_base_url}", flush=True)
    print(f"Using API_KEY: {'yes (' + str(len(api_key)) + ' chars)' if api_key else 'NO'}", flush=True)
    print(f"Using MODEL_NAME: {MODEL_NAME}", flush=True)
    print(flush=True)

    if not api_key:
        print("WARNING: No API_KEY or HF_TOKEN found. Running heuristic fallback.", flush=True)
        return run_heuristic_fallback()

    # Initialize OpenAI client (points at validator's LiteLLM proxy)
    client = OpenAI(base_url=api_base_url, api_key=api_key)
    print(f"OpenAI client initialized: base_url={api_base_url}", flush=True)

    # Optional: verify the proxy is reachable with a tiny probe call
    print("Testing proxy connection...", flush=True)
    try:
        test = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        reply = (test.choices[0].message.content if test.choices else "") or ""
        print(f"Proxy OK: {reply[:40]}", flush=True)
    except Exception as e:
        print(f"Proxy test failed: {str(e)[:200]}", flush=True)

    # Initialize environment — wrapped in try/except for validator safety
    try:
        env = FakeNewsEnvironment()
    except Exception as exc:
        print(f"FakeNewsEnvironment() failed: {exc}. Running heuristic fallback.", flush=True)
        return run_heuristic_fallback()

    all_results = {}
    episodes_per_task = 5

    for task in ["easy", "medium", "hard"]:
        print(f"\n--- Task: {task} ({episodes_per_task} episodes) ---", flush=True)
        scores = []

        for ep in range(episodes_per_task):
            try:
                score = run_episode(client, env, task)
            except Exception as exc:
                print(f"  Episode {ep+1} failed: {str(exc)[:150]}. Score: {MIN_SCORE}", flush=True)
                score = MIN_SCORE
            scores.append(clamp_score(score))
            print(f"  Episode {ep+1}: score={clamp_score(score):.4f}", flush=True)

        avg = clamp_score(sum(scores) / len(scores) if scores else MIN_SCORE)
        all_results[task] = {
            "average_score": round(avg, 4),
            "min_score": clamp_score(min(scores) if scores else MIN_SCORE),
            "max_score": clamp_score(max(scores) if scores else MIN_SCORE),
            "episodes": len(scores),
            "scores": [round(clamp_score(s), 4) for s in scores],
        }
        print(f"  Average: {avg:.4f}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("INFERENCE RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(json.dumps(all_results, indent=2), flush=True)
    return all_results


def run_heuristic_fallback():
    """Heuristic baseline when no API key is available.

    Still emits [START]/[STEP]/[END] blocks so the validator can parse
    the output even without LLM calls. Uses MIN_SCORE as the floor for
    any episode that fails.
    """
    print("Running heuristic fallback (no LLM calls)", flush=True)
    try:
        env = FakeNewsEnvironment()
    except Exception as exc:
        print(f"Even heuristic env init failed: {exc}", flush=True)
        # Emit a minimal valid output so output parsing still passes
        results = {}
        for task in ["easy", "medium", "hard"]:
            print(f"[START] task={task}", flush=True)
            print(f"[STEP] step=1 reward={MIN_SCORE:.4f}", flush=True)
            print(f"[END] task={task} score={MIN_SCORE:.4f} steps=1", flush=True)
            results[task] = {
                "average_score": MIN_SCORE,
                "min_score": MIN_SCORE,
                "max_score": MIN_SCORE,
                "episodes": 1,
                "scores": [MIN_SCORE],
            }
        print(json.dumps(results, indent=2), flush=True)
        return results

    all_results = {}
    for task in ["easy", "medium", "hard"]:
        scores = []
        for _ in range(5):
            print(f"[START] task={task}", flush=True)
            try:
                obs = env.reset(task=task)
            except Exception as exc:
                print(f"  env.reset() failed: {exc}", flush=True)
                print(f"[END] task={task} score={MIN_SCORE:.4f} steps=0", flush=True)
                scores.append(MIN_SCORE)
                continue

            step_count = 0
            try:
                obs = env.step(InvestigateAction(
                    action_type="request_source", source_id="wikipedia"))
                step_count += 1
                reward = clamp_score(obs.reward if obs.reward is not None else MIN_SCORE)
                print(f"[STEP] step={step_count} reward={reward:.4f}", flush=True)

                evidence_text = (obs.source_content or "").lower()
                has_contradiction = any(w in evidence_text for w in [
                    "false", "debunked", "incorrect", "misleading",
                    "contradicts", "refuted", "inaccurate",
                ])
                has_support = any(w in evidence_text for w in [
                    "confirmed", "accurate", "correct", "verified", "supports",
                ])
                verdict, conf = "HALF_TRUE", 0.4
                if has_contradiction and not has_support:
                    verdict, conf = "FALSE", 0.65
                elif has_support and not has_contradiction:
                    verdict, conf = "TRUE", 0.65

                obs = env.step(InvestigateAction(
                    action_type="submit_verdict",
                    verdict=verdict,
                    evidence=["wikipedia"],
                    confidence=conf,
                    reasoning=f"Heuristic: {'contradiction' if has_contradiction else 'support' if has_support else 'ambiguous'} detected.",
                ))
                step_count += 1
                ep_score = clamp_score(obs.reward if obs.reward is not None else MIN_SCORE)
                print(f"[STEP] step={step_count} reward={ep_score:.4f}", flush=True)
                print(f"[END] task={task} score={ep_score:.4f} steps={step_count}", flush=True)
                scores.append(ep_score)
            except Exception as exc:
                print(f"  heuristic step failed: {exc}", flush=True)
                print(f"[END] task={task} score={MIN_SCORE:.4f} steps={step_count}", flush=True)
                scores.append(MIN_SCORE)

        avg = clamp_score(sum(scores) / len(scores) if scores else MIN_SCORE)
        all_results[task] = {
            "average_score": round(avg, 4),
            "min_score": clamp_score(min(scores) if scores else MIN_SCORE),
            "max_score": clamp_score(max(scores) if scores else MIN_SCORE),
            "episodes": len(scores),
            "scores": [round(clamp_score(s), 4) for s in scores],
        }
        print(f"{task:8s} | avg={avg:.4f}", flush=True)

    print("\n" + json.dumps(all_results, indent=2), flush=True)
    return all_results


if __name__ == "__main__":
    main()
