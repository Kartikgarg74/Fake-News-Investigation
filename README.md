---
title: Fake News Investigator
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Fake News Investigator — OpenEnv Environment

An interactive fact-checking environment where AI agents learn to **investigate** and **verify** factual claims through a multi-step process: requesting evidence, cross-referencing sources, checking credibility, and submitting reasoned verdicts.

## Why This Environment?

Misinformation costs the global economy $78B annually. Current AI fact-checkers are classifiers — they label claims but can't explain their reasoning. This environment trains agents to follow the **investigative process** that professional fact-checkers use (IFCN methodology):

1. Gather evidence from multiple source categories
2. Cross-reference claims against authoritative sources
3. Assess source credibility
4. Form a verdict with evidence and calibrated confidence

Agents trained here learn a **transferable investigation process**, not just memorized labels.

## Quick Start

### Install

```bash
pip install openenv-core
git clone <this-repo>
cd fake_news_investigator
```

### Run Locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Use the Environment

```python
from fake_news_investigator import FakeNewsEnv, InvestigateAction

with FakeNewsEnv(base_url="http://localhost:8000").sync() as env:
    obs = env.reset(task="easy")
    print(f"Claim: {obs.claim}")

    # Investigate
    obs = env.step(InvestigateAction(
        action_type="request_source",
        source_id="fact_checks"
    ))
    print(f"Evidence: {obs.source_content}")

    # Submit verdict
    obs = env.step(InvestigateAction(
        action_type="submit_verdict",
        verdict="FALSE",
        evidence=["fact_checks"],
        confidence=0.85,
        reasoning="Fact-checking sources contradict this claim."
    ))
    print(f"Score: {obs.reward}")
```

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `request_source` | `source_id`: category name | Request evidence from a source category |
| `cross_reference` | `source_id`: category name | Run NLI check of claim against source evidence |
| `check_credibility` | `source_id`: source name/URL | Look up source bias and factual reporting rating |
| `submit_verdict` | `verdict`, `evidence`, `confidence`, `reasoning` | Submit final verdict (ends episode) |

### Source Categories
`government_data`, `academic_papers`, `news_articles`, `fact_checks`, `medical_journals`, `statistical_reports`, `international_organizations`, `industry_reports`

### Verdict Labels
`TRUE`, `MOSTLY_TRUE`, `HALF_TRUE`, `MOSTLY_FALSE`, `FALSE`, `PANTS_ON_FIRE`

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `claim` | string | The claim being investigated |
| `available_sources` | list[str] | Source categories available |
| `source_content` | string or null | Evidence from last `request_source` |
| `cross_ref_result` | dict or null | NLI scores: `{entailment, contradiction, neutral}` |
| `credibility_score` | float or null | Source credibility rating (0.0-1.0) |
| `credibility_details` | dict or null | Bias rating, factual reporting level |
| `budget_remaining` | int | Investigation steps left |
| `steps_taken` | int | Steps used so far |
| `message` | string | Environment feedback |
| `done` | bool | Episode complete? |
| `reward` | float | Score (0.0-1.0, set on episode end) |

## Tasks (Difficulty Tiers)

| Tier | Budget | Claim Types | Target Baseline Score |
|------|--------|-------------|----------------------|
| **easy** | 10 steps | Clear TRUE/FALSE claims with abundant evidence | 0.7 - 1.0 |
| **medium** | 8 steps | Distorted/exaggerated claims needing source analysis | 0.3 - 0.6 |
| **hard** | 6 steps | Sophisticated misinformation with misleading framing | 0.0 - 0.3 |

## Reward Function (Multi-Signal)

| Signal | Weight | Measures |
|--------|--------|----------|
| Verdict accuracy | 30% | Exact match = 1.0, adjacent label = 0.5, wrong = 0.0 |
| Evidence quality | 25% | F1 score of cited sources vs gold-standard evidence |
| Efficiency | 15% | `1.0 - (steps_used / budget)` |
| Confidence calibration | 15% | `1.0 - |confidence - was_correct|` |
| Reasoning quality | 15% | Keyword overlap with gold reasoning |

Penalties: irrelevant source requests (-0.03), verdict without evidence (-0.20), contradicting cited evidence (-0.15).

## Baseline Scores

### Heuristic Baseline (no LLM)

| Task | Avg Score | Method |
|------|-----------|--------|
| easy | 0.74 | Request fact_checks + gov_data, keyword analysis, decide |
| medium | 0.67 | Same heuristic |
| hard | 0.53 | Same heuristic |

### LLM Baseline

The baseline script uses the **OpenAI Python client** (as required by the hackathon) but supports multiple free providers:

```bash
# Using Groq (FREE, runs Llama 3 70B)
OPENAI_API_KEY=gsk_... python baseline.py --method llm --provider groq

# Using Together AI (free tier)
OPENAI_API_KEY=... python baseline.py --method llm --provider together

# Using Ollama (local, no API key needed)
python baseline.py --method llm --provider ollama

# Using OpenAI
OPENAI_API_KEY=sk-... python baseline.py --method llm --provider openai

# Heuristic only (no API needed)
python baseline.py --method heuristic
```

## Deployment

### Option 1: Hugging Face Spaces (recommended)

```bash
pip install huggingface_hub
huggingface-cli login
openenv push --repo-id <username>/fake-news-investigator
```

### Option 2: Docker

```bash
docker build -t fake-news-inv -f Dockerfile .
docker run -p 8000:8000 fake-news-inv
```

### Option 3: Local

```bash
pip install openenv-core
uvicorn fake_news_investigator.server.app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode (`{"task": "easy"}`) |
| `/step` | POST | Execute action |
| `/state` | GET | Episode metadata |
| `/ws` | WebSocket | Full API via WebSocket |
| `/tasks` | GET | List 3 difficulty tiers + action schema |
| `/grader` | GET | Scoring weights and methodology |
| `/baseline` | GET | Run heuristic baseline, return scores |
| `/docs` | GET | Swagger API documentation |

## Data Sources

- **Claims**: LIAR dataset (12.8K claims from PolitiFact, 6 veracity labels)
- **Evidence**: Pre-curated evidence passages per claim mapped to source categories
- **Credibility**: 30+ source ratings (bias, factual reporting, credibility score)

## Setup Full Dataset

```bash
pip install datasets
python data/setup_data.py --use-huggingface
```

This downloads the full LIAR dataset (12.8K claims) from HuggingFace. Without this flag, the environment uses 11 built-in sample claims.

## License

BSD 3-Clause License
