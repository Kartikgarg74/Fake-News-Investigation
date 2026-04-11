# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Getting started with Veritas
#
# This notebook walks you through using the Veritas fact-checking environment end-to-end.
#
# **Prerequisites:**
# - Python 3.11+
# - `pip install -r server/requirements.txt`
# - (Optional) `HF_TOKEN` environment variable for real NLI / CLIP inference
# - (Optional) `API_BASE_URL` + `API_KEY` for LiteLLM proxy fallback
#
# If you don't have either token, the environment still works — NLI degrades to a
# neutral (0.33/0.34/0.33) distribution and CLIP returns `ok=False`. All other
# actions (retrieval, credibility, timeline, consensus, search) work with zero
# tokens because they hit public APIs.
#
# To convert this file to a real `.ipynb`:
# ```bash
# pip install jupytext
# jupytext --to ipynb notebooks/getting_started.py
# ```

# %% [markdown]
# ## 1. Initialize the environment
#
# `FakeNewsEnvironment()` spins up all 7 databases and the retrieval + ML layers.
# First run is slower because the DBs get seeded.

# %%
import sys
sys.path.insert(0, "..")  # so `fake_news_investigator` package resolves

from fake_news_investigator.server.environment import FakeNewsEnvironment
from fake_news_investigator.models import InvestigateAction

env = FakeNewsEnvironment()
print("Environment ready.")
print(f"  Claims in DB: {env.claim_manager.get_claim_count()}")
print(f"  Sources in DB: {env.sources_db.count()}")
print(f"  Images in DB: {env.images_db.count()}")

# %% [markdown]
# ## 2. Reset and pull a claim
#
# `reset(task=...)` selects a random claim from the requested difficulty tier.
# The observation has the claim text, available source categories, and the budget.

# %%
obs = env.reset(task="easy", episode_id="notebook_demo_1")
print("CLAIM:", obs.claim)
print("BUDGET:", obs.budget_remaining)
print("IMAGE:", obs.image_url)
print("SOURCES:", obs.available_sources[:5], "...")

# %% [markdown]
# ## 3. Action 1: `request_source` (real Wikipedia retrieval)
#
# This hits the live Wikipedia REST API and caches the result in `evidence.db`.

# %%
obs = env.step(InvestigateAction(
    action_type="request_source",
    source_id="wikipedia",
))
print("MSG:", obs.message)
print("CONTENT:", (obs.source_content or "")[:300])
print("CACHE HIT:", obs.cache_hit)

# %% [markdown]
# ## 4. Action 2: `cross_reference` (real NLI)
#
# Classifies (claim, evidence) as entailment / contradiction / neutral. With a
# valid `HF_TOKEN`, this runs a real DeBERTa model via HF Inference API.

# %%
obs = env.step(InvestigateAction(
    action_type="cross_reference",
    source_id="wikipedia",
))
print("MSG:", obs.message)
print("NLI:", obs.cross_ref_result)

# %% [markdown]
# ## 5. Action 3: `check_credibility` (SourcesDB lookup)
#
# 162 publishers with bias + factual-reporting ratings. Fuzzy domain matching.

# %%
for source in ["reuters.com", "https://www.bbc.com/news/world", "cdc.gov", "infowars.com"]:
    obs = env.step(InvestigateAction(action_type="check_credibility", source_id=source))
    print(f"{source:35s} -> {obs.credibility_score:.2f}  ({obs.credibility_details})")

# %% [markdown]
# ## 6. Action 4: `check_entity` (Wikidata SPARQL)
#
# Resolves named entities to their Wikidata QID + metadata. Cached in `entities.db`.

# %%
obs = env.step(InvestigateAction(
    action_type="check_entity",
    entity="Barack Obama",
))
print("MSG:", obs.message)
print("ENTITY INFO:", obs.entity_info)

# %% [markdown]
# ## 7. Action 5: `search_evidence` (FTS5 + Wikipedia fallback)
#
# Full-text search across the evidence corpus. On miss, falls through to live
# Wikipedia.

# %%
obs = env.step(InvestigateAction(
    action_type="search_evidence",
    query="climate change mitigation",
))
print("MSG:", obs.message)
print("CONTENT:", (obs.source_content or "")[:400])

# %% [markdown]
# ## 8. Action 6: `compute_consensus` (aggregate NLI + credibility)
#
# Single [0, 1] score summarizing all retrieved evidence. Requires at least one
# `cross_reference` to have been called earlier in the episode.

# %%
obs = env.step(InvestigateAction(action_type="compute_consensus"))
print("MSG:", obs.message)
print("CONSENSUS:", obs.consensus_score)

# %% [markdown]
# ## 9. Submit the verdict
#
# This grades the episode and ends it. Reward is a weighted sum of verdict
# accuracy, evidence quality, efficiency, confidence calibration, and reasoning
# quality. Clamped to `(0.01, 0.99)`.

# %%
obs = env.step(InvestigateAction(
    action_type="submit_verdict",
    verdict="HALF_TRUE",
    evidence=["wikipedia"],
    confidence=0.6,
    reasoning="Based on Wikipedia retrieval and NLI cross-reference.",
))
print("DONE:", obs.done)
print("REWARD:", obs.reward)
print("MSG:", obs.message)

# %% [markdown]
# ## 10. Export trajectories for RL training
#
# Every step of every episode was logged to `trajectories.db`. Export to JSONL
# for PPO / DPO / supervised fine-tuning.

# %%
from fake_news_investigator.server.databases import TrajectoriesDB
import json

traj = TrajectoriesDB()
print(f"Total episodes: {traj.count_episodes()}")
print(f"Total steps: {traj.count_steps()}")

# Export the last 10 steps
recent = traj.export_jsonl(limit=10)
print("\nSample step:")
print(json.dumps(recent[-1] if recent else {}, indent=2))

# %% [markdown]
# ## 11. Run the benchmark programmatically
#
# The benchmark CLI can also be imported as a module.

# %%
# from scripts.benchmark import run_benchmark, format_markdown
# results = run_benchmark(mode="heuristic", episodes_per_task=2)
# print(format_markdown(results))

# %% [markdown]
# ## 12. Inspect the chain-of-custody audit log
#
# Every retrieval wrote a row with the source URL, content hash, and timestamp.

# %%
with traj.connect() as conn:
    rows = conn.execute(
        "SELECT episode_id, source_type, status, content_hash, fetched_at "
        "FROM audit ORDER BY id DESC LIMIT 5"
    ).fetchall()
    for r in rows:
        print(dict(r))

# %% [markdown]
# ## Next steps
#
# - Try a harder task: `env.reset(task="hard")`
# - Train a policy: `python scripts/train_ppo.py`
# - Open the live demo: `python -m uvicorn fake_news_investigator.server.app:app`
#   then visit `http://localhost:8000/demo`
# - Read `HACKATHON.md` for the full feature list and judging criteria alignment.
