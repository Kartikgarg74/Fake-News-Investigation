# HACKATHON.md — Meta PyTorch × Scaler OpenEnv Hackathon Submission

**Project**: Veritas — A Production-Grade Fact-Checking RL Environment
**Submitter**: Kartik Garg (solo)
**Team name**: Null Pointers
**Submission URL**: https://huggingface.co/spaces/Kartikgarg00/fake-news-investigator
**GitHub**: https://github.com/kartikgarg00/meta-pytorch-hackathon (if public at submission time)

This document is the **judge-facing one-pager**. The full README is comprehensive; this is the 60-second version with direct links to the evidence behind every claim.

---

## The elevator pitch

Existing AI fact-checkers are classifiers — they output a label without showing their work. Veritas trains AI agents to follow the same investigative process that IFCN fact-checkers use: gather evidence from real sources, cross-reference with NLI, check publisher credibility, analyze images for manipulation, trace temporal provenance, and submit a verdict with calibrated confidence.

**What's new vs. the v1 submission:** after the first round didn't make it, reviewers called out two specific gaps — a monolithic database and a lack of real data flow. Veritas v2 is a rebuild around exactly those two complaints. This document tracks what got shipped vs. what's aspirational.

---

## What's actually shipped (verifiable claims)

Every claim below has a corresponding file you can grep to verify.

| Claim | Evidence (file / test) |
|---|---|
| **7 segregated databases** with a shared base class | `server/databases/{claims,evidence,sources,images,temporal,entities,trajectories}.py` + `base.py` |
| **Real Wikipedia REST retrieval**, cached in FTS5 | `server/retrievers/wikipedia.py` → `server/databases/evidence.py` (FTS5 schema) |
| **Real Wikidata SPARQL entity resolution** | `server/retrievers/wikidata.py` → `server/databases/entities.py` |
| **Google Fact Check Tools API integration** | `server/retrievers/factcheck_api.py` (degrades gracefully without key) |
| **Real DeBERTa NLI via HF Inference API** + LiteLLM proxy fallback | `server/ml/nli.py`, 2-tier strategy |
| **CLIP image-text alignment via HF Inference** | `server/ml/clip_mm.py` |
| **Pure-Python perceptual hashing** (no imagehash dep) | `server/ml/phash.py` |
| **10 action space** (up from 5 in v1) | `server/environment.py` `_VALID_ACTIONS` tuple |
| **162 publisher credibility ratings** (up from 30) | `data/setup_sources.py` curated list |
| **20 known-misattributed image seeds** in ImagesDB | `server/databases/images.py` `_SEED_IMAGES` |
| **Chain-of-custody audit logging** per retrieval | `server/retrievers/orchestrator.py` → `TrajectoriesDB.log_audit()` |
| **Every step logged to trajectories.db** | `server/environment.py` `_log_trajectory()` |
| **Live demo dashboard at /demo with SSE streaming** | `server/app.py` `_DEMO_HTML` + `/demo/stream` endpoint |
| **PPO training recipe** (offline dataset export + SB3 online mode) | `scripts/train_ppo.py` |
| **Benchmark CLI** with markdown output | `scripts/benchmark.py` |
| **FEVER dataset integration script** | `data/setup_fever.py` |
| **93 passing unit tests**, 2 skipped (opt-in live network) | `tests/` (5 test files) |
| **GitHub Actions CI** running tests + validator sanity checks | `.github/workflows/test.yml` |
| **All 5 OpenEnv validator patterns preserved** in inference.py | `tests/test_inference_format.py` (15 regression tests) |
| **Trained Agent + Learning Curve** — heuristic + logistic regression policy, PNG output | `scripts/train_agent.py`, `outputs/learning_curve.png`, `outputs/agent_stats.json` |
| **Adversarial Curriculum** — LLM-generated harder claim variants, self-play loop | `server/adversarial.py`, `/generate_adversarial` endpoint, `/curriculum` endpoint, `env.reset_adversarial()` |
| **Cross-Lingual Fact-Checking** — 100+ language auto-detection + translation | `server/translation.py`, `/translate` endpoint, `env.reset_multilingual()`, `/demo` language selector |

Run `pytest tests/ -v` to verify all of the above in ~11 seconds.

---

## What's aspirational (in the README but not fully shipped)

Being honest so judges don't feel misled:

- **"Production-credible MBFC dataset"** — I ship 162 curated publishers, not the full ~4000. Real MBFC doesn't have a public API. The `SourcesDB.bulk_load()` + `setup_sources.py --csv` path is ready for a real CSV import.
- **"Real pHashes of misattributed images"** — The 20 entries in `images.db` have real descriptions and fact-check URLs but placeholder pHash values. Computing the real hashes would require redistributing the images, which I didn't want to do. Infrastructure for real-hash ingestion is shipped.
- **"Trained policy"** — The PPO script runs end-to-end and exports a JSONL dataset ready for training. It does NOT ship a pretrained checkpoint. The `--mode ppo` path works but requires `stable-baselines3` + `torch` + `gymnasium` installed.

Everything else in the README is verifiable.

---

## How to verify the submission in under 5 minutes

```bash
# 1. Clone
git clone https://huggingface.co/spaces/Kartikgarg00/fake-news-investigator
cd fake-news-investigator

# 2. Install
pip install -r server/requirements.txt
pip install pytest pytest-asyncio pytest-cov httpx

# 3. Run all 93 unit tests (no network needed)
pytest tests/ -v
# Expected: 93 passed, 2 skipped (opt-in live tests)

# 4. Run the benchmark (hits Wikipedia live)
python scripts/benchmark.py --episodes 2
# Expected: markdown table with per-task avg scores

# 5. Export a PPO training dataset
python scripts/train_ppo.py --episodes 6 --out /tmp/training.jsonl
# Expected: 36+ steps with state/action/reward/advantage

# 6. Start the server and hit /demo
python -m uvicorn fake_news_investigator.server.app:app --port 8000 &
curl -s http://localhost:8000/health
curl -s http://localhost:8000/tasks | head
curl -s "http://localhost:8000/trajectories?limit=3" | head

# 7. Run inference.py end-to-end (needs API_KEY + API_BASE_URL)
export API_BASE_URL="https://router.huggingface.co/v1"
export API_KEY="hf_your_token"
python inference.py
# Expected: [START]/[STEP]/[END] blocks, all scores in (0.01, 0.99)
```

---

## Alignment with judging criteria

Based on the hackathon brief:

| Criterion | How Veritas addresses it |
|---|---|
| **Real-world applicability** | Real retrieval (Wikipedia, Wikidata, Fact Check API), real NLI (DeBERTa), real image forensics (CLIP + pHash). Not a simulation. |
| **Multimedia support** | `analyze_image` + `reverse_image_search` actions with real CLIP alignment and perceptual hash matching against 20 known-misattributed images. |
| **PyTorch integration** | Cloud ML uses HF Inference API (PyTorch models under the hood). PPO training recipe uses `stable-baselines3` which is PyTorch-based. |
| **RL trainability** | Every step logged to `trajectories.db`. `scripts/train_ppo.py` exports JSONL ready for offline PPO / DPO / SFT. `--mode ppo` runs SB3 directly. |
| **Scale** | 162 publishers, 12.8k LIAR claims + FEVER integration, FTS5-indexed evidence corpus, Wikipedia REST with aggressive caching. |
| **Production readiness** | Chain-of-custody audit logs, 93 passing unit tests, GitHub Actions CI, graceful degradation on missing API keys, no local ML weights (Docker image ~300 MB). |

---

## v1 → v2 diff in one table

| | v1 (first submission) | v2 (Veritas) |
|---|---|---|
| Databases | 1 (monolithic) | 7 (segregated) |
| Evidence retrieval | Templated strings per label | Live Wikipedia REST + FTS5 cache |
| NLI | `_simulate_nli` (label-based) | Real DeBERTa via HF Inference API |
| Credibility DB | 30 hardcoded | 162 curated + CSV loader |
| Image analysis | Pre-written text | CLIP + pHash |
| Action space | 5 | 10 |
| RL trainability | None | Full trajectory + audit log + PPO recipe |
| Unit tests | 16 | 93 |
| Demo UI | None | `/demo` with SSE streaming |
| CI | None | GitHub Actions |

---

## Limitations & honest trade-offs

1. **Full DeBERTa + CLIP inference requires `HF_TOKEN`**. Without it, the NLI client degrades to a neutral distribution and the env still runs — scores just won't reflect real image/text alignment. The CI test suite runs entirely without any API tokens.

2. **The seeded pHash values in `images.db` are placeholders**. Real hashes would need to be computed from actual misattributed images at deployment time. The ingestion path (`ImagesDB.add()` + `compute_phash()`) is production-ready.

3. **FEVER dataset integration requires `datasets` library** (not in `requirements.txt` to keep the image slim). Optional — run `pip install datasets && python data/setup_fever.py` to enable.

4. **PPO online training requires `stable-baselines3` + `torch` + `gymnasium`** (not in `requirements.txt` for the same reason). Optional — the offline trajectory export mode works with zero extra dependencies.

5. **Live API calls have a 5-second timeout and no retry**. The cache layer IS the retry mechanism — after the first hit, everything is local.

---

## Credits

- LIAR dataset: Wang 2017 (CIKM)
- FEVER dataset: Thorne et al. 2018 (NAACL)
- Wikipedia, Wikidata: CC BY-SA 3.0
- HuggingFace Inference API for DeBERTa NLI and CLIP
- MBFC-compatible schema (ratings curated from public sources)

Built solo by [Kartik Garg](https://github.com/kartikgarg00). MIT License.
