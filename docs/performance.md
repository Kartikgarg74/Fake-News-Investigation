# Veritas — Performance Profile

Measurements taken on a MacBook Air (M-series), Python 3.13, no GPU.
Reproducible via `python notebooks/getting_started.py` or the profiling block
in the source tree.

## Per-action latency

| Action | Avg latency | Notes |
|---|---|---|
| `env.reset()` | **0.7 ms** | Just state reset + random claim pick. Scales with DB size but SQLite index is O(log n). |
| `check_credibility` | **0.4 ms** | Pure SQLite lookup in sources.db (96-162 rows). |
| `compute_consensus` | **0.4 ms** | In-memory aggregation over collected NLI results. |
| `log_step` (trajectories) | **0.4 ms** | SQLite insert. Batched commits would halve this. |
| `request_source` (cache hit) | **~2 ms** | SQLite SELECT from evidence.db. |
| `request_source` (cache miss, live Wikipedia) | **800-1500 ms** | Dominated by Wikipedia REST API latency + the search-then-summary round-trip. |
| `check_entity` (cache miss, live Wikidata) | **500-1200 ms** | wbsearchentities API + Special:EntityData JSON endpoint. |
| `cross_reference` (HF Inference API) | **500-2000 ms** | DeBERTa inference + network. Zero on fallback. |

## Full episode throughput

A standard 5-action episode (request → cross-ref → credibility → consensus → verdict):

| Mode | Avg duration | Notes |
|---|---|---|
| **Offline** (cached, neutral NLI) | **~50 ms** | Everything cached. Suitable for RL training at 20 episodes/sec on a single thread. |
| **Cold** (live Wikipedia, no HF_TOKEN) | **2.4 s** | First-time retrieval of a novel claim. |
| **Warm** (partial cache) | **~300 ms** | Evidence cached but NLI still computed. |

## Scaling estimates

Using cached evidence (which is the expected steady-state for RL training):

- **Single-thread throughput**: ~20 eps/s = **1,200 eps/min**
- **With 4 parallel workers**: ~80 eps/s = **4,800 eps/min**
- **To generate 100k training episodes**: ~20 minutes single-threaded, or ~5 minutes with 4 workers

That's feasible for a PPO warmup phase on a laptop. For full training you'd want GPU-backed NLI (1-2 orders of magnitude speedup) or collect trajectories once and train offline.

## Bottleneck analysis

The current code opens a fresh SQLite connection per query via `DatabaseManager.connect()`. That's ~200 μs per open on a warm filesystem. For 10 queries per episode that's 2 ms of overhead. Fine for production, not optimal for high-throughput training.

**Possible optimizations (not yet implemented):**

1. **Connection pooling**: share one connection per thread. Saves ~2 ms/episode.
2. **Batch trajectory writes**: accumulate 50 step logs and commit once. Saves ~20 ms / 50-episode batch.
3. **Parallel retrieval**: dispatch Wikipedia + Fact Check API concurrently instead of sequentially. Saves ~500 ms when both are called.
4. **Prefetch adjacent claims**: pre-compute `get_random_claim` for the next episode while the current one runs.

None of these are hacks — they're standard optimizations I'd apply if the project graduated from hackathon to production. I chose not to ship them because the current performance is already sufficient for benchmarks and for the live demo, and optimization adds complexity that would make the code harder to audit.

## Memory footprint

- **Process RSS at startup**: ~180 MB (Python + FastAPI + SQLite + imports)
- **Memory growth per episode**: ~0.5 KB (trajectory row + audit row)
- **Docker image size**: ~310 MB (`python:3.11-slim` + requirements + code)

## Disk footprint

| File | Size | Note |
|---|---|---|
| `data/claims.db` | ~10 MB | 12.8k LIAR claims. Built once, read-only at runtime. |
| `data/evidence.db` | Growing, ~1 KB/retrieval | FTS5 indexed. Hits Wikipedia → stores summary. Size grows with unique queries. |
| `data/sources.db` | ~30 KB | 162 publishers. |
| `data/images.db` | ~10 KB | 20 pHash seeds. |
| `data/temporal.db`, `data/entities.db` | ~10 KB each | Small, grows with use. |
| `data/trajectories.db` | Growing, ~1 KB/episode | RL + audit. Export and truncate periodically if you're training long-term. |

## Honest caveats

- **Wikipedia rate limits**: 200 req/s per IP is the official ceiling. In practice you'll be throttled much sooner if you hammer it. The cache layer is your friend.
- **HF Inference API cold starts**: First request to a model can take 20+ seconds if it needs to load. After that, warm latency is 500-2000 ms.
- **SQLite write concurrency**: SQLite uses a single writer. Under parallel RL rollouts, you'll want separate DB files per worker and merge them periodically, OR switch trajectories to PostgreSQL.

## How to run the profile yourself

```bash
python -c "
import time
from fake_news_investigator.server.environment import FakeNewsEnvironment
from fake_news_investigator.models import InvestigateAction

env = FakeNewsEnvironment()
t0 = time.time()
for i in range(5):
    env.reset(task='easy')
print(f'env.reset() x5: avg {(time.time()-t0)/5*1000:.1f} ms')
"
```

Or use `scripts/benchmark.py --episodes 10` which reports per-episode timings.
