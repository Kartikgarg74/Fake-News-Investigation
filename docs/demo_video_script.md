# Veritas — Demo Video Script

A 2-minute screen recording script for the hackathon submission video. Optimized for judge attention: problem statement up front, live demo in the middle, technical depth at the end.

**Target runtime:** 2:00
**Format:** 1080p screen recording with voiceover
**Tools:** QuickTime (Mac) or OBS, plus a mic. No editing required if you can read the script in one take.

---

## Scene 0 — Pre-recording checklist (do this BEFORE you hit record)

1. Kill every app you don't need. Close Slack, Discord, email.
2. Set your terminal to a clean state:
   ```bash
   cd ~/meta-pytorch-hackathon/fake_news_investigator
   clear
   ```
3. Set `API_KEY` and `API_BASE_URL` env vars to valid values:
   ```bash
   export API_BASE_URL="https://router.huggingface.co/v1"
   export API_KEY="hf_your_token_here"
   export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
   ```
4. Start the server in one terminal:
   ```bash
   python -m uvicorn fake_news_investigator.server.app:app --host 127.0.0.1 --port 8000
   ```
5. Open a browser to `http://localhost:8000/demo` — verify it loads.
6. Open a second terminal pre-navigated to the project root — you'll use this to run the benchmark.
7. Open this script in a third window so you can read from it.

**Test shot:** Record 5 seconds, play it back, confirm audio is clear.

---

## Scene 1 — Problem statement (0:00 - 0:15, 15s)

**ON SCREEN:** Open the README.md file in your editor and scroll to the top so the title and tagline are visible.

**SAY:**
> "Misinformation costs the global economy 78 billion dollars a year. Current AI fact-checkers are black boxes — they output a label without showing their work. Veritas trains AI agents on the same investigative process that professional fact-checkers use, with real retrieval, real NLI, and real image forensics. This is a Meta PyTorch hackathon submission. Let me show you how it works."

---

## Scene 2 — Live demo dashboard (0:15 - 0:55, 40s)

**ON SCREEN:** Switch to the browser tab showing `http://localhost:8000/demo`. The Veritas UI is visible.

**SAY:**
> "This is a live fact-checking dashboard. I'll paste in a claim — let's try a classic one."

**ACTION:** Click into the claim textarea. Clear the default. Type or paste:
> `The Great Wall of China is visible from space with the naked eye.`

**SAY:**
> "I'll set difficulty to easy and click Investigate."

**ACTION:** Select "easy" from the dropdown, click the Investigate button.

**SAY:** (while the stream is running)
> "The agent is now running a real investigation. Step one — it's searching the evidence corpus and hitting Wikipedia REST API live. Step two — real NLI cross-reference on the retrieved text. Step three — publisher credibility lookup. Step four — cross-reference against another source. Finally, it computes a multi-source consensus score and submits a verdict with cited evidence."

**ON SCREEN:** The SSE stream populates 5 steps then a green "FINAL VERDICT" box appears.

**SAY:**
> "The final verdict is FALSE. It reached this conclusion by reading the actual Wikipedia article, running real NLI on it, and computing a consensus score — not by looking up the answer in a lookup table."

---

## Scene 3 — Show the chain of custody (0:55 - 1:15, 20s)

**ON SCREEN:** In the address bar, navigate to `http://localhost:8000/trajectories?limit=5`. The browser shows a JSON response.

**SAY:**
> "Every step the agent takes is logged to an RL trajectory database. Every evidence retrieval writes an audit row with the source URL, the exact content hash, and a timestamp. This means you can prove exactly which Wikipedia revision the agent used to reach every verdict. That's chain of custody — the thing you need to actually deploy a fact-checker in production."

**ACTION:** Scroll through the JSON briefly to show episode IDs, reward values, and action types.

---

## Scene 4 — Architecture and scale (1:15 - 1:40, 25s)

**ON SCREEN:** Switch back to your editor. Open `README.md` at the architecture diagram. Scroll slowly.

**SAY:**
> "Underneath, Veritas runs seven segregated databases — one for claims, one for cached evidence with FTS5 full-text search, one for 162 publisher credibility ratings, one for image perceptual hashes, one for claim timelines, one for entity knowledge from Wikidata, and one for RL trajectories and audit logs. Each database has a single responsibility. The heavy ML — DeBERTa for NLI, CLIP for image-text alignment — all runs in the cloud via HuggingFace Inference API. The Docker image stays small and the whole thing runs on a laptop."

---

## Scene 5 — Benchmark + close (1:40 - 2:00, 20s)

**ON SCREEN:** Switch to your second terminal. Run the benchmark:
```bash
python scripts/benchmark.py --episodes 3
```

**SAY:** (while it runs)
> "Here's the benchmark CLI running a real investigation on easy, medium, and hard claims from the LIAR dataset. Each episode hits Wikipedia live, runs real NLI, and outputs a markdown table with per-task scores and stdev."

**ON SCREEN:** The markdown table appears.

**SAY:**
> "Ten actions, seven databases, real retrieval, real NLI, real image forensics, 93 passing unit tests, and a PPO training recipe that exports the full trajectory database for offline RL. This isn't a simulation — it's a production-grade fact-checking environment you can train an agent on today. Thanks for watching."

**ON SCREEN:** Let the terminal output sit for 2 seconds, then stop recording.

---

## Post-production checklist

- [ ] Trim any silence at the start/end
- [ ] Verify audio levels (voice should be consistent, no pops)
- [ ] Optional: add a 1-second title card with "Veritas — Kartik Garg — Meta PyTorch Hackathon 2026"
- [ ] Export as MP4, H.264, 1080p, under 20 MB if possible
- [ ] Upload to YouTube unlisted OR Vimeo OR Loom — whichever the hackathon accepts
- [ ] Drop the URL into the submission form

## If something goes wrong during recording

- **LLM call fails during Scene 2**: No problem — the fallback heuristic in `/demo/stream` still produces output. Just say "the LLM proxy is having a moment; the heuristic fallback is taking over."
- **Server crashes**: Ctrl+C, restart uvicorn, retake Scene 2.
- **You stumble on a word**: Keep going — judges won't care. Only retake if you said something factually wrong.

## Contingency: if `/demo` doesn't render due to a CSP / iframe issue on HF Spaces

Record Scene 2 against `http://localhost:8000/demo` directly (from your local uvicorn) rather than the deployed HF Space URL. The behavior is identical.

---

**Good luck. Ship it.**
