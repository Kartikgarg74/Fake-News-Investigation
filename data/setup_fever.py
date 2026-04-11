"""Load claims from the FEVER dataset into claims.db.

FEVER (Thorne et al. 2018) is the canonical Wikipedia-grounded fact-checking
dataset. It has 185K claims, each labeled SUPPORTS / REFUTES / NOT ENOUGH INFO
and linked to Wikipedia sentences that justify the label. Adding a sample of
FEVER claims demonstrates that the Veritas environment isn't LIAR-specific —
the same action space + evidence pipeline works on any structured claim dataset.

Uses the `datasets` library from HuggingFace (optional dependency). If it's
not installed, the script exits gracefully with a helpful message.

Usage:
    python data/setup_fever.py                 # load 1000 claims
    python data/setup_fever.py --count 5000    # load more
    python data/setup_fever.py --split train   # specific split
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fake_news_investigator.server.databases import ClaimsDB


# FEVER labels map to Veritas difficulty tiers by rough proxy:
# - SUPPORTS  -> easy    (clear positive, well-grounded)
# - REFUTES   -> easy    (clear negative, well-grounded)
# - NOT ENOUGH INFO -> hard (ambiguous, hardest case)
FEVER_LABEL_TO_VERITAS_LABEL = {
    "SUPPORTS": "true",
    "REFUTES": "false",
    "NOT ENOUGH INFO": "half-true",
}

FEVER_LABEL_TO_DIFFICULTY = {
    "SUPPORTS": "easy",
    "REFUTES": "easy",
    "NOT ENOUGH INFO": "hard",
}


def load_fever_dataset(split: str = "train", count: int = 1000) -> List[Dict[str, Any]]:
    """Load the FEVER dataset via HuggingFace datasets library.

    Returns a list of dicts ready for ClaimsDB insertion. Gracefully exits
    with None if datasets lib is unavailable or the download fails.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: The `datasets` library is not installed.")
        print("Install with: pip install datasets")
        print("Then re-run: python data/setup_fever.py")
        return []

    print(f"Loading FEVER dataset (split={split}, count={count})...")
    print("(This may take 30-60 seconds on first run — the dataset is cached afterwards.)")

    try:
        ds = load_dataset("fever", "v1.0", split=split, trust_remote_code=True)  # nosec B615
    except Exception as e:
        # FEVER is a gated/community dataset — may need auth or alternate name
        print(f"Failed to load fever/v1.0: {e}")
        try:
            ds = load_dataset("pminervini/hl-fever", split=split)
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            print("\nTry manually: datasets.load_dataset('fever', 'v1.0')")
            return []

    print(f"Loaded {len(ds)} claims from FEVER, sampling first {count}...")

    out: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        if i >= count:
            break
        label = str(row.get("label", "NOT ENOUGH INFO"))
        claim_text = str(row.get("claim", "")).strip()
        if not claim_text:
            continue
        out.append({
            "id": f"fever_{split}_{i:06d}",
            "claim": claim_text,
            "label": FEVER_LABEL_TO_VERITAS_LABEL.get(label, "half-true"),
            "speaker": "FEVER",
            "topic": "fever",
            "difficulty": FEVER_LABEL_TO_DIFFICULTY.get(label, "hard"),
            "claim_date": None,
            "has_image": 0,
            "image_url": None,
            "gold_evidence": json.dumps(["wikipedia"]),
            "gold_reasoning": f"FEVER label: {label}",
        })
    return out


def insert_claims(claims: List[Dict[str, Any]]) -> int:
    """Bulk-insert FEVER claims into ClaimsDB."""
    if not claims:
        return 0
    db = ClaimsDB()
    count = 0
    with db.connect() as conn:
        for c in claims:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO claims
                       (id, claim, label, speaker, topic, difficulty,
                        claim_date, has_image, image_url,
                        gold_evidence, gold_reasoning)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        c["id"], c["claim"], c["label"], c["speaker"],
                        c["topic"], c["difficulty"], c["claim_date"],
                        c["has_image"], c["image_url"],
                        c["gold_evidence"], c["gold_reasoning"],
                    ),
                )
                count += 1
            except Exception:
                continue
        conn.commit()
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description="Load FEVER claims into Veritas ClaimsDB")
    parser.add_argument("--split", default="train", choices=["train", "labelled_dev", "paper_test"])
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args(argv)

    claims = load_fever_dataset(split=args.split, count=args.count)
    if not claims:
        print("No claims loaded — see messages above for why.")
        return 1

    print(f"\nInserting {len(claims)} claims into ClaimsDB...")
    n = insert_claims(claims)
    print(f"Inserted {n} claims (OR IGNOREd duplicates).")

    db = ClaimsDB()
    print(f"\nClaimsDB now contains:")
    print(f"  total : {db.count()}")
    print(f"  easy  : {db.count('easy')}")
    print(f"  medium: {db.count('medium')}")
    print(f"  hard  : {db.count('hard')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
