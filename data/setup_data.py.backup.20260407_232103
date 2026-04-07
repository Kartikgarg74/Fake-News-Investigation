"""
Download and prepare datasets for the Fake News Investigator environment.

Datasets:
  - LIAR (12.8K claims from PolitiFact) — HuggingFace: ucsbnlp/liar
  - MBFC credibility ratings (built-in, no download needed)

Usage:
    python setup_data.py [--use-huggingface]

Without --use-huggingface, uses the built-in sample claims (11 claims).
With --use-huggingface, downloads the full LIAR dataset (12.8K claims).
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent

DIFFICULTY_MAP = {
    "true": "easy",
    "false": "easy",
    "mostly-true": "medium",
    "barely-true": "medium",
    "half-true": "hard",
    "pants-fire": "hard",
}


def setup_from_huggingface():
    """Download LIAR dataset from HuggingFace and load into SQLite."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets library: pip install datasets")
        print("Falling back to built-in sample claims.")
        return False

    print("Downloading LIAR dataset from HuggingFace...")
    try:
        ds = load_dataset("ucsbnlp/liar", revision="main")  # nosec B615 — trust_remote_code removed; SHA pinning impractical for this dataset
    except Exception:
        # Fallback: try loading without scripts
        try:
            ds = load_dataset("ucsbnlp/liar", revision="main")  # nosec B615
        except Exception:
            # Final fallback: download TSV files directly
            print("HuggingFace loader failed. Downloading TSV files directly...")
            return _download_liar_tsv()

    db_path = DATA_DIR / "claims.db"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE claims (
            id TEXT PRIMARY KEY,
            claim TEXT NOT NULL,
            label TEXT NOT NULL,
            speaker TEXT DEFAULT '',
            topic TEXT DEFAULT '',
            difficulty TEXT NOT NULL,
            gold_evidence TEXT DEFAULT '[]',
            gold_reasoning TEXT DEFAULT '',
            evidence_passages TEXT DEFAULT '{}',
            image_url TEXT DEFAULT NULL
        )
    """)

    count = 0
    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        for i, row in enumerate(ds[split]):
            label = row.get("label", "")
            # LIAR uses integer labels: 0-5
            label_names = [
                "pants-fire", "false", "barely-true",
                "half-true", "mostly-true", "true",
            ]
            if isinstance(label, int) and 0 <= label <= 5:
                label_str = label_names[label]
            else:
                label_str = str(label).lower().replace(" ", "-")

            difficulty = DIFFICULTY_MAP.get(label_str, "medium")
            claim_text = row.get("statement", row.get("claim", ""))
            speaker = row.get("speaker", "")
            topic = row.get("subject", row.get("topic", ""))

            if not claim_text:
                continue

            claim_id = f"{split}_{i:05d}"

            # Generate basic evidence passages based on the label
            evidence_passages = _generate_evidence_template(label_str, claim_text)

            cur.execute(
                """INSERT OR IGNORE INTO claims
                   (id, claim, label, speaker, topic, difficulty,
                    gold_evidence, gold_reasoning, evidence_passages)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    claim_id,
                    claim_text,
                    label_str,
                    speaker,
                    topic if isinstance(topic, str) else str(topic),
                    difficulty,
                    json.dumps(["fact_checks", "government_data"]),
                    f"This claim has been rated as {label_str} by PolitiFact.",
                    json.dumps(evidence_passages),
                ),
            )
            count += 1

    conn.commit()
    conn.close()
    print(f"Loaded {count} claims into {db_path}")

    # Print distribution
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    for diff in ["easy", "medium", "hard"]:
        cur.execute("SELECT COUNT(*) FROM claims WHERE difficulty = ?", (diff,))
        n = cur.fetchone()[0]
        print(f"  {diff}: {n} claims")
    conn.close()

    return True


def _generate_evidence_template(label: str, claim: str) -> dict:
    """Generate templated evidence passages based on claim label.

    These provide basic evidence for the environment to serve.
    In a full system, these would be curated per-claim.
    """
    if label in ("false", "pants-fire"):
        return {
            "fact_checks": (
                f"Fact-checkers have rated this claim as {label}. "
                f"The available evidence contradicts the core assertion. "
                f"Key sources do not support this claim."
            ),
            "government_data": (
                f"Official government data and statistics do not support "
                f"this claim. The evidence points in the opposite direction."
            ),
            "academic_papers": (
                f"Academic research does not support this claim. "
                f"Peer-reviewed studies contradict the main assertion."
            ),
        }
    elif label in ("barely-true",):
        return {
            "fact_checks": (
                f"Fact-checkers have rated this claim as barely true. "
                f"While there may be a kernel of truth, the claim is "
                f"largely exaggerated or misleading."
            ),
            "government_data": (
                f"Government data partially relates to this claim but "
                f"the specific numbers or conclusions are significantly "
                f"different from what is stated."
            ),
            "news_articles": (
                f"News coverage of this topic shows a more nuanced picture "
                f"than what the claim suggests. Key details are omitted."
            ),
        }
    elif label == "half-true":
        return {
            "fact_checks": (
                f"Fact-checkers have rated this claim as half true. "
                f"The claim contains elements of truth but omits important "
                f"context or uses misleading framing."
            ),
            "statistical_reports": (
                f"Statistical analysis shows the claim uses selective data. "
                f"The full picture is more complex than presented."
            ),
            "government_data": (
                f"Government data partially supports the claim but the "
                f"interpretation or framing is misleading."
            ),
        }
    elif label == "mostly-true":
        return {
            "fact_checks": (
                f"Fact-checkers have rated this claim as mostly true. "
                f"The core assertion is accurate but needs minor clarification."
            ),
            "government_data": (
                f"Government data largely supports this claim. "
                f"Minor details may be slightly off."
            ),
        }
    elif label == "true":
        return {
            "fact_checks": (
                f"Fact-checkers have confirmed this claim as true. "
                f"The assertion is accurate and well-supported by evidence."
            ),
            "government_data": (
                f"Official data and statistics confirm this claim. "
                f"The evidence strongly supports the assertion."
            ),
            "academic_papers": (
                f"Academic research supports this claim. "
                f"Peer-reviewed studies confirm the main assertion."
            ),
        }
    else:
        return {
            "fact_checks": f"This claim has been evaluated as {label}.",
        }


def _download_liar_tsv() -> bool:
    """Download LIAR dataset as TSV files directly from the source."""
    import csv
    import urllib.request
    import io
    import zipfile

    url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
    print(f"Downloading from {url}...")

    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Only HTTP/HTTPS URLs are allowed, got: {url}")

    try:
        response = urllib.request.urlopen(url, timeout=60)  # nosec B310 — URL validated above
        zip_data = io.BytesIO(response.read())
    except Exception as e:
        print(f"Download failed: {e}")
        return False

    db_path = DATA_DIR / "claims.db"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE claims (
            id TEXT PRIMARY KEY,
            claim TEXT NOT NULL,
            label TEXT NOT NULL,
            speaker TEXT DEFAULT '',
            topic TEXT DEFAULT '',
            difficulty TEXT NOT NULL,
            gold_evidence TEXT DEFAULT '[]',
            gold_reasoning TEXT DEFAULT '',
            evidence_passages TEXT DEFAULT '{}',
            image_url TEXT DEFAULT NULL
        )
    """)

    count = 0
    with zipfile.ZipFile(zip_data) as zf:
        for filename in zf.namelist():
            if not filename.endswith(".tsv"):
                continue
            split_name = filename.replace(".tsv", "").split("/")[-1]
            with zf.open(filename) as f:
                reader = csv.reader(
                    io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t"
                )
                for i, row in enumerate(reader):
                    if len(row) < 3:
                        continue
                    # LIAR TSV format: id, label, statement, subject, speaker, ...
                    claim_id_raw = row[0] if len(row) > 0 else f"{split_name}_{i}"
                    label_str = row[1].lower().strip() if len(row) > 1 else ""
                    claim_text = row[2] if len(row) > 2 else ""
                    topic = row[3] if len(row) > 3 else ""
                    speaker = row[4] if len(row) > 4 else ""

                    if not claim_text or not label_str:
                        continue

                    difficulty = DIFFICULTY_MAP.get(label_str, "medium")
                    evidence_passages = _generate_evidence_template(label_str, claim_text)

                    cur.execute(
                        """INSERT OR IGNORE INTO claims
                           (id, claim, label, speaker, topic, difficulty,
                            gold_evidence, gold_reasoning, evidence_passages)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            f"{split_name}_{i:05d}",
                            claim_text,
                            label_str,
                            speaker,
                            topic,
                            difficulty,
                            json.dumps(["fact_checks", "government_data"]),
                            f"This claim has been rated as {label_str} by PolitiFact.",
                            json.dumps(evidence_passages),
                        ),
                    )
                    count += 1

    conn.commit()
    conn.close()
    print(f"Loaded {count} claims into {db_path}")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    for diff in ["easy", "medium", "hard"]:
        cur.execute("SELECT COUNT(*) FROM claims WHERE difficulty = ?", (diff,))
        n = cur.fetchone()[0]
        print(f"  {diff}: {n} claims")
    conn.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Setup data for Fake News Investigator")
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Download full LIAR dataset from HuggingFace (12.8K claims)",
    )
    args = parser.parse_args()

    if args.use_huggingface:
        success = setup_from_huggingface()
        if not success:
            print("Using built-in sample claims instead.")
    else:
        print("Using built-in sample claims (11 claims).")
        print("Run with --use-huggingface for the full LIAR dataset (12.8K claims).")

    # Verify DB exists
    db_path = DATA_DIR / "claims.db"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM claims")
        total = cur.fetchone()[0]
        conn.close()
        print(f"\nDatabase ready: {db_path} ({total} claims)")
    else:
        print("\nNo database found. Claims will be created on first environment reset.")


if __name__ == "__main__":
    main()
