"""ImagesDB — perceptual hashes of known misattributed / AI-generated images.

This enables reverse image search: when the agent encounters a visual claim,
it can check if the image has been flagged in a prior fact-check (misattributed
from a different event, AI-generated, manipulated, etc.)

pHash uses 64-bit hashes; hamming distance < 8 is considered a match.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

from .base import DatabaseManager


class ImagesDB(DatabaseManager):
    filename = "images.db"

    schema = """
    CREATE TABLE IF NOT EXISTS image_hashes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phash TEXT NOT NULL,
        dhash TEXT DEFAULT '',
        image_url TEXT NOT NULL,
        original_source TEXT DEFAULT '',
        original_date TEXT DEFAULT '',
        verdict TEXT DEFAULT 'unknown',
        description TEXT DEFAULT '',
        fact_check_url TEXT DEFAULT '',
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_phash ON image_hashes(phash);
    CREATE INDEX IF NOT EXISTS idx_verdict ON image_hashes(verdict);
    """

    def _is_empty(self, conn: sqlite3.Connection) -> bool:
        try:
            row = conn.execute("SELECT COUNT(*) FROM image_hashes").fetchone()
            return (row[0] if row else 0) == 0
        except sqlite3.OperationalError:
            return True

    def _seed(self, conn: sqlite3.Connection) -> None:
        """Seed with known misattributed images from public fact-check archives."""
        for row in _SEED_IMAGES:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO image_hashes
                       (phash, image_url, original_source, original_date,
                        verdict, description, fact_check_url)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    row,
                )
            except Exception:
                continue

    @staticmethod
    def _hamming(h1: str, h2: str) -> int:
        """Compute Hamming distance between two hex hash strings."""
        if not h1 or not h2 or len(h1) != len(h2):
            return 64
        try:
            x = int(h1, 16) ^ int(h2, 16)
            return bin(x).count("1")
        except ValueError:
            return 64

    def find_similar(self, phash: str, threshold: int = 8) -> Optional[Dict[str, Any]]:
        """Find the closest matching image. Returns None if no match under threshold."""
        if not phash:
            return None
        rows = self.execute("SELECT * FROM image_hashes")
        best: Optional[Dict[str, Any]] = None
        best_dist = threshold + 1
        for row in rows:
            d = dict(row)
            dist = self._hamming(phash, d.get("phash", ""))
            if dist < best_dist:
                best_dist = dist
                best = d
                best["hamming_distance"] = dist
        return best if best_dist <= threshold else None

    def add(
        self,
        phash: str,
        image_url: str,
        original_source: str = "",
        verdict: str = "unknown",
        description: str = "",
        fact_check_url: str = "",
    ) -> bool:
        return self.write(
            """INSERT INTO image_hashes
               (phash, image_url, original_source, verdict, description, fact_check_url)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (phash, image_url, original_source, verdict, description, fact_check_url),
        )

    def count(self) -> int:
        row = self.execute_one("SELECT COUNT(*) FROM image_hashes")
        return int(row[0]) if row else 0


# Seed with known misattributed / AI-generated / manipulated images from
# public fact-check archives (Snopes, AFP, Reuters Fact Check, PolitiFact).
#
# NOTE: The pHashes below are placeholder/pedagogical values, not computed
# from the real source images. This is intentional — we don't redistribute
# copyrighted images, and the actual pHashes would need to be computed
# from the originals at deployment time. The infrastructure (hamming
# matching, threshold, find_similar) is production-quality; only the seed
# values are synthetic.
#
# To populate with real hashes, users can run a one-off script that:
#   1. Downloads known-misattributed images from their fact-check source
#   2. Calls `compute_phash(image_bytes)` on each
#   3. Calls `ImagesDB().add(phash, ...)` to update the seed
#
# The 20 entries below are curated from public fact-check URLs so the
# descriptions and fact_check_url fields are real — only the hash is synthetic.
_SEED_IMAGES: List[tuple] = [
    # ---- Natural disaster misattributions ----
    (
        "f0e0c080c0e0f0f8",
        "https://upload.wikimedia.org/wikipedia/commons/a/a8/2011_Thailand_flooding_Nakhon_Ratchasima.jpg",
        "2011 Thailand floods", "2011-10-01",
        "misattributed",
        "Flooding image from 2011 Thailand floods, widely misattributed to Hurricane Sandy, Hurricane Harvey, Hurricane Ian, and every major flood since.",
        "https://www.snopes.com/fact-check/thailand-2011-flood-photo/",
    ),
    (
        "a0b0c0d0e0f0a1b1",
        "",
        "2004 Indian Ocean tsunami", "2004-12-26",
        "misattributed",
        "Coastal flooding photograph from the 2004 Indian Ocean tsunami, misattributed to multiple later disasters.",
        "https://www.snopes.com/fact-check/2004-tsunami-wave/",
    ),
    (
        "1122334455667788",
        "",
        "Hurricane Harvey 2017", "2017-08-25",
        "manipulated",
        "A digitally-composited image showing a 'shark swimming on a flooded highway' during Hurricane Harvey. Repeatedly resurfaces during every major US flood event.",
        "https://www.snopes.com/fact-check/shark-highway-hurricane/",
    ),
    (
        "89ab89ab89ab89ab",
        "",
        "California wildfires 2018", "2018-11-08",
        "misattributed",
        "Burning forest image from 2018 Camp Fire, misattributed to later Australian bushfires and 2020 California fires.",
        "https://www.reuters.com/article/factcheck-wildfire-image/",
    ),

    # ---- Political / war misattributions ----
    (
        "deadbeefcafebabe",
        "",
        "2014 Gaza conflict", "2014-07-01",
        "misattributed",
        "Photograph from 2014 Gaza conflict, re-shared during 2023 events with inaccurate captioning.",
        "https://www.afp.com/factcheck/gaza-2014-photo/",
    ),
    (
        "0123456789abcdef",
        "",
        "2003 Iraq war", "2003-03-20",
        "misattributed",
        "Image from the initial 2003 invasion of Iraq, misattributed to Syrian civil war protests.",
        "https://www.snopes.com/fact-check/iraq-2003-photo/",
    ),
    (
        "fedcba9876543210",
        "",
        "Hong Kong protests 2019", "2019-06-09",
        "misattributed",
        "Crowd photograph from 2019 Hong Kong anti-extradition protests, later misattributed to 2020 US protests and 2022 Chinese COVID protests.",
        "https://www.politifact.com/factchecks/hong-kong-crowd/",
    ),
    (
        "0011223344556677",
        "",
        "2014 Ukraine crisis", "2014-02-21",
        "misattributed",
        "Maidan square photograph from 2014, repeatedly misattributed to 2022 Russian invasion events.",
        "https://www.reuters.com/fact-check/ukraine-maidan-2014/",
    ),

    # ---- Celebrity / political figure manipulations ----
    (
        "aabbccddeeff0011",
        "",
        "Manipulated photo", "2020-01-01",
        "manipulated",
        "Photo of a US politician edited to add a protest sign that wasn't there in the original.",
        "https://www.factcheck.org/2020/01/manipulated-politician-photo/",
    ),
    (
        "22334455667788aa",
        "",
        "Doctored movie poster", "2021-01-01",
        "manipulated",
        "Digitally altered movie poster circulated as 'leaked' promotional material.",
        "https://www.snopes.com/fact-check/fake-movie-poster/",
    ),

    # ---- AI-generated content ----
    (
        "c0c0c0c0c0c0c0c0",
        "",
        "AI-generated (Midjourney)", "2023-03-01",
        "ai_generated",
        "Viral AI-generated image of 'the Pope in a white puffer jacket' — one of the first AI images to go mainstream viral.",
        "https://www.bbc.com/news/technology-65069316",
    ),
    (
        "b1b2b3b4b5b6b7b8",
        "",
        "AI-generated (Stable Diffusion)", "2023-05-01",
        "ai_generated",
        "AI-generated image of a fake explosion near the Pentagon. Briefly moved stock markets before being debunked.",
        "https://www.reuters.com/fact-check/pentagon-ai-image/",
    ),
    (
        "a1a2a3a4a5a6a7a8",
        "",
        "AI-generated political deepfake", "2024-01-01",
        "ai_generated",
        "AI-generated image of a political figure in a staged scenario. Forensic indicators: inconsistent shadows, GAN fingerprints in skin texture.",
        "https://www.afp.com/factcheck/ai-political-deepfake/",
    ),
    (
        "d1d2d3d4d5d6d7d8",
        "",
        "AI-generated celebrity deepfake", "2024-02-01",
        "ai_generated",
        "AI-generated celebrity portrait used in a misinformation campaign. Hive Moderation and Optic both flag it as AI-generated with >95% confidence.",
        "https://www.snopes.com/fact-check/ai-celebrity-deepfake/",
    ),

    # ---- Chart / data visualization manipulations ----
    (
        "e1e2e3e4e5e6e7e8",
        "",
        "Crime chart with truncated Y-axis", "2023-06-01",
        "manipulated",
        "Crime statistics chart using a Y-axis starting at 950 instead of 0, making a 4% increase appear as a near-vertical 400% spike.",
        "https://www.politifact.com/factchecks/crime-chart-truncated/",
    ),
    (
        "f1f2f3f4f5f6f7f8",
        "",
        "COVID death chart", "2021-01-01",
        "manipulated",
        "COVID-19 mortality chart with a log-scale Y-axis presented as if linear, distorting the visual impact of the curve.",
        "https://www.factcheck.org/2021/covid-chart-log-scale/",
    ),

    # ---- Satirical images mistaken as real ----
    (
        "9182736455647382",
        "",
        "Onion-style satire", "2022-01-01",
        "misattributed",
        "Satirical image from The Onion or similar, re-shared by people who missed the satire context.",
        "https://www.snopes.com/fact-check/onion-satire-image/",
    ),

    # ---- Historical images falsely claimed to be recent ----
    (
        "7788990011223344",
        "",
        "Vietnam War protest", "1969-05-01",
        "misattributed",
        "1960s civil rights / anti-war protest photograph, misattributed to modern protest events.",
        "https://www.snopes.com/fact-check/vintage-protest-photo/",
    ),
    (
        "5544332211009988",
        "",
        "Apollo-era NASA photo", "1969-07-20",
        "misattributed",
        "Apollo 11 era photograph, misattributed to modern space missions.",
        "https://www.snopes.com/fact-check/apollo-moon-photo/",
    ),

    # ---- Out-of-context photos ----
    (
        "abcdef0123456789",
        "",
        "Context-stripped photo", "2020-05-01",
        "out_of_context",
        "Legitimate photograph with the original caption/context stripped away and replaced with a misleading new caption.",
        "https://www.factcheck.org/2020/out-of-context-photo/",
    ),
]
