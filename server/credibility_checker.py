"""Source credibility checker using built-in ratings database."""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional

DATA_DIR = Path(__file__).parent.parent / "data"


# Built-in credibility ratings (subset of MBFC-style data)
DEFAULT_RATINGS = {
    "reuters.com": {"bias": "Center", "factual": "Very High", "score": 0.95},
    "apnews.com": {"bias": "Center", "factual": "Very High", "score": 0.95},
    "bbc.com": {"bias": "Center-Left", "factual": "High", "score": 0.90},
    "nytimes.com": {"bias": "Center-Left", "factual": "High", "score": 0.88},
    "washingtonpost.com": {"bias": "Center-Left", "factual": "High", "score": 0.87},
    "wsj.com": {"bias": "Center-Right", "factual": "High", "score": 0.88},
    "nature.com": {"bias": "Center", "factual": "Very High", "score": 0.97},
    "thelancet.com": {"bias": "Center", "factual": "Very High", "score": 0.96},
    "nejm.org": {"bias": "Center", "factual": "Very High", "score": 0.97},
    "who.int": {"bias": "Center", "factual": "High", "score": 0.90},
    "cdc.gov": {"bias": "Center", "factual": "Very High", "score": 0.93},
    "nih.gov": {"bias": "Center", "factual": "Very High", "score": 0.94},
    "nasa.gov": {"bias": "Center", "factual": "Very High", "score": 0.96},
    "scientificamerican.com": {"bias": "Center-Left", "factual": "High", "score": 0.88},
    "snopes.com": {"bias": "Center", "factual": "Very High", "score": 0.92},
    "politifact.com": {"bias": "Center", "factual": "High", "score": 0.90},
    "factcheck.org": {"bias": "Center", "factual": "Very High", "score": 0.92},
    "foxnews.com": {"bias": "Right", "factual": "Mixed", "score": 0.55},
    "breitbart.com": {"bias": "Far Right", "factual": "Low", "score": 0.20},
    "infowars.com": {"bias": "Far Right", "factual": "Very Low", "score": 0.05},
    "dailymail.co.uk": {"bias": "Right", "factual": "Low", "score": 0.35},
    "buzzfeed.com": {"bias": "Left", "factual": "Mixed", "score": 0.50},
    "healthnewsdaily.com": {"bias": "Center-Right", "factual": "Mixed", "score": 0.45},
    "naturalnews.com": {"bias": "Far Right", "factual": "Very Low", "score": 0.08},
    "iea.org": {"bias": "Center", "factual": "High", "score": 0.88},
    "worldbank.org": {"bias": "Center", "factual": "High", "score": 0.90},
    "imf.org": {"bias": "Center", "factual": "High", "score": 0.89},
    "fbi.gov": {"bias": "Center", "factual": "Very High", "score": 0.93},
    "bls.gov": {"bias": "Center", "factual": "Very High", "score": 0.95},
    "irena.org": {"bias": "Center", "factual": "High", "score": 0.88},
}


class CredibilityChecker:
    """Checks source credibility ratings."""

    def __init__(self):
        self.ratings = DEFAULT_RATINGS.copy()

    def check(self, source_id: str) -> Dict:
        """Look up credibility for a source.

        Args:
            source_id: Source name, URL, or identifier.

        Returns:
            Dict with bias, factual rating, and credibility score.
        """
        # Normalize source_id
        source_key = source_id.lower().strip()
        source_key = source_key.replace("https://", "").replace("http://", "")
        source_key = source_key.replace("www.", "")
        source_key = source_key.rstrip("/")

        # Direct match
        if source_key in self.ratings:
            rating = self.ratings[source_key]
            return {
                "source": source_id,
                "bias": rating["bias"],
                "factual_reporting": rating["factual"],
                "credibility_score": rating["score"],
                "found": True,
            }

        # Partial match (check if any key is a substring)
        for key, rating in self.ratings.items():
            if key in source_key or source_key in key:
                return {
                    "source": source_id,
                    "bias": rating["bias"],
                    "factual_reporting": rating["factual"],
                    "credibility_score": rating["score"],
                    "found": True,
                }

        # Unknown source
        return {
            "source": source_id,
            "bias": "Unknown",
            "factual_reporting": "Unknown",
            "credibility_score": 0.5,
            "found": False,
        }
