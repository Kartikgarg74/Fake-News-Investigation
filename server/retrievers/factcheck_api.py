"""Google Fact Check Tools API retriever.

https://developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/claims

Free tier: 10K requests/day (way more than we need). Returns structured
verdicts from PolitiFact, Snopes, AFP Fact Check, FactCheck.org, etc.

Requires GOOGLE_FACTCHECK_API_KEY env var. Without it, the retriever
degrades to an empty response rather than failing — this keeps the
environment bootable in validator containers that don't inject the key.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
TIMEOUT = 5.0


class FactCheckAPIRetriever:
    SOURCE_TYPE = "fact_check_api"
    SOURCE_DOMAIN = "factchecktools.googleapis.com"

    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_FACTCHECK_API_KEY", "")

    def retrieve(self, claim: Dict[str, Any], query: Optional[str] = None) -> Dict[str, Any]:
        """Search the Google Fact Check Tools API for verdicts on this claim."""
        if not self.api_key:
            return self._empty(reason="api_key_missing")

        search_query = query or (claim.get("claim", "") or "")[:200]
        if not search_query:
            return self._empty(reason="empty_query")

        params = {
            "query": search_query,
            "languageCode": "en",
            "pageSize": "5",
        }
        url = f"{API_URL}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(
                url,
                headers={"X-Goog-Api-Key": self.api_key},
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # nosec B310 — fixed Google host
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return self._empty(reason="network_error")

        claims = data.get("claims", [])
        if not claims:
            return self._empty(reason="no_results")

        # Format the top results into a readable summary
        formatted = self._format_results(claims)
        top = claims[0]
        review = (top.get("claimReview") or [{}])[0]

        return {
            "ok": True,
            "content": formatted,
            "source_url": review.get("url", ""),
            "source_domain": self._extract_domain(review.get("publisher", {}).get("site", "")),
            "published_date": (review.get("reviewDate") or "")[:10] or None,
            "num_verdicts": len(claims),
            "top_verdict": review.get("textualRating", ""),
            "top_publisher": review.get("publisher", {}).get("name", ""),
        }

    def _format_results(self, claims: List[Dict[str, Any]]) -> str:
        """Turn a list of Google Fact Check results into a readable passage."""
        lines = []
        for i, c in enumerate(claims[:5], 1):
            text = (c.get("text") or "")[:200]
            reviews = c.get("claimReview") or []
            for r in reviews[:2]:
                publisher = (r.get("publisher") or {}).get("name", "Unknown")
                rating = r.get("textualRating", "No rating")
                review_date = (r.get("reviewDate") or "")[:10]
                lines.append(
                    f"[{publisher}, {review_date}] {rating}: \"{text}\""
                )
        return "\n".join(lines) if lines else "No fact-check reviews found."

    @staticmethod
    def _extract_domain(url_or_domain: str) -> str:
        if not url_or_domain:
            return ""
        s = url_or_domain.lower().strip()
        for p in ("https://", "http://", "www."):
            if s.startswith(p):
                s = s[len(p):]
        return s.split("/")[0]

    def _empty(self, reason: str = "") -> Dict[str, Any]:
        return {
            "ok": False,
            "content": "",
            "source_url": "",
            "source_domain": self.SOURCE_DOMAIN,
            "published_date": None,
            "error": reason,
        }
