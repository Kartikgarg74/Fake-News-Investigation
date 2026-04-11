"""Wikipedia REST API retriever.

Calls en.wikipedia.org/api/rest_v1/page/summary/<title> for a short summary,
and falls back to the search API if direct lookup misses. Results cached in
EvidenceDB keyed by (source_type="wikipedia", query).

Design notes:
- Uses urllib from stdlib, no extra deps. urllib doesn't need retries for
  occasional failures — the cache layer handles that.
- 5-second timeout: the validator container has a real episode budget and
  we shouldn't block a whole episode on one slow retrieval.
- User-Agent is required by Wikipedia's API policy.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

USER_AGENT = "Veritas-FactCheck/1.0 (https://huggingface.co/spaces/Kartikgarg00/fake-news-investigator)"
TIMEOUT = 5.0


class WikipediaRetriever:
    SOURCE_TYPE = "wikipedia"
    SOURCE_DOMAIN = "en.wikipedia.org"

    def retrieve(self, claim: Dict[str, Any], query: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve a Wikipedia summary for a claim.

        Strategy:
        1. Extract likely search terms from the claim (nouns, named entities).
        2. Hit the REST summary endpoint with the best candidate.
        3. If that misses (404), fall back to the search API and retry.

        Never raises. Returns {"ok": False, ...} on any failure.
        """
        search_query = query or self._extract_search_terms(claim.get("claim", ""))
        if not search_query:
            return self._empty()

        # 1. Try direct summary lookup on the top-level term
        summary = self._fetch_summary(search_query)
        if summary and summary.get("ok"):
            return summary

        # 2. Fall back to search API
        top_title = self._search_top_result(search_query)
        if top_title:
            summary = self._fetch_summary(top_title)
            if summary and summary.get("ok"):
                return summary

        return self._empty()

    def _fetch_summary(self, title: str) -> Dict[str, Any]:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # nosec B310 — fixed wikipedia host
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return self._empty()

        # A "disambiguation" result is not useful — skip it
        if data.get("type") == "disambiguation":
            return self._empty()

        extract = data.get("extract", "")
        if not extract:
            return self._empty()

        return {
            "ok": True,
            "content": extract[:2000],  # cap for token budget
            "source_url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "source_domain": self.SOURCE_DOMAIN,
            "published_date": data.get("timestamp", "")[:10] if data.get("timestamp") else None,
            "title": data.get("title", title),
            "wikidata_id": data.get("wikibase_item", ""),
        }

    def _search_top_result(self, query: str) -> Optional[str]:
        """Use the MediaWiki action API to find the top search result title."""
        url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&format=json&list=search&srsearch={urllib.parse.quote(query)}&srlimit=1"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # nosec B310
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

        results = data.get("query", {}).get("search", [])
        if not results:
            return None
        return results[0].get("title")

    @staticmethod
    def _extract_search_terms(claim_text: str) -> str:
        """Pull likely-useful search terms from claim text.

        Strategy: capitalized multi-word phrases (likely named entities) first,
        then fall back to the first N content words. We deliberately don't use
        spaCy or NLTK here — that would add 500MB to the Docker image. The
        heuristic is good enough for Wikipedia search, which is itself fuzzy.
        """
        if not claim_text:
            return ""
        # Pull capitalized word sequences (e.g. "Great Wall", "United Nations")
        cap_phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", claim_text)
        if cap_phrases:
            # Use the longest one — usually the most specific entity
            return max(cap_phrases, key=len)
        # Fallback: first content words, strip punctuation
        words = re.findall(r"\b[a-zA-Z]{4,}\b", claim_text)
        return " ".join(words[:5]) if words else claim_text[:50]

    def _empty(self) -> Dict[str, Any]:
        return {
            "ok": False,
            "content": "",
            "source_url": "",
            "source_domain": self.SOURCE_DOMAIN,
            "published_date": None,
        }
