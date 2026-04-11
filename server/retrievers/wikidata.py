"""Wikidata SPARQL retriever for entity resolution.

Queries the public SPARQL endpoint at query.wikidata.org to resolve entity
names to Wikidata QIDs and pull their canonical description + basic
properties (instance of, country, date of birth, founding date, etc.).

Entity resolution uses the WDQS label search API first (fast, cheap), then
falls back to SPARQL for the full metadata query.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

USER_AGENT = "Veritas-FactCheck/1.0"
SEARCH_URL = "https://www.wikidata.org/w/api.php"
SPARQL_URL = "https://query.wikidata.org/sparql"
TIMEOUT = 5.0


class WikidataRetriever:
    SOURCE_TYPE = "wikidata"
    SOURCE_DOMAIN = "wikidata.org"

    def retrieve(self, entity_name: str) -> Dict[str, Any]:
        """Resolve an entity name and return its Wikidata metadata."""
        if not entity_name or len(entity_name.strip()) < 2:
            return self._empty()

        qid, label, description = self._search_entity(entity_name)
        if not qid:
            return self._empty()

        properties = self._fetch_properties(qid)
        return {
            "ok": True,
            "wikidata_id": qid,
            "name": label or entity_name,
            "description": description or "",
            "type": properties.get("instance_of", "unknown"),
            "properties": properties,
            "source_url": f"https://www.wikidata.org/wiki/{qid}",
            "source_domain": self.SOURCE_DOMAIN,
        }

    def _search_entity(self, name: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Use the wbsearchentities API to find the top-matching QID."""
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "type": "item",
            "search": name,
            "limit": "1",
        }
        url = f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # nosec B310
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None, None, None

        results = data.get("search", [])
        if not results:
            return None, None, None
        top = results[0]
        return top.get("id"), top.get("label"), top.get("description")

    def _fetch_properties(self, qid: str) -> Dict[str, Any]:
        """Fetch a small set of useful properties via the entity data endpoint.

        We use the Special:EntityData JSON endpoint instead of SPARQL here
        because it's cacheable and faster for single-entity lookups.
        """
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # nosec B310
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return {}

        entity = (data.get("entities") or {}).get(qid, {})
        claims = entity.get("claims", {})

        # Extract a handful of useful properties by QID
        # P31 = instance of, P17 = country, P569 = DOB, P571 = inception
        out: Dict[str, Any] = {}
        instance_of = self._get_claim_value(claims, "P31")
        if instance_of:
            out["instance_of"] = instance_of
        country = self._get_claim_value(claims, "P17")
        if country:
            out["country"] = country
        dob = self._get_claim_value(claims, "P569", prop_type="time")
        if dob:
            out["date_of_birth"] = dob
        inception = self._get_claim_value(claims, "P571", prop_type="time")
        if inception:
            out["inception"] = inception
        return out

    @staticmethod
    def _get_claim_value(
        claims: Dict[str, Any], prop: str, prop_type: str = "wikibase-entityid"
    ) -> Optional[str]:
        """Pull the first value of a claim property."""
        entries = claims.get(prop, [])
        if not entries:
            return None
        try:
            datavalue = entries[0].get("mainsnak", {}).get("datavalue", {})
            value = datavalue.get("value", {})
            if prop_type == "time":
                return str(value.get("time", ""))[1:11]  # strip leading + and trailing
            # wikibase-entityid returns a QID; we don't resolve it to label here
            # because that's another API hit. Just return the QID.
            if isinstance(value, dict):
                return value.get("id", str(value))
            return str(value)
        except Exception:
            return None

    def _empty(self) -> Dict[str, Any]:
        return {
            "ok": False,
            "wikidata_id": "",
            "name": "",
            "description": "",
            "type": "unknown",
            "properties": {},
            "source_url": "",
            "source_domain": self.SOURCE_DOMAIN,
        }
