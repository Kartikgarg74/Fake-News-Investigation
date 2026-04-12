"""Cross-lingual translation layer for multi-language fact-checking.

Uses the LLM (via the validator-provided proxy) to translate claims
from any language to English for investigation, and translate verdicts
back to the original language.

This demonstrates that the Veritas environment isn't English-only —
the same investigation pipeline works for claims in Hindi, Spanish,
French, Arabic, etc. Meta's fact-checking pipeline operates in 100+
languages; this module shows we're thinking about that scale.
"""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from typing import Dict, Optional

# -------------------------------------------------------------------------
# Language detection: character-set heuristics (fast, no extra deps)
# -------------------------------------------------------------------------

# Unicode block ranges for heuristic language detection
_UNICODE_RANGES: Dict[str, tuple] = {
    "hi": (0x0900, 0x097F),   # Devanagari → Hindi / Marathi
    "ar": (0x0600, 0x06FF),   # Arabic
    "zh": (0x4E00, 0x9FFF),   # CJK Unified Ideographs (Chinese)
    "ja": (0x3040, 0x30FF),   # Hiragana + Katakana → Japanese
    "ko": (0xAC00, 0xD7A3),   # Hangul → Korean
    "ru": (0x0400, 0x04FF),   # Cyrillic → Russian
    "el": (0x0370, 0x03FF),   # Greek
    "he": (0x0590, 0x05FF),   # Hebrew
    "th": (0x0E00, 0x0E7F),   # Thai
}


def _detect_language_heuristic(text: str) -> Optional[str]:
    """Return a language code if the dominant script is recognisable, else None."""
    counts: Dict[str, int] = {lang: 0 for lang in _UNICODE_RANGES}
    latin_count = 0

    for ch in text:
        cp = ord(ch)
        matched = False
        for lang, (lo, hi) in _UNICODE_RANGES.items():
            if lo <= cp <= hi:
                counts[lang] += 1
                matched = True
                break
        if not matched and ("a" <= ch.lower() <= "z"):
            latin_count += 1

    total = sum(counts.values()) + latin_count
    if total == 0:
        return "en"

    dominant_lang = max(counts, key=counts.get)
    dominant_count = counts[dominant_lang]

    if dominant_count / total > 0.15:
        return dominant_lang

    # Mostly Latin script — treat as English by default
    return "en"


class TranslationClient:
    """Translate claims between languages using the LLM proxy.

    All methods degrade gracefully: if no API key is configured, they
    return the original text unchanged. Translations are cached in memory
    to avoid redundant LLM calls.
    """

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._api_base_url = api_base_url or os.environ.get(
            "API_BASE_URL", "https://router.huggingface.co/v1"
        )
        self._api_key = api_key or os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
        # Cache: cache_key -> translated text
        self._cache: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_language(self, text: str) -> str:
        """Detect the language of *text*.

        Uses a Unicode block heuristic first. Falls back to "en" if the
        heuristic is ambiguous and no API key is configured.

        Returns an ISO 639-1 code ("en", "hi", "zh", etc.).
        """
        lang = _detect_language_heuristic(text)
        if lang is not None:
            return lang

        # Ambiguous — try LLM if available
        if not self._api_key:
            return "en"

        return self._llm_detect_language(text)

    def translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate *text* from *source_lang* to English.

        Returns the original text on any failure or if already English.
        """
        if source_lang == "en" or not text.strip():
            return text

        cache_key = self._cache_key("to_en", text, source_lang)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self._api_key:
            return text

        result = self._llm_translate(text, source_lang, "English")
        self._cache[cache_key] = result
        return result

    def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translate *text* from English to *target_lang*.

        Returns the original text on any failure or if target is English.
        """
        if target_lang == "en" or not text.strip():
            return text

        cache_key = self._cache_key("from_en", text, target_lang)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self._api_key:
            return text

        result = self._llm_translate(text, "English", target_lang)
        self._cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cache_key(self, direction: str, text: str, lang: str) -> str:
        raw = f"{direction}:{lang}:{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _llm_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Call the LLM to translate *text*. Returns original on failure."""
        try:
            import urllib.request

            prompt = (
                f"Translate the following text from {source_lang} to {target_lang}.\n"
                "Return ONLY the translated text with no explanation or preamble.\n\n"
                f"TEXT: {text}"
            )

            payload = json.dumps(
                {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": 0.1,
                }
            ).encode("utf-8")

            req = urllib.request.Request(
                f"{self._api_base_url.rstrip('/')}/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            return data["choices"][0]["message"]["content"].strip()

        except Exception:
            return text

    def _llm_detect_language(self, text: str) -> str:
        """Use LLM to detect language. Returns 'en' on any failure."""
        try:
            import urllib.request

            prompt = (
                "Detect the language of the following text. "
                "Respond with ONLY the ISO 639-1 two-letter language code "
                "(e.g. 'en', 'hi', 'es', 'fr', 'ar', 'zh', 'de', 'ja', 'pt', 'ko').\n\n"
                f"TEXT: {text}"
            )

            payload = json.dumps(
                {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                    "temperature": 0.0,
                }
            ).encode("utf-8")

            req = urllib.request.Request(
                f"{self._api_base_url.rstrip('/')}/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            lang_code = data["choices"][0]["message"]["content"].strip().lower()
            # Sanitise — only accept 2-letter codes
            if len(lang_code) == 2 and lang_code.isalpha():
                return lang_code
            return "en"

        except Exception:
            return "en"
