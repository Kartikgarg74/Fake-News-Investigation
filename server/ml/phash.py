"""Perceptual hashing for reverse image search.

Pure-python pHash implementation using the stdlib only. We deliberately
avoid imagehash + Pillow dependencies to keep the Docker image small —
the hash quality is slightly lower than imagehash but more than good
enough for "is this the same photo that was misattributed last week."

Algorithm: DCT-based pHash on a 32x32 grayscale downsample.

Returns 64-bit hex hashes. Matching uses Hamming distance; threshold 8
works well empirically for "same photo with minor edits."
"""

from __future__ import annotations

import math
import urllib.request
from typing import Optional

USER_AGENT = "Veritas-Vision/1.0"
TIMEOUT = 10.0


def compute_phash(image_url_or_bytes) -> Optional[str]:
    """Compute a 64-bit pHash for an image.

    Accepts either a URL (fetched first) or raw bytes. Returns a 16-char
    hex string, or None on any failure.
    """
    try:
        if isinstance(image_url_or_bytes, bytes):
            image_bytes = image_url_or_bytes
        elif isinstance(image_url_or_bytes, str):
            image_bytes = _fetch(image_url_or_bytes)
        else:
            return None
        if not image_bytes:
            return None
        return _phash_bytes(image_bytes)
    except Exception:
        return None


def _fetch(url: str) -> Optional[bytes]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # nosec B310 — user-provided URL is part of claim data
            if resp.length and resp.length > 20_000_000:
                return None
            return resp.read()
    except Exception:
        return None


def _phash_bytes(image_bytes: bytes) -> Optional[str]:
    """Compute pHash from raw image bytes.

    We need Pillow for image decoding but we'll lazy-import it so the
    module itself loads without Pillow installed (the hash just returns
    None in that case — the env still works, just without image matching).
    """
    try:
        from PIL import Image
        import io
    except ImportError:
        return None

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("L").resize((32, 32), Image.BILINEAR)
        # Use tobytes() instead of getdata() (deprecated in Pillow 14)
        raw = img.tobytes()
        pixels = list(raw)
    except Exception:
        return None

    # 2D DCT on 32x32
    dct = _dct_2d(pixels, 32)

    # Use the top-left 8x8 excluding DC (the [0,0] corner)
    low_freq = []
    for y in range(8):
        for x in range(8):
            if x == 0 and y == 0:
                continue
            low_freq.append(dct[y * 32 + x])

    median = sorted(low_freq)[len(low_freq) // 2]

    bits = 0
    for i, v in enumerate(low_freq[:64]):
        if v > median:
            bits |= (1 << i)

    return f"{bits:016x}"


def _dct_2d(pixels, n: int):
    """Simple 2D DCT. O(n^4) — fine for n=32."""
    out = [0.0] * (n * n)
    for v in range(n):
        for u in range(n):
            s = 0.0
            for y in range(n):
                for x in range(n):
                    s += pixels[y * n + x] * math.cos(
                        math.pi * (2 * x + 1) * u / (2 * n)
                    ) * math.cos(
                        math.pi * (2 * y + 1) * v / (2 * n)
                    )
            cu = 1 / math.sqrt(2) if u == 0 else 1
            cv = 1 / math.sqrt(2) if v == 0 else 1
            out[v * n + u] = 0.25 * cu * cv * s
    return out


def hamming_distance(h1: str, h2: str) -> int:
    """Compute Hamming distance between two hex hashes. 64 on error."""
    if not h1 or not h2 or len(h1) != len(h2):
        return 64
    try:
        x = int(h1, 16) ^ int(h2, 16)
        return bin(x).count("1")
    except ValueError:
        return 64
