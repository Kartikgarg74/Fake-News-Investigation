"""Cloud-hosted ML signal layer.

All heavy models run as API calls to HuggingFace Inference API (or the
validator-provided LiteLLM proxy for text-only models). No local weights
are downloaded — the Docker image stays small and the laptop doesn't
need a GPU.

Modules:
- nli: real NLI via HF Inference API (cross-encoder/nli-deberta-v3-base)
- clip_mm: image-text alignment via CLIP on HF Inference API
- phash: perceptual hashing for reverse image search (pure-python, tiny)
"""

from .nli import NLIClient
from .clip_mm import CLIPClient
from .phash import compute_phash, hamming_distance

__all__ = ["NLIClient", "CLIPClient", "compute_phash", "hamming_distance"]
