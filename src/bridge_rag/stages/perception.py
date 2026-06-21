"""Stage A — Perception (`fA`): the frozen embedder.

The paper's recipe (Step 1): use a pretrained embedder, freeze it. Training does
not touch `fA`; it only ever consumes `hA`. So this module is inference-only.

Profile-as-data: the model name and dimension come from `profile.embedder_model`
/ `profile.embed_dim`. To keep the *whole pipeline runnable on base deps alone*
(no transformers install), there is a deterministic hashing fallback that
produces stable, normalized `embed_dim` vectors. It is not semantically strong,
but it exercises every downstream interface honestly — which is the entire point
of the dev profile. Install the `models` extra to get the real embedder.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from ..config import ModelProfile


@runtime_checkable
class Embedder(Protocol):
    """The `fA` interface. Frozen by contract — no training hook exposed."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized embeddings, shape (len(texts), dim), float32."""
        ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (mat / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic, dependency-free embedder for the dev/base path.

    Hashes token n-grams into a fixed-width vector (the hashing trick). Stable
    across runs (same text -> same vector), L2-normalized so dot-product equals
    cosine similarity. Real semantics come from the `models` extra; this exists
    so CI and the E2E keystone run with zero model downloads.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vec
        grams = tokens + [
            f"{a}_{b}" for a, b in zip(tokens, tokens[1:])
        ]
        for gram in grams:
            h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self._dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        return vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.stack([self._embed_one(t) for t in texts], axis=0)
        return _l2_normalize(mat)


class SentenceTransformerEmbedder:
    """Real frozen embedder via sentence-transformers (the `models` extra)."""

    def __init__(self, model_name: str, expected_dim: int) -> None:
        from sentence_transformers import SentenceTransformer  # lazy

        self._model = SentenceTransformer(model_name)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        if self._dim != expected_dim:
            raise ValueError(
                f"embedder dim {self._dim} != profile.embed_dim {expected_dim} "
                f"for {model_name}; fix the profile, do not hardcode."
            )

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return vecs.astype(np.float32)


def build_embedder(profile: ModelProfile, *, force_fallback: bool = False) -> Embedder:
    """Construct `fA` for the active profile.

    Falls back to the hashing embedder when sentence-transformers is unavailable
    so the pipeline never hard-fails on a base install. The fallback keeps
    `profile.embed_dim`, so downstream bridges see the contracted dimension
    either way.
    """
    if force_fallback:
        return HashingEmbedder(profile.embed_dim)
    try:
        return SentenceTransformerEmbedder(profile.embedder_model, profile.embed_dim)
    except Exception:
        return HashingEmbedder(profile.embed_dim)
