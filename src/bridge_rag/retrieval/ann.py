"""ANN top-k retrieval with a dependency-free numpy fallback.

`ANNIndex` wraps faiss `IndexFlatIP` when faiss is available; when it is not,
it falls back to a pure-numpy inner-product + argsort.  Both paths produce
identical results because the embeddings are L2-normalised (inner product ==
cosine similarity) and `IndexFlatIP` is exact (no approximation).

The index is built once from a `Corpus` and is read-only thereafter.  For the
per-request re-ranking stage, see `DynamicIndex`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..types import Candidate, Document
from .corpus import Corpus

if TYPE_CHECKING:
    pass  # faiss is never imported at the type-checking level

logger = logging.getLogger(__name__)


def _numpy_search(
    embeddings: np.ndarray,
    query: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-numpy inner-product search.  Returns (scores, indices), shape (top_k,)."""
    scores: np.ndarray = embeddings @ query.reshape(-1)  # (N,)
    k = min(top_k, len(scores))
    # argsort ascending; take the last k entries reversed for descending order.
    idx: np.ndarray = np.argsort(scores)[-k:][::-1]
    return scores[idx], idx


class ANNIndex:
    """Exact inner-product index over a frozen `Corpus`.

    Usage::

        index = ANNIndex.build(corpus)
        candidates = index.search(query_embedding, top_k=16)

    The class keeps a reference to the underlying `Corpus` so it can map row
    indices back to `Document` objects without copying the text.
    """

    def __init__(self, corpus: Corpus, *, _faiss_index: object | None = None) -> None:
        self._corpus = corpus
        self._faiss_index = _faiss_index  # None → numpy path
        self._use_faiss = _faiss_index is not None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, corpus: Corpus) -> "ANNIndex":
        """Build the index from a `Corpus`.  Prefers faiss; falls back to numpy."""
        embeddings = corpus.embeddings  # (N, d) float32
        if embeddings.shape[0] == 0:
            logger.warning("building ANNIndex over an empty corpus")
            return cls(corpus, _faiss_index=None)

        try:
            import faiss  # type: ignore[import]  # lazy — may not be installed

            dim = embeddings.shape[1]
            idx = faiss.IndexFlatIP(dim)
            # faiss requires a C-contiguous float32 array.
            idx.add(np.ascontiguousarray(embeddings, dtype=np.float32))
            logger.info(
                "ANNIndex built with faiss: %d vectors, dim=%d", len(corpus), dim
            )
            return cls(corpus, _faiss_index=idx)
        except Exception as exc:  # ImportError or faiss build failure
            logger.info(
                "faiss unavailable (%s); using numpy fallback for ANNIndex", exc
            )
            return cls(corpus, _faiss_index=None)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[Candidate]:
        """Return up to `top_k` candidates ordered by descending inner product.

        `query_embedding` must be L2-normalised (shape (d,) or (1, d)); the
        same normalisation contract as `Embedder.embed`.
        """
        if len(self._corpus) == 0:
            return []

        q = np.ascontiguousarray(query_embedding.reshape(-1), dtype=np.float32)
        k = min(top_k, len(self._corpus))

        if self._use_faiss:
            # faiss search expects shape (1, d); returns (1, k) arrays.
            q2d = q.reshape(1, -1)
            scores_2d, indices_2d = self._faiss_index.search(q2d, k)  # type: ignore[union-attr]
            scores_arr: np.ndarray = scores_2d[0]
            indices_arr: np.ndarray = indices_2d[0]
            # faiss pads with -1 when the corpus is smaller than k.
            valid = indices_arr >= 0
            scores_arr = scores_arr[valid]
            indices_arr = indices_arr[valid]
        else:
            scores_arr, indices_arr = _numpy_search(self._corpus.embeddings, q, k)

        candidates: list[Candidate] = []
        docs: tuple[Document, ...] = self._corpus.documents
        for score, idx in zip(scores_arr.tolist(), indices_arr.tolist()):
            candidates.append(
                Candidate(document=docs[int(idx)], similarity=float(score))
            )
        return candidates
