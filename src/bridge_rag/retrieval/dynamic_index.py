"""Per-request, query-shaped index over the post-ANN candidate subset.

Paper Section 4.3 — "query constructs the index": the QUERY shapes a tiny
index over the already-narrowed candidate set, used once and discarded.  This
is fundamentally different from a static index that is searched by the query.
Here the index *itself* is a function of the query; its construction is part of
the sidecar budget.

Why a second index over an already-narrow set?
- After ANN retrieval we have at most `profile.max_candidates` vectors.
- We re-rank them with the query vector in the exact same embedding space, but
  now we can afford an exact inner-product sweep (tiny N, cheap).
- The resulting ordering is query-conditioned and discarded after the request —
  no state leaks between requests.

The index is cheap because it operates on the narrowed candidate set (typically
≤ 128 vectors) rather than the full corpus.  It is intentionally ephemeral:
do NOT persist, do NOT cache, build fresh per request.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from ..config import ModelProfile
from ..types import Candidate

logger = logging.getLogger(__name__)


def _ip_rerank(
    query: np.ndarray,
    embeddings: np.ndarray,
    candidates: Sequence[Candidate],
) -> list[Candidate]:
    """Re-rank `candidates` by inner product against `query`.

    Pure numpy — no faiss needed for a set this small (≤ max_candidates rows).
    Returns a new list; the original `Candidate` objects are reused unchanged,
    but their order and the `.similarity` field reflects the query-conditioned
    score from this index pass.
    """
    q = query.reshape(-1).astype(np.float32)
    # embeddings: (n, d) — already L2-normalised from the corpus.
    scores: np.ndarray = embeddings @ q  # (n,)
    order = np.argsort(-scores)  # descending
    reranked: list[Candidate] = []
    for rank_idx in order.tolist():
        orig = candidates[int(rank_idx)]
        reranked.append(
            Candidate(
                document=orig.document,
                similarity=float(scores[rank_idx]),
            )
        )
    return reranked


class DynamicIndex:
    """A per-request, query-shaped index over the post-ANN candidate set.

    Construction is the index-build step; `ranked_candidates` exposes the
    result.  This object is intentionally single-use: build → read → discard.

    Implementation note:  we try faiss `IndexFlatIP` first for consistency with
    `ANNIndex`, then fall back to the numpy sweep — both paths yield identical
    results over a set this small.
    """

    def __init__(self, ranked: list[Candidate]) -> None:
        # Frozen after construction — this object is read-only.
        self._ranked: tuple[Candidate, ...] = tuple(ranked)

    @classmethod
    def build_for_query(
        cls,
        query_embedding: np.ndarray,
        candidates: Sequence[Candidate],
        *,
        profile: ModelProfile,
    ) -> "DynamicIndex":
        """Construct a query-conditioned index over `candidates`.

        Steps
        -----
        1. Cap at `profile.max_candidates` (structural budget bound).
        2. Extract embedding matrix — use the document's stored embedding when
           present, otherwise compute a zero vector (should not occur in
           production; corpus.save/load always stores embeddings).
        3. Build a tiny per-request index (faiss if available, numpy fallback).
        4. Return a `DynamicIndex` holding the re-ranked list.

        The index-build itself is part of the sidecar budget window (caller is
        responsible for timing via `Timer.measure`).
        """
        capped: list[Candidate] = list(candidates)[: profile.max_candidates]

        if not capped:
            return cls([])

        # Determine embedding dimension from the first available embedding.
        dim: int = _infer_dim(capped, query_embedding)

        # Build embedding matrix: (n, d).
        emb_rows: list[np.ndarray] = []
        for cand in capped:
            if cand.document.embedding is not None:
                emb_rows.append(
                    cand.document.embedding.reshape(dim).astype(np.float32)
                )
            else:
                # Fallback: use the ANN similarity as a scalar proxy embedded
                # in the query direction — preserves relative order from ANN.
                logger.debug(
                    "doc %r has no stored embedding; using ANN similarity proxy",
                    cand.document.doc_id,
                )
                q_unit = query_embedding.reshape(-1).astype(np.float32)
                norm = float(np.linalg.norm(q_unit))
                if norm > 0.0:
                    q_unit = q_unit / norm
                emb_rows.append(q_unit * float(cand.similarity))

        embeddings = np.stack(emb_rows, axis=0)  # (n, d)

        ranked = _try_faiss_rerank(query_embedding, embeddings, capped)
        logger.debug(
            "DynamicIndex built: %d candidates → %d ranked",
            len(capped),
            len(ranked),
        )
        return cls(ranked)

    @property
    def ranked_candidates(self) -> list[Candidate]:
        """Query-conditioned ranking of the candidate subset, best first."""
        return list(self._ranked)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_dim(candidates: Sequence[Candidate], query_embedding: np.ndarray) -> int:
    """Return embedding dimension, preferring stored doc embeddings."""
    for cand in candidates:
        if cand.document.embedding is not None:
            return int(cand.document.embedding.reshape(-1).shape[0])
    return int(query_embedding.reshape(-1).shape[0])


def _try_faiss_rerank(
    query: np.ndarray,
    embeddings: np.ndarray,
    candidates: Sequence[Candidate],
) -> list[Candidate]:
    """Attempt faiss re-rank; fall back to numpy on import/runtime error."""
    try:
        import faiss  # type: ignore[import]  # lazy — may not be installed

        n, d = embeddings.shape
        idx = faiss.IndexFlatIP(d)
        idx.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        q2d = query.reshape(1, -1).astype(np.float32)
        scores_2d, indices_2d = idx.search(q2d, n)
        scores_arr: np.ndarray = scores_2d[0]
        indices_arr: np.ndarray = indices_2d[0]
        reranked: list[Candidate] = []
        for score, row_idx in zip(scores_arr.tolist(), indices_arr.tolist()):
            if int(row_idx) < 0:
                continue
            orig = candidates[int(row_idx)]
            reranked.append(
                Candidate(document=orig.document, similarity=float(score))
            )
        return reranked
    except Exception:
        # faiss unavailable or raised — use the dependency-free path.
        return _ip_rerank(query, embeddings, candidates)
