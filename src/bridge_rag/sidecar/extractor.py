"""Production span extractor: BGE-M3 cross-encoder reranker (Bridge AB).

Implements the `SpanExtractor` protocol with a real cross-encoder reranker
(sentence-transformers `CrossEncoder`), realising Eq.18-20 from the paper:
attention-style relevance scored by a trained reranker rather than raw dot
products.  When sentence-transformers cannot be loaded, delegates to
`DefaultSpanExtractor` — the same dependency-free path used in dev/CI.

The sidecar budget contract (`sidecar_contract`) is enforced structurally
(bounded candidates in, bounded spans out) and the wall-clock sample is
accumulated via `Timer` so the caller can run empirical p99 checks.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from ..config import ModelProfile
from ..contracts.budget import Timer, sidecar_contract
from ..sidecar.base import DefaultSpanExtractor, SpanExtractor, iter_sentence_spans
from ..stages.perception import Embedder
from ..types import Candidate, Span

logger = logging.getLogger(__name__)


class RerankerSpanExtractor:
    """Cross-encoder reranker span extractor — the production Bridge AB.

    On first call to `extract`, the cross-encoder is loaded lazily so module
    import never touches sentence-transformers.  If loading fails (library
    absent, model unavailable, OOM), we permanently delegate to
    `DefaultSpanExtractor` for the lifetime of this instance.

    Scoring pipeline
    ----------------
    1. Split each candidate document into sentence spans via
       `iter_sentence_spans` (real char offsets → provenance).
    2. Score each (query_text, span_text) pair with the cross-encoder.
       This is the distilled BGE-M3 0.6 B sidecar realising Eq.18-20.
    3. Keep the top `profile.max_spans` by reranker score.
    4. Emit `Span` objects with real `(doc_id, char_start, char_end)` offsets
       and the reranker score — every span is a verifiable citation.
    """

    def __init__(self, profile: ModelProfile, embedder: Embedder) -> None:
        self._profile = profile
        self._embedder = embedder
        self._timer = Timer()
        self._cross_encoder: object | None = None  # loaded lazily
        self._load_failed: bool = False
        self._fallback: DefaultSpanExtractor | None = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _get_cross_encoder(self) -> object | None:
        """Return the CrossEncoder, loading it on first call.

        Returns None and sets `_load_failed` if sentence-transformers is
        unavailable or the model cannot be fetched.
        """
        if self._load_failed:
            return None
        if self._cross_encoder is not None:
            return self._cross_encoder
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]

            logger.info(
                "loading cross-encoder reranker: %s", self._profile.sidecar_model
            )
            self._cross_encoder = CrossEncoder(self._profile.sidecar_model)
            logger.info("cross-encoder loaded")
        except Exception as exc:
            logger.warning(
                "cross-encoder unavailable (%s); falling back to DefaultSpanExtractor",
                exc,
            )
            self._load_failed = True
            self._fallback = DefaultSpanExtractor(self._profile, self._embedder)
        return self._cross_encoder

    def _get_fallback(self) -> DefaultSpanExtractor:
        if self._fallback is None:
            self._fallback = DefaultSpanExtractor(self._profile, self._embedder)
        return self._fallback

    # ------------------------------------------------------------------
    # SpanExtractor protocol
    # ------------------------------------------------------------------

    def extract(
        self,
        query: str,
        query_embedding: np.ndarray,
        candidates: Sequence[Candidate],
    ) -> list[Span]:
        """Extract and rerank spans from `candidates` under the budget contract.

        The cross-encoder scores each (query, sentence_span) pair — this is
        the attention-style relevance of Eq.18-20 produced by a trained
        distilled model rather than raw embedding dot products.

        If the cross-encoder is unavailable, delegates to
        `DefaultSpanExtractor` which uses the embedder dot product instead.
        """
        capped: list[Candidate] = list(candidates)[: self._profile.max_candidates]

        # Structural budget check: bounded candidates in.
        in_contract = sidecar_contract(
            self._profile.sidecar_budget_ms,
            max_candidates=self._profile.max_candidates,
            max_spans=self._profile.max_spans,
            get_candidate_count=lambda: len(capped),
            get_span_count=lambda: 0,
        )
        in_contract.check_structural()

        cross_encoder = self._get_cross_encoder()
        if cross_encoder is None:
            # Delegate entirely to the fallback; it handles its own contract.
            return self._get_fallback().extract(query, query_embedding, candidates)

        with self._timer.measure():
            spans = self._rerank_with_cross_encoder(query, capped, cross_encoder)

        # Structural budget check: bounded spans out.
        out_contract = sidecar_contract(
            self._profile.sidecar_budget_ms,
            max_candidates=self._profile.max_candidates,
            max_spans=self._profile.max_spans,
            get_candidate_count=lambda: len(capped),
            get_span_count=lambda: len(spans),
        )
        out_contract.check_structural()
        return spans

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rerank_with_cross_encoder(
        self,
        query: str,
        candidates: list[Candidate],
        cross_encoder: object,
    ) -> list[Span]:
        """Score sentence spans with the cross-encoder and return top-k spans."""
        # Build the flat list of (query, span_text) pairs alongside provenance.
        pairs: list[list[str]] = []
        provenance: list[tuple[str, int, int, str]] = []  # (doc_id, cs, ce, text)

        for cand in candidates:
            for char_start, char_end, seg_text in iter_sentence_spans(
                cand.document.doc_id, cand.document.text
            ):
                if seg_text.strip():
                    pairs.append([query, seg_text])
                    provenance.append(
                        (cand.document.doc_id, char_start, char_end, seg_text)
                    )

        if not pairs:
            return []

        # Cross-encoder produces a scalar relevance score per pair.
        raw_scores = cross_encoder.predict(pairs)  # type: ignore[union-attr]
        scores: np.ndarray = np.asarray(raw_scores, dtype=np.float32)

        # Top-k by descending score, capped at max_spans.
        k = min(self._profile.max_spans, len(scores))
        top_indices = np.argsort(-scores)[:k]

        spans: list[Span] = []
        for idx in top_indices.tolist():
            doc_id, char_start, char_end, seg_text = provenance[int(idx)]
            spans.append(
                Span(
                    doc_id=doc_id,
                    char_start=char_start,
                    char_end=char_end,
                    text=seg_text,
                    score=float(scores[int(idx)]),
                )
            )
        return spans

    @property
    def timer(self) -> Timer:
        """Access the accumulated wall-clock samples for empirical p99 checks."""
        return self._timer


def build_span_extractor(profile: ModelProfile, embedder: Embedder) -> SpanExtractor:
    """Factory: return a `RerankerSpanExtractor` (which self-falls-back to
    `DefaultSpanExtractor` if sentence-transformers is unavailable).

    Callers depend only on the `SpanExtractor` protocol; the concrete type is
    an implementation detail.
    """
    return RerankerSpanExtractor(profile, embedder)
