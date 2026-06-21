"""Bridge AB — the Sidecar (`gAB`): SpanExtractor protocol + runnable default.

Deliberately NOT a `TensorBridge`. A TensorBridge maps `Tensor -> Tensor`; the
sidecar maps `(query_embedding, candidates) -> list[Span]` under a latency
contract. It *uses* attention-style scoring internally (Eq.18-20) — Q from the
query, K/V from candidate spans, softmax relevance — but its public contract is
span extraction with provenance, which is what makes the downstream
no-hallucination guarantee possible.

The default extractor here is dependency-free: it splits candidate documents
into sentence spans, scores each against the query with the provided embedder
(the same `fA` family, so dot-product = the attention logit), keeps the top
`max_spans`. The production BGE-M3 reranker (distilled from a Qwen3 teacher)
implements the same protocol in `extractor.py`. Both honor the budget contract.
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from ..config import ModelProfile
from ..contracts.budget import Timer, sidecar_contract
from ..stages.perception import Embedder
from ..types import Candidate, Span

_SENTENCE_SPLIT = re.compile(r"[^.!?\n]+[.!?\n]?")


@runtime_checkable
class SpanExtractor(Protocol):
    """The `gAB` interface: needle out of haystack, under budget."""

    def extract(
        self, query: str, query_embedding: np.ndarray, candidates: Sequence[Candidate]
    ) -> list[Span]:
        ...


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(initial=0.0)
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else e


def iter_sentence_spans(doc_id: str, text: str) -> list[tuple[int, int, str]]:
    """Split into (char_start, char_end, text) spans, preserving offsets.

    Offsets are real character positions into the source document — that is what
    makes a span a verifiable citation.
    """
    out: list[tuple[int, int, str]] = []
    for m in _SENTENCE_SPLIT.finditer(text):
        seg = m.group(0)
        if seg.strip():
            out.append((m.start(), m.end(), seg))
    return out


class DefaultSpanExtractor:
    """Attention-style span selection (Eq.18-20), dependency-free.

    Q = query embedding, K = candidate-span embeddings; relevance = softmax(QKᵀ)
    is the attention weight over spans. We select the highest-weight spans up to
    the contract cap. This is the selection face of the bottleneck (Eq.11):
    minimize I(U;X) by keeping few spans, maximize I(U;Y) by keeping the most
    query-relevant ones.
    """

    def __init__(self, profile: ModelProfile, embedder: Embedder) -> None:
        self._profile = profile
        self._embedder = embedder
        self._timer = Timer()

    @property
    def timer(self) -> Timer:
        return self._timer

    def extract(
        self, query: str, query_embedding: np.ndarray, candidates: Sequence[Candidate]
    ) -> list[Span]:
        capped = list(candidates)[: self._profile.max_candidates]
        contract = sidecar_contract(
            self._profile.sidecar_budget_ms,
            max_candidates=self._profile.max_candidates,
            max_spans=self._profile.max_spans,
            get_candidate_count=lambda: len(capped),
            get_span_count=lambda: 0,  # rebound below once spans exist
        )
        contract.check_structural()

        with self._timer.measure():
            raw: list[tuple[str, int, int, str]] = []
            for cand in capped:
                for cs, ce, seg in iter_sentence_spans(
                    cand.document.doc_id, cand.document.text
                ):
                    raw.append((cand.document.doc_id, cs, ce, seg))

            if not raw:
                return []

            seg_texts = [r[3] for r in raw]
            seg_emb = self._embedder.embed(seg_texts)  # (n, d)
            q = query_embedding.reshape(-1)
            logits = seg_emb @ q  # QKᵀ over spans
            weights = _softmax(logits)
            order = np.argsort(-weights)[: self._profile.max_spans]

            spans = [
                Span(
                    doc_id=raw[i][0],
                    char_start=raw[i][1],
                    char_end=raw[i][2],
                    text=raw[i][3],
                    score=float(weights[i]),
                )
                for i in order
            ]

        # Output-cap clause of the budget contract, now that spans exist.
        out_contract = sidecar_contract(
            self._profile.sidecar_budget_ms,
            max_candidates=self._profile.max_candidates,
            max_spans=self._profile.max_spans,
            get_candidate_count=lambda: len(capped),
            get_span_count=lambda: len(spans),
        )
        out_contract.check_structural()
        return spans
