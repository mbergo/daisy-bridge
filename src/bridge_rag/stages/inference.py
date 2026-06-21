"""Stage B — Inference (`fB`): the no-hallucination architectural chokepoint.

`fB` is an identity function on spans: it takes a sequence of `Span` objects
and returns them unchanged as a tuple.  Its value is entirely in what it
FORBIDS: full-document text, raw candidate documents, and any context that has
not been processed by the sidecar span extractor cannot pass this stage.

Paper framing
-------------
hB = uAB = the span set.  fB has no learnable parameters; it is the
information-theoretic checkpoint that enforces context ⊆ spans before the
generator ever sees a token.  Because `assemble_prompt` (Stage C) only accepts
`Sequence[Span]`, and this stage only emits `tuple[Span, ...]`, the
architecture physically cannot route full-document text to the generator.

The structural guarantee is NOT enforced by a runtime heuristic ("does this
look like a full document?").  It is enforced by `assert_spans_only`, which
verifies that every span has:

1. A resolvable `doc_id` in the corpus documents mapping.
2. A non-negative, non-inverted offset window (``char_start < char_end``).
3. Offsets that fall within the declared document's text length.

Any violation raises `InferenceStageError` before the span tuple is returned.
This makes the guarantee hold even if a buggy upstream stage injects a
fabricated span — the error surfaces loudly here rather than silently reaching
the generator.
"""

from __future__ import annotations

import logging
from typing import Sequence

from ..types import Document, Span

logger = logging.getLogger(__name__)


class InferenceStageError(ValueError):
    """Raised when a span fails the provenance gate in `assert_spans_only`.

    Separating this from `ProvenanceError` (in provenance.py) keeps the two
    concerns distinct: provenance.py checks text content, this module checks
    structural validity of the offsets and doc membership.
    """


def assert_spans_only(
    spans: Sequence[Span],
    documents: dict[str, Document],
) -> None:
    """Assert that every span has valid, resolvable provenance.

    This is the structural gate of Stage B.  It must be called before any
    span is forwarded to the generator.  Raises `InferenceStageError` on the
    first violation found.

    Checks performed (per span)
    ---------------------------
    1. ``doc_id`` present in ``documents`` — the span refers to a real corpus
       document, not a fabricated source.
    2. ``char_start >= 0`` and ``char_end > char_start`` — the offset window
       is non-degenerate.
    3. ``char_end <= len(doc.text)`` — the window is within the document's
       bounds; an out-of-bounds offset would mean the span was not actually
       extracted from this document.

    Note: text-content matching (``doc.text[char_start:char_end] == span.text``)
    is the responsibility of `provenance.resolve_citations`.  This function
    validates structure; provenance.py validates content.

    Parameters
    ----------
    spans:
        The candidate spans emitted by the sidecar extractor.
    documents:
        A ``doc_id → Document`` mapping for the corpus.  Must cover every
        ``doc_id`` referenced by the spans.

    Raises
    ------
    InferenceStageError
        On the first span that fails any structural check.
    """
    for i, span in enumerate(spans):
        # Check 1: doc_id resolvable.
        doc = documents.get(span.doc_id)
        if doc is None:
            raise InferenceStageError(
                f"span[{i}] references unknown doc_id={span.doc_id!r}; "
                "full-doc text cannot pass this stage — only extracted spans "
                "with valid corpus provenance are permitted."
            )

        # Check 2: non-degenerate offset window.
        if span.char_start < 0:
            raise InferenceStageError(
                f"span[{i}] doc={span.doc_id!r} has char_start={span.char_start} < 0."
            )
        if span.char_end <= span.char_start:
            raise InferenceStageError(
                f"span[{i}] doc={span.doc_id!r} has inverted or empty window "
                f"[{span.char_start}:{span.char_end})."
            )

        # Check 3: offsets within document bounds.
        doc_len = len(doc.text)
        if span.char_end > doc_len:
            raise InferenceStageError(
                f"span[{i}] doc={span.doc_id!r} offset [{span.char_start}:"
                f"{span.char_end}) exceeds document length {doc_len}; "
                "the span was not extracted from this document."
            )


class InferenceStage:
    """fB — identity on spans, hard gate on provenance.

    This stage carries no parameters.  Its sole function is to be the
    architectural chokepoint: nothing that has not passed `assert_spans_only`
    can reach the generator.

    Construction binds the stage to a fixed corpus document map so the gate
    always operates against the same source of truth that the sidecar extractor
    read from.

    Usage::

        stage = InferenceStage(documents={doc.doc_id: doc for doc in corpus})
        safe_spans = stage.forward(sidecar_output_spans)
        # safe_spans is now a tuple[Span, ...] — safe to pass to assemble_prompt.
    """

    def __init__(self, documents: dict[str, Document]) -> None:
        """Bind the stage to the corpus document map.

        Parameters
        ----------
        documents:
            ``{doc_id: Document}`` covering the full corpus.  Build with::

                {doc.doc_id: doc for doc in corpus}
        """
        # Immutable snapshot — prevents the corpus from changing under the gate.
        self._documents: dict[str, Document] = dict(documents)

    def forward(self, spans: Sequence[Span]) -> tuple[Span, ...]:
        """Gate `spans` through provenance validation; return as a frozen tuple.

        The return type is ``tuple[Span, ...]`` rather than ``list[Span]`` to
        make it clear to downstream callers that the span set is fixed and
        ordered at this point — no further mutation is expected.

        Steps
        -----
        1. Call `assert_spans_only` — raises `InferenceStageError` if any span
           lacks valid corpus provenance.
        2. Return the spans as an immutable tuple.

        The generator (`assemble_prompt`) accepts ``Sequence[Span]``, so the
        tuple passes through without conversion.

        Parameters
        ----------
        spans:
            Candidate spans from the sidecar extractor.

        Returns
        -------
        tuple[Span, ...]
            The same spans, unchanged in content and order, validated.

        Raises
        ------
        InferenceStageError
            If any span fails the structural provenance gate.
        """
        assert_spans_only(spans, self._documents)
        result: tuple[Span, ...] = tuple(spans)
        logger.debug(
            "InferenceStage.forward: %d spans passed the provenance gate",
            len(result),
        )
        return result
