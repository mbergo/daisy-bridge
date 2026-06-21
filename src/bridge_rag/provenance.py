"""Provenance: span-to-citation resolution and async faithfulness check.

The 0%-hallucination guarantee is structural (the generator only ever receives
extracted spans), but provenance makes it *verifiable*: every `Citation` that
reaches the user carries `(doc_id, char_start, char_end)` that can be checked
against the corpus text.

Two public surfaces:

1. `resolve_citations` — synchronous, called before generation.  Assigns
   markers ``[1]``, ``[2]``, … to spans and ASSERTS that each span's text
   matches the corpus slice at its declared offsets.  Raises `ProvenanceError`
   on any mismatch — a mismatch means the span was fabricated or its offsets
   are wrong, both of which violate the provenance contract.

2. `faithfulness_check` — async, runs off the critical path after generation
   (Section 6).  Scans the generated answer for citation markers and confirms
   that every emitted marker corresponds to a real span.  Returns a structured
   report; does not call any model.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from .types import Citation, Document, Span

logger = logging.getLogger(__name__)

_MARKER_RE = re.compile(r"\[(\d+)\]")


class ProvenanceError(ValueError):
    """Raised when a span's text cannot be verified against the corpus source.

    This error means provenance is broken — either the span's offsets are
    wrong, the span text was altered after extraction, or the document was
    mutated after the corpus was built.  Any of these would invalidate the
    no-hallucination guarantee, so we surface it loudly.
    """


def resolve_citations(
    spans: Sequence[Span],
    documents: dict[str, Document],
) -> list[Citation]:
    """Assign citation markers and ASSERT provenance for each span.

    For every span:
    - The marker is ``[i]`` (1-indexed, matching `assemble_prompt`).
    - The corpus slice ``documents[doc_id].text[char_start:char_end]`` is
      retrieved and compared to ``span.text``.
    - A mismatch raises `ProvenanceError`; the comparison is exact (no
      whitespace tolerance) because the span MUST be a verbatim substring.

    Parameters
    ----------
    spans:
        The ordered list of extracted spans to resolve.
    documents:
        A ``doc_id → Document`` mapping covering every doc referenced by the
        spans.  Build this from ``{doc.doc_id: doc for doc in corpus}``.

    Returns
    -------
    list[Citation]
        One `Citation` per span, markers ``[1]`` … ``[N]``.

    Raises
    ------
    ProvenanceError
        If any span's text does not match the corpus slice at its offsets, or
        if the referenced ``doc_id`` is absent from ``documents``.
    """
    citations: list[Citation] = []
    for i, span in enumerate(spans, start=1):
        marker = f"[{i}]"

        doc = documents.get(span.doc_id)
        if doc is None:
            raise ProvenanceError(
                f"span {marker} references unknown doc_id {span.doc_id!r}; "
                "document not present in the provided corpus mapping."
            )

        corpus_slice = doc.text[span.char_start : span.char_end]
        if corpus_slice != span.text:
            raise ProvenanceError(
                f"span {marker} provenance mismatch for doc {span.doc_id!r} "
                f"at [{span.char_start}:{span.char_end}]: "
                f"corpus has {corpus_slice!r}, span carries {span.text!r}."
            )

        citations.append(Citation(marker=marker, span=span))

    return citations


async def faithfulness_check(
    answer_text: str,
    spans: Sequence[Span],
    documents: dict[str, Document],
) -> dict[str, object]:
    """Post-generation faithfulness audit (Section 6, off the critical path).

    Scans `answer_text` for citation markers ``[i]`` and verifies that each
    one refers to a real span from the extraction set.  No model is called —
    this is a structural check on the citation table.

    Parameters
    ----------
    answer_text:
        The generated answer string (may contain markers like ``[1]``, ``[2]``).
    spans:
        The spans that were passed to the generator (``spans[i-1]`` for marker
        ``[i]``).
    documents:
        A ``doc_id → Document`` mapping (same as for `resolve_citations`).

    Returns
    -------
    dict with keys:

    ``grounded`` : bool
        ``True`` iff every marker in the answer resolves to a real span AND
        that span's provenance verifies against the corpus.
    ``cited_markers`` : list[str]
        All ``[i]`` markers found in the answer, in order of appearance.
    ``unsupported_markers`` : list[str]
        Markers that appear in the answer but have no corresponding span
        (index out of range or empty span list).
    ``unverifiable_markers`` : list[str]
        Markers that point to a real span but whose provenance cannot be
        confirmed (doc missing or text mismatch) — should be empty in a
        correctly built corpus.
    ``cited_doc_ids`` : list[str]
        Unique doc_ids of all spans that were actually cited.
    """
    span_list = list(spans)
    n_spans = len(span_list)

    found_markers: list[str] = [m.group(0) for m in _MARKER_RE.finditer(answer_text)]
    cited_markers: list[str] = list(dict.fromkeys(found_markers))  # preserve order, dedupe

    unsupported: list[str] = []
    unverifiable: list[str] = []
    cited_doc_ids_set: set[str] = set()

    for marker in cited_markers:
        # Marker [i] → zero-indexed into span_list.
        raw_idx = int(_MARKER_RE.match(marker).group(1))  # type: ignore[union-attr]
        span_idx = raw_idx - 1

        if span_idx < 0 or span_idx >= n_spans:
            unsupported.append(marker)
            continue

        span = span_list[span_idx]

        doc = documents.get(span.doc_id)
        if doc is None:
            unverifiable.append(marker)
            logger.warning(
                "faithfulness_check: marker %s cites doc %r not in document map",
                marker,
                span.doc_id,
            )
            continue

        corpus_slice = doc.text[span.char_start : span.char_end]
        if corpus_slice != span.text:
            unverifiable.append(marker)
            logger.warning(
                "faithfulness_check: marker %s has provenance mismatch "
                "doc=%r [%d:%d] corpus=%r span=%r",
                marker,
                span.doc_id,
                span.char_start,
                span.char_end,
                corpus_slice,
                span.text,
            )
            continue

        cited_doc_ids_set.add(span.doc_id)

    grounded = (not unsupported) and (not unverifiable)

    return {
        "grounded": grounded,
        "cited_markers": cited_markers,
        "unsupported_markers": unsupported,
        "unverifiable_markers": unverifiable,
        "cited_doc_ids": sorted(cited_doc_ids_set),
    }
