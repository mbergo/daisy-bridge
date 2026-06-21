"""Stage C — Generation (`fC`) protocol + the no-hallucination chokepoint.

The structural 0%-hallucination guarantee (Section 7) is an *architecture*
property, not a runtime checker: the generator can only ever be handed extracted
spans. This module owns the single function that assembles generator input —
`assemble_prompt` — and it takes `list[Span]` and nothing else. There is no
overload that accepts a raw document. If full-doc text cannot reach this
function, it cannot reach the model, and the model cannot hallucinate from
context it never received.

Backends (transformers, vLLM) implement the `Generator` protocol and MUST route
through `assemble_prompt`. A dependency-free `StubGenerator` lets the composition
and the E2E keystone run with zero model downloads.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, Sequence, runtime_checkable

from ..types import Citation, GenerationChunk, ChunkKind, Span

SYSTEM_PROMPT = (
    "You answer strictly from the provided spans. Each span is a citation with "
    "provenance. If the spans do not contain the answer, say so. Never use "
    "outside knowledge."
)


def assemble_prompt(query: str, spans: Sequence[Span]) -> str:
    """The ONLY path to generator input. Spans in, prompt out.

    Markers `[i]` are assigned here so the streamed answer can cite them and the
    provenance layer can resolve them back to `(doc_id, char_start, char_end)`.
    """
    lines = [SYSTEM_PROMPT, "", f"Question: {query}", "", "Spans:"]
    for i, span in enumerate(spans, start=1):
        lines.append(f"[{i}] ({span.doc_id}:{span.char_start}-{span.char_end}) {span.text}")
    lines.append("")
    lines.append("Answer (cite spans as [i]):")
    return "\n".join(lines)


def citations_for(spans: Sequence[Span]) -> list[Citation]:
    """The citation table the markers in `assemble_prompt` refer to."""
    return [Citation(marker=f"[{i}]", span=s) for i, s in enumerate(spans, start=1)]


@runtime_checkable
class Generator(Protocol):
    """The `fC` interface. Implementations never see anything but spans+query."""

    def prime_system_prompt(self) -> None:
        """Warm the system-prompt KV cache (DJ is always-ready, Section 6.3)."""
        ...

    def generate(self, query: str, spans: Sequence[Span]) -> AsyncIterator[GenerationChunk]:
        """Stream the answer token-by-token, then citation chunks, then DONE."""
        ...


class StubGenerator:
    """Deterministic, dependency-free generator.

    Produces a grounded answer by echoing the spans it was given (proving the
    spans-only contract end to end) and emits citation chunks. No torch, no
    transformers — so the E2E keystone runs anywhere. Real text generation comes
    from the transformers/vLLM backends behind the same protocol.
    """

    def __init__(self, *, primed: bool = False) -> None:
        self._primed = primed

    def prime_system_prompt(self) -> None:
        self._primed = True

    async def generate(
        self, query: str, spans: Sequence[Span]
    ) -> AsyncIterator[GenerationChunk]:
        if not self._primed:
            self.prime_system_prompt()
        # assemble_prompt is called for its side effect of being the only door;
        # the stub does not need the string, but the contract must be honored.
        _ = assemble_prompt(query, spans)
        if not spans:
            yield GenerationChunk(kind=ChunkKind.TOKEN, text="No spans were provided; ")
            yield GenerationChunk(kind=ChunkKind.TOKEN, text="I cannot answer.")
            yield GenerationChunk(kind=ChunkKind.DONE)
            return
        cites = citations_for(spans)
        intro = "Based on the retrieved spans: "
        for word in intro.split():
            yield GenerationChunk(kind=ChunkKind.TOKEN, text=word + " ")
        for cite in cites:
            snippet = cite.span.text.strip()
            for word in snippet.split():
                yield GenerationChunk(kind=ChunkKind.TOKEN, text=word + " ")
            yield GenerationChunk(kind=ChunkKind.TOKEN, text=cite.marker + " ")
            yield GenerationChunk(kind=ChunkKind.CITATION, citation=cite)
        yield GenerationChunk(kind=ChunkKind.DONE)
