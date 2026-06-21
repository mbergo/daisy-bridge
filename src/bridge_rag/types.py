"""Core data types — the spine of the system.

`Span` is load-bearing twice over: it carries the provenance that makes the
0%-hallucination guarantee *structural* (a span IS a citation), and it is the
only thing that ever reaches the generator (Section 7, paper).

These types have zero heavy dependencies on purpose: everything else in the
package compiles against them, including modules that never import torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

import numpy as np

Vector = np.ndarray  # shape (d,), float32


@dataclass(frozen=True, slots=True)
class Span:
    """An extracted span with provenance.

    Provenance (`doc_id`, `char_start`, `char_end`) is mandatory — it is what
    turns "the model cited a source" into a verifiable claim. `score` is the
    sidecar relevance (proxy for I(U;Y)). `embedding` is optional; populated
    only when the extractor keeps it around for downstream gating.
    """

    doc_id: str
    char_start: int
    char_end: int
    text: str
    score: float
    embedding: Optional[Vector] = None

    def __post_init__(self) -> None:
        if self.char_start < 0 or self.char_end < self.char_start:
            raise ValueError(
                f"invalid span offsets: [{self.char_start}, {self.char_end})"
            )
        if len(self.text) != self.char_end - self.char_start:
            # Soft invariant: text length should match the offset window. We
            # do not raise (whitespace normalization can legitimately differ)
            # but the provenance check in provenance.py uses offsets, not text.
            pass

    @property
    def length(self) -> int:
        return self.char_end - self.char_start


@dataclass(frozen=True, slots=True)
class Citation:
    """A resolved citation: a span tied to a marker emitted in the answer."""

    marker: str  # e.g. "[1]"
    span: Span

    @property
    def doc_id(self) -> str:
        return self.span.doc_id

    @property
    def offsets(self) -> tuple[int, int]:
        return (self.span.char_start, self.span.char_end)


@dataclass(frozen=True, slots=True)
class Document:
    """A corpus document. `text` retained so spans can resolve to substrings."""

    doc_id: str
    text: str
    embedding: Optional[Vector] = None


@dataclass(frozen=True, slots=True)
class Candidate:
    """An ANN hit: a document plus the similarity that surfaced it."""

    document: Document
    similarity: float


@dataclass(slots=True)
class BridgeOutput:
    """Output of a TensorBridge forward pass plus the telemetry the trainer
    and the invariant probes need (Eq.13 stability story).

    `value` is the bridged representation `u`. `pre_norm`/`post_norm` capture
    the LayerNorm distribution shift the paper tracks (Stabilizer 1)."""

    value: Any  # torch.Tensor — typed Any to avoid a torch import here
    pre_norm: Optional[float] = None
    post_norm: Optional[float] = None
    gate: Optional[Any] = None  # gated routing `r`, when applicable


class ChunkKind(str, Enum):
    TOKEN = "token"
    CITATION = "citation"
    DONE = "done"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class GenerationChunk:
    """One streamed unit from the generator (SSE payload)."""

    kind: ChunkKind
    text: str = ""
    citation: Optional[Citation] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BridgeVersion:
    """A co-adapted (sidecar, qlora) pair (Section 8.3).

    Versioning is paired by construction: the sidecar and the generator's QLoRA
    adapter are trained together, so they must be *served* together. An
    unpaired combination is a load-time error, never a silent quality drop.
    """

    sidecar: str  # e.g. "SIDECAR_V2"
    qlora: str  # e.g. "QLORA_V2"

    @property
    def tag(self) -> str:
        return f"{self.sidecar}+{self.qlora}"

    def paired_with(self, other: "BridgeVersion") -> bool:
        return self.sidecar == other.sidecar and self.qlora == other.qlora


@dataclass(slots=True)
class RequestCtx:
    """Per-request context threaded through the event-driven pipeline (Section 6).

    Holds the query, the timing ledger (so we can attribute the critical-path
    budget), and the artifacts each stage deposits. Mutable on purpose: the
    async tracks fill it concurrently.
    """

    query: str
    request_id: str
    top_k: int = 16
    query_embedding: Optional[Vector] = None
    candidates: Sequence[Candidate] = field(default_factory=tuple)
    spans: Sequence[Span] = field(default_factory=tuple)
    timings_ms: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def mark(self, stage: str, ms: float) -> None:
        self.timings_ms[stage] = ms

    @property
    def critical_path_ms(self) -> float:
        """Sum of the stages on the embed->sidecar->generate chain only."""
        keys = ("embed", "ann", "sidecar")
        return sum(self.timings_ms.get(k, 0.0) for k in keys)
