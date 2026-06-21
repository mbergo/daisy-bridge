"""Pydantic request/response schemas for the FastAPI serving layer."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ..types import Citation


class AnswerRequest(BaseModel):
    """Incoming query payload for ``POST /answer``."""

    query: str = Field(..., min_length=1, description="The user question to answer.")
    top_k: int = Field(
        default=16,
        ge=1,
        le=256,
        description="Number of ANN candidates to retrieve before span extraction.",
    )


class CitationModel(BaseModel):
    """Wire representation of a ``Citation`` for SSE citation events."""

    marker: str = Field(..., description="Inline citation marker, e.g. '[1]'.")
    doc_id: str = Field(..., description="Source document identifier.")
    char_start: int = Field(..., description="Byte-offset start of the span in the source doc.")
    char_end: int = Field(..., description="Byte-offset end of the span in the source doc.")
    text: str = Field(..., description="The literal span text.")
    score: float = Field(..., description="Sidecar relevance score (proxy for I(U;Y)).")

    @classmethod
    def from_citation(cls, citation: Citation) -> "CitationModel":
        """Construct from a domain ``Citation`` object.

        Args:
            citation: A resolved ``Citation`` carrying span provenance.

        Returns:
            A ``CitationModel`` ready for JSON serialisation.
        """
        return cls(
            marker=citation.marker,
            doc_id=citation.span.doc_id,
            char_start=citation.span.char_start,
            char_end=citation.span.char_end,
            text=citation.span.text,
            score=citation.span.score,
        )


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str = Field(default="ok", description="Service liveness indicator.")
    profile: str = Field(..., description="Active profile name (dev or prod).")
    blue_version: str = Field(
        ..., description="Current live version tag, e.g. 'SIDECAR_V1+QLORA_V1'."
    )
    sidecar_budget_ms: float = Field(
        ..., description="Latency budget for the sidecar stage in milliseconds."
    )
