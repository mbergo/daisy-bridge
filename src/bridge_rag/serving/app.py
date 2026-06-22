"""FastAPI + SSE serving application.

Routes
------
GET  /health   -> HealthResponse (JSON)
POST /answer   -> EventSourceResponse (SSE stream)

SSE event mapping
-----------------
ChunkKind.TOKEN    -> event="token",    data=<text>
ChunkKind.CITATION -> event="citation", data=<CitationModel JSON>
ChunkKind.DONE     -> event="done",     data="done"
ChunkKind.ERROR    -> event="error",    data=<error text>

Run with::

    uvicorn bridge_rag.serving.app:app --reload

The ``Orchestrator`` is constructed once at startup (lazy, cached) via a FastAPI
lifespan context manager so it is shared across all requests.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

_STATIC_DIR = Path(__file__).parent / "static"

from ..config import get_settings
from ..types import ChunkKind, GenerationChunk
from ..versioning.blue_green import VersionedDeployment
from .schemas import AnswerRequest, CitationModel, HealthResponse

logger = logging.getLogger(__name__)

# Module-level singletons populated at startup.
_orchestrator: Optional[Any] = None  # Orchestrator; typed Any to defer import
_deployment: Optional[VersionedDeployment] = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Build the ``Orchestrator`` and ``VersionedDeployment`` at startup."""
    global _orchestrator, _deployment

    from ..pipeline.orchestrator import Orchestrator  # noqa: PLC0415

    logger.info("Starting bridge-rag serving layer…")
    _orchestrator = Orchestrator.from_settings()

    settings = get_settings()
    _deployment = VersionedDeployment(
        initial_sidecar_tag="SIDECAR_V1",
        initial_qlora_tag="QLORA_V1",
    )

    logger.info("bridge-rag ready (profile=%s)", settings.profile.name)
    yield

    logger.info("bridge-rag shutting down")
    _orchestrator = None
    _deployment = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Construct and return the FastAPI application.

    Returns:
        A configured ``FastAPI`` instance with ``/health`` and ``/answer``
        routes registered.
    """
    app = FastAPI(
        title="bridge-rag",
        description="Multi-stage neural RAG with grounded span-only generation.",
        version="0.1.0",
        lifespan=_lifespan,
    )

    # ------------------------------------------------------------------
    # GET /  -> the console UI
    # ------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """Serve the single-page console that consumes the /answer SSE stream."""
        return FileResponse(_STATIC_DIR / "index.html")

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Return service liveness, active profile, and current version tag."""
        if _deployment is None:
            raise HTTPException(status_code=503, detail="Service not initialised")
        settings = get_settings()
        profile = settings.profile
        return HealthResponse(
            status="ok",
            profile=profile.name.value,
            blue_version=_deployment.current().tag,
            sidecar_budget_ms=profile.sidecar_budget_ms,
        )

    # ------------------------------------------------------------------
    # POST /answer
    # ------------------------------------------------------------------

    @app.post("/answer")
    async def answer(request: AnswerRequest) -> EventSourceResponse:
        """Stream the grounded answer as SSE events.

        SSE event types
        ~~~~~~~~~~~~~~~
        ``token``    : incremental answer text (data = plain text)
        ``citation`` : resolved citation (data = CitationModel JSON)
        ``done``     : terminal event (data = "done")
        ``error``    : error event (data = error message text)
        """
        if _orchestrator is None:
            raise HTTPException(status_code=503, detail="Service not initialised")

        async def _event_generator() -> AsyncIterator[dict[str, str]]:
            try:
                async for chunk in _orchestrator.answer(
                    request.query, top_k=request.top_k
                ):
                    yield _chunk_to_sse(chunk)
            except Exception as exc:
                logger.error("SSE stream error: %s", exc)
                yield {"event": "error", "data": str(exc)}

        return EventSourceResponse(_event_generator())

    return app


def _chunk_to_sse(chunk: GenerationChunk) -> dict[str, str]:
    """Map a ``GenerationChunk`` to an SSE event dict.

    Args:
        chunk: A streamed chunk from the generator.

    Returns:
        A dict with ``"event"`` and ``"data"`` keys consumed by
        ``EventSourceResponse``.
    """
    if chunk.kind == ChunkKind.TOKEN:
        return {"event": "token", "data": chunk.text}

    if chunk.kind == ChunkKind.CITATION:
        if chunk.citation is not None:
            data = CitationModel.from_citation(chunk.citation).model_dump_json()
        else:
            data = "{}"
        return {"event": "citation", "data": data}

    if chunk.kind == ChunkKind.DONE:
        # Carry the critical-path ledger so the client can render timings.
        return {"event": "done", "data": json.dumps(chunk.meta or {})}

    if chunk.kind == ChunkKind.ERROR:
        return {"event": "error", "data": chunk.text}

    # Unknown kind — surface as error rather than silently dropping.
    return {"event": "error", "data": f"unknown chunk kind: {chunk.kind!r}"}


# Module-level app instance so ``uvicorn bridge_rag.serving.app:app`` works.
app = create_app()
