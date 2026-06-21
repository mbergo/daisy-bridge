"""Event-driven request lifecycle — parallel tracks A/B/C (Section 6).

The pipeline has three concurrent tracks:

* **Track A** (critical path): embed -> ANN -> sidecar -> generate.
  Every stage is timed and recorded in ``RequestCtx``.  Nothing on this track
  blocks on tracks B or C.

* **Track B** (non-blocking fire-and-forget): auth/rate-limit/cache/logging/
  guardrail stubs.  Launched as ``asyncio`` tasks at request start; never
  awaited on the critical path.  If they fail, the failure is logged and ignored.

* **Track C** (generation prep): ``generator.prime_system_prompt()`` kicked off
  concurrently with the critical path so the generator is warm when spans arrive.

After the generator emits DONE, faithfulness checking and feedback logging are
scheduled as background tasks — they do not block the response stream.

Imports of ``retrieval`` and ``provenance`` are deferred inside functions to
avoid import-order coupling with the parallel agent delivering those modules.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, AsyncIterator

from ..types import ChunkKind, GenerationChunk, RequestCtx

if TYPE_CHECKING:
    from ..retrieval.ann import ANNIndex
    from ..sidecar.base import SpanExtractor
    from ..stages.generation import Generator
    from ..stages.perception import Embedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Track B stubs — fire-and-forget, never block the critical path
# ---------------------------------------------------------------------------

async def _stub_auth_check(ctx: RequestCtx) -> None:
    """No-op auth/rate-limit stub (Track B)."""


async def _stub_cache_lookup(ctx: RequestCtx) -> None:
    """No-op cache-lookup stub (Track B)."""


async def _stub_guardrail(ctx: RequestCtx) -> None:
    """No-op guardrail stub (Track B)."""


async def _stub_request_log(ctx: RequestCtx) -> None:
    """No-op request-logging stub (Track B)."""


# ---------------------------------------------------------------------------
# Post-response background tasks
# ---------------------------------------------------------------------------

async def _run_faithfulness_check(ctx: RequestCtx) -> None:
    """Async faithfulness probe — runs after DONE, never on the hot path."""
    try:
        from ..provenance import faithfulness_check  # lazy import  # noqa: PLC0415
        await faithfulness_check(ctx)
    except Exception as exc:
        logger.debug("faithfulness_check skipped or failed: %s", exc)


async def _run_feedback_log(ctx: RequestCtx) -> None:
    """Async feedback logger — runs after DONE, never on the hot path."""
    try:
        logger.debug(
            "feedback_log: request_id=%s critical_path_ms=%.2f timings=%s",
            ctx.request_id,
            ctx.critical_path_ms,
            ctx.timings_ms,
        )
    except Exception as exc:
        logger.debug("feedback_log failed: %s", exc)


def _fire_and_forget(coro: object) -> None:
    """Schedule a coroutine as a background task, logging any exception."""
    import asyncio  # noqa: PLC0415 (already imported at module level, kept for clarity)

    async def _guarded(c: object) -> None:
        try:
            await c  # type: ignore[misc]
        except Exception as exc:
            logger.debug("background task raised: %s", exc)

    asyncio.create_task(_guarded(coro))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_request(
    ctx: RequestCtx,
    *,
    embedder: "Embedder",
    ann_index: "ANNIndex",
    span_extractor: "SpanExtractor",
    generator: "Generator",
) -> AsyncIterator[GenerationChunk]:
    """Drive one request through the three-track event-driven pipeline.

    Track B tasks are created (fire-and-forget) immediately.
    Track C (prime_system_prompt) is launched concurrently with Track A.
    Track A (embed -> ANN -> sidecar) runs on the critical path, timed.
    Generation streams start as soon as spans are available.
    Post-DONE tasks are scheduled but never awaited.

    Args:
        ctx: Mutable per-request context; stages deposit their outputs here.
        embedder: Stage A frozen embedder.
        ann_index: Approximate nearest-neighbour index over the corpus.
        span_extractor: Bridge AB sidecar.
        generator: Stage C generator backend.

    Yields:
        ``GenerationChunk`` objects from the generator in streaming order.
    """
    # ------------------------------------------------------------------
    # Track B — fire-and-forget (never awaited on the critical path)
    # ------------------------------------------------------------------
    _fire_and_forget(_stub_auth_check(ctx))
    _fire_and_forget(_stub_cache_lookup(ctx))
    _fire_and_forget(_stub_guardrail(ctx))
    _fire_and_forget(_stub_request_log(ctx))

    # ------------------------------------------------------------------
    # Track C — generation prep (concurrent with Track A)
    # ------------------------------------------------------------------
    prime_task: asyncio.Task[None] = asyncio.create_task(
        asyncio.to_thread(generator.prime_system_prompt)
    )

    # ------------------------------------------------------------------
    # Track A — critical path: embed -> ANN -> sidecar
    # ------------------------------------------------------------------

    # Stage A: embed
    t0 = time.perf_counter()
    ctx.query_embedding = embedder.embed([ctx.query])[0]
    ctx.mark("embed", (time.perf_counter() - t0) * 1000.0)

    # ANN search
    t1 = time.perf_counter()
    ctx.candidates = ann_index.search(ctx.query_embedding, ctx.top_k)
    ctx.mark("ann", (time.perf_counter() - t1) * 1000.0)

    # Bridge AB: sidecar span extraction
    t2 = time.perf_counter()
    ctx.spans = span_extractor.extract(
        ctx.query, ctx.query_embedding, ctx.candidates
    )
    ctx.mark("sidecar", (time.perf_counter() - t2) * 1000.0)

    # Wait for Track C to finish before calling generate so the generator is
    # warm.  prime_system_prompt is typically fast (it is either a cached KV
    # lookup or a no-op); this join does not materially delay TTFT because it
    # runs concurrently with the embed+ANN+sidecar stages above.
    try:
        await asyncio.wait_for(prime_task, timeout=30.0)
    except (asyncio.TimeoutError, Exception) as exc:
        logger.warning("prime_system_prompt task did not complete cleanly: %s", exc)

    # ------------------------------------------------------------------
    # Generation — stream to caller
    # ------------------------------------------------------------------
    async for chunk in generator.generate(ctx.query, ctx.spans):
        # Suppress the backend's bare DONE; we emit an enriched one below that
        # carries the critical-path ledger (the paper's timing tracks).
        if chunk.kind == ChunkKind.DONE:
            break
        yield chunk
        if chunk.kind == ChunkKind.ERROR:
            break

    yield GenerationChunk(
        kind=ChunkKind.DONE,
        meta={
            "timings_ms": dict(ctx.timings_ms),
            "critical_path_ms": ctx.critical_path_ms,
            "span_count": len(ctx.spans),
            "candidate_count": len(ctx.candidates),
            "doc_ids": sorted({s.doc_id for s in ctx.spans}),
        },
    )

    # ------------------------------------------------------------------
    # Post-response background tasks (never block the response stream)
    # ------------------------------------------------------------------
    _fire_and_forget(_run_faithfulness_check(ctx))
    _fire_and_forget(_run_feedback_log(ctx))
