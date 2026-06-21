"""Console-script entrypoints referenced by ``pyproject.toml``.

bridge-rag-corpus   -> precompute_corpus_main
bridge-rag-bench    -> benchmark_main

Both are thin CLI wrappers; heavy imports (ANNIndex, Corpus, Orchestrator) are
deferred inside functions to avoid import cost on the base install.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import sys
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# bridge-rag-corpus
# ---------------------------------------------------------------------------

def precompute_corpus_main(argv: Optional[list[str]] = None) -> None:
    """Build and persist a corpus embedding store from a JSONL file.

    Reads ``{"doc_id": "...", "text": "..."}`` lines, embeds each document
    using the active profile's embedder, builds an ANNIndex, and saves the
    corpus to ``settings.corpus_path``.

    Usage::

        bridge-rag-corpus --input data/docs.jsonl

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]`` when ``None``).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="bridge-rag-corpus",
        description="Pre-compute embeddings and build the ANN index from a JSONL corpus.",
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Path to a JSONL file with {doc_id, text} records.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        metavar="NAME",
        help="Profile override (dev|prod). Defaults to BRIDGE_RAG_PROFILE_NAME env var.",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    # Apply profile override before importing settings.
    if args.profile:
        import os  # noqa: PLC0415
        os.environ.setdefault("BRIDGE_RAG_PROFILE_NAME", args.profile)

    from .config import get_settings  # noqa: PLC0415
    from .retrieval.corpus import Corpus  # noqa: PLC0415
    from .retrieval.ann import ANNIndex  # noqa: PLC0415
    from .stages.perception import build_embedder  # noqa: PLC0415

    settings = get_settings()
    profile = settings.profile

    logger.info("Profile: %s", profile.name)
    logger.info("Embedder: %s (dim=%d)", profile.embedder_model, profile.embed_dim)

    embedder = build_embedder(profile)
    logger.info("Building corpus from %s …", input_path)
    corpus = Corpus.from_jsonl(input_path, embedder)
    logger.info("Corpus built: %d documents", len(corpus))

    corpus_out = Path(settings.corpus_path)
    # Corpus.save appends nothing; use .npz extension by convention.
    if not corpus_out.suffix:
        corpus_out = corpus_out.with_suffix(".npz")
    corpus.save(corpus_out)
    logger.info("Corpus saved: %s", corpus_out)

    index = ANNIndex.build(corpus)
    logger.info("ANNIndex built (%d vectors)", len(corpus))

    # ANNIndex itself does not persist separately; corpus.save() stores the
    # vectors and ANNIndex.build() reconstructs in O(N) on load.  Log the
    # effective index path so operators know where to point settings.index_path.
    logger.info(
        "Index ready. Set BRIDGE_RAG_CORPUS_PATH=%s to use this corpus.", corpus_out
    )


# ---------------------------------------------------------------------------
# bridge-rag-bench
# ---------------------------------------------------------------------------

def benchmark_main(argv: Optional[list[str]] = None) -> None:
    """Run synthetic requests through the Orchestrator and report latency.

    Collects per-stage timings from ``RequestCtx.timings_ms`` and reports
    p50/p99 for embed, ANN, sidecar, and critical_path.  Compares against
    ``profile.sidecar_budget_ms`` and reports pass/fail — no hard assertions
    so CI on a laptop does not fail on prod numbers.

    Usage::

        bridge-rag-bench --requests 20 --query "What is bridge-rag?"

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]`` when ``None``).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="bridge-rag-bench",
        description="Benchmark the Orchestrator pipeline with synthetic requests.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=10,
        metavar="N",
        help="Number of synthetic requests to run (default: 10).",
    )
    parser.add_argument(
        "--query",
        default="What are the key contributions of bridge-rag?",
        metavar="TEXT",
        help="Query text to repeat across all benchmark requests.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=16,
        metavar="K",
        help="top_k candidates for each request (default: 16).",
    )
    parser.add_argument(
        "--profile",
        default=None,
        metavar="NAME",
        help="Profile override (dev|prod).",
    )
    args = parser.parse_args(argv)

    if args.profile:
        import os  # noqa: PLC0415
        os.environ.setdefault("BRIDGE_RAG_PROFILE_NAME", args.profile)

    asyncio.run(_run_benchmark(args.requests, args.query, args.top_k))


async def _run_benchmark(n: int, query: str, top_k: int) -> None:
    """Async benchmark driver."""
    from .config import get_settings  # noqa: PLC0415
    from .pipeline.orchestrator import Orchestrator  # noqa: PLC0415
    from .types import ChunkKind  # noqa: PLC0415

    settings = get_settings()
    profile = settings.profile
    budget_ms = profile.sidecar_budget_ms

    logger.info(
        "Benchmark: profile=%s n=%d top_k=%d sidecar_budget_ms=%.1f",
        profile.name,
        n,
        top_k,
        budget_ms,
    )

    orchestrator = Orchestrator.from_settings(settings)

    # Timing buckets
    embed_times: list[float] = []
    ann_times: list[float] = []
    sidecar_times: list[float] = []
    critical_times: list[float] = []

    for i in range(n):
        request_id = str(uuid.uuid4())
        # We need access to the RequestCtx timings; thread them back through
        # a thin wrapper that captures the ctx after each run.
        from .types import RequestCtx  # noqa: PLC0415
        import time  # noqa: PLC0415

        ctx = RequestCtx(query=query, request_id=request_id, top_k=top_k)

        from .pipeline.lifecycle import run_request  # noqa: PLC0415

        t_total_start = time.perf_counter()
        async for chunk in run_request(
            ctx,
            embedder=orchestrator._embedder,  # type: ignore[attr-defined]
            ann_index=orchestrator._ann_index,  # type: ignore[attr-defined]
            span_extractor=orchestrator._span_extractor,  # type: ignore[attr-defined]
            generator=orchestrator._generator,  # type: ignore[attr-defined]
        ):
            if chunk.kind in (ChunkKind.DONE, ChunkKind.ERROR):
                break

        embed_times.append(ctx.timings_ms.get("embed", 0.0))
        ann_times.append(ctx.timings_ms.get("ann", 0.0))
        sidecar_times.append(ctx.timings_ms.get("sidecar", 0.0))
        critical_times.append(ctx.critical_path_ms)

        logger.info(
            "request %d/%d  embed=%.1fms ann=%.1fms sidecar=%.1fms critical=%.1fms",
            i + 1,
            n,
            ctx.timings_ms.get("embed", 0.0),
            ctx.timings_ms.get("ann", 0.0),
            ctx.timings_ms.get("sidecar", 0.0),
            ctx.critical_path_ms,
        )

    def _pct(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = max(0, int(len(sorted_data) * p / 100.0) - 1)
        return sorted_data[idx]

    print("\n=== Benchmark Results ===")
    print(f"Profile:          {profile.name}")
    print(f"Requests:         {n}")
    print(f"Query:            {query!r}")
    print()
    print(f"{'Stage':<20} {'p50 (ms)':>10} {'p99 (ms)':>10}")
    print("-" * 42)
    for label, times in [
        ("embed", embed_times),
        ("ann", ann_times),
        ("sidecar", sidecar_times),
        ("critical_path", critical_times),
    ]:
        p50 = _pct(times, 50)
        p99 = _pct(times, 99)
        print(f"{label:<20} {p50:>10.2f} {p99:>10.2f}")

    print()
    sidecar_p99 = _pct(sidecar_times, 99)
    status = "PASS" if sidecar_p99 <= budget_ms else "WARN"
    print(
        f"Sidecar budget: {budget_ms:.1f} ms  |  p99 sidecar: {sidecar_p99:.2f} ms  |  {status}"
    )
    if status == "WARN":
        print(
            "  NOTE: p99 sidecar exceeds the profile budget. "
            "This is informational — not a hard failure. "
            "Tune hardware, batch size, or profile.sidecar_budget_ms accordingly."
        )
    print()
