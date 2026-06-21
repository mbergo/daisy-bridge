"""Serving-layer orchestrator — assembles components for the active profile.

``Orchestrator`` is the single entry point for the serving layer (FastAPI app,
benchmark scripts, etc.).  It wires together:

* Stage A embedder (frozen, from ``build_embedder``)
* Corpus + ANNIndex (lazy; loaded from paths in ``Settings`` when present,
  otherwise an empty corpus)
* Bridge AB sidecar (``DefaultSpanExtractor``)
* Stage C generator (via ``build_generator`` factory)

``answer()`` is the sole public method: it builds a ``RequestCtx`` and delegates
to ``lifecycle.run_request``, keeping the serving wrapper separate from the
differentiable training graph in ``pipeline/composition.py``.

Heavy imports (``ANNIndex``, ``Corpus``) are deferred inside methods to avoid
import-order coupling with the retrieval agent.
"""

from __future__ import annotations

import logging
import uuid
from typing import AsyncIterator, Optional

from ..config import ModelProfile, Settings, get_settings
from ..sidecar.base import DefaultSpanExtractor, SpanExtractor
from ..stages.generation import Generator
from ..stages.perception import Embedder, build_embedder
from ..types import GenerationChunk, RequestCtx

logger = logging.getLogger(__name__)


class Orchestrator:
    """Assembled pipeline for the active ``ModelProfile``.

    Args:
        embedder: Frozen Stage A embedder.
        ann_index: ANN index over the corpus (may be empty).
        span_extractor: Bridge AB sidecar.
        generator: Stage C generator backend.
        profile: The ``ModelProfile`` that configured this orchestrator.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        ann_index: object,  # ANNIndex — typed as object to avoid eager import
        span_extractor: SpanExtractor,
        generator: Generator,
        profile: ModelProfile,
    ) -> None:
        self._embedder = embedder
        self._ann_index = ann_index
        self._span_extractor = span_extractor
        self._generator = generator
        self._profile = profile

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "Orchestrator":
        """Build a fully wired ``Orchestrator`` from ``Settings``.

        Loads the corpus and ANN index from ``settings.corpus_path`` /
        ``settings.index_path`` when those paths exist; otherwise uses an empty
        corpus so the service starts without a pre-built index.

        Args:
            settings: Runtime settings; defaults to ``get_settings()`` when
                ``None``.

        Returns:
            A ready-to-use ``Orchestrator`` instance.
        """
        if settings is None:
            settings = get_settings()

        profile = settings.profile

        # Stage A — frozen embedder
        embedder = build_embedder(profile)

        # Corpus + ANN index (lazy imports to avoid order coupling)
        ann_index = cls._build_index(settings, embedder, profile)

        # Bridge AB sidecar
        span_extractor: SpanExtractor = DefaultSpanExtractor(profile, embedder)

        # Stage C generator
        from ..stages.generation_vllm import build_generator  # noqa: PLC0415
        generator = build_generator(profile)

        logger.info(
            "Orchestrator ready: profile=%s backend=%s",
            profile.name,
            profile.generator_backend,
        )
        return cls(
            embedder=embedder,
            ann_index=ann_index,
            span_extractor=span_extractor,
            generator=generator,
            profile=profile,
        )

    @staticmethod
    def _build_index(settings: Settings, embedder: Embedder, profile: ModelProfile) -> object:
        """Construct the ANNIndex, loading from disk when available."""
        from ..retrieval.ann import ANNIndex  # noqa: PLC0415
        from ..retrieval.corpus import Corpus  # noqa: PLC0415
        import os  # noqa: PLC0415

        corpus_path = settings.corpus_path

        # Prefer a pre-built .npz (fast, no re-embed). The corpus saver writes
        # "<corpus_path>.npz", so check that explicitly first.
        npz_path = corpus_path if corpus_path.endswith(".npz") else corpus_path + ".npz"
        if os.path.exists(npz_path):
            try:
                corpus = Corpus.load(npz_path)
                logger.info("Corpus loaded from %s (%d docs)", npz_path, len(corpus))
                return ANNIndex.build(corpus)
            except Exception as exc:
                logger.warning("Failed to load corpus npz %s: %s", npz_path, exc)

        # Otherwise, build from a raw .jsonl by re-embedding (point at the file).
        if os.path.exists(corpus_path) and corpus_path.endswith(".jsonl"):
            try:
                corpus = Corpus.from_jsonl(corpus_path, embedder)
                logger.info("Corpus embedded from %s (%d docs)", corpus_path, len(corpus))
                return ANNIndex.build(corpus)
            except Exception as exc:
                logger.warning("Failed to embed corpus jsonl %s: %s", corpus_path, exc)

        # Fallback: empty corpus — service starts, no retrieval results.
        logger.info(
            "No corpus found at %s; starting with empty index", corpus_path
        )
        empty_corpus = Corpus([], __import__("numpy").zeros((0, profile.embed_dim), dtype="float32"))
        return ANNIndex.build(empty_corpus)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def answer(
        self,
        query: str,
        *,
        top_k: int = 16,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[GenerationChunk]:
        """Answer *query* by running the full event-driven pipeline.

        Builds a ``RequestCtx`` and delegates to ``lifecycle.run_request``.

        Args:
            query: The user question.
            top_k: Number of ANN candidates to retrieve.
            request_id: Optional caller-supplied request identifier; a UUID is
                generated when not provided.

        Yields:
            ``GenerationChunk`` objects in streaming order: TOKEN*, CITATION*,
            DONE | ERROR.
        """
        rid = request_id or str(uuid.uuid4())
        ctx = RequestCtx(query=query, request_id=rid, top_k=top_k)

        from ..pipeline.lifecycle import run_request  # noqa: PLC0415

        async for chunk in run_request(
            ctx,
            embedder=self._embedder,
            ann_index=self._ann_index,  # type: ignore[arg-type]
            span_extractor=self._span_extractor,
            generator=self._generator,
        ):
            yield chunk
