"""vLLM backend for stage C generation (`fC`) — production path.

``VLLMGenerator`` uses vLLM's ``AsyncLLMEngine`` with automatic prefix caching
(the system prompt lands in the prefix cache so its KV state is reused across
requests — the same "DJ always ready" property as the transformers backend, but
handled by vLLM's scheduler rather than explicit ``past_key_values``).

``build_generator`` is the canonical factory: callers should use it rather than
instantiating backends directly.

Heavy import (``vllm``) is lazy. Unlike the transformers backend, a missing
``vllm`` install raises ``RuntimeError`` immediately — the prod backend must be
explicit; silent degradation would mask deployment misconfiguration.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional, Sequence

from ..config import GeneratorBackend, ModelProfile
from ..types import ChunkKind, GenerationChunk, Span
from .generation import (
    Generator,
    StubGenerator,
    assemble_prompt,
    citations_for,
)

if TYPE_CHECKING:
    pass  # vllm never imported at type-check time

logger = logging.getLogger(__name__)


class VLLMGenerator:
    """vLLM ``AsyncLLMEngine`` generator with LoRA adapter and prefix caching.

    ``prime_system_prompt`` is a no-op: vLLM's automatic prefix caching
    absorbs the system-prompt tokens transparently on first use and reuses
    the cached KV state on subsequent requests.

    Args:
        profile: Active ``ModelProfile`` (carries model name, qlora_rank, etc.).

    Raises:
        RuntimeError: At construction time if ``vllm`` is not installed.
            Install the ``prod`` extra: ``pip install bridge-rag[prod]``.
    """

    def __init__(self, profile: ModelProfile) -> None:
        self._profile = profile
        self._engine: Optional[Any] = None
        self._lora_request: Optional[Any] = None
        self._ensure_vllm_available()

    # ------------------------------------------------------------------
    # Generator protocol
    # ------------------------------------------------------------------

    def prime_system_prompt(self) -> None:
        """No-op: vLLM's automatic prefix caching handles KV reuse.

        The system prompt is cached implicitly on its first appearance in any
        request; subsequent requests reuse the cached KV state.  No explicit
        warm-up step is required.
        """
        self._ensure_engine()
        logger.debug(
            "VLLMGenerator.prime_system_prompt: prefix caching is automatic; no-op"
        )

    async def generate(
        self, query: str, spans: Sequence[Span]
    ) -> AsyncIterator[GenerationChunk]:
        """Stream tokens for *query* grounded in *spans*.

        Builds the prompt exclusively through ``assemble_prompt`` then streams
        from ``AsyncLLMEngine``, yielding TOKEN chunks per token, CITATION
        chunks after the last token, then DONE.

        Args:
            query: The user query string.
            spans: Retrieved spans — the only content that can reach the model.

        Yields:
            ``GenerationChunk`` objects in order: TOKEN*, CITATION*, DONE | ERROR.
        """
        self._ensure_engine()

        prompt = assemble_prompt(query, spans)
        cites = citations_for(spans)
        request_id = str(uuid.uuid4())

        try:
            async for chunk in self._stream_from_engine(prompt, request_id):
                yield chunk
            for cite in cites:
                yield GenerationChunk(kind=ChunkKind.CITATION, citation=cite)
            yield GenerationChunk(kind=ChunkKind.DONE)
        except Exception as exc:
            logger.error("VLLMGenerator streaming error: %s", exc)
            yield GenerationChunk(kind=ChunkKind.ERROR, text=str(exc))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_vllm_available(self) -> None:
        """Raise ``RuntimeError`` immediately if vllm is not importable.

        The prod backend must be explicit — silent fallback would mask a broken
        deployment.
        """
        try:
            import vllm  # type: ignore[import]  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "vllm is not installed. "
                "Install the prod extra to use VLLMGenerator: "
                "pip install bridge-rag[prod]"
            ) from exc

    def _ensure_engine(self) -> None:
        """Lazily construct the ``AsyncLLMEngine`` on first use."""
        if self._engine is not None:
            return
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine  # type: ignore[import]
            from vllm.lora.request import LoRARequest  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "vllm is not installed. Install bridge-rag[prod]."
            ) from exc

        engine_args = AsyncEngineArgs(
            model=self._profile.generator_model,
            enable_lora=True,
            max_loras=1,
            max_lora_rank=self._profile.qlora_rank,
            enable_prefix_caching=True,
            dtype="auto",
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        # LoRARequest references a conceptual adapter path; in production this
        # would point to the actual adapter checkpoint directory.
        self._lora_request = LoRARequest(
            lora_name=f"qlora_r{self._profile.qlora_rank}",
            lora_int_id=1,
            lora_path=f"adapters/{self._profile.generator_model}/qlora",
        )
        logger.info(
            "VLLMGenerator engine started: model=%s, qlora_rank=%d",
            self._profile.generator_model,
            self._profile.qlora_rank,
        )

    async def _stream_from_engine(
        self, prompt: str, request_id: str
    ) -> AsyncIterator[GenerationChunk]:
        """Yield TOKEN chunks from the vLLM async engine."""
        from vllm import SamplingParams  # type: ignore[import]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
        )

        prev_text = ""
        async for request_output in self._engine.generate(  # type: ignore[union-attr]
            prompt,
            sampling_params,
            request_id=request_id,
            lora_request=self._lora_request,
        ):
            if request_output.outputs:
                current_text = request_output.outputs[0].text
                delta = current_text[len(prev_text):]
                if delta:
                    yield GenerationChunk(kind=ChunkKind.TOKEN, text=delta)
                prev_text = current_text


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def build_generator(profile: ModelProfile) -> Generator:
    """Construct the appropriate ``Generator`` backend for *profile*.

    Selection order:
    - ``STUB`` -> ``StubGenerator`` (no model deps, deterministic)
    - ``TRANSFORMERS`` -> ``TransformersGenerator`` (dev/reference backend)
    - ``VLLM`` -> ``VLLMGenerator`` (prod backend; raises if vllm missing)

    Args:
        profile: Active ``ModelProfile`` (carries ``generator_backend`` enum).

    Returns:
        A ``Generator`` instance ready to be primed and used.

    Raises:
        RuntimeError: If backend is ``VLLM`` and ``vllm`` is not installed.
        ValueError: If ``profile.generator_backend`` is an unrecognised value.
    """
    backend = profile.generator_backend

    if backend == GeneratorBackend.STUB:
        logger.info("build_generator: using StubGenerator")
        return StubGenerator()

    if backend == GeneratorBackend.TRANSFORMERS:
        # Lazy import — avoids hard dep on transformers for callers that only
        # use STUB or VLLM.
        from .generation_transformers import TransformersGenerator  # noqa: PLC0415

        logger.info(
            "build_generator: using TransformersGenerator (%s)",
            profile.generator_model,
        )
        return TransformersGenerator(profile)

    if backend == GeneratorBackend.VLLM:
        logger.info(
            "build_generator: using VLLMGenerator (%s)", profile.generator_model
        )
        return VLLMGenerator(profile)

    raise ValueError(
        f"unrecognised generator_backend: {backend!r}. "
        f"Expected one of {list(GeneratorBackend)}."
    )
