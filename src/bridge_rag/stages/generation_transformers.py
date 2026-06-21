"""Transformers/PEFT backend for stage C generation (`fC`).

Dev/reference backend. Loads a causal-LM via HuggingFace ``transformers`` and
attaches a QLoRA adapter via ``peft``. The system-prompt KV-cache is pre-built
once (``prime_system_prompt``) and reused on every request so the system-prompt
tokens are never re-computed — the "DJ always ready" property from Section 6.3.

Heavy imports (``torch``, ``transformers``, ``peft``) are lazy: if they are
absent the class falls back internally to ``StubGenerator`` so the file compiles
and the dev profile runs anywhere.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional, Sequence

from ..config import ModelProfile
from ..types import ChunkKind, GenerationChunk, Span
from .generation import (
    Generator,
    StubGenerator,
    assemble_prompt,
    citations_for,
)

if TYPE_CHECKING:
    pass  # torch / transformers / peft never imported at type-check time

logger = logging.getLogger(__name__)

_SENTINEL = object()  # marks end of streamer queue


class TransformersGenerator:
    """HuggingFace Transformers causal-LM generator with QLoRA + KV-cache priming.

    Implements the ``Generator`` protocol. Falls back to ``StubGenerator``
    transparently when ``torch``/``transformers``/``peft`` are not installed.

    Args:
        profile: Active ``ModelProfile`` (carries model name, qlora_rank, etc.).
    """

    def __init__(self, profile: ModelProfile) -> None:
        self._profile = profile
        self._fallback: Optional[StubGenerator] = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._past_key_values: Any = None  # cached KV state for the system prompt
        self._primed: bool = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Generator protocol
    # ------------------------------------------------------------------

    def prime_system_prompt(self) -> None:
        """Tokenize the system prompt and cache its ``past_key_values``.

        The warm KV state is reused by every subsequent ``generate`` call so
        the system-prompt tokens are never re-computed (Section 6.3).  If heavy
        deps are absent, initialises the stub fallback instead.
        """
        if self._primed:
            return

        with self._lock:
            if self._primed:
                return
            try:
                self._load_model()
                self._build_kv_cache()
                self._primed = True
                logger.info(
                    "TransformersGenerator primed for model %s (qlora_rank=%d)",
                    self._profile.generator_model,
                    self._profile.qlora_rank,
                )
            except Exception as exc:
                logger.warning(
                    "transformers/peft unavailable (%s); using StubGenerator fallback",
                    exc,
                )
                self._fallback = StubGenerator(primed=True)
                self._primed = True

    async def generate(
        self, query: str, spans: Sequence[Span]
    ) -> AsyncIterator[GenerationChunk]:
        """Stream tokens for *query* grounded in *spans*.

        Builds the prompt exclusively through ``assemble_prompt`` (the only
        permitted path to generator input) then streams tokens from a background
        thread via ``TextIteratorStreamer``, followed by CITATION chunks and a
        terminal DONE chunk.
        """
        if not self._primed:
            self.prime_system_prompt()

        if self._fallback is not None:
            async for chunk in self._fallback.generate(query, spans):
                yield chunk
            return

        prompt = assemble_prompt(query, spans)
        cites = citations_for(spans)

        try:
            async for chunk in self._stream_tokens(prompt):
                yield chunk
            for cite in cites:
                yield GenerationChunk(kind=ChunkKind.CITATION, citation=cite)
            yield GenerationChunk(kind=ChunkKind.DONE)
        except Exception as exc:
            logger.error("TransformersGenerator streaming error: %s", exc)
            yield GenerationChunk(kind=ChunkKind.ERROR, text=str(exc))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load model + tokenizer + QLoRA adapter.

        Raises ``ImportError`` if the heavy deps are absent (caller catches and
        falls back to stub).
        """
        import torch  # type: ignore[import]
        from peft import LoraConfig, get_peft_model  # type: ignore[import]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

        model_name = self._profile.generator_model
        logger.info("loading tokenizer: %s", model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info("loading model: %s", model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        lora_config = LoraConfig(
            r=self._profile.qlora_rank,
            lora_alpha=self._profile.qlora_rank * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._model = get_peft_model(self._model, lora_config)
        self._model.eval()

    def _build_kv_cache(self) -> None:
        """Run a forward pass over the system prompt and cache ``past_key_values``."""
        import torch  # type: ignore[import]
        from .generation import SYSTEM_PROMPT

        inputs = self._tokenizer(SYSTEM_PROMPT, return_tensors="pt")
        with torch.no_grad():
            out = self._model(**inputs, use_cache=True)
        self._past_key_values = out.past_key_values
        logger.debug("system-prompt KV cache built (%d tokens)", inputs["input_ids"].shape[1])

    async def _stream_tokens(self, prompt: str) -> AsyncIterator[GenerationChunk]:
        """Drive ``TextIteratorStreamer`` in a background thread, yield tokens async."""
        import torch  # type: ignore[import]
        from transformers import TextIteratorStreamer  # type: ignore[import]

        loop = asyncio.get_running_loop()
        token_queue: queue.Queue[Any] = queue.Queue()
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        def _generate_thread() -> None:
            try:
                with torch.no_grad():
                    self._model.generate(
                        input_ids,
                        past_key_values=self._past_key_values,
                        streamer=streamer,
                        max_new_tokens=512,
                        do_sample=False,
                    )
            except Exception as exc:
                token_queue.put(exc)
            finally:
                token_queue.put(_SENTINEL)

        thread = threading.Thread(target=_generate_thread, daemon=True)
        thread.start()

        for text in streamer:
            chunk = await loop.run_in_executor(None, lambda t=text: t)  # type: ignore[misc]
            if chunk:
                yield GenerationChunk(kind=ChunkKind.TOKEN, text=chunk)

        thread.join(timeout=60.0)

        # Drain any error placed by the thread.
        try:
            item = token_queue.get_nowait()
            if item is not _SENTINEL and isinstance(item, Exception):
                raise item
        except queue.Empty:
            pass
