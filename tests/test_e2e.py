"""The keystone E2E test: the full Eq.6 chain on the dev profile, CPU, no torch.

embed -> ANN -> sidecar -> fB -> generate -> cite. Asserts the properties the
whole architecture exists to provide:
  - spans are extracted WITH provenance,
  - the generator context is a subset of those spans (fB chokepoint),
  - an answer streams,
  - every citation resolves to real document offsets.
"""

from __future__ import annotations

import pytest

from bridge_rag.pipeline.lifecycle import run_request
from bridge_rag.provenance import faithfulness_check, resolve_citations
from bridge_rag.stages.inference import InferenceStage
from bridge_rag.types import ChunkKind, RequestCtx


@pytest.mark.asyncio
async def test_full_chain(settings, embedder, ann_index, extractor, generator, documents):
    ctx = RequestCtx(query="what bounds the Jacobian spectral norm?", request_id="e2e", top_k=3)
    chunks = []
    async for ch in run_request(
        ctx,
        embedder=embedder,
        ann_index=ann_index,
        span_extractor=extractor,
        generator=generator,
    ):
        chunks.append(ch)

    text = "".join(c.text for c in chunks if c.kind == ChunkKind.TOKEN)
    cited = [c.citation for c in chunks if c.kind == ChunkKind.CITATION]

    # 1. spans extracted, with provenance, under the span cap
    assert len(ctx.spans) > 0
    assert len(ctx.spans) <= settings.profile.max_spans

    # 2. fB chokepoint: everything downstream is a provenanced span
    passed = InferenceStage(documents).forward(ctx.spans)
    assert passed == tuple(ctx.spans)

    # 3. an answer streamed
    assert text.strip() != ""

    # 4. citations resolve to real offsets
    resolved = resolve_citations(ctx.spans, documents)
    for c in resolved:
        doc = documents[c.span.doc_id]
        assert doc.text[c.span.char_start : c.span.char_end] == c.span.text

    # 5. the answer is grounded — every emitted marker maps to a real span
    faith = await faithfulness_check(text, ctx.spans, documents)
    assert faith["grounded"] is True

    # 6. critical-path timing recorded (embed/ann/sidecar)
    assert ctx.critical_path_ms >= 0.0
    assert "sidecar" in ctx.timings_ms


@pytest.mark.asyncio
async def test_relevant_doc_outranks_filler(embedder, ann_index, extractor, generator):
    # The Jacobian query should surface d2 (Jacobian/LayerNorm), not the filler.
    ctx = RequestCtx(query="LayerNorm Jacobian spectral norm", request_id="rank", top_k=3)
    async for _ in run_request(
        ctx,
        embedder=embedder,
        ann_index=ann_index,
        span_extractor=extractor,
        generator=generator,
    ):
        pass
    top_doc = ctx.spans[0].doc_id
    assert top_doc == "d2"
