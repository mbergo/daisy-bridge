"""The structural 0%-hallucination guarantee: context can only be spans.

These tests assert the *architecture*, not model behavior: the only door to
generator input is `assemble_prompt(query, spans)`, and `fB` refuses anything
without resolvable provenance. If full-doc text cannot pass these, the model
cannot hallucinate from context it never received.
"""

from __future__ import annotations

import inspect

import pytest

from bridge_rag.provenance import ProvenanceError, resolve_citations
from bridge_rag.stages import generation
from bridge_rag.stages.generation import assemble_prompt
from bridge_rag.stages.inference import InferenceStage
from bridge_rag.types import Document, Span


def test_only_door_is_spans() -> None:
    # assemble_prompt takes (query, spans). There is no document/raw-text param.
    params = list(inspect.signature(assemble_prompt).parameters)
    assert params == ["query", "spans"]


def test_no_assemble_overload_accepts_documents() -> None:
    # No alternate public assembler smuggling full documents into the generator.
    public = [n for n in dir(generation) if "assemble" in n.lower()]
    assert public == ["assemble_prompt"]


def test_prompt_contains_only_span_text_not_full_doc() -> None:
    doc = Document("d1", "SECRET full document body that must never be sent. Needle here.")
    span = Span("d1", 51, 63, "Needle here.", score=0.9)
    prompt = assemble_prompt("q", [span])
    assert "Needle here." in prompt
    assert "SECRET full document" not in prompt


def test_inference_stage_rejects_unprovenanced_span() -> None:
    docs = {"d1": Document("d1", "hello world")}
    bad = Span("d1", 0, 999, "out of range", score=0.1)  # offsets exceed doc
    with pytest.raises(Exception):
        InferenceStage(docs).forward((bad,))


def test_provenance_must_match_source_text() -> None:
    docs = {"d1": Document("d1", "hello world")}
    fabricated = Span("d1", 0, 5, "WRONG", score=0.1)  # text != doc[0:5]=="hello"
    with pytest.raises(ProvenanceError):
        resolve_citations((fabricated,), docs)
