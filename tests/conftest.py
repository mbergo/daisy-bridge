"""Shared fixtures: a tiny in-memory corpus on the dev profile, no downloads."""

from __future__ import annotations

import pytest

from bridge_rag.config import Settings, ProfileName
from bridge_rag.retrieval.ann import ANNIndex
from bridge_rag.retrieval.corpus import Corpus
from bridge_rag.sidecar.base import DefaultSpanExtractor
from bridge_rag.stages.generation import StubGenerator
from bridge_rag.stages.perception import build_embedder
from bridge_rag.types import Document

_DOCS = [
    Document(
        "d1",
        "The information bottleneck minimizes mutual information with the input. "
        "It preserves the answer signal.",
    ),
    Document(
        "d2",
        "LayerNorm bounds the Jacobian spectral norm. "
        "Residual paths keep the product near identity.",
    ),
    Document("d3", "Unrelated filler about gardening and the weather today."),
]


@pytest.fixture
def settings() -> Settings:
    return Settings(profile_name=ProfileName.DEV)


@pytest.fixture
def embedder(settings: Settings):
    # Force the dependency-free embedder so tests never hit the network.
    return build_embedder(settings.profile, force_fallback=True)


@pytest.fixture
def documents() -> dict[str, Document]:
    return {d.doc_id: d for d in _DOCS}


@pytest.fixture
def corpus(embedder) -> Corpus:
    vecs = embedder.embed([d.text for d in _DOCS])
    docs = tuple(Document(d.doc_id, d.text, v) for d, v in zip(_DOCS, vecs))
    return Corpus(docs, vecs)


@pytest.fixture
def ann_index(corpus) -> ANNIndex:
    return ANNIndex.build(corpus)


@pytest.fixture
def extractor(settings, embedder) -> DefaultSpanExtractor:
    return DefaultSpanExtractor(settings.profile, embedder)


@pytest.fixture
def generator() -> StubGenerator:
    return StubGenerator()
