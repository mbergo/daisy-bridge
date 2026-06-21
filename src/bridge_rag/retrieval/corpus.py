"""Immutable precomputed embedding store.

`Corpus` is built once — either from a JSONL file (offline indexing) or
incrementally via `add` — then frozen.  It holds no faiss logic; it is the
pure vector table that `ANNIndex` and `DynamicIndex` read from.

Persistence is via `np.savez` so the store loads with zero model dependencies:
ids and texts are stored as object arrays, embeddings as float32.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from ..stages.perception import Embedder
from ..types import Document

logger = logging.getLogger(__name__)

_BATCH_SIZE = 64  # embed in batches to bound peak memory


class Corpus:
    """Immutable after build: documents and their embeddings in aligned order.

    The store is deliberately separate from any index structure so it can be
    shared across an `ANNIndex` (static, global) and multiple per-request
    `DynamicIndex` instances without copying the vectors.
    """

    def __init__(
        self,
        documents: Sequence[Document],
        embeddings: np.ndarray,
    ) -> None:
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"document count {len(documents)} != embedding rows {embeddings.shape[0]}"
            )
        # Freeze as tuples so nothing downstream can mutate the store.
        self._documents: tuple[Document, ...] = tuple(documents)
        # Ensure contiguous float32 — required by faiss and numpy matmul alike.
        self._embeddings: np.ndarray = np.ascontiguousarray(
            embeddings, dtype=np.float32
        )
        # doc_id -> row-index for O(1) lookup.
        self._index: dict[str, int] = {
            doc.doc_id: i for i, doc in enumerate(self._documents)
        }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_jsonl(cls, path: str | Path, embedder: Embedder) -> "Corpus":
        """Build a corpus by reading `{"doc_id", "text"}` lines and embedding.

        The file is consumed once; embeddings are produced in batches of
        `_BATCH_SIZE` to avoid loading all texts into memory simultaneously.
        """
        path = Path(path)
        docs: list[Document] = []
        all_embeddings: list[np.ndarray] = []

        def _embed_batch(batch: list[Document]) -> None:
            texts = [d.text for d in batch]
            vecs = embedder.embed(texts)  # (n, d) float32, L2-normalised
            all_embeddings.append(vecs)

        batch: list[Document] = []
        with path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{lineno}: invalid JSON — {exc}"
                    ) from exc
                doc = Document(doc_id=str(obj["doc_id"]), text=str(obj["text"]))
                docs.append(doc)
                batch.append(doc)
                if len(batch) >= _BATCH_SIZE:
                    _embed_batch(batch)
                    batch = []

        if batch:
            _embed_batch(batch)

        if not docs:
            # Return a valid but empty corpus with a (0, d) embedding matrix.
            empty = np.zeros((0, embedder.dim), dtype=np.float32)
            return cls([], empty)

        embeddings = np.concatenate(all_embeddings, axis=0)
        logger.info("corpus built: %d documents, dim=%d", len(docs), embeddings.shape[1])
        return cls(docs, embeddings)

    def add(self, doc: Document, embedder: Embedder) -> "Corpus":
        """Return a NEW corpus with `doc` appended (immutable update pattern).

        The caller must supply an `embedder` so the new vector is produced in
        the same space as the existing ones.  A new `Corpus` is returned rather
        than mutating `self`.
        """
        if doc.doc_id in self._index:
            raise ValueError(f"duplicate doc_id: {doc.doc_id!r}")
        vec = embedder.embed([doc.text])  # (1, d)
        new_embeddings = np.concatenate([self._embeddings, vec], axis=0)
        embedded_doc = Document(
            doc_id=doc.doc_id,
            text=doc.text,
            embedding=vec[0],
        )
        return Corpus(list(self._documents) + [embedded_doc], new_embeddings)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def documents(self) -> tuple[Document, ...]:
        """Ordered tuple of all documents — index aligned with `embeddings`."""
        return self._documents

    @property
    def embeddings(self) -> np.ndarray:
        """Shape (N, d) float32, L2-normalised — index aligned with `documents`."""
        return self._embeddings

    def get_by_doc_id(self, doc_id: str) -> Document | None:
        """O(1) lookup by doc_id."""
        idx = self._index.get(doc_id)
        return self._documents[idx] if idx is not None else None

    def __len__(self) -> int:
        return len(self._documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self._documents)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist to a `.npz` archive (ids, texts, vectors).

        Texts are retained so span provenance can be validated against the
        original source without re-fetching.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ids = np.array([d.doc_id for d in self._documents], dtype=object)
        texts = np.array([d.text for d in self._documents], dtype=object)
        np.savez(str(path), ids=ids, texts=texts, vectors=self._embeddings)
        logger.info("corpus saved: %s (%d docs)", path, len(self._documents))

    @classmethod
    def load(cls, path: str | Path) -> "Corpus":
        """Reload from a `.npz` archive produced by `save`.

        No embedder required — vectors are stored directly.
        """
        path = Path(path)
        data = np.load(str(path), allow_pickle=True)
        ids: list[str] = [str(x) for x in data["ids"]]
        texts: list[str] = [str(x) for x in data["texts"]]
        vectors: np.ndarray = data["vectors"].astype(np.float32)
        docs = [
            Document(doc_id=doc_id, text=text, embedding=vectors[i])
            for i, (doc_id, text) in enumerate(zip(ids, texts))
        ]
        logger.info("corpus loaded: %s (%d docs)", path, len(docs))
        return cls(docs, vectors)
