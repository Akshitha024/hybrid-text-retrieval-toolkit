"""Dense bi-encoder index over FAISS IndexFlatIP.

We use L2-normalized embeddings so inner product == cosine similarity. The
flat index is exact and trades index-build time for query time; for corpora
below ~1M docs it is the right default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..types import Doc, Hit
from .base import Index

if TYPE_CHECKING:
    import faiss
    from sentence_transformers import SentenceTransformer


class DenseIndex(Index):
    name = "dense"

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        max_seq_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self._model: SentenceTransformer | None = None
        self._faiss: faiss.IndexFlatIP | None = None
        self._doc_ids: list[str] = []

    def _load(self) -> SentenceTransformer:
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            self._model.max_seq_length = self.max_seq_length
        return self._model

    def _encode(self, texts: list[str]) -> NDArray[Any]:
        m = self._load()
        emb = m.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 256,
        )
        return emb.astype(np.float32)

    def add(self, docs: list[Doc]) -> None:
        import faiss

        self._doc_ids = [d.doc_id for d in docs]
        texts = [_doc_text(d) for d in docs]
        emb = self._encode(texts)
        self._faiss = faiss.IndexFlatIP(emb.shape[1])
        self._faiss.add(emb)

    def query(self, text: str, top_k: int) -> list[Hit]:
        return self.query_batch([text], top_k)[0]

    def query_batch(self, texts: list[str], top_k: int) -> list[list[Hit]]:
        if self._faiss is None:
            raise RuntimeError("call .add() before .query()")
        q = self._encode(texts)
        top_k = min(top_k, len(self._doc_ids))
        scores, idxs = self._faiss.search(q, top_k)
        out: list[list[Hit]] = []
        for s, ix in zip(scores, idxs, strict=True):
            out.append(
                [
                    Hit(doc_id=self._doc_ids[int(i)], score=float(sc), rank=rank + 1)
                    for rank, (i, sc) in enumerate(zip(ix, s, strict=True))
                    if i >= 0
                ]
            )
        return out

    def size(self) -> int:
        return len(self._doc_ids)


def _doc_text(d: Doc) -> str:
    return f"{d.title}. {d.text}" if d.title else d.text
