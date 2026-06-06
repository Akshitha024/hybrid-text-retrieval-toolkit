"""ColBERT-style late-interaction scorer (token-level).

A lightweight reimplementation of the MaxSim scorer from the ColBERT paper
(Khattab & Zaharia, 2020). We use any sentence-transformers model that exposes
per-token embeddings and compute per-document scores as

    score(q, d) = sum_{i in q_tokens} max_{j in d_tokens} <q_i, d_j>

This is NOT a re-implementation of PLAID/ColBERTv2 retrieval (which uses
quantized centroids and approximate nearest neighbor over token vectors). It
is the brute-force version that runs cleanly on CPU for small corpora and
shows whether token-level late interaction helps on a given dataset before
investing in the production-scale variant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray

from ..types import Doc, Hit
from .base import Index

if TYPE_CHECKING:
    from transformers import AutoModel, AutoTokenizer


class LateInteractionIndex(Index):
    name = "late_interaction"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_doc_tokens: int = 180,
        max_query_tokens: int = 32,
    ) -> None:
        self.model_name = model_name
        self.max_doc_tokens = max_doc_tokens
        self.max_query_tokens = max_query_tokens
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModel | None = None
        self._doc_embeds: list[NDArray[Any]] = []
        self._doc_ids: list[str] = []

    def _load(self) -> tuple[Any, Any]:
        from transformers import AutoModel, AutoTokenizer

        if self._tokenizer is None or self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)  # type: ignore[no-untyped-call]
            model: Any = AutoModel.from_pretrained(self.model_name)
            model.train(False)  # inference mode
            self._model = model
        return self._tokenizer, self._model

    @torch.no_grad()
    def _embed(self, text: str, max_tokens: int) -> NDArray[Any]:
        tok, model = self._load()
        enc = tok(text, truncation=True, max_length=max_tokens, return_tensors="pt")
        out = model(**enc)
        hidden = out.last_hidden_state[0]  # (T, H)
        normed = torch.nn.functional.normalize(hidden, p=2, dim=1)
        return normed.cpu().numpy().astype(np.float32)

    def add(self, docs: list[Doc]) -> None:
        self._doc_ids = [d.doc_id for d in docs]
        self._doc_embeds = []
        for d in docs:
            text = f"{d.title}. {d.text}" if d.title else d.text
            self._doc_embeds.append(self._embed(text, self.max_doc_tokens))

    def query(self, text: str, top_k: int) -> list[Hit]:
        if not self._doc_embeds:
            raise RuntimeError("call .add() before .query()")
        q_emb = self._embed(text, self.max_query_tokens)  # (Q, H)
        scores = np.empty(len(self._doc_embeds), dtype=np.float32)
        for i, d_emb in enumerate(self._doc_embeds):
            sim = q_emb @ d_emb.T  # (Q, D_i)
            scores[i] = float(sim.max(axis=1).sum())
        top_k = min(top_k, len(self._doc_ids))
        idxs = scores.argsort()[-top_k:][::-1]
        return [
            Hit(doc_id=self._doc_ids[int(i)], score=float(scores[i]), rank=rank + 1)
            for rank, i in enumerate(idxs)
        ]

    def size(self) -> int:
        return len(self._doc_ids)
