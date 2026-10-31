"""Cross-encoder reranking with FlagEmbedding."""

import logging
from dataclasses import replace
from typing import Protocol

from rag_chatbot.domain import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]: ...


class FlagReranker:
    def __init__(self, model_name: str, *, device: str | None = None, max_length: int = 512) -> None:
        from FlagEmbedding import FlagReranker as _FlagReranker

        logger.info("Loading reranker %s", model_name)
        self._max_length = max_length
        self._model = _FlagReranker(model_name, use_fp16=_wants_fp16(device), devices=device)

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = self._model.compute_score(
            [[query, c.text] for c in candidates], max_length=self._max_length, normalize=True
        )
        if not isinstance(scores, list):  # single pair returns a bare float
            scores = [scores]
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [replace(chunk, score=float(score)) for chunk, score in ranked[:top_n]]


def _wants_fp16(device: str | None) -> bool:
    import torch

    return (device or ("cuda" if torch.cuda.is_available() else "cpu")).startswith("cuda")
