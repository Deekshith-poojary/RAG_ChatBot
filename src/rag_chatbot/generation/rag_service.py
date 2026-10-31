"""RAG orchestration: retrieve -> prompt -> generate -> verify citations."""

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from rag_chatbot.domain import CitationReport, RetrievedChunk, Source
from rag_chatbot.generation.citations import check_citations
from rag_chatbot.generation.ollama_client import OllamaClient
from rag_chatbot.generation.prompts import NOT_FOUND_MESSAGE, build_messages
from rag_chatbot.retrieval.retriever import RetrievalResult, Retriever
from rag_chatbot.utils import Stopwatch

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 300


@dataclass(slots=True)
class RagAnswer:
    answer: str
    sources: list[Source]
    citations: CitationReport
    model: str


def to_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    return [
        Source(
            index=i,
            doc_id=c.doc_id,
            filename=c.filename,
            page_start=c.page_start,
            page_end=c.page_end,
            score=round(c.score, 4),
            snippet=c.text[:SNIPPET_CHARS],
        )
        for i, c in enumerate(chunks, start=1)
    ]


def _mark_cited(sources: list[Source], citations: CitationReport) -> None:
    for source in sources:
        source.cited = source.index in citations.cited


class RagService:
    def __init__(self, retriever: Retriever, llm: OllamaClient) -> None:
        self._retriever = retriever
        self._llm = llm

    async def retrieve(
        self, question: str, *, top_k: int | None, rerank: bool, doc_ids: list[str] | None
    ) -> RetrievalResult:
        # Embedding, ANN search and reranking are blocking; keep them off the event loop.
        return await asyncio.to_thread(self._retriever.retrieve, question, top_k=top_k, rerank=rerank, doc_ids=doc_ids)

    async def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        rerank: bool = True,
        doc_ids: list[str] | None = None,
        model: str | None = None,
    ) -> RagAnswer:
        watch = Stopwatch()
        retrieval = await self.retrieve(question, top_k=top_k, rerank=rerank, doc_ids=doc_ids)
        sources = to_sources(retrieval.chunks)
        used_model = model or self._llm.model

        if not sources:
            return self._finish(NOT_FOUND_MESSAGE, sources, retrieval, watch, used_model)

        llm_watch = Stopwatch()
        answer = await self._llm.chat(build_messages(question, retrieval.chunks), model=model)
        return self._finish(answer.strip(), sources, retrieval, watch, used_model, llm_ms=llm_watch.elapsed_ms())

    async def answer_stream(
        self,
        question: str,
        *,
        top_k: int | None = None,
        rerank: bool = True,
        doc_ids: list[str] | None = None,
        model: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield ``sources``, then ``token`` events, then a final ``done`` event."""
        watch = Stopwatch()
        retrieval = await self.retrieve(question, top_k=top_k, rerank=rerank, doc_ids=doc_ids)
        sources = to_sources(retrieval.chunks)
        used_model = model or self._llm.model
        yield {"type": "sources", "sources": [s.model_dump() for s in sources]}

        if not sources:
            final = self._finish(NOT_FOUND_MESSAGE, sources, retrieval, watch, used_model)
            yield {"type": "token", "text": final.answer}
            yield self._done_event(final)
            return

        llm_watch = Stopwatch()
        parts: list[str] = []
        async for piece in self._llm.chat_stream(build_messages(question, retrieval.chunks), model=model):
            parts.append(piece)
            yield {"type": "token", "text": piece}

        final = self._finish(
            "".join(parts).strip(), sources, retrieval, watch, used_model, llm_ms=llm_watch.elapsed_ms()
        )
        yield self._done_event(final)

    @staticmethod
    def _finish(
        answer: str,
        sources: list[Source],
        retrieval: RetrievalResult,
        watch: Stopwatch,
        model: str,
        *,
        llm_ms: float = 0.0,
    ) -> RagAnswer:
        citations = check_citations(answer, len(sources))
        _mark_cited(sources, citations)
        # Stage timings are for diagnosing slow requests; they are logged, not returned to clients.
        logger.info(
            "Answered in %.0f ms: embed %.0f, search %.0f, rerank %.0f, llm %.0f (%d sources, invalid citations: %s)",
            watch.elapsed_ms(),
            retrieval.embed_ms,
            retrieval.search_ms,
            retrieval.rerank_ms,
            llm_ms,
            len(sources),
            citations.invalid,
        )
        return RagAnswer(answer, sources, citations, model)

    @staticmethod
    def _done_event(result: RagAnswer) -> dict[str, Any]:
        return {"type": "done", "citations": result.citations.model_dump(), "model": result.model}
