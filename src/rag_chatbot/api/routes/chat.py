import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from rag_chatbot.api.dependencies import ContainerDep
from rag_chatbot.api.schemas import ChatRequest, ChatResponse, SearchRequest, SearchResponse
from rag_chatbot.errors import RagError
from rag_chatbot.generation.rag_service import to_sources

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, container: ContainerDep) -> SearchResponse:
    """Retrieval only (no LLM): top-K passages with provenance and scores."""
    retrieval = await container.rag.retrieve(
        request.query, top_k=request.top_k, rerank=request.rerank, doc_ids=request.doc_ids
    )
    return SearchResponse(sources=to_sources(retrieval.chunks))


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, container: ContainerDep) -> ChatResponse:
    """Full RAG answer with sources (PDF name + pages) and citation check."""
    result = await container.rag.answer(
        request.question,
        top_k=request.top_k,
        rerank=request.rerank,
        doc_ids=request.doc_ids,
        model=request.model,
    )
    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        citations=result.citations,
        model=result.model,
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, container: ContainerDep) -> StreamingResponse:
    """NDJSON stream of ``sources``, ``token`` and ``done`` events (``error`` on failure)."""

    async def events() -> AsyncIterator[str]:
        try:
            async for event in container.rag.answer_stream(
                request.question,
                top_k=request.top_k,
                rerank=request.rerank,
                doc_ids=request.doc_ids,
                model=request.model,
            ):
                yield json.dumps(event) + "\n"
        except RagError as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
        except Exception:
            logger.exception("Streaming chat failed")
            yield json.dumps({"type": "error", "message": "Internal error"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")
