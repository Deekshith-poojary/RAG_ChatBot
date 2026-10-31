import asyncio

from fastapi import APIRouter

from rag_chatbot.api.dependencies import ContainerDep
from rag_chatbot.api.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(container: ContainerDep) -> HealthResponse:
    ollama_ok = await container.llm.is_healthy()
    chunks = await asyncio.to_thread(container.store.count)
    return HealthResponse(
        status="ok" if ollama_ok else "degraded",
        ollama=ollama_ok,
        llm_model=container.llm.model,
        embedding_model=container.embedder.model_name,
        reranker=container.retriever.reranker_available,
        indexed_chunks=chunks,
    )
