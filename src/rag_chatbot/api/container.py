"""Dependency wiring for the API process."""

import logging
from dataclasses import dataclass

from rag_chatbot.config import Settings
from rag_chatbot.embeddings.embedder import Embedder
from rag_chatbot.generation.ollama_client import OllamaClient
from rag_chatbot.generation.rag_service import RagService
from rag_chatbot.ingestion.chunker import TokenChunker
from rag_chatbot.ingestion.extractor import PdfExtractor
from rag_chatbot.ingestion.ocr import PaddleOcrEngine
from rag_chatbot.ingestion.pipeline import IngestionPipeline
from rag_chatbot.ingestion.registry import DocumentRegistry
from rag_chatbot.ingestion.service import DocumentService
from rag_chatbot.retrieval.reranker import FlagReranker, Reranker
from rag_chatbot.retrieval.retriever import Retriever
from rag_chatbot.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Container:
    settings: Settings
    embedder: Embedder
    store: QdrantVectorStore
    retriever: Retriever
    llm: OllamaClient
    rag: RagService
    documents: DocumentService

    async def aclose(self) -> None:
        self.documents.shutdown()
        await self.llm.aclose()
        self.store.close()


def build_container(settings: Settings) -> Container:
    """Create all long-lived components. Blocking: loads models, so run it off the event loop."""
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    embedder = Embedder(
        settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
    )
    store = QdrantVectorStore(
        collection=settings.qdrant_collection,
        vector_size=embedder.dimension,
        path=settings.qdrant_path,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    reranker: Reranker | None = None
    if settings.reranker_enabled:
        reranker = FlagReranker(settings.reranker_model, device=settings.embedding_device)

    retriever = Retriever(
        embedder,
        store,
        reranker,
        candidates=settings.retrieve_candidates,
        top_k=settings.final_top_k,
    )

    llm = OllamaClient(
        settings.ollama_base_url,
        settings.ollama_model,
        api_key=settings.ollama_api_key,
        timeout_s=settings.ollama_timeout_s,
        keep_alive=settings.ollama_keep_alive,
        num_ctx=settings.ollama_num_ctx,
        max_tokens=settings.ollama_max_tokens,
        temperature=settings.ollama_temperature,
    )

    ocr = (
        PaddleOcrEngine(
            det_model=settings.ocr_det_model,
            rec_model=settings.ocr_rec_model,
            det_max_side=settings.ocr_det_max_side,
            min_confidence=settings.ocr_min_confidence,
        )
        if settings.ocr_enabled
        else None
    )
    extractor = PdfExtractor(
        ocr,
        dpi=settings.ocr_dpi,
        min_native_chars=settings.ocr_min_native_chars,
        ocr_embedded_images=settings.ocr_embedded_images,
        min_image_px=settings.ocr_min_image_px,
    )
    chunker = TokenChunker(
        embedder.count_tokens,
        chunk_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        min_chunk_chars=settings.min_chunk_chars,
    )
    registry = DocumentRegistry(settings.registry_path)
    pipeline = IngestionPipeline(
        extractor=extractor,
        chunker=chunker,
        embedder=embedder,
        store=store,
        registry=registry,
        upload_dir=settings.upload_dir,
    )

    return Container(
        settings=settings,
        embedder=embedder,
        store=store,
        retriever=retriever,
        llm=llm,
        rag=RagService(retriever, llm),
        documents=DocumentService(registry=registry, pipeline=pipeline, store=store),
    )
