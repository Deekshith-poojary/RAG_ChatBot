"""Ingestion pipeline: extract -> clean -> chunk -> embed -> index."""

import logging
from pathlib import Path

from rag_chatbot.domain import DocumentStatus, TextSource
from rag_chatbot.embeddings.embedder import Embedder
from rag_chatbot.errors import PdfExtractionError
from rag_chatbot.ingestion.chunker import TokenChunker
from rag_chatbot.ingestion.cleaning import clean_pages, detect_language
from rag_chatbot.ingestion.extractor import PdfExtractor
from rag_chatbot.ingestion.registry import DocumentRegistry
from rag_chatbot.retrieval.vector_store import QdrantVectorStore
from rag_chatbot.utils import batched

logger = logging.getLogger(__name__)

EMBED_BATCH = 128
EXTRACTION_SHARE = 0.6  # share of overall progress attributed to extraction/OCR


class IngestionPipeline:
    def __init__(
        self,
        *,
        extractor: PdfExtractor,
        chunker: TokenChunker,
        embedder: Embedder,
        store: QdrantVectorStore,
        registry: DocumentRegistry,
        upload_dir: Path,
    ) -> None:
        self._extractor = extractor
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._registry = registry
        self._upload_dir = upload_dir

    def pdf_path(self, doc_id: str) -> Path:
        return self._upload_dir / f"{doc_id}.pdf"

    def run(self, doc_id: str) -> None:
        """Ingest a stored PDF. Never raises; failures are recorded on the document."""
        try:
            self._run(doc_id)
        except Exception as exc:
            logger.exception("Ingestion failed for %s", doc_id)
            self._registry.update(doc_id, status=DocumentStatus.FAILED, error=str(exc)[:500])

    def _run(self, doc_id: str) -> None:
        record = self._registry.update(doc_id, status=DocumentStatus.EXTRACTING, progress=0.0, error=None)
        logger.info("Ingesting %s (%s)", record.filename, doc_id)

        def on_page(done: int, total: int) -> None:
            self._registry.update(doc_id, pages=total, progress=EXTRACTION_SHARE * done / total)

        raw_pages = self._extractor.extract(self.pdf_path(doc_id), on_page)
        pages = clean_pages(raw_pages)
        ocr_pages = sum(1 for p in raw_pages if p.source is TextSource.OCR)

        language = detect_language(pages)
        chunks = self._chunker.chunk(doc_id=doc_id, filename=record.filename, pages=pages, language=language)
        if not chunks:
            raise PdfExtractionError("No extractable text found in the PDF")

        self._registry.update(
            doc_id,
            status=DocumentStatus.EMBEDDING,
            pages=len(pages),
            ocr_pages=ocr_pages,
            language=language,
            chunks=len(chunks),
        )

        self._store.delete_document(doc_id)  # makes re-ingestion idempotent
        done = 0
        for batch in batched(chunks, EMBED_BATCH):
            self._store.upsert_chunks(batch, self._embedder.embed_documents([c.text for c in batch]))
            done += len(batch)
            self._registry.update(doc_id, progress=EXTRACTION_SHARE + (1 - EXTRACTION_SHARE) * done / len(chunks))

        self._registry.update(doc_id, status=DocumentStatus.READY, progress=1.0)
        logger.info("Ingested %s: %d pages (%d OCR), %d chunks", record.filename, len(pages), ocr_pages, len(chunks))
