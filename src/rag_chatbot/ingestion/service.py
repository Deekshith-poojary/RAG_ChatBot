"""Document lifecycle: register uploads, schedule ingestion, delete."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag_chatbot.domain import ACTIVE_STATUSES, DocumentRecord, DocumentStatus
from rag_chatbot.errors import DocumentBusyError
from rag_chatbot.ingestion.pipeline import IngestionPipeline
from rag_chatbot.ingestion.registry import DocumentRegistry
from rag_chatbot.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

DOC_ID_LENGTH = 16


class DocumentService:
    def __init__(
        self,
        *,
        registry: DocumentRegistry,
        pipeline: IngestionPipeline,
        store: QdrantVectorStore,
    ) -> None:
        self._registry = registry
        self._pipeline = pipeline
        self._store = store
        # OCR engines and the local vector store are not safe to drive concurrently: one job at a time.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")

    def register_upload(
        self, tmp_path: Path, *, sha256: str, filename: str, size_bytes: int
    ) -> tuple[DocumentRecord, bool]:
        """Store an uploaded PDF and queue it. Returns (record, queued); identical content is not re-ingested."""
        doc_id = sha256[:DOC_ID_LENGTH]
        existing = self._registry.find(doc_id)
        if existing and existing.status is not DocumentStatus.FAILED:
            tmp_path.unlink(missing_ok=True)
            return existing, False

        os.replace(tmp_path, self._pipeline.pdf_path(doc_id))
        record = DocumentRecord(doc_id=doc_id, filename=filename, size_bytes=size_bytes)
        self._registry.put(record)
        self._executor.submit(self._pipeline.run, doc_id)
        return record, True

    def list_documents(self) -> list[DocumentRecord]:
        return self._registry.list_all()

    def get_document(self, doc_id: str) -> DocumentRecord:
        return self._registry.get(doc_id)

    def delete_document(self, doc_id: str) -> None:
        record = self._registry.get(doc_id)
        if record.status in ACTIVE_STATUSES:
            raise DocumentBusyError(f"'{record.filename}' is still being processed")
        self._store.delete_document(doc_id)
        self._pipeline.pdf_path(doc_id).unlink(missing_ok=True)
        self._registry.delete(doc_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
