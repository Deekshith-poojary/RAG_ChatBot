"""Qdrant vector store (embedded local file mode by default, or a Qdrant server via URL)."""

import logging
import threading
from collections.abc import Sequence
from pathlib import Path

from qdrant_client import QdrantClient, models

from rag_chatbot.domain import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

HNSW_M = 16
HNSW_EF_CONSTRUCT = 128
HNSW_EF_SEARCH = 128


class QdrantVectorStore:
    def __init__(
        self,
        *,
        collection: str,
        vector_size: int,
        path: Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._collection = collection
        self._is_remote = url is not None
        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            if path is None:
                raise ValueError("Either path or url is required")
            path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(path))
        self._lock = threading.RLock()  # local mode is not guaranteed thread-safe
        self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        with self._lock:
            if self._client.collection_exists(self._collection):
                existing = self._client.get_collection(self._collection).config.params.vectors
                size = existing.size if isinstance(existing, models.VectorParams) else None
                if size != vector_size:
                    raise RuntimeError(
                        f"Collection '{self._collection}' has vector size {size}, but the embedding model produces "
                        f"{vector_size}. Use a different data dir/collection or re-ingest."
                    )
                return
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT),
            )
            if self._is_remote:  # payload indexes are not supported in local mode
                self._client.create_payload_index(
                    self._collection, field_name="doc_id", field_schema=models.PayloadSchemaType.KEYWORD
                )

    def upsert_chunks(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        points = [
            models.PointStruct(id=chunk.chunk_id, vector=list(vector), payload=chunk.payload())
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        with self._lock:
            self._client.upsert(collection_name=self._collection, points=points)

    def search(
        self, vector: Sequence[float], *, limit: int, doc_ids: Sequence[str] | None = None
    ) -> list[RetrievedChunk]:
        query_filter = None
        if doc_ids:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))]
            )
        with self._lock:
            response = self._client.query_points(
                collection_name=self._collection,
                query=list(vector),
                limit=limit,
                query_filter=query_filter,
                search_params=models.SearchParams(hnsw_ef=HNSW_EF_SEARCH),
                with_payload=True,
            )
        return [
            RetrievedChunk(
                chunk_id=str(point.id),
                doc_id=point.payload["doc_id"],
                filename=point.payload["filename"],
                page_start=point.payload["page_start"],
                page_end=point.payload["page_end"],
                text=point.payload["text"],
                score=point.score,
            )
            for point in response.points
            if point.payload
        ]

    def delete_document(self, doc_id: str) -> None:
        selector = models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
        )
        with self._lock:
            self._client.delete(collection_name=self._collection, points_selector=selector)

    def count(self) -> int:
        with self._lock:
            return self._client.count(collection_name=self._collection, exact=True).count

    def close(self) -> None:
        with self._lock:
            self._client.close()
