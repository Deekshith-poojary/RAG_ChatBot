"""Query -> embed -> ANN search -> optional rerank."""

from dataclasses import dataclass

from rag_chatbot.domain import RetrievedChunk
from rag_chatbot.embeddings.embedder import Embedder
from rag_chatbot.retrieval.reranker import Reranker
from rag_chatbot.retrieval.vector_store import QdrantVectorStore
from rag_chatbot.utils import Stopwatch


@dataclass(slots=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    embed_ms: float = 0.0
    search_ms: float = 0.0
    rerank_ms: float = 0.0


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: QdrantVectorStore,
        reranker: Reranker | None,
        *,
        candidates: int,
        top_k: int,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._reranker = reranker
        self._candidates = candidates
        self._top_k = top_k

    @property
    def reranker_available(self) -> bool:
        return self._reranker is not None

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        rerank: bool = True,
        doc_ids: list[str] | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self._top_k
        use_reranker = rerank and self._reranker is not None
        watch = Stopwatch()

        vector = self._embedder.embed_query(query)
        embed_ms = watch.lap_ms()

        limit = max(self._candidates, top_k) if use_reranker else top_k
        candidates = self._store.search(vector, limit=limit, doc_ids=doc_ids)
        search_ms = watch.lap_ms()

        if not use_reranker:
            return RetrievalResult(candidates[:top_k], embed_ms, search_ms)

        assert self._reranker is not None
        ranked = self._reranker.rerank(query, candidates, top_k)
        return RetrievalResult(ranked, embed_ms, search_ms, watch.lap_ms())
