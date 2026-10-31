"""Sentence-transformers embedding wrapper."""

import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(
        self,
        model_name: str,
        *,
        query_prefix: str = "",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        logger.info("Loading embedding model %s", model_name)
        self._model = SentenceTransformer(model_name, device=device)
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self.model_name = model_name
        self.dimension = len(self._model.encode("dimension probe"))  # also warms up the model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, batch_size=self._batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        vector = self._model.encode(self._query_prefix + query, normalize_embeddings=True, show_progress_bar=False)
        return vector.tolist()

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer.encode(text, add_special_tokens=False, verbose=False))
