"""Application settings, loaded from environment variables and an optional ``.env`` file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Storage
    data_dir: Path = Path("data")
    qdrant_collection: str = "pdf_chunks"
    qdrant_url: str | None = None  # None -> embedded local store under data_dir
    qdrant_api_key: str | None = None
    max_upload_mb: int = 500

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_api_key: str | None = None
    ollama_timeout_s: float = 120.0
    ollama_keep_alive: str = "30m"
    ollama_num_ctx: int = 4096
    ollama_max_tokens: int = 400
    ollama_temperature: float = 0.1

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = 32
    embedding_device: str | None = None  # None -> auto (cuda if available)

    # Reranker
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"

    # Chunking (tokens are counted with the embedding model's tokenizer)
    chunk_tokens: int = 500
    chunk_overlap_tokens: int = 100
    min_chunk_chars: int = 40

    # Extraction / OCR
    ocr_enabled: bool = True
    ocr_det_model: str = "PP-OCRv5_mobile_det"  # server-size models take ~20 s/page on CPU, mobile ~1 s
    ocr_rec_model: str = "en_PP-OCRv5_mobile_rec"  # language-specific, e.g. latin_PP-OCRv5_mobile_rec
    ocr_det_max_side: int = 1280
    ocr_dpi: int = 200
    ocr_min_native_chars: int = 40
    ocr_embedded_images: bool = True
    ocr_min_image_px: int = 150
    ocr_min_confidence: float = 0.5

    # Retrieval
    retrieve_candidates: int = 15
    final_top_k: int = 5

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "documents.json"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
