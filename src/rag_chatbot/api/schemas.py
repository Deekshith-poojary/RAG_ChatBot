"""API request/response models."""

from pydantic import BaseModel, Field

from rag_chatbot.domain import CitationReport, Source


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    rerank: bool = True
    doc_ids: list[str] | None = None


class SearchResponse(BaseModel):
    sources: list[Source]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    rerank: bool = True
    doc_ids: list[str] | None = None
    model: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    citations: CitationReport
    model: str


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    system: str | None = None
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)


class GenerateResponse(BaseModel):
    model: str
    response: str


class OllamaModel(BaseModel):
    name: str
    size_bytes: int | None = None


class UploadResult(BaseModel):
    filename: str
    doc_id: str | None = None
    status: str  # "queued" | "duplicate" | "rejected"
    error: str | None = None


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    ollama: bool
    llm_model: str
    embedding_model: str
    reranker: bool
    indexed_chunks: int
