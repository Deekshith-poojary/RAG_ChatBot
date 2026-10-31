"""Core data types shared across ingestion, retrieval, generation and the API."""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class TextSource(StrEnum):
    NATIVE = "native"
    OCR = "ocr"
    MIXED = "mixed"  # native text plus OCR of embedded images


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"


ACTIVE_STATUSES = frozenset({DocumentStatus.QUEUED, DocumentStatus.EXTRACTING, DocumentStatus.EMBEDDING})


@dataclass(slots=True, frozen=True)
class PageContent:
    """Text of one PDF page. ``body`` is page text; ``image_text`` is OCR of embedded images."""

    page_number: int  # 1-based
    body: str
    source: TextSource
    image_text: str = ""

    @property
    def text(self) -> str:
        return f"{self.body}\n\n{self.image_text}".strip() if self.image_text else self.body


@dataclass(slots=True, frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    language: str

    def payload(self) -> dict[str, str | int]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "language": self.language,
            "text": self.text,
        }


@dataclass(slots=True, frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    filename: str
    page_start: int
    page_end: int
    text: str
    score: float


def _utcnow() -> datetime:
    return datetime.now(UTC)


class DocumentRecord(BaseModel):
    """Persisted ingestion state of an uploaded PDF (also returned by the API)."""

    doc_id: str
    filename: str
    size_bytes: int
    status: DocumentStatus = DocumentStatus.QUEUED
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    pages: int = 0
    ocr_pages: int = 0
    chunks: int = 0
    language: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class Source(BaseModel):
    """A passage used to build an answer, with provenance."""

    index: int  # 1-based, matches [n] citations in the answer
    doc_id: str
    filename: str
    page_start: int
    page_end: int
    score: float
    snippet: str
    cited: bool = False


class CitationReport(BaseModel):
    cited: list[int] = Field(default_factory=list)
    invalid: list[int] = Field(default_factory=list)  # cited indexes with no matching source
