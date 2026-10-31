"""Deterministic token-based chunking that preserves page provenance."""

import uuid
from bisect import bisect_right
from collections.abc import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_chatbot.domain import Chunk, PageContent

_NAMESPACE = uuid.UUID("6f1d6a52-3c7b-4f0e-9b0d-2f8c1a5e7d11")
_PAGE_SEPARATOR = "\n\n"


class TokenChunker:
    def __init__(
        self,
        token_counter: Callable[[str], int],
        *,
        chunk_tokens: int = 500,
        overlap_tokens: int = 100,
        min_chunk_chars: int = 40,
    ) -> None:
        self._min_chunk_chars = min_chunk_chars
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_tokens,
            chunk_overlap=overlap_tokens,
            length_function=token_counter,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, *, doc_id: str, filename: str, pages: list[PageContent], language: str) -> list[Chunk]:
        page_starts: list[int] = []
        page_numbers: list[int] = []
        parts: list[str] = []
        offset = 0
        for page in pages:
            if not page.text:
                continue
            page_starts.append(offset)
            page_numbers.append(page.page_number)
            parts.append(page.text)
            offset += len(page.text) + len(_PAGE_SEPARATOR)

        if not parts:
            return []

        def page_at(position: int) -> int:
            return page_numbers[max(0, bisect_right(page_starts, position) - 1)]

        chunks: list[Chunk] = []
        cursor = 0
        for doc in self._splitter.create_documents([_PAGE_SEPARATOR.join(parts)]):
            text = doc.page_content
            if len(text) < self._min_chunk_chars:
                continue
            start = doc.metadata.get("start_index", -1)
            start = start if start >= 0 else cursor
            cursor = start
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid5(_NAMESPACE, f"{doc_id}:{index}")),
                    doc_id=doc_id,
                    filename=filename,
                    chunk_index=index,
                    text=text,
                    page_start=page_at(start),
                    page_end=page_at(start + len(text) - 1),
                    language=language,
                )
            )
        return chunks
