import hashlib
import uuid
from pathlib import Path

import anyio
from fastapi import APIRouter, UploadFile, status

from rag_chatbot.api.dependencies import ContainerDep
from rag_chatbot.api.schemas import UploadResult
from rag_chatbot.domain import DocumentRecord

router = APIRouter(prefix="/documents", tags=["documents"])

_CHUNK_SIZE = 1024 * 1024
_PDF_MAGIC = b"%PDF-"


class _UploadRejectedError(Exception):
    pass


async def _save_pdf(file: UploadFile, directory: Path, max_bytes: int) -> tuple[Path, str, int]:
    """Stream an upload to a temp file, validating the PDF header and size. Returns (path, sha256, size)."""
    tmp_path = directory / f".upload-{uuid.uuid4().hex}.part"
    digest = hashlib.sha256()
    size = 0
    try:
        async with await anyio.open_file(tmp_path, "wb") as out:
            while chunk := await file.read(_CHUNK_SIZE):
                if size == 0 and not chunk.startswith(_PDF_MAGIC):
                    raise _UploadRejectedError("Not a PDF file")
                size += len(chunk)
                if size > max_bytes:
                    raise _UploadRejectedError(f"File exceeds the {max_bytes // (1024 * 1024)} MB limit")
                digest.update(chunk)
                await out.write(chunk)
        if size == 0:
            raise _UploadRejectedError("Empty file")
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return tmp_path, digest.hexdigest(), size


@router.post("", response_model=list[UploadResult], status_code=status.HTTP_202_ACCEPTED)
async def upload_documents(files: list[UploadFile], container: ContainerDep) -> list[UploadResult]:
    """Upload one or more PDFs. Ingestion (text extraction, OCR, embedding) runs in the background."""
    results: list[UploadResult] = []
    for file in files:
        filename = Path(file.filename or "document.pdf").name[:255]  # never used to build paths
        if not filename.lower().endswith(".pdf"):
            results.append(UploadResult(filename=filename, status="rejected", error="Only .pdf files are accepted"))
            continue
        try:
            tmp_path, sha256, size = await _save_pdf(
                file, container.settings.upload_dir, container.settings.max_upload_bytes
            )
        except _UploadRejectedError as exc:
            results.append(UploadResult(filename=filename, status="rejected", error=str(exc)))
            continue
        record, queued = container.documents.register_upload(
            tmp_path, sha256=sha256, filename=filename, size_bytes=size
        )
        results.append(
            UploadResult(filename=filename, doc_id=record.doc_id, status="queued" if queued else "duplicate")
        )
    return results


@router.get("", response_model=list[DocumentRecord])
def list_documents(container: ContainerDep) -> list[DocumentRecord]:
    return container.documents.list_documents()


@router.get("/{doc_id}", response_model=DocumentRecord)
def get_document(doc_id: str, container: ContainerDep) -> DocumentRecord:
    return container.documents.get_document(doc_id)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, container: ContainerDep) -> None:
    container.documents.delete_document(doc_id)
