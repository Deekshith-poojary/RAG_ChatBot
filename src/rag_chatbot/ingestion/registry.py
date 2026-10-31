"""JSON-file backed registry of uploaded documents and their ingestion state."""

import json
import logging
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_chatbot.domain import ACTIVE_STATUSES, DocumentRecord, DocumentStatus
from rag_chatbot.errors import DocumentNotFoundError

logger = logging.getLogger(__name__)

PROGRESS_SAVE_INTERVAL_S = 1.0
REPLACE_ATTEMPTS = 5
REPLACE_BACKOFF_S = 0.1


class DocumentRegistry:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._records: dict[str, DocumentRecord] = {}
        self._last_save = 0.0
        self._load()
        self._fail_interrupted()

    def get(self, doc_id: str) -> DocumentRecord:
        with self._lock:
            try:
                return self._records[doc_id].model_copy()
            except KeyError:
                raise DocumentNotFoundError(doc_id) from None

    def find(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            record = self._records.get(doc_id)
            return record.model_copy() if record else None

    def list_all(self) -> list[DocumentRecord]:
        with self._lock:
            return sorted((r.model_copy() for r in self._records.values()), key=lambda r: r.created_at, reverse=True)

    def put(self, record: DocumentRecord) -> None:
        with self._lock:
            self._records[record.doc_id] = record
            self._save()

    def update(self, doc_id: str, **fields: Any) -> DocumentRecord:
        with self._lock:
            record = self._records.get(doc_id)
            if record is None:
                raise DocumentNotFoundError(doc_id)
            updated = record.model_copy(update={**fields, "updated_at": datetime.now(UTC)})
            self._records[doc_id] = updated
            self._save(force="status" in fields)  # progress-only updates are written at most once per second
            return updated.model_copy()

    def delete(self, doc_id: str) -> None:
        with self._lock:
            if self._records.pop(doc_id, None) is None:
                raise DocumentNotFoundError(doc_id)
            self._save()

    def _load(self) -> None:
        if not self._path.exists():
            return
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        self._records = {item["doc_id"]: DocumentRecord.model_validate(item) for item in raw}

    def _save(self, *, force: bool = True) -> None:
        if not force and time.monotonic() - self._last_save < PROGRESS_SAVE_INTERVAL_S:
            return
        payload = [r.model_dump(mode="json") for r in self._records.values()]
        try:
            self._write(payload)
        except OSError:
            # In-memory state stays correct; a failed snapshot must not abort an ingestion.
            logger.exception("Could not persist the document registry to %s", self._path)
            return
        self._last_save = time.monotonic()

    def _write(self, payload: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for attempt in range(1, REPLACE_ATTEMPTS + 1):
            try:
                os.replace(tmp, self._path)
                return
            except PermissionError:  # Windows: target briefly locked by a scanner or sync client
                if attempt == REPLACE_ATTEMPTS:
                    raise
                time.sleep(REPLACE_BACKOFF_S * attempt)

    def _fail_interrupted(self) -> None:
        """Ingestion runs in-process, so anything still active after a restart was interrupted."""
        stale = [r for r in self._records.values() if r.status in ACTIVE_STATUSES]
        for record in stale:
            logger.warning("Marking interrupted ingestion as failed: %s", record.filename)
            self._records[record.doc_id] = record.model_copy(
                update={"status": DocumentStatus.FAILED, "error": "Interrupted by server restart"}
            )
        if stale:
            self._save()
