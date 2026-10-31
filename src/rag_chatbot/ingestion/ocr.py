"""OCR engines."""

import logging
import os
import threading
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class OcrEngine(Protocol):
    def read(self, image: np.ndarray) -> str:
        """Return recognised text for a BGR/RGB image array, lines separated by newlines."""
        ...


class PaddleOcrEngine:
    """PaddleOCR wrapper. Models are loaded lazily on first use and calls are serialised."""

    def __init__(
        self,
        *,
        det_model: str,
        rec_model: str,
        det_max_side: int = 1280,
        min_confidence: float = 0.5,
    ) -> None:
        self._det_model = det_model
        self._rec_model = rec_model
        self._det_max_side = det_max_side
        self._min_confidence = min_confidence
        self._engine = None
        self._lock = threading.Lock()

    def _load(self):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        logger.info("Loading PaddleOCR (%s + %s)", self._det_model, self._rec_model)
        return PaddleOCR(
            text_detection_model_name=self._det_model,
            text_recognition_model_name=self._rec_model,
            text_det_limit_side_len=self._det_max_side,
            text_det_limit_type="max",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,  # oneDNN/PIR path raises NotImplementedError on some Windows CPUs
        )

    def read(self, image: np.ndarray) -> str:
        with self._lock:
            if self._engine is None:
                self._engine = self._load()
            results = self._engine.predict(image)

        lines: list[str] = []
        for result in results:
            for text, score in zip(result["rec_texts"], result["rec_scores"], strict=False):
                if score >= self._min_confidence and text.strip():
                    lines.append(text.strip())
        return "\n".join(lines)
