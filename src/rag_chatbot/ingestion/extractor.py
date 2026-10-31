"""Native text extraction (PyMuPDF) with OCR for scanned pages and embedded images."""

import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from rag_chatbot.domain import PageContent, TextSource
from rag_chatbot.errors import PdfExtractionError
from rag_chatbot.ingestion.ocr import OcrEngine

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


class PdfExtractor:
    """Extracts per-page text.

    Pages with a usable text layer use it directly (plus OCR of large embedded images);
    pages without one (scans) are rendered and OCR'd as a whole.
    """

    def __init__(
        self,
        ocr: OcrEngine | None,
        *,
        dpi: int = 200,
        min_native_chars: int = 40,
        ocr_embedded_images: bool = True,
        min_image_px: int = 150,
    ) -> None:
        self._ocr = ocr
        self._dpi = dpi
        self._min_native_chars = min_native_chars
        self._ocr_embedded_images = ocr_embedded_images
        self._min_image_px = min_image_px

    def extract(self, pdf_path: Path, on_progress: ProgressCallback | None = None) -> list[PageContent]:
        try:
            doc = pymupdf.open(pdf_path)
        except Exception as exc:
            raise PdfExtractionError(f"Cannot open PDF: {exc}") from exc

        with doc:
            if doc.needs_pass:
                raise PdfExtractionError("PDF is password protected")
            total = doc.page_count
            seen_xrefs: set[int] = set()  # repeated logos/backgrounds are OCR'd only once
            pages: list[PageContent] = []
            for index, page in enumerate(doc):
                pages.append(self._extract_page(doc, page, seen_xrefs))
                if on_progress:
                    on_progress(index + 1, total)
        return pages

    def _extract_page(self, doc: pymupdf.Document, page: pymupdf.Page, seen_xrefs: set[int]) -> PageContent:
        number = page.number + 1
        native = page.get_text("text", sort=True).strip()

        if len(native) >= self._min_native_chars:
            image_text = self._ocr_images(doc, page, seen_xrefs) if self._can_ocr_images else ""
            return PageContent(number, native, TextSource.MIXED if image_text else TextSource.NATIVE, image_text)

        if self._ocr is None:
            return PageContent(number, native, TextSource.NATIVE)

        ocr_text = self._safe_ocr(self._render_page(page), f"page {number}")
        return PageContent(number, ocr_text or native, TextSource.OCR)

    @property
    def _can_ocr_images(self) -> bool:
        return self._ocr is not None and self._ocr_embedded_images

    def _render_page(self, page: pymupdf.Page) -> np.ndarray:
        pix = page.get_pixmap(dpi=self._dpi, colorspace=pymupdf.csRGB, alpha=False)
        return cv2.cvtColor(_pixmap_to_array(pix), cv2.COLOR_RGB2BGR)

    def _ocr_images(self, doc: pymupdf.Document, page: pymupdf.Page, seen_xrefs: set[int]) -> str:
        texts: list[str] = []
        for info in page.get_images(full=True):
            xref, width, height = info[0], info[2], info[3]
            if xref in seen_xrefs or min(width, height) < self._min_image_px:
                continue
            seen_xrefs.add(xref)
            try:
                image = cv2.cvtColor(_pixmap_to_array(_to_rgb(pymupdf.Pixmap(doc, xref))), cv2.COLOR_RGB2BGR)
            except Exception:
                logger.debug("Skipping unreadable image xref=%s", xref, exc_info=True)
                continue
            text = self._safe_ocr(image, f"image xref={xref} on page {page.number + 1}")
            if text:
                texts.append(text)
        return "\n".join(texts)

    def _safe_ocr(self, image: np.ndarray, what: str) -> str:
        assert self._ocr is not None
        try:
            return self._ocr.read(image)
        except Exception:
            logger.exception("OCR failed for %s", what)
            return ""


def _to_rgb(pix: pymupdf.Pixmap) -> pymupdf.Pixmap:
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    if pix.colorspace is None or pix.colorspace.n != 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    return pix


def _pixmap_to_array(pix: pymupdf.Pixmap) -> np.ndarray:
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
