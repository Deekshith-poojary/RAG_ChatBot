"""Text normalisation, header/footer removal and language detection."""

import re
import unicodedata
from collections import Counter
from dataclasses import replace

from langdetect import DetectorFactory, LangDetectException, detect

from rag_chatbot.domain import PageContent

DetectorFactory.seed = 0  # deterministic detection

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_LINEBREAK = re.compile(r"(?<=[a-z])-\n(?=[a-z])")
_INLINE_SPACES = re.compile(r"[ \t\u00a0]+")
_BLANK_RUNS = re.compile(r"\n{3,}")
_DIGITS = re.compile(r"\d+")
MAX_MARGIN_LINE_CHARS = 100


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_LINEBREAK.sub("", text)
    lines = (_INLINE_SPACES.sub(" ", line).strip() for line in text.splitlines())
    return _BLANK_RUNS.sub("\n\n", "\n".join(lines)).strip()


def _signature(line: str) -> str:
    """Page-number-insensitive form of a line, so 'Page 3' and 'Page 4' collide.

    Long lines are body text, never headers/footers, so they get an empty (ignored) signature.
    """
    if len(line) > MAX_MARGIN_LINE_CHARS:
        return ""
    return _DIGITS.sub("#", line.lower()).strip()


def _strip_margins(body: str, repeated: set[str], margin_lines: int) -> str:
    lines = body.splitlines()
    non_empty = [i for i, line in enumerate(lines) if line.strip()]
    margin_idx = set(non_empty[:margin_lines] + non_empty[-margin_lines:])
    kept = [line for i, line in enumerate(lines) if not (i in margin_idx and _signature(line) in repeated)]
    return "\n".join(kept).strip()


def strip_repeated_margins(
    pages: list[PageContent],
    *,
    margin_lines: int = 3,
    min_ratio: float = 0.4,
    min_pages: int = 5,
) -> list[PageContent]:
    """Drop lines at the top/bottom of pages that repeat across a large share of the document."""
    if len(pages) < min_pages:
        return pages

    counts: Counter[str] = Counter()
    for page in pages:
        lines = [line for line in page.body.splitlines() if line.strip()]
        margin = {_signature(line) for line in lines[:margin_lines] + lines[-margin_lines:]}
        counts.update(margin)

    threshold = max(2, int(len(pages) * min_ratio))
    repeated = {sig for sig, count in counts.items() if sig and count >= threshold}
    if not repeated:
        return pages
    return [replace(page, body=_strip_margins(page.body, repeated, margin_lines)) for page in pages]


def clean_pages(pages: list[PageContent]) -> list[PageContent]:
    normalized = [
        replace(page, body=normalize_text(page.body), image_text=normalize_text(page.image_text)) for page in pages
    ]
    return strip_repeated_margins(normalized)


def detect_language(pages: list[PageContent], sample_chars: int = 3000) -> str:
    sample = ""
    for page in pages:
        sample += page.text + "\n"
        if len(sample) >= sample_chars:
            break
    try:
        return detect(sample[:sample_chars])
    except LangDetectException:
        return "unknown"
