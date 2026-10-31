"""Extraction and validation of [n] citations in generated answers."""

import re

from rag_chatbot.domain import CitationReport

_CITATION_GROUP = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def check_citations(answer: str, source_count: int) -> CitationReport:
    found: set[int] = set()
    for group in _CITATION_GROUP.findall(answer):
        found.update(int(n) for n in re.findall(r"\d+", group))
    return CitationReport(
        cited=sorted(n for n in found if 1 <= n <= source_count),
        invalid=sorted(n for n in found if not 1 <= n <= source_count),
    )
