"""Small shared helpers."""

import time
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def batched(items: Iterable[T], size: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


class Stopwatch:
    """Monotonic stopwatch reporting milliseconds."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0

    def lap_ms(self) -> float:
        now = time.perf_counter()
        elapsed = (now - self._start) * 1000.0
        self._start = now
        return elapsed
