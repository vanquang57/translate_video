"""Thin tqdm wrapper used to report pipeline progress."""

from __future__ import annotations

from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - hard dependency, but keep gentle failure
    tqdm = None  # type: ignore[assignment]


class ProgressReporter:
    """Pipeline-friendly progress bar.

    Supports re-using the same instance for several stages: call
    :meth:`start` for each stage with the new total + name.
    """

    def __init__(self) -> None:
        self._bar: Any | None = None
        self._stage: str = ""

    def start(self, total: int, stage_name: str) -> None:
        self.close()
        self._stage = stage_name
        if tqdm is None:
            return
        self._bar = tqdm(total=total, desc=stage_name, unit="frame", leave=False)

    def update(self, n: int = 1) -> None:
        if self._bar is not None:
            self._bar.update(n)

    def set_stage(self, name: str) -> None:
        self._stage = name
        if self._bar is not None:
            self._bar.set_description(name)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
