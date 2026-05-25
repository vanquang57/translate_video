"""Translate Chinese segment text to Vietnamese with caching and retries.

The implementation wraps :class:`deep_translator.GoogleTranslator` and
adds:
  - in-pipeline cache keyed on the stripped source text,
  - exponential backoff retries on network/quota errors,
  - graceful fallback to the original text when translation fails.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Protocol, Sequence

from .models import Text_Segment, Translation_Result
from .text_utils import normalize_text

logger = logging.getLogger(__name__)


class ITranslator(Protocol):
    def translate(self, text: str) -> Translation_Result:
        ...

    def translate_segments(
        self, segments: Sequence[Text_Segment]
    ) -> dict[str, Translation_Result]:
        ...


class GoogleTranslator:
    """deep-translator backed translator with cache + retry."""

    def __init__(
        self,
        timeout_seconds: float = 10.0,
        max_chars: int = 5000,
        max_retries: int = 3,
        backoff_seconds: tuple[float, ...] = (1.0, 2.0, 4.0),
        source_lang: str = "zh-CN",
        target_lang: str = "vi",
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout_seconds <= 0 or timeout_seconds > 60:
            raise ValueError(
                f"timeout_seconds must be in (0, 60] (got {timeout_seconds})"
            )
        if max_chars < 1:
            raise ValueError(f"max_chars must be >= 1 (got {max_chars})")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0 (got {max_retries})")

        self._timeout = timeout_seconds
        self._max_chars = max_chars
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._source = source_lang
        self._target = target_lang
        self._sleep = sleep
        self._cache: dict[str, Translation_Result] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, text: str) -> Translation_Result:
        # Empty / whitespace-only -> passthrough (Req 5.6).
        if not normalize_text(text):
            return Translation_Result(
                source_text=text,
                translated_text=text,
                status="passthrough",
            )

        stripped = normalize_text(text)
        # Oversize -> untranslated (Req 5.7).
        if len(stripped) > self._max_chars:
            logger.warning(
                "translator: text exceeds max_chars=%d (len=%d), skipping",
                self._max_chars,
                len(stripped),
            )
            return Translation_Result(
                source_text=text,
                translated_text=text,
                status="untranslated",
                error_message=f"text length {len(stripped)} > max_chars {self._max_chars}",
            )

        # Cache hit -> no backend call (Req 5.3).
        cached = self._cache.get(stripped)
        if cached is not None:
            return cached

        # Backend call with retry/backoff.
        last_error: str | None = None
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            try:
                translated = self._call_backend(stripped)
                result = Translation_Result(
                    source_text=text,
                    translated_text=translated,
                    status="translated",
                )
                self._cache[stripped] = result
                return result
            except Exception as exc:  # noqa: BLE001 - intentional broad catch
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "translator attempt %d/%d failed: %s",
                    attempt + 1,
                    attempts,
                    last_error,
                )
                # Sleep only between retries, not after the final one.
                if attempt < self._max_retries and attempt < len(self._backoff):
                    self._sleep(self._backoff[attempt])

        logger.error(
            "translator: giving up after %d attempts (last error: %s)",
            attempts,
            last_error,
        )
        return Translation_Result(
            source_text=text,
            translated_text=text,
            status="untranslated",
            error_message=last_error,
        )

    def translate_segments(
        self, segments: Sequence[Text_Segment]
    ) -> dict[str, Translation_Result]:
        results: dict[str, Translation_Result] = {}
        for seg in segments:
            results[seg.segment_id] = self.translate(seg.canonical_text)
        return results

    # ------------------------------------------------------------------
    # Backend wrapper (split out so unit tests can subclass / monkeypatch)
    # ------------------------------------------------------------------

    def _call_backend(self, text: str) -> str:  # pragma: no cover - thin wrapper
        # Imported lazily so tests that monkeypatch _call_backend never need
        # the real package to be installed.
        from deep_translator import GoogleTranslator as _Backend  # type: ignore

        backend = _Backend(source=self._source, target=self._target)
        translated = backend.translate(text)
        if not isinstance(translated, str):
            raise RuntimeError(f"backend returned non-string: {type(translated)!r}")
        return translated
