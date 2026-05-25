"""Gemini-backed translator for higher-quality Chinese -> Vietnamese translation.

Uses the official ``google-genai`` SDK against the public Gemini Developer
API (free tier is generous enough for our use case).

Key behaviour:
  * Returns the same :class:`Translation_Result` shape as
    :class:`GoogleTranslator`, so the pipeline does not need to know
    which backend produced the translation.
  * In-process cache keyed on stripped source text.
  * Soft RPM limiter prevents 429s from the free-tier quota; if a 429
    still occurs we apply exponential backoff just like the deep-translator
    fallback path.
  * The prompt explicitly asks for a translation that is short enough to
    fit a target character budget when ``max_chars_target > 0``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Sequence

from .errors import InvalidConfigError
from .models import Gemini_Config, Text_Segment, Translation_Result
from .text_utils import normalize_text

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    "Bạn là biên dịch viên Hán-Việt. Dịch chuỗi tiếng Trung dưới đây sang "
    "tiếng Việt theo các quy tắc sau:\n"
    "1. Trả về DUY NHẤT bản dịch tiếng Việt, không thêm chú thích, không "
    "đặt dấu nháy, không xuống dòng.\n"
    "2. Giữ giọng văn tự nhiên, phù hợp ngữ cảnh phụ đề video hài.\n"
    "3. Bản dịch nên ngắn gọn, tương đương độ dài câu gốc nếu có thể.\n"
    "{length_constraint}"
    "Câu gốc:\n{source}"
)


class GeminiTranslator:
    """Gemini-backed translator. Same protocol as :class:`GoogleTranslator`."""

    def __init__(
        self,
        config: Gemini_Config,
        max_retries: int = 3,
        backoff_seconds: tuple[float, ...] = (1.0, 2.0, 4.0),
    ) -> None:
        api_key = os.environ.get(config.api_key_env, "").strip()
        if not api_key:
            raise InvalidConfigError(
                f"Gemini translator enabled but environment variable "
                f"{config.api_key_env!r} is empty. Set it with:\n"
                f'  setx {config.api_key_env} "AIza..."\n'
                "and open a new terminal."
            )

        self._config = config
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._cache: dict[str, Translation_Result] = {}

        # Soft RPM limiter: keep the most recent N call timestamps and
        # sleep just long enough to stay under the configured rate.
        self._lock = threading.Lock()
        self._call_history: list[float] = []

        self._client = self._build_client(api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, text: str) -> Translation_Result:
        if not normalize_text(text):
            return Translation_Result(
                source_text=text,
                translated_text=text,
                status="passthrough",
            )

        stripped = normalize_text(text)
        cached = self._cache.get(stripped)
        if cached is not None:
            return cached

        last_error: str | None = None
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            try:
                self._respect_rpm()
                translated = self._call_backend(stripped)
                result = Translation_Result(
                    source_text=text,
                    translated_text=translated,
                    status="translated",
                )
                self._cache[stripped] = result
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "gemini translator attempt %d/%d failed: %s",
                    attempt + 1,
                    attempts,
                    last_error,
                )
                if attempt < self._max_retries and attempt < len(self._backoff):
                    time.sleep(self._backoff[attempt])

        logger.error(
            "gemini translator: giving up after %d attempts (last error: %s)",
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
    # Internals
    # ------------------------------------------------------------------

    def _build_client(self, api_key: str):
        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise InvalidConfigError(
                "google-genai is not installed. Run: pip install google-genai"
            ) from exc
        return genai.Client(api_key=api_key)

    def _respect_rpm(self) -> None:
        """Sleep just enough to keep us below the configured RPM."""
        rpm = self._config.rpm
        if rpm <= 0:
            return
        with self._lock:
            now = time.monotonic()
            # Drop calls older than 60 seconds.
            self._call_history = [t for t in self._call_history if now - t < 60.0]
            if len(self._call_history) >= rpm:
                wait_until = self._call_history[0] + 60.0
                wait = max(0.0, wait_until - now)
                if wait > 0:
                    logger.debug(
                        "gemini: throttling for %.2fs to respect RPM=%d", wait, rpm
                    )
                    time.sleep(wait)
                    now = time.monotonic()
                    self._call_history = [
                        t for t in self._call_history if now - t < 60.0
                    ]
            self._call_history.append(now)

    def _call_backend(self, text: str) -> str:
        from google.genai import types as genai_types  # type: ignore

        prompt = self._build_prompt(text)
        response = self._client.models.generate_content(
            model=self._config.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=512,
            ),
        )
        translated = (response.text or "").strip()
        # Strip any surrounding quotes the model sometimes adds despite the prompt.
        if len(translated) >= 2 and translated[0] in ("\"", "'") and translated[-1] == translated[0]:
            translated = translated[1:-1].strip()
        if not translated:
            raise RuntimeError("gemini returned an empty translation")
        return translated

    def _build_prompt(self, source: str) -> str:
        if self._config.max_chars_target > 0:
            length_constraint = (
                f"4. Cố gắng giữ độ dài ≤ {self._config.max_chars_target} ký tự.\n"
            )
        else:
            length_constraint = ""
        return _PROMPT_TEMPLATE.format(
            length_constraint=length_constraint, source=source
        )
