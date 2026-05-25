"""Gemini-backed translator for higher-quality Chinese -> Vietnamese translation.

Uses the official ``google-genai`` SDK against the public Gemini Developer
API (free tier is generous enough for our use case).

When ``base_url`` is configured, switches to the OpenAI-compatible client
so that proxies like 9Router (http://localhost:20128/v1) work seamlessly.

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
import re
import threading
import time
from typing import Any, Sequence

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

_BATCH_PROMPT_TEMPLATE = (
    "Bạn là biên dịch viên Hán-Việt. Dịch các chuỗi tiếng Trung dưới đây sang "
    "tiếng Việt theo các quy tắc sau:\n"
    "1. Trả về DUY NHẤT bản dịch tiếng Việt cho mỗi dòng, không thêm chú thích.\n"
    "2. Giữ giọng văn tự nhiên, phù hợp ngữ cảnh phụ đề video hài.\n"
    "3. Mỗi bản dịch trên một dòng riêng, đánh số theo thứ tự (1. bản dịch, 2. bản dịch, ...).\n"
    "4. Không thêm dấu nháy, không thêm chú thích, không thêm dòng trống.\n"
    "{length_constraint}"
    "Các câu gốc:\n{numbered_sources}"
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
        # Step 1: Handle empty input
        if not segments:
            return {}

        results: dict[str, Translation_Result] = {}

        # Step 2: Cache lookup — check each segment's normalized text
        segments_needing_translation: list[Text_Segment] = []
        for seg in segments:
            normalized = normalize_text(seg.canonical_text)

            # Passthrough: empty/whitespace-only text
            if not normalized:
                result = Translation_Result(
                    source_text=seg.canonical_text,
                    translated_text=seg.canonical_text,
                    status="passthrough",
                )
                self._cache[normalized] = result
                results[seg.segment_id] = result
                continue

            cached = self._cache.get(normalized)
            if cached is not None:
                if cached.status in ("translated", "passthrough"):
                    # Use cached result directly
                    results[seg.segment_id] = cached
                    continue
                else:
                    # Status is "untranslated" — re-include for translation
                    del self._cache[normalized]
                    segments_needing_translation.append(seg)
            else:
                segments_needing_translation.append(seg)

        # If all segments were resolved from cache, return early
        if not segments_needing_translation:
            return results

        # Step 3: Deduplicate — collect unique normalized texts
        # Maps normalized_text -> list of segment_ids that share it
        text_to_segment_ids: dict[str, list[str]] = {}
        unique_texts_ordered: list[str] = []
        for seg in segments_needing_translation:
            normalized = normalize_text(seg.canonical_text)
            if normalized not in text_to_segment_ids:
                text_to_segment_ids[normalized] = []
                unique_texts_ordered.append(normalized)
            text_to_segment_ids[normalized].append(seg.segment_id)

        # Step 4: Partition unique texts into batches of batch_size
        batch_size = self._config.batch_size
        batches: list[list[str]] = []
        for i in range(0, len(unique_texts_ordered), batch_size):
            batches.append(unique_texts_ordered[i : i + batch_size])

        # Step 5: Call _translate_batch for each partition
        # Step 6: Store results in cache keyed by normalized text
        for batch in batches:
            batch_results = self._translate_batch(batch)
            for text, result in zip(batch, batch_results):
                self._cache[text] = result

        # Step 7: Merge — assemble output dict keyed by segment_id
        for seg in segments_needing_translation:
            normalized = normalize_text(seg.canonical_text)
            results[seg.segment_id] = self._cache[normalized]

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_client(self, api_key: str) -> Any:
        """Khởi tạo client phù hợp với cấu hình.

        - Nếu base_url được set → dùng OpenAI SDK (tương thích 9Router, OpenRouter, v.v.)
        - Nếu base_url rỗng → dùng Google GenAI SDK (gọi trực tiếp Gemini API)
        """
        if self._config.base_url:
            # --- Chế độ OpenAI-compatible (9Router, OpenRouter, v.v.) ---
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:
                raise InvalidConfigError(
                    "openai package is not installed. Run: pip install openai"
                ) from exc
            logger.info(
                "gemini: sử dụng OpenAI-compatible endpoint: %s (model: %s)",
                self._config.base_url,
                self._config.model,
            )
            self._use_openai = True
            return OpenAI(base_url=self._config.base_url, api_key=api_key)
        else:
            # --- Chế độ Google GenAI SDK (mặc định) ---
            try:
                from google import genai  # type: ignore
            except ImportError as exc:
                raise InvalidConfigError(
                    "google-genai is not installed. Run: pip install google-genai"
                ) from exc
            self._use_openai = False
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
        prompt = self._build_prompt(text)

        if self._use_openai:
            # --- OpenAI-compatible mode (9Router, OpenRouter, v.v.) ---
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
            )
            translated = (response.choices[0].message.content or "").strip()
        else:
            # --- Google GenAI SDK mode ---
            from google.genai import types as genai_types  # type: ignore

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

    def _build_batch_prompt(self, texts: list[str]) -> str:
        """Construct a batch translation prompt with numbered source texts."""
        if self._config.max_chars_target > 0:
            length_constraint = (
                f"5. Mỗi bản dịch không vượt quá {self._config.max_chars_target} ký tự.\n"
            )
        else:
            length_constraint = ""
        numbered_sources = "\n".join(
            f"{i}. {text}" for i, text in enumerate(texts, start=1)
        )
        return _BATCH_PROMPT_TEMPLATE.format(
            length_constraint=length_constraint,
            numbered_sources=numbered_sources,
        )

    def _translate_batch(self, texts: list[str]) -> list[Translation_Result]:
        """Translate a batch of texts in a single API call with fallback.

        Each text in *texts* is already normalized (stripped). On success,
        returns one Translation_Result per text with status "translated".
        On failure (API exception or parse mismatch), falls back to
        individual translate() calls for each text.
        """
        prompt = self._build_batch_prompt(texts)
        last_error: str | None = None
        attempts = self._max_retries + 1

        for attempt in range(attempts):
            try:
                self._respect_rpm()

                if self._use_openai:
                    # --- OpenAI-compatible mode ---
                    response = self._client.chat.completions.create(
                        model=self._config.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=1024 + 256 * len(texts),
                    )
                    raw_text = (response.choices[0].message.content or "").strip()
                else:
                    # --- Google GenAI SDK mode ---
                    from google.genai import types as genai_types  # type: ignore

                    response = self._client.models.generate_content(
                        model=self._config.model,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            temperature=0.2,
                            max_output_tokens=1024 + 256 * len(texts),
                        ),
                    )
                    raw_text = (response.text or "").strip()

                parsed = self._parse_batch_response(raw_text, len(texts))

                if parsed is None:
                    # Count mismatch — fall back to individual translation
                    logger.warning(
                        "gemini batch: parse failure (count mismatch) for batch "
                        "of %d texts, falling back to individual translation",
                        len(texts),
                    )
                    return [self.translate(t) for t in texts]

                # Check for empty parsed items and handle them
                results: list[Translation_Result] = []
                empty_indices: list[int] = []

                for i, translated in enumerate(parsed):
                    if not translated.strip():
                        empty_indices.append(i)
                        results.append(None)  # type: ignore[arg-type]  # placeholder
                    else:
                        results.append(
                            Translation_Result(
                                source_text=texts[i],
                                translated_text=translated,
                                status="translated",
                            )
                        )

                # Retry empty items individually
                if empty_indices:
                    logger.warning(
                        "gemini batch: %d/%d items were empty, retrying individually",
                        len(empty_indices),
                        len(texts),
                    )
                    for idx in empty_indices:
                        results[idx] = self.translate(texts[idx])

                return results

            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "gemini batch attempt %d/%d failed for batch of %d texts: %s",
                    attempt + 1,
                    attempts,
                    len(texts),
                    last_error,
                )
                if attempt < self._max_retries and attempt < len(self._backoff):
                    time.sleep(self._backoff[attempt])

        # All retries exhausted — fall back to individual translation
        logger.warning(
            "gemini batch: all %d attempts failed (last error: %s) for batch "
            "of %d texts, falling back to individual translation",
            attempts,
            last_error,
            len(texts),
        )
        return [self.translate(t) for t in texts]

    def _parse_batch_response(
        self, response: str, expected_count: int
    ) -> list[str] | None:
        """Parse a numbered-list batch response into individual translations.

        Splits the response on line boundaries, skips empty lines, strips
        the numeric prefix (pattern ``^\\d+\\.\\s+``), strips surrounding
        whitespace and quotes from each translation.

        Returns ``None`` if the number of parsed translations does not equal
        *expected_count*, signaling that the caller should fall back to
        individual translation.
        """
        _numbered_prefix_re = re.compile(r"^\d+\.\s+")

        lines = response.splitlines()
        translations: list[str] = []

        for line in lines:
            # Skip empty lines (after stripping whitespace)
            stripped_line = line.strip()
            if not stripped_line:
                continue

            # Strip the numeric prefix if present (e.g., "1. ", "12. ")
            text = _numbered_prefix_re.sub("", stripped_line)

            # Strip leading/trailing whitespace
            text = text.strip()

            # Strip surrounding quotes (single or double)
            if (
                len(text) >= 2
                and text[0] in ('"', "'")
                and text[-1] == text[0]
            ):
                text = text[1:-1].strip()

            translations.append(text)

        if len(translations) != expected_count:
            return None

        return translations
