"""Translator sử dụng OpenAI-compatible API (9Router, OpenRouter, v.v.)

Gọi LLM qua endpoint OpenAI-compatible để dịch Trung → Việt.
Khi API thất bại → tự động fallback về Google Translate (deep-translator, miễn phí).

Tính năng:
  * Batch translation: gộp nhiều text trong 1 lần gọi API (cấu hình batch_size)
  * Cache: không dịch lại text đã dịch trước đó
  * RPM limiter: giới hạn số request/phút tránh bị rate limit
  * Fallback: nếu LLM API lỗi → dùng Google Translate miễn phí
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Sequence

from .errors import InvalidConfigError
from .models import Gemini_Config, Text_Segment, Translation_Result
from .text_utils import normalize_text

logger = logging.getLogger(__name__)

# --- Prompt cho dịch 1 câu ---
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

# --- Prompt cho dịch batch (nhiều câu 1 lần) ---
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
    """Translator dùng OpenAI-compatible API với fallback về Google Translate.

    Yêu cầu:
      - base_url phải được cấu hình (ví dụ: http://localhost:20128/v1)
      - model là tên model/combo trên proxy (ví dụ: "free")
      - api_key_env chứa tên biến môi trường có API key
    """

    def __init__(
        self,
        config: Gemini_Config,
        max_retries: int = 3,
        backoff_seconds: tuple[float, ...] = (1.0, 2.0, 4.0),
    ) -> None:
        api_key = os.environ.get(config.api_key_env, "").strip()
        if not api_key:
            raise InvalidConfigError(
                f"Translator cần API key nhưng biến môi trường "
                f"{config.api_key_env!r} đang rỗng. Set bằng:\n"
                f'  setx {config.api_key_env} "your-key"\n'
                "rồi mở terminal mới."
            )

        if not config.base_url:
            raise InvalidConfigError(
                "base_url phải được cấu hình để dùng OpenAI-compatible API.\n"
                "Ví dụ: base_url: \"http://localhost:20128/v1\" (9Router)"
            )

        self._config = config
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._cache: dict[str, Translation_Result] = {}

        # RPM limiter
        self._lock = threading.Lock()
        self._call_history: list[float] = []

        # Khởi tạo OpenAI client
        self._client = self._build_client(api_key)

        # Google Translate fallback (lazy init)
        self._google_fallback = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, text: str) -> Translation_Result:
        """Dịch 1 câu. Thử LLM API trước, nếu lỗi thì fallback Google Translate."""
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

        # Thử gọi LLM API
        last_error: str | None = None
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            try:
                self._respect_rpm()
                translated = self._call_llm(stripped)
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
                    "LLM API attempt %d/%d thất bại: %s",
                    attempt + 1,
                    attempts,
                    last_error,
                )
                if attempt < self._max_retries and attempt < len(self._backoff):
                    time.sleep(self._backoff[attempt])

        # LLM API thất bại hoàn toàn → fallback Google Translate
        logger.warning(
            "LLM API thất bại sau %d lần thử, fallback sang Google Translate: %s",
            attempts,
            last_error,
        )
        return self._translate_google_fallback(text, stripped)

    def translate_segments(
        self, segments: Sequence[Text_Segment]
    ) -> dict[str, Translation_Result]:
        """Dịch nhiều segments với batch translation + cache + fallback."""
        if not segments:
            return {}

        results: dict[str, Translation_Result] = {}

        # Bước 1: Cache lookup
        segments_needing_translation: list[Text_Segment] = []
        for seg in segments:
            normalized = normalize_text(seg.canonical_text)

            # Passthrough: text rỗng
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
                    results[seg.segment_id] = cached
                    continue
                else:
                    # "untranslated" → thử lại
                    del self._cache[normalized]
                    segments_needing_translation.append(seg)
            else:
                segments_needing_translation.append(seg)

        if not segments_needing_translation:
            return results

        # Bước 2: Deduplicate
        text_to_segment_ids: dict[str, list[str]] = {}
        unique_texts_ordered: list[str] = []
        for seg in segments_needing_translation:
            normalized = normalize_text(seg.canonical_text)
            if normalized not in text_to_segment_ids:
                text_to_segment_ids[normalized] = []
                unique_texts_ordered.append(normalized)
            text_to_segment_ids[normalized].append(seg.segment_id)

        # Bước 3: Chia batch
        batch_size = self._config.batch_size
        batches: list[list[str]] = []
        for i in range(0, len(unique_texts_ordered), batch_size):
            batches.append(unique_texts_ordered[i : i + batch_size])

        # Bước 4: Dịch từng batch
        for batch in batches:
            batch_results = self._translate_batch(batch)
            for text, result in zip(batch, batch_results):
                self._cache[text] = result

        # Bước 5: Gộp kết quả
        for seg in segments_needing_translation:
            normalized = normalize_text(seg.canonical_text)
            results[seg.segment_id] = self._cache[normalized]

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_client(self, api_key: str):
        """Khởi tạo OpenAI client trỏ về base_url (9Router, OpenRouter, v.v.)"""
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise InvalidConfigError(
                "Package 'openai' chưa cài. Chạy: pip install openai"
            ) from exc

        logger.info(
            "Translator: dùng endpoint %s (model: %s)",
            self._config.base_url,
            self._config.model,
        )
        return OpenAI(base_url=self._config.base_url, api_key=api_key)

    def _get_google_fallback(self):
        """Lazy init Google Translate fallback (deep-translator)."""
        if self._google_fallback is None:
            try:
                from deep_translator import GoogleTranslator  # type: ignore
                self._google_fallback = GoogleTranslator(source="zh-CN", target="vi")
                logger.info("Google Translate fallback đã sẵn sàng")
            except ImportError:
                logger.error("deep-translator chưa cài, không thể fallback")
                self._google_fallback = None
        return self._google_fallback

    def _translate_google_fallback(
        self, original_text: str, stripped: str
    ) -> Translation_Result:
        """Dịch bằng Google Translate miễn phí (fallback khi LLM API lỗi)."""
        fallback = self._get_google_fallback()
        if fallback is None:
            return Translation_Result(
                source_text=original_text,
                translated_text=original_text,
                status="untranslated",
                error_message="Cả LLM API và Google Translate đều không khả dụng",
            )
        try:
            translated = fallback.translate(stripped)
            if not isinstance(translated, str) or not translated.strip():
                raise RuntimeError("Google Translate trả về kết quả rỗng")
            result = Translation_Result(
                source_text=original_text,
                translated_text=translated.strip(),
                status="translated",
            )
            self._cache[stripped] = result
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("Google Translate fallback cũng thất bại: %s", exc)
            return Translation_Result(
                source_text=original_text,
                translated_text=original_text,
                status="untranslated",
                error_message=f"Fallback failed: {exc}",
            )

    def _respect_rpm(self) -> None:
        """Chờ nếu cần để không vượt quá RPM limit."""
        rpm = self._config.rpm
        if rpm <= 0:
            return
        with self._lock:
            now = time.monotonic()
            self._call_history = [t for t in self._call_history if now - t < 60.0]
            if len(self._call_history) >= rpm:
                wait_until = self._call_history[0] + 60.0
                wait = max(0.0, wait_until - now)
                if wait > 0:
                    logger.debug("Throttling %.2fs (RPM=%d)", wait, rpm)
                    time.sleep(wait)
                    now = time.monotonic()
                    self._call_history = [
                        t for t in self._call_history if now - t < 60.0
                    ]
            self._call_history.append(now)

    def _call_llm(self, text: str) -> str:
        """Gọi LLM API (OpenAI-compatible) để dịch 1 câu."""
        prompt = self._build_prompt(text)
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )
        translated = (response.choices[0].message.content or "").strip()

        # Bỏ dấu nháy bao quanh nếu có
        if len(translated) >= 2 and translated[0] in ('"', "'") and translated[-1] == translated[0]:
            translated = translated[1:-1].strip()
        if not translated:
            raise RuntimeError("LLM trả về kết quả rỗng")
        return translated

    def _call_llm_batch(self, prompt: str) -> str:
        """Gọi LLM API cho batch prompt, trả về raw text response."""
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2048,
        )
        return (response.choices[0].message.content or "").strip()

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
        """Tạo prompt batch với danh sách đánh số."""
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
        """Dịch batch texts qua LLM API. Nếu lỗi → fallback Google Translate từng câu."""
        prompt = self._build_batch_prompt(texts)
        last_error: str | None = None
        attempts = self._max_retries + 1

        for attempt in range(attempts):
            try:
                self._respect_rpm()
                raw_text = self._call_llm_batch(prompt)
                parsed = self._parse_batch_response(raw_text, len(texts))

                if parsed is None:
                    logger.warning(
                        "Batch parse lỗi (số dòng không khớp) cho %d texts, "
                        "fallback từng câu",
                        len(texts),
                    )
                    return self._fallback_individual(texts)

                # Kiểm tra item rỗng
                results: list[Translation_Result] = []
                empty_indices: list[int] = []

                for i, translated in enumerate(parsed):
                    if not translated.strip():
                        empty_indices.append(i)
                        results.append(None)  # type: ignore[arg-type]
                    else:
                        results.append(
                            Translation_Result(
                                source_text=texts[i],
                                translated_text=translated,
                                status="translated",
                            )
                        )

                # Retry item rỗng bằng Google Translate
                if empty_indices:
                    logger.warning(
                        "Batch: %d/%d items rỗng, fallback Google Translate",
                        len(empty_indices),
                        len(texts),
                    )
                    for idx in empty_indices:
                        results[idx] = self._translate_google_fallback(
                            texts[idx], texts[idx]
                        )

                return results

            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Batch attempt %d/%d thất bại (%d texts): %s",
                    attempt + 1,
                    attempts,
                    len(texts),
                    last_error,
                )
                if attempt < self._max_retries and attempt < len(self._backoff):
                    time.sleep(self._backoff[attempt])

        # Tất cả retry thất bại → fallback Google Translate từng câu
        logger.warning(
            "Batch thất bại sau %d lần (%s), fallback Google Translate cho %d texts",
            attempts,
            last_error,
            len(texts),
        )
        return self._fallback_individual(texts)

    def _fallback_individual(self, texts: list[str]) -> list[Translation_Result]:
        """Fallback: dịch từng câu bằng Google Translate."""
        results: list[Translation_Result] = []
        for text in texts:
            results.append(self._translate_google_fallback(text, text))
        return results

    def _parse_batch_response(
        self, response: str, expected_count: int
    ) -> list[str] | None:
        """Parse response dạng danh sách đánh số.

        Trả về None nếu số dòng không khớp expected_count.
        """
        _numbered_prefix_re = re.compile(r"^\d+\.\s+")

        lines = response.splitlines()
        translations: list[str] = []

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            text = _numbered_prefix_re.sub("", stripped_line)
            text = text.strip()

            # Bỏ dấu nháy bao quanh
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
