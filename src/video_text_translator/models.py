"""Frozen dataclasses for the Video Text Translator domain models.

All models are immutable (`frozen=True, slots=True`) and validated in
`__post_init__` so that an invalid model can never exist at runtime.
This module deliberately has no dependencies on other modules in the
package to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Bounding_Box:
    """Axis-aligned rectangle in pixel coordinates.

    Coordinates use the OpenCV convention: origin at the top-left corner,
    x increases to the right, y increases downward.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Bounding_Box dimensions must be positive (got w={self.width}, h={self.height})"
            )
        if self.x < 0 or self.y < 0:
            raise ValueError(
                f"Bounding_Box origin must be non-negative (got x={self.x}, y={self.y})"
            )

    @property
    def x2(self) -> int:
        """Right edge (exclusive)."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom edge (exclusive)."""
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        """Geometric center of the box in pixel coordinates."""
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def area(self) -> int:
        """Area in square pixels."""
        return self.width * self.height


# ---------------------------------------------------------------------------
# OCR / Tracking
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Text_Region:
    """A single text detection result for one frame."""

    box: Bounding_Box
    text: str
    confidence: float
    frame_index: int
    timestamp: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0] (got {self.confidence})")
        if self.frame_index < 0:
            raise ValueError(f"frame_index must be >= 0 (got {self.frame_index})")
        if self.timestamp < 0.0:
            raise ValueError(f"timestamp must be >= 0.0 (got {self.timestamp})")


@dataclass(frozen=True, slots=True)
class Frame_Region_Entry:
    """One frame-level entry inside a Text_Segment."""

    frame_index: int
    timestamp: float
    box: Bounding_Box
    text: str
    interpolated: bool = False

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError(f"frame_index must be >= 0 (got {self.frame_index})")
        if self.timestamp < 0.0:
            raise ValueError(f"timestamp must be >= 0.0 (got {self.timestamp})")


@dataclass(frozen=True, slots=True)
class Text_Segment:
    """A contiguous run of frames where the same text appears."""

    segment_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    canonical_text: str
    entries: tuple[Frame_Region_Entry, ...]

    def __post_init__(self) -> None:
        if self.start_frame > self.end_frame:
            raise ValueError(
                f"start_frame ({self.start_frame}) must be <= end_frame ({self.end_frame})"
            )
        if self.start_time > self.end_time:
            raise ValueError(
                f"start_time ({self.start_time}) must be <= end_time ({self.end_time})"
            )
        if not self.segment_id:
            raise ValueError("segment_id must be non-empty")


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


Translation_Status = Literal["translated", "passthrough", "untranslated"]


@dataclass(frozen=True, slots=True)
class Translation_Result:
    """Result of attempting to translate one source string."""

    source_text: str
    translated_text: str
    status: Translation_Status
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Rendering style
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Style_Preset:
    """All visual options for rendering Vietnamese text on a frame."""

    font_path: str
    font_size_max: int = 64
    font_size_min: int = 12
    text_rgb: tuple[int, int, int] = (255, 255, 255)
    stroke_enabled: bool = True
    stroke_rgb: tuple[int, int, int] = (0, 0, 0)
    stroke_width: int = 2
    background_enabled: bool = True
    background_rgb: tuple[int, int, int] = (0, 0, 0)
    background_alpha: int = 128
    shadow_enabled: bool = True
    shadow_rgb: tuple[int, int, int] = (0, 0, 0)
    shadow_offset: tuple[int, int] = (2, 2)
    overflow: "Overflow_Config | None" = None

    def __post_init__(self) -> None:
        if not self.font_path:
            raise ValueError("font_path must be non-empty")
        if not (8 <= self.font_size_max <= 512):
            raise ValueError(f"font_size_max must be in [8, 512] (got {self.font_size_max})")
        if not (6 <= self.font_size_min <= self.font_size_max):
            raise ValueError(
                f"font_size_min must be in [6, font_size_max={self.font_size_max}] "
                f"(got {self.font_size_min})"
            )
        for name, rgb in (
            ("text_rgb", self.text_rgb),
            ("stroke_rgb", self.stroke_rgb),
            ("background_rgb", self.background_rgb),
            ("shadow_rgb", self.shadow_rgb),
        ):
            if len(rgb) != 3 or any(not (0 <= c <= 255) for c in rgb):
                raise ValueError(f"{name} must be a tuple of 3 ints in [0, 255] (got {rgb})")
        if not (0 <= self.background_alpha <= 255):
            raise ValueError(
                f"background_alpha must be in [0, 255] (got {self.background_alpha})"
            )
        if not (0 <= self.stroke_width <= 20):
            raise ValueError(f"stroke_width must be in [0, 20] (got {self.stroke_width})")
        if len(self.shadow_offset) != 2 or any(
            not (-50 <= v <= 50) for v in self.shadow_offset
        ):
            raise ValueError(
                f"shadow_offset must be a tuple of 2 ints in [-50, 50] (got {self.shadow_offset})"
            )


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


Compute_Mode = Literal["cpu", "gpu"]


@dataclass(frozen=True, slots=True)
class Detector_Config:
    confidence_threshold: float = 0.5
    batch_size: int = 4
    model_variant: Literal["mobile", "server"] = "mobile"
    cpu_threads: int = 0  # 0 = auto (all logical cores)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"detector.confidence_threshold must be in [0.0, 1.0] "
                f"(got {self.confidence_threshold})"
            )
        if not (1 <= self.batch_size <= 32):
            raise ValueError(
                f"detector.batch_size must be in [1, 32] (got {self.batch_size})"
            )
        if self.model_variant not in ("mobile", "server"):
            raise ValueError(
                f'detector.model_variant must be "mobile" or "server" '
                f'(got "{self.model_variant}")'
            )
        if not (0 <= self.cpu_threads <= 64):
            raise ValueError(
                f"detector.cpu_threads must be in [0, 64] (got {self.cpu_threads})"
            )


@dataclass(frozen=True, slots=True)
class Tracker_Config:
    iou_threshold: float = 0.5
    content_similarity_threshold: float = 0.7
    center_distance_ratio: float = 0.10
    n_inactive: int = 3
    max_active_segments: int = 100
    smooth_lock_threshold: int = 3  # px — jitter below this is locked out
    smooth_ema_alpha: float = 0.3   # EMA responsiveness (0, 1]

    def __post_init__(self) -> None:
        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ValueError(
                f"tracker.iou_threshold must be in [0.0, 1.0] (got {self.iou_threshold})"
            )
        if not (0.0 <= self.content_similarity_threshold <= 1.0):
            raise ValueError(
                f"tracker.content_similarity_threshold must be in [0.0, 1.0] "
                f"(got {self.content_similarity_threshold})"
            )
        if not (0.0 < self.center_distance_ratio <= 1.0):
            raise ValueError(
                f"tracker.center_distance_ratio must be in (0.0, 1.0] "
                f"(got {self.center_distance_ratio})"
            )
        if not (1 <= self.n_inactive <= 30):
            raise ValueError(
                f"tracker.n_inactive must be in [1, 30] (got {self.n_inactive})"
            )
        if not (1 <= self.max_active_segments <= 1000):
            raise ValueError(
                f"tracker.max_active_segments must be in [1, 1000] "
                f"(got {self.max_active_segments})"
            )
        if not (0 <= self.smooth_lock_threshold <= 20):
            raise ValueError(
                f"tracker.smooth_lock_threshold must be in [0, 20] "
                f"(got {self.smooth_lock_threshold})"
            )
        if not (0.0 < self.smooth_ema_alpha <= 1.0):
            raise ValueError(
                f"tracker.smooth_ema_alpha must be in (0.0, 1.0] "
                f"(got {self.smooth_ema_alpha})"
            )


@dataclass(frozen=True, slots=True)
class Inpainter_Config:
    algorithm: Literal["telea", "ns"] = "telea"
    radius: int = 3
    padding: int = 4

    def __post_init__(self) -> None:
        if self.algorithm not in ("telea", "ns"):
            raise ValueError(
                f'inpainter.algorithm must be "telea" or "ns" (got "{self.algorithm}")'
            )
        if not (1 <= self.radius <= 20):
            raise ValueError(f"inpainter.radius must be in [1, 20] (got {self.radius})")
        if not (0 <= self.padding <= 20):
            raise ValueError(f"inpainter.padding must be in [0, 20] (got {self.padding})")


@dataclass(frozen=True, slots=True)
class Llm_Config:
    """Cấu hình LLM translator (OpenAI-compatible API qua 9Router, OpenRouter, v.v.)"""
    enabled: bool = False
    model: str = "free"
    api_key_env: str = "LLM_API_KEY"
    base_url: str = ""  # endpoint URL (ví dụ: http://localhost:20128/v1)
    max_chars_target: int = 0  # 0 = không giới hạn; khác 0 = yêu cầu dịch ngắn hơn N ký tự
    rpm: int = 30  # giới hạn request/phút
    timeout_seconds: float = 30.0
    batch_size: int = 10  # [1, 50] — số text gộp trong 1 lần gọi API

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("translator.llm.model must be non-empty")
        if not self.api_key_env:
            raise ValueError("translator.llm.api_key_env must be non-empty")
        if self.max_chars_target < 0:
            raise ValueError(
                f"translator.llm.max_chars_target must be >= 0 "
                f"(got {self.max_chars_target})"
            )
        if not (1 <= self.rpm <= 1000):
            raise ValueError(
                f"translator.llm.rpm must be in [1, 1000] (got {self.rpm})"
            )
        if not (0.0 < self.timeout_seconds <= 120.0):
            raise ValueError(
                f"translator.llm.timeout_seconds must be in (0, 120] "
                f"(got {self.timeout_seconds})"
            )
        if not (1 <= self.batch_size <= 50):
            raise ValueError(
                f"translator.llm.batch_size must be in [1, 50] "
                f"(got {self.batch_size})"
            )


# Giữ alias cũ để không break import cũ
Gemini_Config = Llm_Config


@dataclass(frozen=True, slots=True)
class Translator_Config:
    backend: Literal["google", "llm"] = "google"
    timeout_seconds: float = 10.0
    max_chars: int = 5000
    max_retries: int = 3
    llm: Llm_Config = field(default_factory=Llm_Config)

    def __post_init__(self) -> None:
        if self.backend not in ("google", "llm"):
            raise ValueError(
                f'translator.backend must be "google" or "llm" '
                f'(got "{self.backend}")'
            )
        if not (0.0 < self.timeout_seconds <= 60.0):
            raise ValueError(
                f"translator.timeout_seconds must be in (0, 60] (got {self.timeout_seconds})"
            )
        if not (1 <= self.max_chars <= 10000):
            raise ValueError(
                f"translator.max_chars must be in [1, 10000] (got {self.max_chars})"
            )
        if not (0 <= self.max_retries <= 10):
            raise ValueError(
                f"translator.max_retries must be in [0, 10] (got {self.max_retries})"
            )


@dataclass(frozen=True, slots=True)
class Overflow_Config:
    """Strategy for handling Vietnamese text that doesn't fit the original box.

    Cascade order: expand bbox -> word wrap -> condensed font.
    Set the ``*_enabled`` flags to disable individual strategies.
    """

    expand_bbox_enabled: bool = True
    expand_bbox_max: float = 1.5  # multiplier of original bbox dimensions
    word_wrap_enabled: bool = True
    word_wrap_max_lines: int = 3
    condensed_enabled: bool = True
    condensed_font_path: str = "fonts/NotoSans-Condensed.ttf"

    def __post_init__(self) -> None:
        if not (1.0 <= self.expand_bbox_max <= 4.0):
            raise ValueError(
                f"renderer.overflow.expand_bbox_max must be in [1.0, 4.0] "
                f"(got {self.expand_bbox_max})"
            )
        if not (1 <= self.word_wrap_max_lines <= 10):
            raise ValueError(
                f"renderer.overflow.word_wrap_max_lines must be in [1, 10] "
                f"(got {self.word_wrap_max_lines})"
            )
        if not self.condensed_font_path:
            raise ValueError(
                "renderer.overflow.condensed_font_path must be non-empty"
            )


Encoder_Mode = Literal["auto", "cpu", "nvenc", "qsv", "amf"]
Pass2_Mode = Literal["auto", "sequential", "parallel"]


@dataclass(frozen=True, slots=True)
class Performance_Config:
    ocr_stride: int = 3
    ocr_downscale: float = 1.5
    io_buffer_frames: int = 8
    max_duration_seconds: int = 7200
    max_file_size_bytes: int = 5 * 1024 * 1024 * 1024  # 5 GB
    encoder: Encoder_Mode = "auto"
    encoder_preset: str = "fast"
    pass2_mode: Pass2_Mode = "auto"
    parallel_workers: int = 0  # 0 = auto (number of CPU cores)

    def __post_init__(self) -> None:
        if not (1 <= self.ocr_stride <= 10):
            raise ValueError(
                f"performance.ocr_stride must be in [1, 10] (got {self.ocr_stride})"
            )
        if not (1.0 <= self.ocr_downscale <= 4.0):
            raise ValueError(
                f"performance.ocr_downscale must be in [1.0, 4.0] (got {self.ocr_downscale})"
            )
        if not (0 <= self.io_buffer_frames <= 64):
            raise ValueError(
                f"performance.io_buffer_frames must be in [0, 64] "
                f"(got {self.io_buffer_frames})"
            )
        if self.max_duration_seconds <= 0:
            raise ValueError(
                f"performance.max_duration_seconds must be > 0 "
                f"(got {self.max_duration_seconds})"
            )
        if self.max_file_size_bytes <= 0:
            raise ValueError(
                f"performance.max_file_size_bytes must be > 0 "
                f"(got {self.max_file_size_bytes})"
            )
        if self.encoder not in ("auto", "cpu", "nvenc", "qsv", "amf"):
            raise ValueError(
                f'performance.encoder must be one of "auto", "cpu", "nvenc", "qsv", "amf" '
                f'(got "{self.encoder}")'
            )
        if self.encoder_preset not in ("ultrafast", "fast", "medium"):
            raise ValueError(
                f'performance.encoder_preset must be "ultrafast", "fast", or "medium" '
                f'(got "{self.encoder_preset}")'
            )
        if self.pass2_mode not in ("auto", "sequential", "parallel"):
            raise ValueError(
                f'performance.pass2_mode must be "auto", "sequential", or "parallel" '
                f'(got "{self.pass2_mode}")'
            )
        if not (0 <= self.parallel_workers <= 32):
            raise ValueError(
                f"performance.parallel_workers must be in [0, 32] "
                f"(got {self.parallel_workers})"
            )


@dataclass(frozen=True, slots=True)
class Config:
    """Full pipeline configuration assembled from YAML + CLI overrides."""

    input_path: str
    output_path: str
    compute_mode: Compute_Mode = "cpu"
    detector: Detector_Config = field(default_factory=Detector_Config)
    tracker: Tracker_Config = field(default_factory=Tracker_Config)
    inpainter: Inpainter_Config = field(default_factory=Inpainter_Config)
    translator: Translator_Config = field(default_factory=Translator_Config)
    renderer: Style_Preset = field(default_factory=lambda: Style_Preset(font_path=""))
    performance: Performance_Config = field(default_factory=Performance_Config)

    def __post_init__(self) -> None:
        if self.compute_mode not in ("cpu", "gpu"):
            raise ValueError(
                f'compute_mode must be "cpu" or "gpu" (got "{self.compute_mode}")'
            )
