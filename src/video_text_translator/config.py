"""Configuration loading: YAML + CLI overrides + validation.

Resolution order (later wins): defaults < YAML file < CLI args.
The final ``Config`` object is immutable and fully validated.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from .errors import InvalidConfigError
from .models import (
    Config,
    Detector_Config,
    Gemini_Config,
    Inpainter_Config,
    Overflow_Config,
    Performance_Config,
    Style_Preset,
    Tracker_Config,
    Translator_Config,
)

# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML file and return the parsed dictionary.

    Raises :class:`InvalidConfigError` when the file is missing, unreadable
    or does not parse to a mapping.
    """
    p = Path(path)
    if not p.is_file():
        raise InvalidConfigError(f"config file not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise InvalidConfigError(f"failed to parse YAML {p}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise InvalidConfigError(
            f"config root must be a mapping, got {type(data).__name__}"
        )
    return data


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return a new dict.

    Lists and scalar values from ``override`` replace those in ``base``;
    nested dicts are merged key-by-key.
    """
    result: dict[str, Any] = {**base}
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-text-translator",
        description=(
            "Translate hard-coded Chinese text in a video to Vietnamese."
        ),
    )
    parser.add_argument(
        "-i", "--input", dest="input_path", required=False,
        help="Input video path (.mp4)",
    )
    parser.add_argument(
        "-o", "--output", dest="output_path", required=False,
        help="Output video path (.mp4)",
    )
    parser.add_argument(
        "-c", "--config", dest="config_path", default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )

    parser.add_argument("--compute-mode", choices=["cpu", "gpu", "CPU", "GPU"])
    parser.add_argument("--ocr-stride", type=int, dest="ocr_stride")
    parser.add_argument("--ocr-downscale", type=float, dest="ocr_downscale")
    parser.add_argument("--confidence", type=float, dest="confidence_threshold")
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument(
        "--model-variant", choices=["mobile", "server"], dest="model_variant",
        help="OCR model size: 'mobile' (fast, default) or 'server' (accurate, slow)",
    )
    parser.add_argument(
        "--cpu-threads", type=int, dest="cpu_threads",
        help="CPU threads for OCR (0 = auto, all cores)",
    )
    parser.add_argument(
        "--translator", choices=["google", "gemini"], dest="translator_backend",
        help="Translation backend (default: google; gemini requires GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--gemini-model", dest="gemini_model",
        help="Gemini model name (default: gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--inpaint-algo", choices=["telea", "ns"], dest="inpaint_algo"
    )
    parser.add_argument("--inpaint-radius", type=int, dest="inpaint_radius")
    parser.add_argument("--inpaint-padding", type=int, dest="inpaint_padding")
    parser.add_argument("--font", dest="font_path", help="Font file (.ttf/.otf)")
    parser.add_argument("--font-size-max", type=int, dest="font_size_max")
    parser.add_argument("--font-size-min", type=int, dest="font_size_min")
    parser.add_argument("--max-chars", type=int, dest="max_chars")
    parser.add_argument(
        "--translate-timeout", type=float, dest="translate_timeout"
    )
    parser.add_argument("--n-inactive", type=int, dest="n_inactive")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging"
    )
    parser.add_argument("--quiet", action="store_true", help="Errors only")
    return parser


def cli_overrides(ns: argparse.Namespace) -> dict[str, Any]:
    """Convert an argparse Namespace into a sparse override dict.

    Only keys actually provided by the user (i.e. non-None) are
    returned; this lets ``deep_merge`` leave the rest of the config
    untouched.
    """
    out: dict[str, Any] = {}

    if ns.input_path is not None:
        out["input_path"] = ns.input_path
    if ns.output_path is not None:
        out["output_path"] = ns.output_path
    if ns.compute_mode is not None:
        out["compute_mode"] = ns.compute_mode.lower()

    detector: dict[str, Any] = {}
    if ns.confidence_threshold is not None:
        detector["confidence_threshold"] = ns.confidence_threshold
    if ns.batch_size is not None:
        detector["batch_size"] = ns.batch_size
    if ns.model_variant is not None:
        detector["model_variant"] = ns.model_variant
    if ns.cpu_threads is not None:
        detector["cpu_threads"] = ns.cpu_threads
    if detector:
        out["detector"] = detector

    tracker: dict[str, Any] = {}
    if ns.n_inactive is not None:
        tracker["n_inactive"] = ns.n_inactive
    if tracker:
        out["tracker"] = tracker

    inpainter: dict[str, Any] = {}
    if ns.inpaint_algo is not None:
        inpainter["algorithm"] = ns.inpaint_algo
    if ns.inpaint_radius is not None:
        inpainter["radius"] = ns.inpaint_radius
    if ns.inpaint_padding is not None:
        inpainter["padding"] = ns.inpaint_padding
    if inpainter:
        out["inpainter"] = inpainter

    translator: dict[str, Any] = {}
    if ns.translate_timeout is not None:
        translator["timeout_seconds"] = ns.translate_timeout
    if ns.max_chars is not None:
        translator["max_chars"] = ns.max_chars
    if ns.translator_backend is not None:
        translator["backend"] = ns.translator_backend
    if ns.gemini_model is not None:
        translator.setdefault("gemini", {})["model"] = ns.gemini_model
        translator.setdefault("gemini", {})["enabled"] = True
        translator["backend"] = "gemini"
    if translator:
        out["translator"] = translator

    renderer: dict[str, Any] = {}
    if ns.font_path is not None:
        renderer["font_path"] = ns.font_path
    if ns.font_size_max is not None:
        renderer["font_size_max"] = ns.font_size_max
    if ns.font_size_min is not None:
        renderer["font_size_min"] = ns.font_size_min
    if renderer:
        out["renderer"] = renderer

    performance: dict[str, Any] = {}
    if ns.ocr_stride is not None:
        performance["ocr_stride"] = ns.ocr_stride
    if ns.ocr_downscale is not None:
        performance["ocr_downscale"] = ns.ocr_downscale
    if performance:
        out["performance"] = performance

    return out


# ---------------------------------------------------------------------------
# Build Config from merged dict
# ---------------------------------------------------------------------------


def _require(d: dict[str, Any], key: str, kind: str) -> Any:
    if key not in d:
        raise InvalidConfigError(f"{kind}: missing required field '{key}'")
    return d[key]


def _coerce_color(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise InvalidConfigError(
            f"{name} must be a 3-element list of integers (got {value!r})"
        )
    return (int(value[0]), int(value[1]), int(value[2]))


def _coerce_offset(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise InvalidConfigError(
            f"{name} must be a 2-element list of integers (got {value!r})"
        )
    return (int(value[0]), int(value[1]))


def build_config(merged: dict[str, Any]) -> Config:
    """Validate ``merged`` and assemble an immutable :class:`Config`.

    All range/enum checks live inside the dataclass ``__post_init__``
    methods, so any out-of-range value reliably surfaces as an
    :class:`InvalidConfigError`.
    """
    try:
        compute_mode = str(merged.get("compute_mode", "cpu")).lower()

        det_d = merged.get("detector", {}) or {}
        detector = Detector_Config(
            confidence_threshold=float(det_d.get("confidence_threshold", 0.5)),
            batch_size=int(det_d.get("batch_size", 4)),
            model_variant=str(det_d.get("model_variant", "mobile")).lower(),  # type: ignore[arg-type]
            cpu_threads=int(det_d.get("cpu_threads", 0)),
        )

        trk_d = merged.get("tracker", {}) or {}
        tracker = Tracker_Config(
            iou_threshold=float(trk_d.get("iou_threshold", 0.5)),
            content_similarity_threshold=float(
                trk_d.get("content_similarity_threshold", 0.7)
            ),
            center_distance_ratio=float(trk_d.get("center_distance_ratio", 0.10)),
            n_inactive=int(trk_d.get("n_inactive", 3)),
            max_active_segments=int(trk_d.get("max_active_segments", 100)),
        )

        ip_d = merged.get("inpainter", {}) or {}
        inpainter = Inpainter_Config(
            algorithm=str(ip_d.get("algorithm", "telea")).lower(),
            radius=int(ip_d.get("radius", 3)),
            padding=int(ip_d.get("padding", 4)),
        )

        tr_d = merged.get("translator", {}) or {}
        gemini_d = tr_d.get("gemini", {}) or {}
        gemini = Gemini_Config(
            enabled=bool(gemini_d.get("enabled", False)),
            model=str(gemini_d.get("model", "gemini-2.5-flash-lite")),
            api_key_env=str(gemini_d.get("api_key_env", "GEMINI_API_KEY")),
            base_url=str(gemini_d.get("base_url", "")),
            max_chars_target=int(gemini_d.get("max_chars_target", 0)),
            rpm=int(gemini_d.get("rpm", 15)),
            timeout_seconds=float(gemini_d.get("timeout_seconds", 30.0)),
            batch_size=int(gemini_d.get("batch_size", 10)),
        )
        # Resolve effective backend: explicit `backend` takes precedence,
        # otherwise the legacy `gemini.enabled: true` flag enables it.
        backend = str(tr_d.get("backend", "gemini" if gemini.enabled else "google")).lower()
        translator = Translator_Config(
            backend=backend,  # type: ignore[arg-type]
            timeout_seconds=float(tr_d.get("timeout_seconds", 10.0)),
            max_chars=int(tr_d.get("max_chars", 5000)),
            max_retries=int(tr_d.get("max_retries", 3)),
            gemini=gemini,
        )

        rd_d = merged.get("renderer", {}) or {}
        font_path = str(_require(rd_d, "font_path", "renderer"))
        ovf_d = rd_d.get("overflow", {}) or {}
        overflow = Overflow_Config(
            expand_bbox_enabled=bool(ovf_d.get("expand_bbox_enabled", True)),
            expand_bbox_max=float(ovf_d.get("expand_bbox_max", 1.5)),
            word_wrap_enabled=bool(ovf_d.get("word_wrap_enabled", True)),
            word_wrap_max_lines=int(ovf_d.get("word_wrap_max_lines", 3)),
            condensed_enabled=bool(ovf_d.get("condensed_enabled", True)),
            condensed_font_path=str(
                ovf_d.get("condensed_font_path", "fonts/NotoSans-Condensed.ttf")
            ),
        )
        renderer = Style_Preset(
            font_path=font_path,
            font_size_max=int(rd_d.get("font_size_max", 64)),
            font_size_min=int(rd_d.get("font_size_min", 12)),
            text_rgb=_coerce_color(rd_d.get("text_rgb", [255, 255, 255]), "renderer.text_rgb"),
            stroke_enabled=bool(rd_d.get("stroke_enabled", True)),
            stroke_rgb=_coerce_color(rd_d.get("stroke_rgb", [0, 0, 0]), "renderer.stroke_rgb"),
            stroke_width=int(rd_d.get("stroke_width", 2)),
            background_enabled=bool(rd_d.get("background_enabled", True)),
            background_rgb=_coerce_color(
                rd_d.get("background_rgb", [0, 0, 0]), "renderer.background_rgb"
            ),
            background_alpha=int(rd_d.get("background_alpha", 128)),
            shadow_enabled=bool(rd_d.get("shadow_enabled", True)),
            shadow_rgb=_coerce_color(rd_d.get("shadow_rgb", [0, 0, 0]), "renderer.shadow_rgb"),
            shadow_offset=_coerce_offset(
                rd_d.get("shadow_offset", [2, 2]), "renderer.shadow_offset"
            ),
            overflow=overflow,
        )

        perf_d = merged.get("performance", {}) or {}
        performance = Performance_Config(
            ocr_stride=int(perf_d.get("ocr_stride", 3)),
            ocr_downscale=float(perf_d.get("ocr_downscale", 1.5)),
            io_buffer_frames=int(perf_d.get("io_buffer_frames", 8)),
            max_duration_seconds=int(perf_d.get("max_duration_seconds", 7200)),
            max_file_size_bytes=int(
                perf_d.get("max_file_size_bytes", 5 * 1024 * 1024 * 1024)
            ),
        )

        return Config(
            input_path=str(merged.get("input_path", "")),
            output_path=str(merged.get("output_path", "")),
            compute_mode=compute_mode,  # type: ignore[arg-type]
            detector=detector,
            tracker=tracker,
            inpainter=inpainter,
            translator=translator,
            renderer=renderer,
            performance=performance,
        )
    except (ValueError, KeyError, TypeError) as exc:
        # Translate validation failures into a single error type so the
        # caller can format them uniformly.
        raise InvalidConfigError(str(exc)) from exc


def load_config(args: argparse.Namespace) -> Config:
    """Resolve YAML + CLI overrides into a final :class:`Config`."""
    yaml_dict = load_yaml(args.config_path) if args.config_path else {}
    overrides = cli_overrides(args)
    merged = deep_merge(yaml_dict, overrides)
    config = build_config(merged)
    return config
