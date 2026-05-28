"""Perturbation configuration: dataclasses, presets, validation, and YAML loading.

This module defines the immutable configuration for the video perturbation
pipeline, including preset definitions, data models for scheduling and scene
detection, and validation logic that ensures all parameters fall within their
documented valid ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .config import load_yaml
from .errors import InvalidConfigError

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SegmentParam:
    """A parameter value for one time segment."""

    start_time: float
    end_time: float
    value: float


@dataclass(frozen=True, slots=True)
class SceneBoundary:
    """A detected scene boundary."""

    frame_index: int
    timestamp: float
    histogram_diff: float


@dataclass(frozen=True, slots=True)
class Scene:
    """A video scene between two boundaries."""

    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    is_micro: bool  # duration < max_scene_duration


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    "light": {
        "speed_min": 0.995,
        "speed_max": 1.005,
        "max_frame_drop_percent": 0.5,
        "micro_offset_ms": 20.0,
        "max_crop_percent": 3.0,
        "max_zoom": 1.03,
        "audio_tempo_min": 0.995,
        "audio_tempo_max": 1.005,
        "eq_range_db": 1.0,
        "ambience_volume_min_db": -40.0,
        "ambience_volume_max_db": -35.0,
        "change_interval": 15.0,
        # Wave 1: Color Drift
        "gamma_range": 0.01,
        "saturation_range": 0.02,
        "contrast_range": 0.02,
        "hue_range": 1.0,
        # Wave 1: Rotation Drift
        "max_rotation_degrees": 0.5,
        # Wave 1: Overlay
        "overlay_opacity_max": 0.02,
        # Wave 1: GOP Perturbation
        "gop_min": 30,
        "gop_max": 90,
    },
    "medium": {
        "speed_min": 0.99,
        "speed_max": 1.01,
        "max_frame_drop_percent": 1.0,
        "micro_offset_ms": 50.0,
        "max_crop_percent": 5.0,
        "max_zoom": 1.05,
        "audio_tempo_min": 0.99,
        "audio_tempo_max": 1.01,
        "eq_range_db": 2.0,
        "ambience_volume_min_db": -38.0,
        "ambience_volume_max_db": -30.0,
        "change_interval": 10.0,
        # Wave 1: Color Drift
        "gamma_range": 0.02,
        "saturation_range": 0.03,
        "contrast_range": 0.03,
        "hue_range": 2.0,
        # Wave 1: Rotation Drift
        "max_rotation_degrees": 1.0,
        # Wave 1: Overlay
        "overlay_opacity_max": 0.03,
        # Wave 1: GOP Perturbation
        "gop_min": 15,
        "gop_max": 120,
    },
    "heavy": {
        "speed_min": 0.97,
        "speed_max": 1.03,
        "max_frame_drop_percent": 3.0,
        "micro_offset_ms": 100.0,
        "max_crop_percent": 10.0,
        "max_zoom": 1.15,
        "audio_tempo_min": 0.97,
        "audio_tempo_max": 1.03,
        "eq_range_db": 4.0,
        "ambience_volume_min_db": -35.0,
        "ambience_volume_max_db": -25.0,
        "change_interval": 5.0,
        # Wave 1: Color Drift
        "gamma_range": 0.04,
        "saturation_range": 0.05,
        "contrast_range": 0.05,
        "hue_range": 3.0,
        # Wave 1: Rotation Drift
        "max_rotation_degrees": 2.0,
        # Wave 1: Overlay
        "overlay_opacity_max": 0.05,
        # Wave 1: GOP Perturbation
        "gop_min": 10,
        "gop_max": 150,
    },
}


# ---------------------------------------------------------------------------
# PerturbationConfig dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PerturbationConfig:
    """Immutable configuration for the perturbation pipeline."""

    input_path: str
    output_path: str
    preset: Literal["light", "medium", "heavy"] = "medium"
    seed: int | None = None  # None = random

    # Temporal Drift
    temporal_enabled: bool = True
    speed_min: float = 0.99  # [0.90, 1.0]
    speed_max: float = 1.01  # [1.0, 1.10]
    max_frame_drop_percent: float = 1.0  # [0, 20]
    micro_offset_ms: float = 50.0  # [0, 500]

    # Spatial Transform
    spatial_enabled: bool = True
    max_crop_percent: float = 5.0  # [0, 50]
    max_zoom: float = 1.05  # [1.0, 2.0]

    # Scene Recomposition
    scene_enabled: bool = False
    scene_threshold: float = 0.3  # [0.1, 1.0]
    max_scene_duration: float = 5.0  # [1.0, 60.0]
    transition_duration_ms: float = 500.0  # [100, 2000]

    # Audio Perturbation
    audio_enabled: bool = True
    audio_tempo_min: float = 0.99  # [0.90, 1.0]
    audio_tempo_max: float = 1.01  # [1.0, 1.10]
    eq_range_db: float = 2.0  # [0, 12.0]
    ambience_volume_min_db: float = -40.0  # [-60, 0]
    ambience_volume_max_db: float = -30.0  # [-60, 0]

    # Multi-Transform Combo
    combo_enabled: bool = False
    combo_intensity_factor: float = 0.5  # [0.1, 1.0]

    # Color Drift (Wave 1)
    color_drift_enabled: bool = True
    gamma_range: float = 0.02  # [0.0, 0.1] deviation from 1.0
    saturation_range: float = 0.03  # [0.0, 0.1] deviation from 1.0
    contrast_range: float = 0.03  # [0.0, 0.1] deviation from 1.0
    hue_range: float = 2.0  # [0, 5] degrees

    # Rotation Drift (Wave 1)
    rotation_enabled: bool = True
    max_rotation_degrees: float = 1.0  # [0.0, 5.0]

    # Overlay (Wave 1)
    overlay_enabled: bool = True
    overlay_opacity_max: float = 0.03  # [0.0, 0.1] max opacity
    overlay_grain_enabled: bool = True
    overlay_vignette_enabled: bool = True

    # GOP Perturbation (Wave 1)
    gop_perturbation_enabled: bool = True
    gop_min: int = 15  # [1, 300]
    gop_max: int = 120  # [1, 300]

    # Timing
    change_interval: float = 10.0  # [0.5, 300.0] seconds

    # Performance
    encoder: str = "auto"
    encoder_preset: str = "fast"
    parallel_workers: int = 0  # 0 = auto-detect based on CPU cores


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Valid ranges: (min_inclusive, max_inclusive)
_VALID_RANGES: dict[str, tuple[float, float]] = {
    "speed_min": (0.90, 1.0),
    "speed_max": (1.0, 1.10),
    "max_frame_drop_percent": (0.0, 20.0),
    "micro_offset_ms": (0.0, 500.0),
    "max_crop_percent": (0.0, 50.0),
    "max_zoom": (1.0, 2.0),
    "scene_threshold": (0.1, 1.0),
    "max_scene_duration": (1.0, 60.0),
    "transition_duration_ms": (100.0, 2000.0),
    "audio_tempo_min": (0.90, 1.0),
    "audio_tempo_max": (1.0, 1.10),
    "eq_range_db": (0.0, 12.0),
    "ambience_volume_min_db": (-60.0, 0.0),
    "ambience_volume_max_db": (-60.0, 0.0),
    "combo_intensity_factor": (0.1, 1.0),
    "change_interval": (0.5, 300.0),
    # Wave 1: Color Drift
    "gamma_range": (0.0, 0.1),
    "saturation_range": (0.0, 0.1),
    "contrast_range": (0.0, 0.1),
    "hue_range": (0.0, 5.0),
    # Wave 1: Rotation Drift
    "max_rotation_degrees": (0.0, 5.0),
    # Wave 1: Overlay
    "overlay_opacity_max": (0.0, 0.1),
    # Wave 1: GOP Perturbation
    "gop_min": (1, 300),
    "gop_max": (1, 300),
}

_VALID_ENCODERS = {"auto", "cpu", "nvenc", "qsv", "amf"}
_VALID_ENCODER_PRESETS = {"ultrafast", "fast", "medium"}
_VALID_PRESETS = {"light", "medium", "heavy"}


def validate_config(config: PerturbationConfig) -> None:
    """Validate all parameter ranges in a PerturbationConfig.

    Raises :class:`InvalidConfigError` if any parameter is out of range,
    with a message identifying the parameter name and valid range.
    """
    # Validate preset name
    if config.preset not in _VALID_PRESETS:
        raise InvalidConfigError(
            f"Invalid preset '{config.preset}' "
            f"(valid: {', '.join(sorted(_VALID_PRESETS))})"
        )

    # Validate encoder
    if config.encoder not in _VALID_ENCODERS:
        raise InvalidConfigError(
            f"Invalid encoder '{config.encoder}' "
            f"(valid: {', '.join(sorted(_VALID_ENCODERS))})"
        )

    # Validate encoder_preset
    if config.encoder_preset not in _VALID_ENCODER_PRESETS:
        raise InvalidConfigError(
            f"Invalid encoder_preset '{config.encoder_preset}' "
            f"(valid: {', '.join(sorted(_VALID_ENCODER_PRESETS))})"
        )

    # Validate numeric ranges
    for param_name, (lo, hi) in _VALID_RANGES.items():
        value = getattr(config, param_name)
        if not (lo <= value <= hi):
            raise InvalidConfigError(
                f"Invalid {param_name}: {value} (valid range: {lo}-{hi})"
            )


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def _flatten_yaml_to_config_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested YAML structure into flat PerturbationConfig field names.

    The YAML uses nested sections (temporal, spatial, scene, audio, combo,
    timing, performance) but PerturbationConfig uses flat field names.
    """
    flat: dict[str, Any] = {}

    # Top-level fields
    if "preset" in data:
        flat["preset"] = data["preset"]
    if "seed" in data:
        flat["seed"] = data["seed"]
    if "input_path" in data:
        flat["input_path"] = data["input_path"]
    if "output_path" in data:
        flat["output_path"] = data["output_path"]

    # Temporal section
    temporal = data.get("temporal", {}) or {}
    if "enabled" in temporal:
        flat["temporal_enabled"] = temporal["enabled"]
    if "speed_min" in temporal:
        flat["speed_min"] = temporal["speed_min"]
    if "speed_max" in temporal:
        flat["speed_max"] = temporal["speed_max"]
    if "max_frame_drop_percent" in temporal:
        flat["max_frame_drop_percent"] = temporal["max_frame_drop_percent"]
    if "micro_offset_ms" in temporal:
        flat["micro_offset_ms"] = temporal["micro_offset_ms"]

    # Spatial section
    spatial = data.get("spatial", {}) or {}
    if "enabled" in spatial:
        flat["spatial_enabled"] = spatial["enabled"]
    if "max_crop_percent" in spatial:
        flat["max_crop_percent"] = spatial["max_crop_percent"]
    if "max_zoom" in spatial:
        flat["max_zoom"] = spatial["max_zoom"]

    # Scene section
    scene = data.get("scene", {}) or {}
    if "enabled" in scene:
        flat["scene_enabled"] = scene["enabled"]
    if "scene_threshold" in scene:
        flat["scene_threshold"] = scene["scene_threshold"]
    if "max_scene_duration" in scene:
        flat["max_scene_duration"] = scene["max_scene_duration"]
    if "transition_duration_ms" in scene:
        flat["transition_duration_ms"] = scene["transition_duration_ms"]

    # Audio section
    audio = data.get("audio", {}) or {}
    if "enabled" in audio:
        flat["audio_enabled"] = audio["enabled"]
    if "tempo_min" in audio:
        flat["audio_tempo_min"] = audio["tempo_min"]
    if "tempo_max" in audio:
        flat["audio_tempo_max"] = audio["tempo_max"]
    if "eq_range_db" in audio:
        flat["eq_range_db"] = audio["eq_range_db"]
    if "ambience_volume_min_db" in audio:
        flat["ambience_volume_min_db"] = audio["ambience_volume_min_db"]
    if "ambience_volume_max_db" in audio:
        flat["ambience_volume_max_db"] = audio["ambience_volume_max_db"]

    # Combo section
    combo = data.get("combo", {}) or {}
    if "enabled" in combo:
        flat["combo_enabled"] = combo["enabled"]
    if "intensity_factor" in combo:
        flat["combo_intensity_factor"] = combo["intensity_factor"]

    # Color Drift section (Wave 1)
    color = data.get("color_drift", {}) or {}
    if "enabled" in color:
        flat["color_drift_enabled"] = color["enabled"]
    if "gamma_range" in color:
        flat["gamma_range"] = color["gamma_range"]
    if "saturation_range" in color:
        flat["saturation_range"] = color["saturation_range"]
    if "contrast_range" in color:
        flat["contrast_range"] = color["contrast_range"]
    if "hue_range" in color:
        flat["hue_range"] = color["hue_range"]

    # Rotation Drift section (Wave 1)
    rotation = data.get("rotation", {}) or {}
    if "enabled" in rotation:
        flat["rotation_enabled"] = rotation["enabled"]
    if "max_rotation_degrees" in rotation:
        flat["max_rotation_degrees"] = rotation["max_rotation_degrees"]

    # Overlay section (Wave 1)
    overlay = data.get("overlay", {}) or {}
    if "enabled" in overlay:
        flat["overlay_enabled"] = overlay["enabled"]
    if "opacity_max" in overlay:
        flat["overlay_opacity_max"] = overlay["opacity_max"]
    if "grain_enabled" in overlay:
        flat["overlay_grain_enabled"] = overlay["grain_enabled"]
    if "vignette_enabled" in overlay:
        flat["overlay_vignette_enabled"] = overlay["vignette_enabled"]

    # GOP Perturbation section (Wave 1)
    gop = data.get("gop", {}) or {}
    if "enabled" in gop:
        flat["gop_perturbation_enabled"] = gop["enabled"]
    if "gop_min" in gop:
        flat["gop_min"] = gop["gop_min"]
    if "gop_max" in gop:
        flat["gop_max"] = gop["gop_max"]

    # Timing section
    timing = data.get("timing", {}) or {}
    if "change_interval" in timing:
        flat["change_interval"] = timing["change_interval"]

    # Performance section
    performance = data.get("performance", {}) or {}
    if "encoder" in performance:
        flat["encoder"] = performance["encoder"]
    if "encoder_preset" in performance:
        flat["encoder_preset"] = performance["encoder_preset"]
    if "parallel_workers" in performance:
        flat["parallel_workers"] = performance["parallel_workers"]

    return flat


def load_perturbation_config(
    yaml_path: str | Path,
    preset_override: str | None = None,
    param_overrides: dict[str, Any] | None = None,
) -> PerturbationConfig:
    """Load perturbation config from YAML with preset defaults and overrides.

    Resolution order (later wins):
    1. Preset defaults (from PRESETS dict)
    2. YAML file values
    3. preset_override (changes which preset is used)
    4. param_overrides (explicit parameter overrides)

    Args:
        yaml_path: Path to the perturbation YAML config file.
        preset_override: If provided, overrides the preset specified in YAML.
        param_overrides: If provided, dict of field_name -> value overrides
            applied on top of everything else.

    Returns:
        A validated PerturbationConfig instance.

    Raises:
        InvalidConfigError: If the YAML file is missing, has invalid syntax,
            or any parameter value is out of range.
    """
    # Step 1: Load YAML file
    data = load_yaml(yaml_path)

    # Step 2: Flatten YAML into config field names
    yaml_fields = _flatten_yaml_to_config_fields(data)

    # Step 3: Determine preset name
    preset_name = preset_override or yaml_fields.get("preset", "medium")
    if preset_name not in _VALID_PRESETS:
        raise InvalidConfigError(
            f"Invalid preset '{preset_name}' "
            f"(valid: {', '.join(sorted(_VALID_PRESETS))})"
        )

    # Step 4: Start with preset defaults as base.
    # Try to read preset values from YAML's "presets" section first;
    # fall back to hardcoded PRESETS dict if not found in YAML.
    yaml_presets = data.get("presets", {}) or {}
    yaml_preset_data = yaml_presets.get(preset_name, {}) or {}

    if yaml_preset_data:
        # Flatten the YAML preset section into config field names
        merged: dict[str, Any] = _flatten_yaml_to_config_fields(yaml_preset_data)
    else:
        # Fallback to hardcoded Python PRESETS dict
        merged = dict(PRESETS[preset_name])

    merged["preset"] = preset_name

    # Step 5: Apply YAML overrides on top of preset
    # (exclude 'preset' from yaml_fields since we already handled it)
    for key, value in yaml_fields.items():
        if key != "preset":
            merged[key] = value

    # Step 6: Apply param_overrides on top
    if param_overrides:
        merged.update(param_overrides)

    # Step 7: Ensure required fields have defaults
    merged.setdefault("input_path", "")
    merged.setdefault("output_path", "")
    merged.setdefault("seed", None)
    merged.setdefault("temporal_enabled", True)
    merged.setdefault("spatial_enabled", True)
    merged.setdefault("scene_enabled", False)
    merged.setdefault("audio_enabled", True)
    merged.setdefault("combo_enabled", False)
    merged.setdefault("combo_intensity_factor", 0.5)
    merged.setdefault("scene_threshold", 0.3)
    merged.setdefault("max_scene_duration", 5.0)
    merged.setdefault("transition_duration_ms", 500.0)
    merged.setdefault("encoder", "auto")
    merged.setdefault("encoder_preset", "fast")
    # Wave 1 defaults
    merged.setdefault("color_drift_enabled", True)
    merged.setdefault("gamma_range", 0.02)
    merged.setdefault("saturation_range", 0.03)
    merged.setdefault("contrast_range", 0.03)
    merged.setdefault("hue_range", 2.0)
    merged.setdefault("rotation_enabled", True)
    merged.setdefault("max_rotation_degrees", 1.0)
    merged.setdefault("overlay_enabled", True)
    merged.setdefault("overlay_opacity_max", 0.03)
    merged.setdefault("overlay_grain_enabled", True)
    merged.setdefault("overlay_vignette_enabled", True)
    merged.setdefault("gop_perturbation_enabled", True)
    merged.setdefault("gop_min", 15)
    merged.setdefault("gop_max", 120)

    # Step 8: Build config
    try:
        config = PerturbationConfig(**merged)
    except TypeError as exc:
        raise InvalidConfigError(
            f"Invalid perturbation config fields: {exc}"
        ) from exc

    # Step 9: Validate
    validate_config(config)

    return config
