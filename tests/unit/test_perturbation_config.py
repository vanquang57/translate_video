"""Unit tests for perturbation_config module."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path so we can import without the full package chain
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from video_text_translator.errors import InvalidConfigError
from video_text_translator.perturbation_config import (
    PRESETS,
    PerturbationConfig,
    Scene,
    SceneBoundary,
    SegmentParam,
    _flatten_yaml_to_config_fields,
    load_perturbation_config,
    validate_config,
)


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestSegmentParam:
    def test_creation(self) -> None:
        sp = SegmentParam(start_time=0.0, end_time=10.0, value=0.995)
        assert sp.start_time == 0.0
        assert sp.end_time == 10.0
        assert sp.value == 0.995

    def test_frozen(self) -> None:
        sp = SegmentParam(start_time=0.0, end_time=10.0, value=0.995)
        with pytest.raises(AttributeError):
            sp.value = 1.0  # type: ignore[misc]


class TestSceneBoundary:
    def test_creation(self) -> None:
        sb = SceneBoundary(frame_index=150, timestamp=5.0, histogram_diff=0.45)
        assert sb.frame_index == 150
        assert sb.timestamp == 5.0
        assert sb.histogram_diff == 0.45

    def test_frozen(self) -> None:
        sb = SceneBoundary(frame_index=150, timestamp=5.0, histogram_diff=0.45)
        with pytest.raises(AttributeError):
            sb.frame_index = 200  # type: ignore[misc]


class TestScene:
    def test_creation(self) -> None:
        s = Scene(
            start_frame=0, end_frame=150,
            start_time=0.0, end_time=5.0, is_micro=True,
        )
        assert s.start_frame == 0
        assert s.end_frame == 150
        assert s.start_time == 0.0
        assert s.end_time == 5.0
        assert s.is_micro is True

    def test_frozen(self) -> None:
        s = Scene(
            start_frame=0, end_frame=150,
            start_time=0.0, end_time=5.0, is_micro=True,
        )
        with pytest.raises(AttributeError):
            s.is_micro = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PerturbationConfig tests
# ---------------------------------------------------------------------------


class TestPerturbationConfig:
    def test_default_values(self) -> None:
        cfg = PerturbationConfig(input_path="in.mp4", output_path="out.mp4")
        assert cfg.preset == "medium"
        assert cfg.seed is None
        assert cfg.temporal_enabled is True
        assert cfg.speed_min == 0.99
        assert cfg.speed_max == 1.01
        assert cfg.max_frame_drop_percent == 1.0
        assert cfg.micro_offset_ms == 50.0
        assert cfg.spatial_enabled is True
        assert cfg.max_crop_percent == 5.0
        assert cfg.max_zoom == 1.05
        assert cfg.scene_enabled is False
        assert cfg.scene_threshold == 0.3
        assert cfg.max_scene_duration == 5.0
        assert cfg.transition_duration_ms == 500.0
        assert cfg.audio_enabled is True
        assert cfg.audio_tempo_min == 0.99
        assert cfg.audio_tempo_max == 1.01
        assert cfg.eq_range_db == 2.0
        assert cfg.ambience_volume_min_db == -40.0
        assert cfg.ambience_volume_max_db == -30.0
        assert cfg.combo_enabled is False
        assert cfg.combo_intensity_factor == 0.5
        assert cfg.change_interval == 10.0
        assert cfg.encoder == "auto"
        assert cfg.encoder_preset == "fast"

    def test_frozen(self) -> None:
        cfg = PerturbationConfig(input_path="in.mp4", output_path="out.mp4")
        with pytest.raises(AttributeError):
            cfg.speed_min = 0.95  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PRESETS tests
# ---------------------------------------------------------------------------


class TestPresets:
    def test_light_preset_exists(self) -> None:
        assert "light" in PRESETS
        assert PRESETS["light"]["speed_min"] == 0.995
        assert PRESETS["light"]["max_crop_percent"] == 3.0

    def test_medium_preset_exists(self) -> None:
        assert "medium" in PRESETS
        assert PRESETS["medium"]["speed_min"] == 0.99
        assert PRESETS["medium"]["max_crop_percent"] == 5.0

    def test_heavy_preset_exists(self) -> None:
        assert "heavy" in PRESETS
        assert PRESETS["heavy"]["speed_min"] == 0.97
        assert PRESETS["heavy"]["max_crop_percent"] == 10.0

    def test_all_presets_have_same_keys(self) -> None:
        keys = set(PRESETS["light"].keys())
        assert set(PRESETS["medium"].keys()) == keys
        assert set(PRESETS["heavy"].keys()) == keys


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_default_config(self) -> None:
        cfg = PerturbationConfig(input_path="in.mp4", output_path="out.mp4")
        validate_config(cfg)  # Should not raise

    def test_invalid_preset(self) -> None:
        cfg = PerturbationConfig(
            input_path="in.mp4", output_path="out.mp4",
            preset="extreme",  # type: ignore[arg-type]
        )
        with pytest.raises(InvalidConfigError, match="Invalid preset"):
            validate_config(cfg)

    def test_invalid_encoder(self) -> None:
        cfg = PerturbationConfig(
            input_path="in.mp4", output_path="out.mp4",
            encoder="h265",
        )
        with pytest.raises(InvalidConfigError, match="Invalid encoder"):
            validate_config(cfg)

    def test_invalid_encoder_preset(self) -> None:
        cfg = PerturbationConfig(
            input_path="in.mp4", output_path="out.mp4",
            encoder_preset="slow",
        )
        with pytest.raises(InvalidConfigError, match="Invalid encoder_preset"):
            validate_config(cfg)

    def test_speed_min_too_low(self) -> None:
        cfg = PerturbationConfig(
            input_path="in.mp4", output_path="out.mp4",
            speed_min=0.5,
        )
        with pytest.raises(InvalidConfigError, match="speed_min"):
            validate_config(cfg)

    def test_speed_max_too_high(self) -> None:
        cfg = PerturbationConfig(
            input_path="in.mp4", output_path="out.mp4",
            speed_max=1.5,
        )
        with pytest.raises(InvalidConfigError, match="speed_max"):
            validate_config(cfg)

    def test_max_crop_percent_too_high(self) -> None:
        cfg = PerturbationConfig(
            input_path="in.mp4", output_path="out.mp4",
            max_crop_percent=60.0,
        )
        with pytest.raises(InvalidConfigError, match="max_crop_percent"):
            validate_config(cfg)

    def test_max_zoom_too_low(self) -> None:
        cfg = PerturbationConfig(
            input_path="in.mp4", output_path="out.mp4",
            max_zoom=0.5,
        )
        with pytest.raises(InvalidConfigError, match="max_zoom"):
            validate_config(cfg)

    def test_change_interval_too_low(self) -> None:
        cfg = PerturbationConfig(
            input_path="in.mp4", output_path="out.mp4",
            change_interval=0.1,
        )
        with pytest.raises(InvalidConfigError, match="change_interval"):
            validate_config(cfg)

    def test_combo_intensity_factor_too_low(self) -> None:
        cfg = PerturbationConfig(
            input_path="in.mp4", output_path="out.mp4",
            combo_intensity_factor=0.05,
        )
        with pytest.raises(InvalidConfigError, match="combo_intensity_factor"):
            validate_config(cfg)

    def test_eq_range_db_too_high(self) -> None:
        cfg = PerturbationConfig(
            input_path="in.mp4", output_path="out.mp4",
            eq_range_db=15.0,
        )
        with pytest.raises(InvalidConfigError, match="eq_range_db"):
            validate_config(cfg)

    def test_ambience_volume_min_db_too_low(self) -> None:
        cfg = PerturbationConfig(
            input_path="in.mp4", output_path="out.mp4",
            ambience_volume_min_db=-70.0,
        )
        with pytest.raises(InvalidConfigError, match="ambience_volume_min_db"):
            validate_config(cfg)

    def test_transition_duration_ms_too_low(self) -> None:
        cfg = PerturbationConfig(
            input_path="in.mp4", output_path="out.mp4",
            transition_duration_ms=50.0,
        )
        with pytest.raises(InvalidConfigError, match="transition_duration_ms"):
            validate_config(cfg)

    def test_scene_threshold_too_low(self) -> None:
        cfg = PerturbationConfig(
            input_path="in.mp4", output_path="out.mp4",
            scene_threshold=0.05,
        )
        with pytest.raises(InvalidConfigError, match="scene_threshold"):
            validate_config(cfg)

    def test_boundary_values_valid(self) -> None:
        """Boundary values at the edge of valid ranges should pass."""
        cfg = PerturbationConfig(
            input_path="in.mp4", output_path="out.mp4",
            speed_min=0.90,
            speed_max=1.10,
            max_frame_drop_percent=20.0,
            micro_offset_ms=500.0,
            max_crop_percent=50.0,
            max_zoom=2.0,
            scene_threshold=1.0,
            max_scene_duration=60.0,
            transition_duration_ms=2000.0,
            audio_tempo_min=0.90,
            audio_tempo_max=1.10,
            eq_range_db=12.0,
            ambience_volume_min_db=-60.0,
            ambience_volume_max_db=0.0,
            combo_intensity_factor=1.0,
            change_interval=300.0,
        )
        validate_config(cfg)  # Should not raise


# ---------------------------------------------------------------------------
# YAML loading tests
# ---------------------------------------------------------------------------


class TestLoadPerturbationConfig:
    def _write_yaml(self, content: str) -> Path:
        """Write YAML content to a temp file and return its path."""
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        f.write(content)
        f.close()
        return Path(f.name)

    def test_load_minimal_yaml(self) -> None:
        path = self._write_yaml("preset: light\n")
        cfg = load_perturbation_config(path)
        assert cfg.preset == "light"
        # Should have light preset values
        assert cfg.speed_min == 0.995
        assert cfg.max_crop_percent == 3.0

    def test_load_with_preset_override(self) -> None:
        path = self._write_yaml("preset: light\n")
        cfg = load_perturbation_config(path, preset_override="heavy")
        assert cfg.preset == "heavy"
        assert cfg.speed_min == 0.97

    def test_load_with_param_overrides(self) -> None:
        path = self._write_yaml("preset: medium\n")
        cfg = load_perturbation_config(
            path, param_overrides={"max_crop_percent": 8.0}
        )
        assert cfg.max_crop_percent == 8.0
        # Other medium values should remain
        assert cfg.speed_min == 0.99

    def test_yaml_overrides_preset(self) -> None:
        yaml_content = """\
preset: light
temporal:
  speed_min: 0.99
"""
        path = self._write_yaml(yaml_content)
        cfg = load_perturbation_config(path)
        # YAML override takes precedence over preset
        assert cfg.speed_min == 0.99
        # Other light values remain
        assert cfg.max_crop_percent == 3.0

    def test_param_overrides_take_precedence(self) -> None:
        yaml_content = """\
preset: medium
temporal:
  speed_min: 0.98
"""
        path = self._write_yaml(yaml_content)
        cfg = load_perturbation_config(
            path, param_overrides={"speed_min": 0.95}
        )
        # param_overrides win over YAML
        assert cfg.speed_min == 0.95

    def test_missing_file_raises(self) -> None:
        with pytest.raises(InvalidConfigError, match="not found"):
            load_perturbation_config("/nonexistent/path.yaml")

    def test_invalid_yaml_raises(self) -> None:
        path = self._write_yaml("{{invalid yaml content")
        with pytest.raises(InvalidConfigError):
            load_perturbation_config(path)

    def test_invalid_preset_in_yaml_raises(self) -> None:
        path = self._write_yaml("preset: extreme\n")
        with pytest.raises(InvalidConfigError, match="Invalid preset"):
            load_perturbation_config(path)

    def test_invalid_preset_override_raises(self) -> None:
        path = self._write_yaml("preset: medium\n")
        with pytest.raises(InvalidConfigError, match="Invalid preset"):
            load_perturbation_config(path, preset_override="ultra")

    def test_out_of_range_param_raises(self) -> None:
        yaml_content = """\
preset: medium
temporal:
  speed_min: 0.5
"""
        path = self._write_yaml(yaml_content)
        with pytest.raises(InvalidConfigError, match="speed_min"):
            load_perturbation_config(path)

    def test_load_existing_perturbation_yaml(self) -> None:
        """Load the actual configs/perturbation.yaml file."""
        yaml_path = Path(__file__).resolve().parents[2] / "configs" / "perturbation.yaml"
        if yaml_path.exists():
            cfg = load_perturbation_config(yaml_path)
            assert cfg.preset == "medium"
            assert cfg.temporal_enabled is True

    def test_empty_yaml_uses_defaults(self) -> None:
        path = self._write_yaml("")
        cfg = load_perturbation_config(path)
        # Should use medium preset defaults
        assert cfg.preset == "medium"
        assert cfg.speed_min == 0.99


# ---------------------------------------------------------------------------
# Flatten YAML tests
# ---------------------------------------------------------------------------


class TestFlattenYaml:
    def test_flatten_temporal(self) -> None:
        data = {"temporal": {"enabled": False, "speed_min": 0.95}}
        flat = _flatten_yaml_to_config_fields(data)
        assert flat["temporal_enabled"] is False
        assert flat["speed_min"] == 0.95

    def test_flatten_audio(self) -> None:
        data = {"audio": {"tempo_min": 0.98, "tempo_max": 1.02}}
        flat = _flatten_yaml_to_config_fields(data)
        assert flat["audio_tempo_min"] == 0.98
        assert flat["audio_tempo_max"] == 1.02

    def test_flatten_combo(self) -> None:
        data = {"combo": {"enabled": True, "intensity_factor": 0.7}}
        flat = _flatten_yaml_to_config_fields(data)
        assert flat["combo_enabled"] is True
        assert flat["combo_intensity_factor"] == 0.7

    def test_flatten_empty(self) -> None:
        flat = _flatten_yaml_to_config_fields({})
        assert flat == {}
