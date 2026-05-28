"""Unit tests for perturbation_combo module."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

# Add src to path so we can import without the full package chain
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from video_text_translator.perturbation_config import PerturbationConfig
from video_text_translator.perturbation_combo import (
    APPLICATION_ORDER,
    MultiTransformCombo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> PerturbationConfig:
    """Create a PerturbationConfig with sensible defaults and overrides."""
    defaults = {
        "input_path": "test.mp4",
        "output_path": "out.mp4",
        "preset": "medium",
        "temporal_enabled": True,
        "spatial_enabled": True,
        "audio_enabled": True,
        "combo_enabled": True,
        "combo_intensity_factor": 0.5,
        "speed_min": 0.99,
        "speed_max": 1.01,
        "max_frame_drop_percent": 1.0,
        "micro_offset_ms": 50.0,
        "max_crop_percent": 5.0,
        "max_zoom": 1.05,
        "audio_tempo_min": 0.99,
        "audio_tempo_max": 1.01,
        "eq_range_db": 2.0,
        "change_interval": 10.0,
    }
    defaults.update(overrides)
    return PerturbationConfig(**defaults)


# ---------------------------------------------------------------------------
# Group selection tests
# ---------------------------------------------------------------------------


class TestSelectGroups:
    """Tests for MultiTransformCombo.select_groups()."""

    def test_selects_two_or_three_groups(self) -> None:
        """Should select between 2 and 3 groups."""
        config = _make_config()
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)

        groups = combo.select_groups()
        assert 2 <= len(groups) <= 3

    def test_only_selects_enabled_groups(self) -> None:
        """Should only select groups that are enabled in config."""
        config = _make_config(temporal_enabled=False)
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)

        groups = combo.select_groups()
        assert "temporal" not in groups

    def test_all_three_enabled_can_select_all(self) -> None:
        """With all 3 groups enabled, can select all 3."""
        config = _make_config()
        # Try multiple seeds to find one that selects 3
        found_three = False
        for seed in range(100):
            rng = random.Random(seed)
            combo = MultiTransformCombo(config, rng)
            groups = combo.select_groups()
            if len(groups) == 3:
                found_three = True
                break
        assert found_three, "Should be able to select 3 groups with some seed"

    def test_two_enabled_always_selects_two(self) -> None:
        """With only 2 groups enabled, always selects exactly 2."""
        config = _make_config(audio_enabled=False)
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)

        groups = combo.select_groups()
        assert len(groups) == 2
        assert "audio" not in groups

    def test_one_enabled_returns_one(self) -> None:
        """With only 1 group enabled, returns that single group."""
        config = _make_config(temporal_enabled=False, spatial_enabled=False)
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)

        groups = combo.select_groups()
        assert groups == ["audio"]

    def test_none_enabled_returns_empty(self) -> None:
        """With no groups enabled, returns empty list."""
        config = _make_config(
            temporal_enabled=False, spatial_enabled=False, audio_enabled=False
        )
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)

        groups = combo.select_groups()
        assert groups == []

    def test_groups_sorted_in_application_order(self) -> None:
        """Selected groups should always be in Temporal → Spatial → Audio order."""
        config = _make_config()
        for seed in range(50):
            rng = random.Random(seed)
            combo = MultiTransformCombo(config, rng)
            groups = combo.select_groups()

            # Verify order matches APPLICATION_ORDER
            indices = [APPLICATION_ORDER.index(g) for g in groups]
            assert indices == sorted(indices), (
                f"Groups {groups} not in application order with seed {seed}"
            )

    def test_reproducible_with_same_seed(self) -> None:
        """Same seed should produce same group selection."""
        config = _make_config()
        combo1 = MultiTransformCombo(config, random.Random(42))
        combo2 = MultiTransformCombo(config, random.Random(42))

        assert combo1.select_groups() == combo2.select_groups()

    def test_different_seeds_can_produce_different_selections(self) -> None:
        """Different seeds should be able to produce different selections."""
        config = _make_config()
        selections = set()
        for seed in range(100):
            rng = random.Random(seed)
            combo = MultiTransformCombo(config, rng)
            groups = combo.select_groups()
            selections.add(tuple(groups))

        # With 3 groups, possible selections are: any 2 of 3 (3 combos) + all 3 (1)
        assert len(selections) > 1


# ---------------------------------------------------------------------------
# Intensity scaling tests
# ---------------------------------------------------------------------------


class TestScaleIntensity:
    """Tests for MultiTransformCombo.scale_intensity()."""

    def test_factor_one_returns_base_value(self) -> None:
        """With factor=1.0, result should equal base_value."""
        config = _make_config(combo_intensity_factor=1.0)
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)

        result = combo.scale_intensity(5.0, 0.0, 10.0)
        assert result == pytest.approx(5.0)

    def test_factor_zero_point_one_near_minimum(self) -> None:
        """With factor=0.1, result should be close to param_min."""
        config = _make_config(combo_intensity_factor=0.1)
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)

        result = combo.scale_intensity(5.0, 0.0, 10.0)
        # result = 0.0 + (5.0 - 0.0) * 0.1 = 0.5
        assert result == pytest.approx(0.5)

    def test_factor_half_scales_correctly(self) -> None:
        """With factor=0.5, result should be halfway between min and base."""
        config = _make_config(combo_intensity_factor=0.5)
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)

        result = combo.scale_intensity(10.0, 0.0, 20.0)
        # result = 0.0 + (10.0 - 0.0) * 0.5 = 5.0
        assert result == pytest.approx(5.0)

    def test_base_at_minimum_returns_minimum(self) -> None:
        """When base_value equals param_min, result should be param_min."""
        config = _make_config(combo_intensity_factor=0.5)
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)

        result = combo.scale_intensity(2.0, 2.0, 10.0)
        # result = 2.0 + (2.0 - 2.0) * 0.5 = 2.0
        assert result == pytest.approx(2.0)

    def test_base_at_maximum_with_factor_half(self) -> None:
        """When base_value equals param_max, result is midpoint with factor=0.5."""
        config = _make_config(combo_intensity_factor=0.5)
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)

        result = combo.scale_intensity(10.0, 0.0, 10.0)
        # result = 0.0 + (10.0 - 0.0) * 0.5 = 5.0
        assert result == pytest.approx(5.0)

    def test_result_clamped_to_range(self) -> None:
        """Result should be clamped within [param_min, param_max]."""
        config = _make_config(combo_intensity_factor=1.0)
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)

        # base_value at max, factor=1.0 → result = max (within range)
        result = combo.scale_intensity(10.0, 0.0, 10.0)
        assert 0.0 <= result <= 10.0

    def test_result_within_range_always(self) -> None:
        """Result should always be within [param_min, param_max]."""
        config = _make_config(combo_intensity_factor=0.5)
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)

        for base in [0.0, 2.5, 5.0, 7.5, 10.0]:
            result = combo.scale_intensity(base, 0.0, 10.0)
            assert 0.0 <= result <= 10.0, f"Result {result} out of range for base={base}"


# ---------------------------------------------------------------------------
# Fixed application order tests
# ---------------------------------------------------------------------------


class TestApplicationOrder:
    """Tests for fixed application order: Temporal → Spatial → Audio."""

    def test_apply_groups_in_order(self) -> None:
        """Groups should be applied in Temporal → Spatial → Audio order."""
        config = _make_config()
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)

        # Force all 3 groups selected
        combo._selected_groups = ["temporal", "spatial", "audio"]

        applied_order: list[str] = []

        def make_handler(name: str):
            def handler():
                applied_order.append(name)
            return handler

        handlers = {
            "temporal": make_handler("temporal"),
            "spatial": make_handler("spatial"),
            "audio": make_handler("audio"),
        }

        combo.apply_groups(handlers)
        assert applied_order == ["temporal", "spatial", "audio"]

    def test_skips_unselected_groups(self) -> None:
        """Only selected groups should be applied."""
        config = _make_config()
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)

        # Only temporal and audio selected
        combo._selected_groups = ["temporal", "audio"]

        applied_order: list[str] = []

        def make_handler(name: str):
            def handler():
                applied_order.append(name)
            return handler

        handlers = {
            "temporal": make_handler("temporal"),
            "spatial": make_handler("spatial"),
            "audio": make_handler("audio"),
        }

        combo.apply_groups(handlers)
        assert applied_order == ["temporal", "audio"]

    def test_order_maintained_regardless_of_selection(self) -> None:
        """Even if spatial and audio are selected (no temporal), order is maintained."""
        config = _make_config()
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)

        combo._selected_groups = ["spatial", "audio"]

        applied_order: list[str] = []

        def make_handler(name: str):
            def handler():
                applied_order.append(name)
            return handler

        handlers = {
            "temporal": make_handler("temporal"),
            "spatial": make_handler("spatial"),
            "audio": make_handler("audio"),
        }

        combo.apply_groups(handlers)
        assert applied_order == ["spatial", "audio"]


# ---------------------------------------------------------------------------
# Graceful degradation tests
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Tests for graceful degradation when a group fails."""

    def test_continues_after_group_failure(self) -> None:
        """If a group fails, remaining groups should still be applied."""
        config = _make_config()
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)

        combo._selected_groups = ["temporal", "spatial", "audio"]

        applied_order: list[str] = []

        def temporal_handler():
            applied_order.append("temporal")

        def spatial_handler():
            raise RuntimeError("Spatial processing failed")

        def audio_handler():
            applied_order.append("audio")

        handlers = {
            "temporal": temporal_handler,
            "spatial": spatial_handler,
            "audio": audio_handler,
        }

        result = combo.apply_groups(handlers)
        assert applied_order == ["temporal", "audio"]
        assert result == ["temporal", "audio"]

    def test_returns_only_successful_groups(self) -> None:
        """apply_groups should return only successfully applied groups."""
        config = _make_config()
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)

        combo._selected_groups = ["temporal", "spatial", "audio"]

        def failing_handler():
            raise ValueError("Something went wrong")

        def success_handler():
            pass

        handlers = {
            "temporal": failing_handler,
            "spatial": success_handler,
            "audio": failing_handler,
        }

        result = combo.apply_groups(handlers)
        assert result == ["spatial"]

    def test_all_groups_fail_returns_empty(self) -> None:
        """If all groups fail, returns empty list."""
        config = _make_config()
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)

        combo._selected_groups = ["temporal", "spatial", "audio"]

        def failing_handler():
            raise RuntimeError("Failed")

        handlers = {
            "temporal": failing_handler,
            "spatial": failing_handler,
            "audio": failing_handler,
        }

        result = combo.apply_groups(handlers)
        assert result == []

    def test_missing_handler_skipped_gracefully(self) -> None:
        """If a handler is not registered for a group, skip it."""
        config = _make_config()
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)

        combo._selected_groups = ["temporal", "spatial", "audio"]

        applied_order: list[str] = []

        def audio_handler():
            applied_order.append("audio")

        # Only audio handler registered
        handlers = {
            "audio": audio_handler,
        }

        result = combo.apply_groups(handlers)
        assert applied_order == ["audio"]
        assert result == ["audio"]


# ---------------------------------------------------------------------------
# Scaled config tests
# ---------------------------------------------------------------------------


class TestCreateScaledConfig:
    """Tests for MultiTransformCombo.create_scaled_config()."""

    def test_temporal_params_scaled(self) -> None:
        """Temporal parameters should be scaled toward no-effect values."""
        config = _make_config(
            combo_intensity_factor=0.5,
            speed_min=0.98,
            speed_max=1.02,
            max_frame_drop_percent=2.0,
            micro_offset_ms=100.0,
        )
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)
        combo._selected_groups = ["temporal"]

        scaled = combo.create_scaled_config()

        # speed_min: 1.0 + (0.98 - 1.0) * 0.5 = 1.0 + (-0.02 * 0.5) = 0.99
        assert scaled.speed_min == pytest.approx(0.99)
        # speed_max: 1.0 + (1.02 - 1.0) * 0.5 = 1.0 + (0.02 * 0.5) = 1.01
        assert scaled.speed_max == pytest.approx(1.01)
        # max_frame_drop_percent: 0.0 + (2.0 - 0.0) * 0.5 = 1.0
        assert scaled.max_frame_drop_percent == pytest.approx(1.0)
        # micro_offset_ms: 0.0 + (100.0 - 0.0) * 0.5 = 50.0
        assert scaled.micro_offset_ms == pytest.approx(50.0)

    def test_spatial_params_scaled(self) -> None:
        """Spatial parameters should be scaled toward no-effect values."""
        config = _make_config(
            combo_intensity_factor=0.5,
            max_crop_percent=10.0,
            max_zoom=1.10,
        )
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)
        combo._selected_groups = ["spatial"]

        scaled = combo.create_scaled_config()

        # max_crop_percent: 0.0 + (10.0 - 0.0) * 0.5 = 5.0
        assert scaled.max_crop_percent == pytest.approx(5.0)
        # max_zoom: 1.0 + (1.10 - 1.0) * 0.5 = 1.05
        assert scaled.max_zoom == pytest.approx(1.05)

    def test_audio_params_scaled(self) -> None:
        """Audio parameters should be scaled toward no-effect values."""
        config = _make_config(
            combo_intensity_factor=0.5,
            audio_tempo_min=0.98,
            audio_tempo_max=1.02,
            eq_range_db=4.0,
        )
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)
        combo._selected_groups = ["audio"]

        scaled = combo.create_scaled_config()

        # audio_tempo_min: 1.0 + (0.98 - 1.0) * 0.5 = 0.99
        assert scaled.audio_tempo_min == pytest.approx(0.99)
        # audio_tempo_max: 1.0 + (1.02 - 1.0) * 0.5 = 1.01
        assert scaled.audio_tempo_max == pytest.approx(1.01)
        # eq_range_db: 0.0 + (4.0 - 0.0) * 0.5 = 2.0
        assert scaled.eq_range_db == pytest.approx(2.0)

    def test_unselected_groups_not_scaled(self) -> None:
        """Parameters for unselected groups should remain unchanged."""
        config = _make_config(
            combo_intensity_factor=0.5,
            speed_min=0.98,
            speed_max=1.02,
            max_crop_percent=10.0,
            max_zoom=1.10,
        )
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)
        combo._selected_groups = ["temporal"]  # Only temporal selected

        scaled = combo.create_scaled_config()

        # Spatial params should be unchanged
        assert scaled.max_crop_percent == 10.0
        assert scaled.max_zoom == 1.10

    def test_factor_one_preserves_values(self) -> None:
        """With factor=1.0, scaled config should match original."""
        config = _make_config(
            combo_intensity_factor=1.0,
            speed_min=0.98,
            speed_max=1.02,
            max_frame_drop_percent=2.0,
        )
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)
        combo._selected_groups = ["temporal"]

        scaled = combo.create_scaled_config()

        # speed_min: 1.0 + (0.98 - 1.0) * 1.0 = 0.98
        assert scaled.speed_min == pytest.approx(0.98)
        # speed_max: 1.0 + (1.02 - 1.0) * 1.0 = 1.02
        assert scaled.speed_max == pytest.approx(1.02)
        # max_frame_drop_percent: 0.0 + (2.0 - 0.0) * 1.0 = 2.0
        assert scaled.max_frame_drop_percent == pytest.approx(2.0)

    def test_scaled_config_is_immutable(self) -> None:
        """Scaled config should be a new frozen dataclass instance."""
        config = _make_config(combo_intensity_factor=0.5)
        rng = random.Random(42)
        combo = MultiTransformCombo(config, rng)
        combo._selected_groups = ["temporal", "spatial"]

        scaled = combo.create_scaled_config()

        # Should be a different object
        assert scaled is not config
        # Should still be frozen
        with pytest.raises(Exception):
            scaled.speed_min = 0.5  # type: ignore[misc]
