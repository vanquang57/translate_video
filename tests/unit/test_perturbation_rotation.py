"""Unit tests for perturbation_rotation module."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from video_text_translator.perturbation_config import PerturbationConfig
from video_text_translator.perturbation_rotation import RotationDriftProcessor


def _make_config(
    max_rotation_degrees: float = 1.0,
    change_interval: float = 10.0,
    **kwargs,
) -> PerturbationConfig:
    """Create a PerturbationConfig with rotation parameters."""
    return PerturbationConfig(
        input_path="test.mp4",
        output_path="out.mp4",
        max_rotation_degrees=max_rotation_degrees,
        change_interval=change_interval,
        **kwargs,
    )


def _make_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a dummy frame with a pattern for rotation detection."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw a cross pattern that makes rotation visible
    cx, cy = width // 2, height // 2
    frame[cy - 2:cy + 2, :, :] = 200  # Horizontal line
    frame[:, cx - 2:cx + 2, :] = 200  # Vertical line
    return frame


# ---------------------------------------------------------------------------
# Skip condition tests
# ---------------------------------------------------------------------------


class TestSkipCondition:
    """Tests for the skip condition (max_rotation_degrees == 0)."""

    def test_skip_when_rotation_zero(self) -> None:
        """Frames pass through unmodified when max_rotation_degrees is 0."""
        config = _make_config(max_rotation_degrees=0.0)
        rng = random.Random(42)
        processor = RotationDriftProcessor(config, 640, 480, rng)

        assert processor.skip is True

        frame = _make_frame()
        result = processor.transform_frame(frame, 5.0)
        assert result is frame

    def test_no_skip_when_rotation_nonzero(self) -> None:
        """Processing occurs when max_rotation_degrees > 0."""
        config = _make_config(max_rotation_degrees=1.0)
        rng = random.Random(42)
        processor = RotationDriftProcessor(config, 640, 480, rng)
        assert processor.skip is False


# ---------------------------------------------------------------------------
# Output resolution tests
# ---------------------------------------------------------------------------


class TestOutputResolution:
    """Tests that output frames always match original resolution."""

    def test_output_resolution_matches_input(self) -> None:
        """Output frame has same dimensions as original."""
        config = _make_config(max_rotation_degrees=2.0)
        rng = random.Random(42)
        processor = RotationDriftProcessor(config, 1920, 1080, rng)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = processor.transform_frame(frame, 5.0)
        assert result.shape == (1080, 1920, 3)

    def test_output_resolution_small_frame(self) -> None:
        """Output resolution preserved for small frames."""
        config = _make_config(max_rotation_degrees=2.0)
        rng = random.Random(42)
        processor = RotationDriftProcessor(config, 320, 240, rng)

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = processor.transform_frame(frame, 15.0)
        assert result.shape == (240, 320, 3)

    def test_output_resolution_at_various_timestamps(self) -> None:
        """Output resolution is consistent across different timestamps."""
        config = _make_config(max_rotation_degrees=2.0)
        rng = random.Random(42)
        processor = RotationDriftProcessor(config, 1280, 720, rng)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for t in [0.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
            result = processor.transform_frame(frame, t)
            assert result.shape == (720, 1280, 3), f"Failed at t={t}"


# ---------------------------------------------------------------------------
# Transform produces different output
# ---------------------------------------------------------------------------


class TestTransformEffect:
    """Tests that rotation produces different output than input."""

    def test_transform_modifies_frame(self) -> None:
        """Rotation produces a different frame than input."""
        config = _make_config(max_rotation_degrees=2.0)
        rng = random.Random(42)
        processor = RotationDriftProcessor(config, 640, 480, rng)

        frame = _make_frame()
        result = processor.transform_frame(frame, 5.0)

        # Should not be identical to input (rotation changes pixel positions)
        assert not np.array_equal(result, frame)

    def test_output_varies_over_time(self) -> None:
        """Rotation produces different results at different timestamps."""
        config = _make_config(max_rotation_degrees=2.0)
        rng = random.Random(42)
        processor = RotationDriftProcessor(config, 640, 480, rng)

        frame = _make_frame()
        results = set()
        for t in [0.0, 15.0, 30.0, 45.0, 60.0]:
            result = processor.transform_frame(frame, t)
            results.add(result.tobytes())

        assert len(results) > 1, "Rotation should vary over time"


# ---------------------------------------------------------------------------
# Pixel value range tests
# ---------------------------------------------------------------------------


class TestPixelRange:
    """Tests that output pixel values stay within [0, 255]."""

    def test_pixel_values_in_valid_range(self) -> None:
        """All output pixels are in [0, 255]."""
        config = _make_config(max_rotation_degrees=5.0)
        rng = random.Random(42)
        processor = RotationDriftProcessor(config, 640, 480, rng)

        frame = _make_frame()
        for t in [0.0, 5.0, 10.0, 20.0, 30.0]:
            result = processor.transform_frame(frame, t)
            assert result.dtype == np.uint8
            assert result.min() >= 0
            assert result.max() <= 255


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Tests for deterministic behavior with same seed."""

    def test_same_seed_same_result(self) -> None:
        """Same seed produces identical results."""
        config = _make_config(max_rotation_degrees=2.0)
        frame = _make_frame()

        proc1 = RotationDriftProcessor(config, 640, 480, random.Random(42))
        proc2 = RotationDriftProcessor(config, 640, 480, random.Random(42))

        for t in [0.0, 5.0, 10.0, 25.0]:
            r1 = proc1.transform_frame(frame.copy(), t)
            r2 = proc2.transform_frame(frame.copy(), t)
            assert np.array_equal(r1, r2), f"Results differ at t={t}"

    def test_different_seeds_different_results(self) -> None:
        """Different seeds produce different results."""
        config = _make_config(max_rotation_degrees=2.0)
        frame = _make_frame()

        proc1 = RotationDriftProcessor(config, 640, 480, random.Random(1))
        proc2 = RotationDriftProcessor(config, 640, 480, random.Random(999))

        results_differ = False
        for t in [5.0, 15.0, 25.0]:
            r1 = proc1.transform_frame(frame.copy(), t)
            r2 = proc2.transform_frame(frame.copy(), t)
            if not np.array_equal(r1, r2):
                results_differ = True
                break

        assert results_differ, "Different seeds should produce different results"
