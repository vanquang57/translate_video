"""Unit tests for perturbation_color module."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from video_text_translator.perturbation_config import PerturbationConfig
from video_text_translator.perturbation_color import ColorDriftProcessor


def _make_config(
    gamma_range: float = 0.02,
    saturation_range: float = 0.03,
    contrast_range: float = 0.03,
    hue_range: float = 2.0,
    change_interval: float = 10.0,
    **kwargs,
) -> PerturbationConfig:
    """Create a PerturbationConfig with color drift parameters."""
    return PerturbationConfig(
        input_path="test.mp4",
        output_path="out.mp4",
        gamma_range=gamma_range,
        saturation_range=saturation_range,
        contrast_range=contrast_range,
        hue_range=hue_range,
        change_interval=change_interval,
        **kwargs,
    )


def _make_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a dummy frame with a color gradient."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Create a gradient pattern for meaningful color tests
    frame[:, :, 0] = np.tile(np.linspace(50, 200, width, dtype=np.uint8), (height, 1))
    frame[:, :, 1] = np.tile(
        np.linspace(30, 180, height, dtype=np.uint8).reshape(-1, 1), (1, width)
    )
    frame[:, :, 2] = 128
    return frame


# ---------------------------------------------------------------------------
# Skip condition tests
# ---------------------------------------------------------------------------


class TestSkipCondition:
    """Tests for the skip condition (all ranges at zero)."""

    def test_skip_when_all_ranges_zero(self) -> None:
        """Frames pass through unmodified when all color ranges are zero."""
        config = _make_config(
            gamma_range=0.0, saturation_range=0.0,
            contrast_range=0.0, hue_range=0.0,
        )
        rng = random.Random(42)
        processor = ColorDriftProcessor(config, rng)

        assert processor.skip is True

        frame = _make_frame()
        result = processor.transform_frame(frame, 5.0)
        assert result is frame

    def test_no_skip_when_gamma_nonzero(self) -> None:
        """Processing occurs when gamma_range > 0."""
        config = _make_config(
            gamma_range=0.02, saturation_range=0.0,
            contrast_range=0.0, hue_range=0.0,
        )
        rng = random.Random(42)
        processor = ColorDriftProcessor(config, rng)
        assert processor.skip is False

    def test_no_skip_when_hue_nonzero(self) -> None:
        """Processing occurs when hue_range > 0."""
        config = _make_config(
            gamma_range=0.0, saturation_range=0.0,
            contrast_range=0.0, hue_range=2.0,
        )
        rng = random.Random(42)
        processor = ColorDriftProcessor(config, rng)
        assert processor.skip is False


# ---------------------------------------------------------------------------
# Output resolution tests
# ---------------------------------------------------------------------------


class TestOutputResolution:
    """Tests that output frames always match input resolution."""

    def test_output_resolution_matches_input(self) -> None:
        """Output frame has same dimensions as input."""
        config = _make_config()
        rng = random.Random(42)
        processor = ColorDriftProcessor(config, rng)

        frame = _make_frame(640, 480)
        result = processor.transform_frame(frame, 5.0)
        assert result.shape == (480, 640, 3)

    def test_output_resolution_at_various_timestamps(self) -> None:
        """Output resolution is consistent across different timestamps."""
        config = _make_config()
        rng = random.Random(42)
        processor = ColorDriftProcessor(config, rng)

        frame = _make_frame(1280, 720)
        for t in [0.0, 5.0, 10.0, 25.0, 50.0]:
            result = processor.transform_frame(frame, t)
            assert result.shape == (720, 1280, 3), f"Failed at t={t}"


# ---------------------------------------------------------------------------
# Transform produces different output
# ---------------------------------------------------------------------------


class TestTransformEffect:
    """Tests that transforms produce different output than input."""

    def test_transform_modifies_frame(self) -> None:
        """Color drift produces a different frame than input."""
        config = _make_config(
            gamma_range=0.05, saturation_range=0.05,
            contrast_range=0.05, hue_range=3.0,
        )
        rng = random.Random(42)
        processor = ColorDriftProcessor(config, rng)

        frame = _make_frame()
        result = processor.transform_frame(frame, 5.0)

        # Should not be identical to input
        assert not np.array_equal(result, frame)

    def test_output_varies_over_time(self) -> None:
        """Color drift produces different results at different timestamps."""
        config = _make_config(
            gamma_range=0.05, saturation_range=0.05,
            contrast_range=0.05, hue_range=3.0,
        )
        rng = random.Random(42)
        processor = ColorDriftProcessor(config, rng)

        frame = _make_frame()
        results = set()
        for t in [0.0, 15.0, 30.0, 45.0, 60.0]:
            result = processor.transform_frame(frame, t)
            results.add(result.tobytes())

        assert len(results) > 1, "Color drift should vary over time"


# ---------------------------------------------------------------------------
# Pixel value range tests
# ---------------------------------------------------------------------------


class TestPixelRange:
    """Tests that output pixel values stay within [0, 255]."""

    def test_pixel_values_in_valid_range(self) -> None:
        """All output pixels are in [0, 255]."""
        config = _make_config(
            gamma_range=0.1, saturation_range=0.1,
            contrast_range=0.1, hue_range=5.0,
        )
        rng = random.Random(42)
        processor = ColorDriftProcessor(config, rng)

        frame = _make_frame()
        for t in [0.0, 5.0, 10.0, 20.0, 30.0]:
            result = processor.transform_frame(frame, t)
            assert result.dtype == np.uint8
            assert result.min() >= 0
            assert result.max() <= 255

    def test_pixel_values_with_extreme_input(self) -> None:
        """Handles frames with extreme pixel values (0 and 255)."""
        config = _make_config(
            gamma_range=0.1, saturation_range=0.1,
            contrast_range=0.1, hue_range=5.0,
        )
        rng = random.Random(42)
        processor = ColorDriftProcessor(config, rng)

        # Frame with all zeros
        frame_black = np.zeros((100, 100, 3), dtype=np.uint8)
        result = processor.transform_frame(frame_black, 5.0)
        assert result.dtype == np.uint8

        # Frame with all 255
        frame_white = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = processor.transform_frame(frame_white, 5.0)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Tests for deterministic behavior with same seed."""

    def test_same_seed_same_result(self) -> None:
        """Same seed produces identical results."""
        config = _make_config()
        frame = _make_frame()

        proc1 = ColorDriftProcessor(config, random.Random(42))
        proc2 = ColorDriftProcessor(config, random.Random(42))

        for t in [0.0, 5.0, 10.0, 25.0]:
            r1 = proc1.transform_frame(frame.copy(), t)
            r2 = proc2.transform_frame(frame.copy(), t)
            assert np.array_equal(r1, r2), f"Results differ at t={t}"

    def test_different_seeds_different_results(self) -> None:
        """Different seeds produce different results."""
        config = _make_config(
            gamma_range=0.05, saturation_range=0.05,
            contrast_range=0.05, hue_range=3.0,
        )
        frame = _make_frame()

        proc1 = ColorDriftProcessor(config, random.Random(1))
        proc2 = ColorDriftProcessor(config, random.Random(999))

        results_differ = False
        for t in [5.0, 15.0, 25.0]:
            r1 = proc1.transform_frame(frame.copy(), t)
            r2 = proc2.transform_frame(frame.copy(), t)
            if not np.array_equal(r1, r2):
                results_differ = True
                break

        assert results_differ, "Different seeds should produce different results"
