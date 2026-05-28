"""Unit tests for perturbation_overlay module."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from video_text_translator.perturbation_config import PerturbationConfig
from video_text_translator.perturbation_overlay import OverlayProcessor


def _make_config(
    overlay_opacity_max: float = 0.03,
    overlay_grain_enabled: bool = True,
    overlay_vignette_enabled: bool = True,
    change_interval: float = 10.0,
    **kwargs,
) -> PerturbationConfig:
    """Create a PerturbationConfig with overlay parameters."""
    return PerturbationConfig(
        input_path="test.mp4",
        output_path="out.mp4",
        overlay_opacity_max=overlay_opacity_max,
        overlay_grain_enabled=overlay_grain_enabled,
        overlay_vignette_enabled=overlay_vignette_enabled,
        change_interval=change_interval,
        **kwargs,
    )


def _make_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a dummy frame with mid-range values."""
    frame = np.full((height, width, 3), 128, dtype=np.uint8)
    return frame


# ---------------------------------------------------------------------------
# Skip condition tests
# ---------------------------------------------------------------------------


class TestSkipCondition:
    """Tests for the skip condition."""

    def test_skip_when_opacity_zero(self) -> None:
        """Frames pass through unmodified when overlay_opacity_max is 0."""
        config = _make_config(overlay_opacity_max=0.0)
        rng = random.Random(42)
        processor = OverlayProcessor(config, 640, 480, rng)

        assert processor.skip is True

        frame = _make_frame()
        result = processor.transform_frame(frame, 5.0)
        assert result is frame

    def test_skip_when_both_overlays_disabled(self) -> None:
        """Frames pass through when both grain and vignette are disabled."""
        config = _make_config(
            overlay_opacity_max=0.05,
            overlay_grain_enabled=False,
            overlay_vignette_enabled=False,
        )
        rng = random.Random(42)
        processor = OverlayProcessor(config, 640, 480, rng)

        assert processor.skip is True

        frame = _make_frame()
        result = processor.transform_frame(frame, 5.0)
        assert result is frame

    def test_no_skip_when_grain_enabled(self) -> None:
        """Processing occurs when grain is enabled with nonzero opacity."""
        config = _make_config(
            overlay_opacity_max=0.03,
            overlay_grain_enabled=True,
            overlay_vignette_enabled=False,
        )
        rng = random.Random(42)
        processor = OverlayProcessor(config, 640, 480, rng)
        assert processor.skip is False

    def test_no_skip_when_vignette_enabled(self) -> None:
        """Processing occurs when vignette is enabled with nonzero opacity."""
        config = _make_config(
            overlay_opacity_max=0.03,
            overlay_grain_enabled=False,
            overlay_vignette_enabled=True,
        )
        rng = random.Random(42)
        processor = OverlayProcessor(config, 640, 480, rng)
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
        processor = OverlayProcessor(config, 1920, 1080, rng)

        frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        result = processor.transform_frame(frame, 5.0)
        assert result.shape == (1080, 1920, 3)

    def test_output_resolution_small_frame(self) -> None:
        """Output resolution preserved for small frames."""
        config = _make_config()
        rng = random.Random(42)
        processor = OverlayProcessor(config, 320, 240, rng)

        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        result = processor.transform_frame(frame, 15.0)
        assert result.shape == (240, 320, 3)

    def test_output_resolution_at_various_timestamps(self) -> None:
        """Output resolution is consistent across different timestamps."""
        config = _make_config()
        rng = random.Random(42)
        processor = OverlayProcessor(config, 1280, 720, rng)

        frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
        for t in [0.0, 5.0, 10.0, 25.0, 50.0]:
            result = processor.transform_frame(frame, t)
            assert result.shape == (720, 1280, 3), f"Failed at t={t}"


# ---------------------------------------------------------------------------
# Transform produces different output
# ---------------------------------------------------------------------------


class TestTransformEffect:
    """Tests that overlays produce different output than input."""

    def test_grain_modifies_frame(self) -> None:
        """Noise grain produces a different frame than input."""
        config = _make_config(
            overlay_opacity_max=0.05,
            overlay_grain_enabled=True,
            overlay_vignette_enabled=False,
        )
        rng = random.Random(42)
        processor = OverlayProcessor(config, 640, 480, rng)

        frame = _make_frame()
        result = processor.transform_frame(frame, 5.0)

        # With grain, output should differ from input
        assert not np.array_equal(result, frame)

    def test_vignette_modifies_frame(self) -> None:
        """Vignette produces a different frame than input."""
        config = _make_config(
            overlay_opacity_max=0.05,
            overlay_grain_enabled=False,
            overlay_vignette_enabled=True,
        )
        rng = random.Random(42)
        processor = OverlayProcessor(config, 640, 480, rng)

        frame = _make_frame()
        result = processor.transform_frame(frame, 5.0)

        # With vignette, output should differ from input
        assert not np.array_equal(result, frame)

    def test_grain_varies_between_frames(self) -> None:
        """Noise grain produces different results for different frames."""
        config = _make_config(
            overlay_opacity_max=0.05,
            overlay_grain_enabled=True,
            overlay_vignette_enabled=False,
        )
        rng = random.Random(42)
        processor = OverlayProcessor(config, 640, 480, rng)

        frame = _make_frame()
        r1 = processor.transform_frame(frame.copy(), 5.0)
        r2 = processor.transform_frame(frame.copy(), 5.1)

        # Noise should be different between frames
        assert not np.array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Pixel value range tests
# ---------------------------------------------------------------------------


class TestPixelRange:
    """Tests that output pixel values stay within [0, 255]."""

    def test_pixel_values_in_valid_range(self) -> None:
        """All output pixels are in [0, 255]."""
        config = _make_config(overlay_opacity_max=0.1)
        rng = random.Random(42)
        processor = OverlayProcessor(config, 640, 480, rng)

        frame = _make_frame()
        for t in [0.0, 5.0, 10.0, 20.0, 30.0]:
            result = processor.transform_frame(frame, t)
            assert result.dtype == np.uint8
            assert result.min() >= 0
            assert result.max() <= 255

    def test_pixel_values_with_bright_frame(self) -> None:
        """Handles frames with high pixel values (near 255)."""
        config = _make_config(overlay_opacity_max=0.1)
        rng = random.Random(42)
        processor = OverlayProcessor(config, 640, 480, rng)

        frame = np.full((480, 640, 3), 250, dtype=np.uint8)
        result = processor.transform_frame(frame, 5.0)
        assert result.dtype == np.uint8
        assert result.max() <= 255

    def test_pixel_values_with_dark_frame(self) -> None:
        """Handles frames with low pixel values (near 0)."""
        config = _make_config(overlay_opacity_max=0.1)
        rng = random.Random(42)
        processor = OverlayProcessor(config, 640, 480, rng)

        frame = np.full((480, 640, 3), 5, dtype=np.uint8)
        result = processor.transform_frame(frame, 5.0)
        assert result.dtype == np.uint8
        assert result.min() >= 0


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Tests for deterministic behavior with same seed."""

    def test_same_seed_same_result_vignette_only(self) -> None:
        """Same seed produces identical vignette results."""
        config = _make_config(
            overlay_opacity_max=0.05,
            overlay_grain_enabled=False,
            overlay_vignette_enabled=True,
        )
        frame = _make_frame()

        proc1 = OverlayProcessor(config, 640, 480, random.Random(42))
        proc2 = OverlayProcessor(config, 640, 480, random.Random(42))

        for t in [0.0, 5.0, 10.0, 25.0]:
            r1 = proc1.transform_frame(frame.copy(), t)
            r2 = proc2.transform_frame(frame.copy(), t)
            assert np.array_equal(r1, r2), f"Results differ at t={t}"

    def test_same_seed_same_result_grain(self) -> None:
        """Same seed produces identical grain results."""
        config = _make_config(
            overlay_opacity_max=0.05,
            overlay_grain_enabled=True,
            overlay_vignette_enabled=False,
        )
        frame = _make_frame()

        proc1 = OverlayProcessor(config, 640, 480, random.Random(42))
        proc2 = OverlayProcessor(config, 640, 480, random.Random(42))

        for t in [0.0, 5.0, 10.0]:
            r1 = proc1.transform_frame(frame.copy(), t)
            r2 = proc2.transform_frame(frame.copy(), t)
            assert np.array_equal(r1, r2), f"Results differ at t={t}"
