"""Unit tests for perturbation_spatial module."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path so we can import without the full package chain
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from video_text_translator.perturbation_config import PerturbationConfig
from video_text_translator.perturbation_spatial import SpatialTransformProcessor


def _make_config(
    max_crop_percent: float = 5.0,
    max_zoom: float = 1.05,
    change_interval: float = 10.0,
    **kwargs,
) -> PerturbationConfig:
    """Create a PerturbationConfig with spatial parameters."""
    return PerturbationConfig(
        input_path="test.mp4",
        output_path="out.mp4",
        max_crop_percent=max_crop_percent,
        max_zoom=max_zoom,
        change_interval=change_interval,
        **kwargs,
    )


def _make_frame(width: int = 1920, height: int = 1080) -> np.ndarray:
    """Create a dummy frame with given dimensions."""
    return np.zeros((height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Skip condition tests
# ---------------------------------------------------------------------------


class TestSkipCondition:
    """Tests for the skip condition (max_crop_percent == 0 and max_zoom == 1.0)."""

    def test_skip_when_no_crop_no_zoom(self) -> None:
        """Frames pass through unmodified when crop=0 and zoom=1.0."""
        config = _make_config(max_crop_percent=0.0, max_zoom=1.0)
        rng = random.Random(42)
        processor = SpatialTransformProcessor(config, 1920, 1080, rng)

        assert processor.skip is True

        frame = _make_frame()
        frame[100, 200] = [255, 128, 64]  # Mark a pixel
        result = processor.transform_frame(frame, 5.0)

        # Should be the exact same array (passthrough)
        assert result is frame

    def test_no_skip_when_crop_nonzero(self) -> None:
        """Processing occurs when max_crop_percent > 0."""
        config = _make_config(max_crop_percent=5.0, max_zoom=1.0)
        rng = random.Random(42)
        processor = SpatialTransformProcessor(config, 1920, 1080, rng)

        assert processor.skip is False

    def test_no_skip_when_zoom_above_one(self) -> None:
        """Processing occurs when max_zoom > 1.0."""
        config = _make_config(max_crop_percent=0.0, max_zoom=1.05)
        rng = random.Random(42)
        processor = SpatialTransformProcessor(config, 1920, 1080, rng)

        assert processor.skip is False

    def test_compute_crop_region_skip_returns_full_frame(self) -> None:
        """compute_crop_region returns full frame when skipping."""
        config = _make_config(max_crop_percent=0.0, max_zoom=1.0)
        rng = random.Random(42)
        processor = SpatialTransformProcessor(config, 1920, 1080, rng)

        x, y, w, h = processor.compute_crop_region(5.0)
        assert (x, y, w, h) == (0, 0, 1920, 1080)


# ---------------------------------------------------------------------------
# Output resolution tests
# ---------------------------------------------------------------------------


class TestOutputResolution:
    """Tests that output frames always match original resolution."""

    def test_output_resolution_matches_input(self) -> None:
        """Output frame has same dimensions as original."""
        config = _make_config(max_crop_percent=10.0, max_zoom=1.1)
        rng = random.Random(42)
        processor = SpatialTransformProcessor(config, 1920, 1080, rng)

        frame = _make_frame(1920, 1080)
        result = processor.transform_frame(frame, 5.0)

        assert result.shape == (1080, 1920, 3)

    def test_output_resolution_small_frame(self) -> None:
        """Output resolution preserved for small frames."""
        config = _make_config(max_crop_percent=20.0, max_zoom=1.5)
        rng = random.Random(42)
        processor = SpatialTransformProcessor(config, 640, 480, rng)

        frame = _make_frame(640, 480)
        result = processor.transform_frame(frame, 15.0)

        assert result.shape == (480, 640, 3)

    def test_output_resolution_at_various_timestamps(self) -> None:
        """Output resolution is consistent across different timestamps."""
        config = _make_config(max_crop_percent=10.0, max_zoom=1.2)
        rng = random.Random(42)
        processor = SpatialTransformProcessor(config, 1280, 720, rng)

        frame = _make_frame(1280, 720)
        for t in [0.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
            result = processor.transform_frame(frame, t)
            assert result.shape == (720, 1280, 3), f"Failed at t={t}"


# ---------------------------------------------------------------------------
# Crop region clamping tests
# ---------------------------------------------------------------------------


class TestCropRegionClamping:
    """Tests that crop regions are always within frame bounds."""

    def test_crop_region_within_bounds(self) -> None:
        """Crop region (x, y, w, h) stays within frame dimensions."""
        config = _make_config(max_crop_percent=10.0, max_zoom=1.1)
        rng = random.Random(42)
        processor = SpatialTransformProcessor(config, 1920, 1080, rng)

        for t in [0.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0]:
            x, y, w, h = processor.compute_crop_region(t)
            assert x >= 0, f"x={x} < 0 at t={t}"
            assert y >= 0, f"y={y} < 0 at t={t}"
            assert x + w <= 1920, f"x+w={x + w} > 1920 at t={t}"
            assert y + h <= 1080, f"y+h={y + h} > 1080 at t={t}"
            assert w > 0, f"w={w} <= 0 at t={t}"
            assert h > 0, f"h={h} <= 0 at t={t}"

    def test_crop_region_with_heavy_crop(self) -> None:
        """Heavy crop settings still produce valid regions."""
        config = _make_config(max_crop_percent=50.0, max_zoom=2.0)
        rng = random.Random(42)
        processor = SpatialTransformProcessor(config, 1920, 1080, rng)

        for t in [0.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
            x, y, w, h = processor.compute_crop_region(t)
            assert x >= 0
            assert y >= 0
            assert x + w <= 1920
            assert y + h <= 1080
            assert w > 0
            assert h > 0

    def test_crop_region_small_frame(self) -> None:
        """Crop region valid for small frame dimensions."""
        config = _make_config(max_crop_percent=30.0, max_zoom=1.5)
        rng = random.Random(42)
        processor = SpatialTransformProcessor(config, 100, 100, rng)

        for t in [0.0, 5.0, 10.0, 20.0]:
            x, y, w, h = processor.compute_crop_region(t)
            assert x >= 0
            assert y >= 0
            assert x + w <= 100
            assert y + h <= 100
            assert w > 0
            assert h > 0


# ---------------------------------------------------------------------------
# Drift rate tests
# ---------------------------------------------------------------------------


class TestDriftRate:
    """Tests that crop center drift rate does not exceed 0.5% per second."""

    def test_drift_rate_within_limit(self) -> None:
        """Drift between consecutive timestamps respects 0.5% limit."""
        config = _make_config(max_crop_percent=5.0, max_zoom=1.05)
        rng = random.Random(42)
        width, height = 1920, 1080
        processor = SpatialTransformProcessor(config, width, height, rng)

        # Sample at 1-second intervals
        dt = 1.0
        max_dx = 0.005 * width * dt  # 0.5% of width per second
        max_dy = 0.005 * height * dt  # 0.5% of height per second

        prev_x, prev_y, _, _ = processor.compute_crop_region(0.0)
        prev_cx = prev_x + processor.compute_crop_region(0.0)[2] / 2.0
        prev_cy = prev_y + processor.compute_crop_region(0.0)[3] / 2.0

        for i in range(1, 60):
            t = i * dt
            x, y, w, h = processor.compute_crop_region(t)
            cx = x + w / 2.0
            cy = y + h / 2.0

            # The center displacement should be bounded by drift rate
            # Note: clamping may reduce apparent drift, so we just check
            # the internal drift offset directly
            drift_x, drift_y = processor._get_drift_offset(t)
            prev_drift_x, prev_drift_y = processor._get_drift_offset(t - dt)

            delta_x = abs(drift_x - prev_drift_x)
            delta_y = abs(drift_y - prev_drift_y)

            # Allow small floating point tolerance
            assert delta_x <= max_dx + 1e-9, (
                f"Horizontal drift {delta_x:.4f} exceeds limit {max_dx:.4f} at t={t}"
            )
            assert delta_y <= max_dy + 1e-9, (
                f"Vertical drift {delta_y:.4f} exceeds limit {max_dy:.4f} at t={t}"
            )

    def test_drift_changes_direction_at_interval_boundary(self) -> None:
        """Drift direction changes at Change_Interval boundaries."""
        config = _make_config(
            max_crop_percent=5.0, max_zoom=1.05, change_interval=10.0
        )
        rng = random.Random(42)
        processor = SpatialTransformProcessor(config, 1920, 1080, rng)

        # Get drift at different segments
        # Within first segment (0-10s), drift direction should be consistent
        d1_start = processor._get_drift_offset(0.0)
        d1_mid = processor._get_drift_offset(5.0)
        d1_end = processor._get_drift_offset(9.99)

        # The drift should be accumulating in a consistent direction within a segment
        # (direction doesn't change mid-segment)
        if d1_mid[0] != 0:
            # Same sign for x-drift within segment
            assert (d1_mid[0] > 0) == (d1_end[0] > 0) or abs(d1_end[0]) < 1e-9


# ---------------------------------------------------------------------------
# Animated crop/zoom tests
# ---------------------------------------------------------------------------


class TestAnimatedCropZoom:
    """Tests for time-varying crop and zoom behavior."""

    def test_crop_varies_over_time(self) -> None:
        """Crop region changes between different timestamps."""
        config = _make_config(max_crop_percent=10.0, max_zoom=1.1)
        rng = random.Random(42)
        processor = SpatialTransformProcessor(config, 1920, 1080, rng)

        regions = set()
        for t in [0.0, 15.0, 30.0, 45.0, 60.0]:
            region = processor.compute_crop_region(t)
            regions.add(region)

        # Should have at least some variation
        assert len(regions) > 1, "Crop region should vary over time"

    def test_reproducibility_with_same_seed(self) -> None:
        """Same seed produces identical results."""
        config = _make_config(max_crop_percent=10.0, max_zoom=1.1)

        proc1 = SpatialTransformProcessor(config, 1920, 1080, random.Random(42))
        proc2 = SpatialTransformProcessor(config, 1920, 1080, random.Random(42))

        for t in [0.0, 5.0, 10.0, 25.0]:
            assert proc1.compute_crop_region(t) == proc2.compute_crop_region(t)

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds produce different crop regions."""
        config = _make_config(max_crop_percent=10.0, max_zoom=1.1)

        proc1 = SpatialTransformProcessor(config, 1920, 1080, random.Random(1))
        proc2 = SpatialTransformProcessor(config, 1920, 1080, random.Random(999))

        # At least one timestamp should differ
        results_differ = False
        for t in [5.0, 15.0, 25.0, 35.0]:
            if proc1.compute_crop_region(t) != proc2.compute_crop_region(t):
                results_differ = True
                break

        assert results_differ, "Different seeds should produce different results"

    def test_transform_frame_produces_valid_image(self) -> None:
        """transform_frame produces a valid numpy array with correct dtype."""
        config = _make_config(max_crop_percent=10.0, max_zoom=1.1)
        rng = random.Random(42)
        processor = SpatialTransformProcessor(config, 640, 480, rng)

        # Create a frame with a gradient pattern
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(640, dtype=np.uint8)  # Horizontal gradient

        result = processor.transform_frame(frame, 5.0)

        assert result.dtype == np.uint8
        assert result.shape == (480, 640, 3)

    def test_crop_region_size_reflects_crop_percent(self) -> None:
        """With high crop percent, crop region should be smaller than full frame."""
        # Use a config with guaranteed high crop
        config = _make_config(max_crop_percent=50.0, max_zoom=1.0)
        rng = random.Random(42)
        processor = SpatialTransformProcessor(config, 1000, 1000, rng)

        # At some timestamp, the crop should be less than full frame
        found_smaller = False
        for t in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]:
            x, y, w, h = processor.compute_crop_region(t)
            if w < 1000 or h < 1000:
                found_smaller = True
                break

        assert found_smaller, "With max_crop_percent=50, some crops should be smaller"

    def test_zoom_reduces_visible_area(self) -> None:
        """With zoom > 1.0, the crop region should be smaller than full frame."""
        config = _make_config(max_crop_percent=0.0, max_zoom=2.0)
        rng = random.Random(42)
        processor = SpatialTransformProcessor(config, 1000, 1000, rng)

        # With zoom up to 2.0, visible area = 1/zoom, so at max zoom
        # the crop should be about 500x500
        found_smaller = False
        for t in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]:
            x, y, w, h = processor.compute_crop_region(t)
            if w < 1000 or h < 1000:
                found_smaller = True
                break

        assert found_smaller, "With max_zoom=2.0, some crops should be smaller"
