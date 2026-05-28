"""Unit tests for perturbation_warp module (Wave 2)."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path so we can import without the full package chain
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from video_text_translator.perturbation_config import PerturbationConfig
from video_text_translator.perturbation_warp import LocalizedWarpProcessor


def _make_config(**overrides) -> PerturbationConfig:
    """Create a PerturbationConfig with sensible defaults for testing."""
    defaults = {
        "input_path": "test.mp4",
        "output_path": "out.mp4",
        "preset": "medium",
        "warp_enabled": True,
        "warp_grid_size": 8,
        "warp_max_displacement": 3.0,
        "base_frequency": 0.1,
        "scheduling_mode": "correlated",
    }
    defaults.update(overrides)
    return PerturbationConfig(**defaults)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestWarpInit:
    """Tests for LocalizedWarpProcessor initialization."""

    def test_creates_with_valid_config(self) -> None:
        """Processor initializes without error."""
        config = _make_config()
        rng = random.Random(42)
        proc = LocalizedWarpProcessor(config, 1920, 1080, rng)
        assert proc.grid_size == 8
        assert proc.max_displacement == 3.0
        assert not proc.skip

    def test_skip_when_displacement_zero(self) -> None:
        """Processor skips when max_displacement is 0."""
        config = _make_config(warp_max_displacement=0.0)
        rng = random.Random(42)
        proc = LocalizedWarpProcessor(config, 1920, 1080, rng)
        assert proc.skip

    def test_accepts_base_phase(self) -> None:
        """Processor accepts an external base_phase for correlation."""
        config = _make_config()
        rng = random.Random(42)
        proc = LocalizedWarpProcessor(config, 1920, 1080, rng, base_phase=1.5)
        assert proc._base_phase == 1.5

    def test_generates_random_phase_when_none(self) -> None:
        """Processor generates a random base_phase when not provided."""
        config = _make_config()
        rng = random.Random(42)
        proc = LocalizedWarpProcessor(config, 1920, 1080, rng)
        assert 0 <= proc._base_phase <= 2 * math.pi


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------


class TestWarpTransform:
    """Tests for LocalizedWarpProcessor.transform_frame()."""

    def test_output_same_shape_as_input(self) -> None:
        """Output frame has same dimensions as input."""
        config = _make_config(warp_grid_size=4, warp_max_displacement=2.0)
        rng = random.Random(42)
        proc = LocalizedWarpProcessor(config, 320, 240, rng)

        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        result = proc.transform_frame(frame, 1.0)
        assert result.shape == frame.shape

    def test_output_dtype_preserved(self) -> None:
        """Output frame preserves uint8 dtype."""
        config = _make_config(warp_grid_size=4, warp_max_displacement=2.0)
        rng = random.Random(42)
        proc = LocalizedWarpProcessor(config, 320, 240, rng)

        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        result = proc.transform_frame(frame, 1.0)
        assert result.dtype == np.uint8

    def test_skip_returns_same_frame(self) -> None:
        """When skip=True, returns the exact same frame."""
        config = _make_config(warp_max_displacement=0.0)
        rng = random.Random(42)
        proc = LocalizedWarpProcessor(config, 320, 240, rng)

        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        result = proc.transform_frame(frame, 1.0)
        np.testing.assert_array_equal(result, frame)

    def test_warp_modifies_frame(self) -> None:
        """Warp with non-zero displacement should modify the frame."""
        config = _make_config(warp_grid_size=4, warp_max_displacement=5.0)
        rng = random.Random(42)
        proc = LocalizedWarpProcessor(config, 320, 240, rng)

        # Use a frame with clear structure (gradient) so warp is visible
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for y in range(240):
            frame[y, :, :] = y  # Horizontal gradient

        result = proc.transform_frame(frame, 1.0)
        # The frame should be different (warp applied)
        assert not np.array_equal(result, frame)

    def test_different_timestamps_produce_different_results(self) -> None:
        """Different timestamps should produce different warps."""
        config = _make_config(warp_grid_size=4, warp_max_displacement=3.0)
        rng = random.Random(42)
        proc = LocalizedWarpProcessor(config, 320, 240, rng)

        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        result_t0 = proc.transform_frame(frame, 0.0)
        result_t5 = proc.transform_frame(frame, 5.0)
        # Should be different (oscillation changes over time)
        assert not np.array_equal(result_t0, result_t5)

    def test_reproducibility_with_same_seed(self) -> None:
        """Same seed produces same warp result."""
        config = _make_config(warp_grid_size=4, warp_max_displacement=3.0)
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

        proc1 = LocalizedWarpProcessor(config, 320, 240, random.Random(42))
        result1 = proc1.transform_frame(frame, 2.5)

        proc2 = LocalizedWarpProcessor(config, 320, 240, random.Random(42))
        result2 = proc2.transform_frame(frame, 2.5)

        np.testing.assert_array_equal(result1, result2)

    def test_grayscale_frame(self) -> None:
        """Warp works on grayscale (H, W) frames."""
        config = _make_config(warp_grid_size=4, warp_max_displacement=3.0)
        rng = random.Random(42)
        proc = LocalizedWarpProcessor(config, 320, 240, rng)

        frame = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
        result = proc.transform_frame(frame, 1.0)
        assert result.shape == (240, 320)

    def test_small_frame(self) -> None:
        """Warp works on very small frames."""
        config = _make_config(warp_grid_size=4, warp_max_displacement=1.0)
        rng = random.Random(42)
        proc = LocalizedWarpProcessor(config, 16, 16, rng)

        frame = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        result = proc.transform_frame(frame, 1.0)
        assert result.shape == (16, 16, 3)


# ---------------------------------------------------------------------------
# Displacement grid tests
# ---------------------------------------------------------------------------


class TestDisplacementGrid:
    """Tests for displacement computation internals."""

    def test_displacement_within_bounds(self) -> None:
        """All displacements should be within max_displacement."""
        config = _make_config(warp_grid_size=8, warp_max_displacement=5.0)
        rng = random.Random(42)
        proc = LocalizedWarpProcessor(config, 320, 240, rng)

        for t in [0.0, 1.0, 5.0, 10.0, 20.0]:
            dx_grid, dy_grid = proc._compute_displacement_grid(t)
            # Max possible is max_displacement * 1.0 (amplitude) * 1.0 (edge_factor)
            assert np.all(np.abs(dx_grid) <= 5.0 + 1e-6)
            assert np.all(np.abs(dy_grid) <= 5.0 + 1e-6)

    def test_edge_attenuation(self) -> None:
        """Edge control points should have reduced displacement."""
        config = _make_config(warp_grid_size=8, warp_max_displacement=5.0)
        rng = random.Random(42)
        proc = LocalizedWarpProcessor(config, 320, 240, rng)

        # Find a timestamp where there's non-zero displacement
        dx_grid, dy_grid = proc._compute_displacement_grid(1.0)

        # Corner points (0,0) should have zero displacement due to edge factor
        assert dx_grid[0, 0] == 0.0
        assert dy_grid[0, 0] == 0.0

    def test_interpolation_produces_full_resolution(self) -> None:
        """Interpolated displacement field matches frame dimensions."""
        config = _make_config(warp_grid_size=4, warp_max_displacement=3.0)
        rng = random.Random(42)
        proc = LocalizedWarpProcessor(config, 320, 240, rng)

        dx_grid = np.ones((4, 4), dtype=np.float32)
        dy_grid = np.ones((4, 4), dtype=np.float32)

        dx_field, dy_field = proc._interpolate_displacement_field(dx_grid, dy_grid)
        assert dx_field.shape == (240, 320)
        assert dy_field.shape == (240, 320)
