"""Localized dynamic warp processor for per-region mesh distortion.

This module implements the LocalizedWarpProcessor class which applies
time-varying mesh distortion to video frames using a grid of control
points with oscillating displacements. This creates subtle per-region
warping that is much harder for detectors to normalize than global transforms.

Wave 2 feature: uses cv2.remap for efficient per-pixel mapping with
bilinear interpolation between control points.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

from .perturbation_config import PerturbationConfig


class LocalizedWarpProcessor:
    """Applies time-varying mesh distortion to video frames.

    Creates a grid of control points where each point has a displacement
    (dx, dy) that oscillates over time with its own phase but a shared
    frequency. The displacement field is interpolated smoothly between
    control points using bilinear interpolation via cv2.remap.
    """

    def __init__(
        self,
        config: PerturbationConfig,
        width: int,
        height: int,
        rng: random.Random,
        base_phase: float | None = None,
    ) -> None:
        """Initialize the warp processor.

        Args:
            config: Perturbation configuration with warp parameters.
            width: Frame width in pixels.
            height: Frame height in pixels.
            rng: Seeded Random instance for reproducibility.
            base_phase: Optional shared base phase for correlation with
                other processors. If None, a random phase is generated.
        """
        self.config = config
        self.width = width
        self.height = height
        self.rng = rng

        self.grid_size = config.warp_grid_size
        self.max_displacement = config.warp_max_displacement

        # Skip if displacement is zero
        self.skip = self.max_displacement <= 0.0

        if self.skip:
            return

        # Shared base phase for correlation with scheduling
        self._base_phase = base_phase if base_phase is not None else rng.uniform(0, 2 * math.pi)

        # Base frequency from config (with slight variation per control point)
        self._base_frequency = config.base_frequency

        # Generate per-control-point parameters
        # Each control point has: phase_x, phase_y, amplitude_x, amplitude_y, freq_x, freq_y
        n_points = self.grid_size * self.grid_size
        self._phases_x = [rng.uniform(0, 2 * math.pi) for _ in range(n_points)]
        self._phases_y = [rng.uniform(0, 2 * math.pi) for _ in range(n_points)]
        # Amplitude per point: random fraction of max_displacement [0.3, 1.0]
        self._amplitudes_x = [
            self.max_displacement * rng.uniform(0.3, 1.0) for _ in range(n_points)
        ]
        self._amplitudes_y = [
            self.max_displacement * rng.uniform(0.3, 1.0) for _ in range(n_points)
        ]
        # Frequency per point: slight variation around base (±30%)
        self._freqs_x = [
            self._base_frequency * rng.uniform(0.7, 1.3) for _ in range(n_points)
        ]
        self._freqs_y = [
            self._base_frequency * rng.uniform(0.7, 1.3) for _ in range(n_points)
        ]

        # Precompute the base coordinate grids for remap (pixel coordinates)
        # These are the identity mapping — each pixel maps to itself
        self._base_map_x = np.zeros((height, width), dtype=np.float32)
        self._base_map_y = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            self._base_map_x[y, :] = np.arange(width, dtype=np.float32)
            self._base_map_y[y, :] = float(y)

    def _compute_displacement_grid(self, timestamp: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute displacement at each control point for the given timestamp.

        Args:
            timestamp: Current time in seconds.

        Returns:
            Tuple of (dx_grid, dy_grid) arrays of shape (grid_size, grid_size)
            containing pixel displacements at each control point.
        """
        dx_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        dy_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                idx = row * self.grid_size + col
                # Sinusoidal oscillation with per-point phase and frequency
                dx = self._amplitudes_x[idx] * math.sin(
                    2 * math.pi * self._freqs_x[idx] * timestamp
                    + self._phases_x[idx] + self._base_phase
                )
                dy = self._amplitudes_y[idx] * math.sin(
                    2 * math.pi * self._freqs_y[idx] * timestamp
                    + self._phases_y[idx] + self._base_phase
                )

                # Reduce displacement at edges to avoid border artifacts
                # Edge factor: 0 at border, 1 in interior
                edge_factor_x = min(col, self.grid_size - 1 - col) / max(1, self.grid_size // 4)
                edge_factor_y = min(row, self.grid_size - 1 - row) / max(1, self.grid_size // 4)
                edge_factor = min(1.0, edge_factor_x) * min(1.0, edge_factor_y)

                dx_grid[row, col] = dx * edge_factor
                dy_grid[row, col] = dy * edge_factor

        return dx_grid, dy_grid

    def _interpolate_displacement_field(
        self, dx_grid: np.ndarray, dy_grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate control point displacements to full-resolution displacement field.

        Uses cv2.resize with bilinear interpolation to smoothly spread
        the control point displacements across the entire frame.

        Args:
            dx_grid: Control point X displacements (grid_size x grid_size).
            dy_grid: Control point Y displacements (grid_size x grid_size).

        Returns:
            Tuple of (dx_field, dy_field) arrays of shape (height, width).
        """
        # Resize from grid_size x grid_size to height x width using bilinear
        dx_field = cv2.resize(
            dx_grid, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        dy_field = cv2.resize(
            dy_grid, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        return dx_field, dy_field

    def transform_frame(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """Apply time-varying mesh warp to the frame.

        1. Compute current displacement for each control point based on timestamp
        2. Interpolate displacements to full resolution
        3. Build remap arrays (map_x, map_y) = base_coords + displacement
        4. Apply cv2.remap to frame
        5. Return warped frame at original resolution

        Args:
            frame: Input frame as numpy array (H, W, C) or (H, W).
            timestamp: Current time in seconds.

        Returns:
            Warped frame at original resolution.
        """
        if self.skip:
            return frame

        # Compute control point displacements
        dx_grid, dy_grid = self._compute_displacement_grid(timestamp)

        # Interpolate to full resolution
        dx_field, dy_field = self._interpolate_displacement_field(dx_grid, dy_grid)

        # Build remap arrays: destination pixel (x, y) maps FROM source (x + dx, y + dy)
        # cv2.remap maps: dst(x,y) = src(map_x(x,y), map_y(x,y))
        # So to shift pixels by (dx, dy), we set map = base - displacement
        # (pulling from the displaced source position)
        map_x = self._base_map_x + dx_field
        map_y = self._base_map_y + dy_field

        # Apply remap with bilinear interpolation
        warped = cv2.remap(
            frame, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        return warped
