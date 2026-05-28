"""Overlay processor for time-varying semi-transparent overlays.

This module implements the OverlayProcessor class which applies
time-varying noise grain and vignette overlays to video frames.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from .perturbation_config import PerturbationConfig, SegmentParam
from .perturbation_scheduler import ParameterScheduler


class OverlayProcessor:
    """Applies time-varying semi-transparent overlays per frame.

    Supports two overlay types:
    - Noise grain: low-alpha random noise texture
    - Vignette: animated vignette with varying intensity

    Both are controlled by a master opacity parameter that changes
    per Change_Interval with smooth interpolation.
    """

    def __init__(
        self,
        config: PerturbationConfig,
        width: int,
        height: int,
        rng: random.Random,
    ) -> None:
        """Initialize the overlay processor.

        Args:
            config: Perturbation configuration with overlay parameters.
            width: Frame width in pixels.
            height: Frame height in pixels.
            rng: Seeded Random instance for reproducibility.
        """
        self.config = config
        self.width = width
        self.height = height
        self.rng = rng

        # Skip condition: opacity at zero or both overlays disabled
        self.skip = (
            config.overlay_opacity_max == 0.0
            or (not config.overlay_grain_enabled and not config.overlay_vignette_enabled)
        )

        if self.skip:
            self._opacity_schedule: list[SegmentParam] = []
            self._vignette_schedule: list[SegmentParam] = []
            return

        self._scheduler = ParameterScheduler(
            duration=3600.0,  # 1 hour max
            change_interval=config.change_interval,
            rng=rng,
            interpolation_window=config.transition_window,
        )

        # Opacity schedule: values in [0.0, overlay_opacity_max]
        self._opacity_schedule = self._scheduler.schedule(
            0.0, config.overlay_opacity_max
        )

        # Vignette intensity schedule: values in [0.3, 1.0] (relative intensity)
        self._vignette_schedule = self._scheduler.schedule(0.3, 1.0)

        # Pre-compute base vignette mask (static shape, intensity varies)
        self._vignette_mask = self._create_vignette_mask(width, height)

        # NumPy RNG for noise generation (seeded from the main rng)
        self._np_rng = np.random.default_rng(rng.randint(0, 2**31))

    def transform_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> np.ndarray:
        """Apply overlay effects for the given timestamp.

        Args:
            frame: Input frame as numpy array (H, W, C) BGR uint8.
            timestamp: Current time in seconds.

        Returns:
            Frame with overlays applied, same resolution and dtype.
        """
        if self.skip:
            return frame

        opacity = self._scheduler.interpolate(self._opacity_schedule, timestamp)

        # If opacity is effectively zero, skip
        if opacity < 1e-6:
            return frame

        result = frame.astype(np.float32)

        # Apply noise grain overlay
        if self.config.overlay_grain_enabled:
            noise = self._generate_noise(frame.shape)
            result = result + opacity * noise

        # Apply vignette overlay
        if self.config.overlay_vignette_enabled:
            vignette_intensity = self._scheduler.interpolate(
                self._vignette_schedule, timestamp
            )
            vignette = self._vignette_mask * vignette_intensity * opacity
            # Vignette darkens edges: subtract from frame
            result = result - (vignette * 255.0)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _generate_noise(self, shape: tuple[int, ...]) -> np.ndarray:
        """Generate a noise texture for the current frame.

        Returns noise values centered around 0 with range [-128, 128].
        The noise is applied with the opacity multiplier, so actual
        effect is very subtle.

        Args:
            shape: Shape of the frame (H, W, C).

        Returns:
            Noise array as float32 with values in [-128, 128].
        """
        noise = self._np_rng.integers(
            -128, 129, size=shape, dtype=np.int16
        ).astype(np.float32)
        return noise

    @staticmethod
    def _create_vignette_mask(width: int, height: int) -> np.ndarray:
        """Create a vignette mask (darker at edges, zero at center).

        Returns a float32 array of shape (H, W, 1) with values in [0, 1],
        where 0 = center (no darkening) and 1 = corners (max darkening).

        Args:
            width: Frame width.
            height: Frame height.

        Returns:
            Vignette mask as float32 array (H, W, 1).
        """
        # Create coordinate grids normalized to [-1, 1]
        x = np.linspace(-1, 1, width, dtype=np.float32)
        y = np.linspace(-1, 1, height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        # Radial distance from center, normalized
        dist = np.sqrt(xx ** 2 + yy ** 2)

        # Smooth vignette: ramp from 0 at center to 1 at corners
        # Use a power curve for natural-looking falloff
        vignette = np.clip(dist / np.sqrt(2), 0, 1) ** 2

        # Add channel dimension for broadcasting
        return vignette[:, :, np.newaxis]
