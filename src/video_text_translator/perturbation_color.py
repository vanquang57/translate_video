"""Color drift processor for time-varying color adjustments.

This module implements the ColorDriftProcessor class which applies
time-varying gamma, saturation, contrast, and hue adjustments to
video frames using smooth interpolation between segments.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from .perturbation_config import PerturbationConfig, SegmentParam
from .perturbation_scheduler import ParameterScheduler


class ColorDriftProcessor:
    """Applies time-varying color adjustments per frame.

    Parameters drift over time using ParameterScheduler:
    - Gamma: applied via LUT (lookup table) for efficiency
    - Saturation: multiplied in HSV space
    - Contrast: multiplied around mean intensity
    - Hue: shifted in HSV space
    """

    def __init__(
        self,
        config: PerturbationConfig,
        rng: random.Random,
    ) -> None:
        """Initialize the color drift processor.

        Args:
            config: Perturbation configuration with color drift parameters.
            rng: Seeded Random instance for reproducibility.
        """
        self.config = config
        self.rng = rng

        # Skip condition: all ranges at zero means passthrough
        self.skip = (
            config.gamma_range == 0.0
            and config.saturation_range == 0.0
            and config.contrast_range == 0.0
            and config.hue_range == 0.0
        )

        if self.skip:
            self._gamma_schedule: list[SegmentParam] = []
            self._saturation_schedule: list[SegmentParam] = []
            self._contrast_schedule: list[SegmentParam] = []
            self._hue_schedule: list[SegmentParam] = []
            return

        self._scheduler = ParameterScheduler(
            duration=3600.0,  # 1 hour max
            change_interval=config.change_interval,
            rng=rng,
            interpolation_window=config.transition_window,
        )

        # Gamma schedule: values in [1.0 - gamma_range, 1.0 + gamma_range]
        # Clamped to [0.9, 1.1]
        gamma_min = max(0.9, 1.0 - config.gamma_range)
        gamma_max = min(1.1, 1.0 + config.gamma_range)
        self._gamma_schedule = self._scheduler.schedule(gamma_min, gamma_max)

        # Saturation schedule: multiplier in [1.0 - sat_range, 1.0 + sat_range]
        # Clamped to [0.9, 1.1]
        sat_min = max(0.9, 1.0 - config.saturation_range)
        sat_max = min(1.1, 1.0 + config.saturation_range)
        self._saturation_schedule = self._scheduler.schedule(sat_min, sat_max)

        # Contrast schedule: multiplier in [1.0 - contrast_range, 1.0 + contrast_range]
        # Clamped to [0.9, 1.1]
        con_min = max(0.9, 1.0 - config.contrast_range)
        con_max = min(1.1, 1.0 + config.contrast_range)
        self._contrast_schedule = self._scheduler.schedule(con_min, con_max)

        # Hue schedule: shift in [-hue_range, +hue_range] degrees
        # Clamped to [-5, 5]
        hue_min = max(-5.0, -config.hue_range)
        hue_max = min(5.0, config.hue_range)
        self._hue_schedule = self._scheduler.schedule(hue_min, hue_max)

    def transform_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> np.ndarray:
        """Apply color drift adjustments for the given timestamp.

        Args:
            frame: Input frame as numpy array (H, W, C) BGR uint8.
            timestamp: Current time in seconds.

        Returns:
            Color-adjusted frame at same resolution and dtype.
        """
        if self.skip:
            return frame

        # Get interpolated parameter values
        gamma = self._scheduler.interpolate(self._gamma_schedule, timestamp)
        saturation = self._scheduler.interpolate(self._saturation_schedule, timestamp)
        contrast = self._scheduler.interpolate(self._contrast_schedule, timestamp)
        hue_shift = self._scheduler.interpolate(self._hue_schedule, timestamp)

        result = frame

        # Apply gamma via LUT (efficient for per-frame processing)
        if gamma != 1.0:
            result = self._apply_gamma(result, gamma)

        # Apply contrast (multiply around mean)
        if contrast != 1.0:
            result = self._apply_contrast(result, contrast)

        # Apply saturation and hue in HSV space
        if saturation != 1.0 or hue_shift != 0.0:
            result = self._apply_hsv_adjustments(result, saturation, hue_shift)

        return result

    @staticmethod
    def _apply_gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction using a lookup table."""
        inv_gamma = 1.0 / gamma
        table = np.array(
            [(i / 255.0) ** inv_gamma * 255 for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(frame, table)

    @staticmethod
    def _apply_contrast(frame: np.ndarray, factor: float) -> np.ndarray:
        """Apply contrast adjustment by multiplying around mean intensity."""
        mean = np.mean(frame).astype(np.float32)
        adjusted = mean + factor * (frame.astype(np.float32) - mean)
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_hsv_adjustments(
        frame: np.ndarray, saturation: float, hue_shift: float
    ) -> np.ndarray:
        """Apply saturation multiplier and hue shift in HSV space."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Hue channel is in [0, 180] in OpenCV (half-degrees)
        if hue_shift != 0.0:
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift / 2.0) % 180.0

        # Saturation channel is in [0, 255]
        if saturation != 1.0:
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)

        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
