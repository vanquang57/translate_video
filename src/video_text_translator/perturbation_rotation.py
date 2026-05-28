"""Rotation drift processor for time-varying micro rotation.

This module implements the RotationDriftProcessor class which applies
time-varying small rotation angles to video frames with smooth
interpolation between segments.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from .perturbation_config import PerturbationConfig, SegmentParam
from .perturbation_scheduler import ParameterScheduler


class RotationDriftProcessor:
    """Applies time-varying micro rotation per frame.

    The rotation angle oscillates within ±max_rotation_degrees,
    changing per Change_Interval with smooth interpolation.
    Output maintains original resolution by cropping black borders.
    """

    def __init__(
        self,
        config: PerturbationConfig,
        width: int,
        height: int,
        rng: random.Random,
    ) -> None:
        """Initialize the rotation drift processor.

        Args:
            config: Perturbation configuration with rotation parameters.
            width: Frame width in pixels.
            height: Frame height in pixels.
            rng: Seeded Random instance for reproducibility.
        """
        self.config = config
        self.width = width
        self.height = height
        self.rng = rng

        # Skip condition: no rotation means passthrough
        self.skip = config.max_rotation_degrees == 0.0

        if self.skip:
            self._rotation_schedule: list[SegmentParam] = []
            return

        self._scheduler = ParameterScheduler(
            duration=3600.0,  # 1 hour max
            change_interval=config.change_interval,
            rng=rng,
        )

        # Rotation schedule: angle in [-max_rotation_degrees, +max_rotation_degrees]
        self._rotation_schedule = self._scheduler.schedule(
            -config.max_rotation_degrees, config.max_rotation_degrees
        )

    def transform_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> np.ndarray:
        """Apply rotation for the given timestamp, return frame at original resolution.

        Uses cv2.getRotationMatrix2D + cv2.warpAffine, then crops to
        the largest axis-aligned rectangle that fits within the rotated
        image (avoiding black borders).

        Args:
            frame: Input frame as numpy array (H, W, C) or (H, W).
            timestamp: Current time in seconds.

        Returns:
            Rotated frame at original resolution (self.width x self.height).
        """
        if self.skip:
            return frame

        angle = self._scheduler.interpolate(self._rotation_schedule, timestamp)

        # If angle is effectively zero, skip processing
        if abs(angle) < 1e-6:
            return frame

        h, w = frame.shape[:2]
        center = (w / 2.0, h / 2.0)

        # Get rotation matrix
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotate the frame (black fill for borders)
        rotated = cv2.warpAffine(
            frame, rot_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Crop to the largest inscribed rectangle to avoid border artifacts,
        # then resize back to original dimensions.
        crop_w, crop_h = self._inscribed_rect_size(w, h, angle)
        x = int((w - crop_w) / 2)
        y = int((h - crop_h) / 2)

        # Ensure valid crop bounds
        x = max(0, x)
        y = max(0, y)
        crop_w = min(crop_w, w - x)
        crop_h = min(crop_h, h - y)

        cropped = rotated[y:y + crop_h, x:x + crop_w]

        # Resize back to original dimensions
        result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return result

    @staticmethod
    def _inscribed_rect_size(
        width: int, height: int, angle_degrees: float
    ) -> tuple[int, int]:
        """Compute the largest axis-aligned rectangle inscribed in a rotated rectangle.

        For small angles (< 5°), this is a good approximation that avoids
        black borders after rotation.

        Args:
            width: Original width.
            height: Original height.
            angle_degrees: Rotation angle in degrees.

        Returns:
            (crop_width, crop_height) of the inscribed rectangle.
        """
        import math

        angle_rad = abs(math.radians(angle_degrees))
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        if sin_a == 0:
            return width, height

        # For small angles, the inscribed rectangle dimensions are:
        # w' = w * cos(a) - h * sin(a)
        # h' = h * cos(a) - w * sin(a)
        # But this can go negative for large angles, so we use a simpler approach
        # that works well for small angles (< 5 degrees):
        new_w = int(width * cos_a - height * sin_a)
        new_h = int(height * cos_a - width * sin_a)

        # For very small angles, the above might still be close to original
        # Ensure we don't get negative or zero dimensions
        new_w = max(new_w, int(width * 0.9))
        new_h = max(new_h, int(height * 0.9))

        # Don't exceed original dimensions
        new_w = min(new_w, width)
        new_h = min(new_h, height)

        return new_w, new_h
