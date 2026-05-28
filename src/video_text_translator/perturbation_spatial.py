"""Spatial transform processor for animated crop/zoom perturbation.

This module implements the SpatialTransformProcessor class which applies
time-varying crop and zoom transformations to video frames, with smooth
drift of the crop center position.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

from .perturbation_config import PerturbationConfig, SegmentParam
from .perturbation_scheduler import ParameterScheduler


class SpatialTransformProcessor:
    """Applies animated crop/zoom per frame.

    The processor uses ParameterScheduler to generate time-varying crop
    percentages and zoom levels, and applies a drifting crop center that
    moves at most 0.5% of frame dimensions per second.
    """

    # Maximum drift rate: 0.5% of frame dimension per second.
    MAX_DRIFT_RATE: float = 0.005

    def __init__(
        self,
        config: PerturbationConfig,
        width: int,
        height: int,
        rng: random.Random,
    ) -> None:
        """Initialize the spatial transform processor.

        Args:
            config: Perturbation configuration with spatial parameters.
            width: Frame width in pixels.
            height: Frame height in pixels.
            rng: Seeded Random instance for reproducibility.
        """
        self.config = config
        self.width = width
        self.height = height
        self.rng = rng

        # Skip condition: no crop and no zoom means passthrough
        self.skip = (config.max_crop_percent == 0 and config.max_zoom == 1.0)

        if self.skip:
            self._crop_schedule: list[SegmentParam] = []
            self._zoom_schedule: list[SegmentParam] = []
            self._drift_directions: list[tuple[float, float]] = []
            return

        # We need a duration estimate for scheduling. Use a large default
        # that will be sufficient for most videos. The scheduler generates
        # segments up to this duration.
        # In practice, the pipeline should set this based on actual video duration.
        # For now, we use change_interval to build schedules on-demand or
        # pre-build for a reasonable max duration.
        self._change_interval = config.change_interval
        self._scheduler = ParameterScheduler(
            duration=3600.0,  # 1 hour max; segments beyond video are unused
            change_interval=config.change_interval,
            rng=rng,
            interpolation_window=config.transition_window,
        )

        # Generate crop schedule: values in [0, max_crop_percent]
        self._crop_schedule = self._scheduler.schedule(0.0, config.max_crop_percent)

        # Generate zoom schedule: values in [1.0, max_zoom]
        self._zoom_schedule = self._scheduler.schedule(1.0, config.max_zoom)

        # Generate drift directions for each change_interval segment.
        # Each direction is a unit vector (dx, dy) scaled by a random magnitude.
        n_segments = len(self._crop_schedule)
        self._drift_directions = self._generate_drift_directions(n_segments)

    def _generate_drift_directions(
        self, n_segments: int
    ) -> list[tuple[float, float]]:
        """Generate random drift direction vectors for each segment.

        Each direction is a normalized (dx, dy) pair representing the
        drift direction during that segment. The actual drift rate is
        clamped to MAX_DRIFT_RATE in transform_frame.

        Returns:
            List of (dx_factor, dy_factor) tuples where each component
            is in [-1, 1], representing the fraction of max drift rate
            to apply in each axis.
        """
        directions: list[tuple[float, float]] = []
        for _ in range(n_segments):
            # Random angle for drift direction
            angle = self.rng.uniform(0, 2 * math.pi)
            # Random magnitude factor [0.3, 1.0] to add variety
            magnitude = self.rng.uniform(0.3, 1.0)
            dx = math.cos(angle) * magnitude
            dy = math.sin(angle) * magnitude
            directions.append((dx, dy))
        return directions

    def _get_drift_offset(self, timestamp: float) -> tuple[float, float]:
        """Compute accumulated drift offset at the given timestamp.

        The drift accumulates over time, with direction changing at each
        Change_Interval boundary. The rate is clamped to MAX_DRIFT_RATE
        (0.5% of frame dimension per second).

        Args:
            timestamp: Current time in seconds.

        Returns:
            (offset_x, offset_y) in pixels, representing the drift from center.
        """
        if not self._drift_directions:
            return (0.0, 0.0)

        offset_x = 0.0
        offset_y = 0.0

        # Accumulate drift through each segment up to the current timestamp
        for i, (dx_factor, dy_factor) in enumerate(self._drift_directions):
            seg = self._crop_schedule[i]
            if timestamp <= seg.start_time:
                break

            # Duration spent in this segment
            seg_end = min(timestamp, seg.end_time)
            dt = seg_end - seg.start_time

            # Drift rate: MAX_DRIFT_RATE * dimension * direction_factor
            offset_x += self.MAX_DRIFT_RATE * self.width * dx_factor * dt
            offset_y += self.MAX_DRIFT_RATE * self.height * dy_factor * dt

            if timestamp <= seg.end_time:
                break

        return (offset_x, offset_y)

    def compute_crop_region(
        self, timestamp: float
    ) -> tuple[int, int, int, int]:
        """Compute (x, y, w, h) crop region for the given timestamp.

        The crop region is determined by:
        1. Current crop_percent from the crop schedule (interpolated)
        2. Current zoom level from the zoom schedule (interpolated)
        3. Combined effective crop from both crop_percent and zoom
        4. Crop center position based on accumulated drift
        5. Clamping to frame bounds

        Args:
            timestamp: Current time in seconds.

        Returns:
            Tuple of (x, y, w, h) representing the crop region in pixels,
            clamped to frame bounds.
        """
        if self.skip:
            return (0, 0, self.width, self.height)

        # Get interpolated crop percent and zoom
        crop_percent = self._scheduler.interpolate(self._crop_schedule, timestamp)
        zoom = self._scheduler.interpolate(self._zoom_schedule, timestamp)

        # Compute effective visible region:
        # crop_percent reduces the frame, zoom further reduces the visible area
        # crop_fraction: fraction of frame to remove (0 to max_crop_percent/100)
        crop_fraction = crop_percent / 100.0

        # Zoom reduces visible area: visible = 1/zoom of the frame
        # Combined: visible fraction = (1 - crop_fraction) / zoom
        visible_fraction = (1.0 - crop_fraction) / zoom

        # Ensure visible fraction is at least a small minimum to avoid degenerate crops
        visible_fraction = max(visible_fraction, 0.01)

        # Compute crop dimensions
        crop_w = int(round(self.width * visible_fraction))
        crop_h = int(round(self.height * visible_fraction))

        # Ensure minimum 1 pixel
        crop_w = max(crop_w, 1)
        crop_h = max(crop_h, 1)

        # Compute center position with drift
        center_x = self.width / 2.0
        center_y = self.height / 2.0

        drift_x, drift_y = self._get_drift_offset(timestamp)
        center_x += drift_x
        center_y += drift_y

        # Compute top-left corner from center
        x = int(round(center_x - crop_w / 2.0))
        y = int(round(center_y - crop_h / 2.0))

        # Clamp to frame bounds without reducing crop size
        x = max(0, min(x, self.width - crop_w))
        y = max(0, min(y, self.height - crop_h))

        return (x, y, crop_w, crop_h)

    def transform_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> np.ndarray:
        """Apply crop + zoom for the given timestamp, return frame at original resolution.

        Args:
            frame: Input frame as numpy array (H, W, C) or (H, W).
            timestamp: Current time in seconds.

        Returns:
            Transformed frame at original resolution (self.width x self.height).
        """
        if self.skip:
            return frame

        x, y, crop_w, crop_h = self.compute_crop_region(timestamp)

        # Crop the frame
        cropped = frame[y : y + crop_h, x : x + crop_w]

        # Resize back to original dimensions
        result = cv2.resize(
            cropped,
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR,
        )

        return result
