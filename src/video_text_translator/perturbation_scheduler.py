"""Parameter scheduling for time-varying perturbation values.

This module implements the ParameterScheduler class which generates
time-segmented parameter schedules and provides smooth interpolation
between segment values at boundaries.
"""

from __future__ import annotations

import math
import random

from .perturbation_config import SegmentParam


class ParameterScheduler:
    """Generates parameter schedules for the entire video duration.

    The scheduler divides the video into segments of `change_interval` length
    and assigns a random parameter value to each segment. At segment boundaries,
    linear interpolation over 500ms provides smooth transitions.
    """

    # Duration of the interpolation window at boundaries (in seconds).
    INTERPOLATION_WINDOW: float = 0.5  # 500ms total (250ms each side)

    def __init__(
        self, duration: float, change_interval: float, rng: random.Random
    ) -> None:
        """Initialize the scheduler.

        Args:
            duration: Total video duration in seconds (>= 0).
            change_interval: Time between parameter changes in seconds (> 0).
            rng: Seeded Random instance for reproducibility.
        """
        self.duration = duration
        self.change_interval = change_interval
        self.rng = rng

    def schedule(
        self, param_min: float, param_max: float
    ) -> list[SegmentParam]:
        """Generate a list of segments with random values in [min, max].

        If change_interval > duration, produces exactly 1 segment covering
        the entire duration. Otherwise produces ceil(duration / change_interval)
        segments.

        Args:
            param_min: Minimum parameter value (inclusive).
            param_max: Maximum parameter value (inclusive).

        Returns:
            List of SegmentParam with contiguous time coverage from 0 to duration.
        """
        if self.duration <= 0:
            return []

        # Edge case: change_interval > duration → single segment
        if self.change_interval >= self.duration:
            value = self.rng.uniform(param_min, param_max)
            return [SegmentParam(start_time=0.0, end_time=self.duration, value=value)]

        n_segments = math.ceil(self.duration / self.change_interval)
        segments: list[SegmentParam] = []

        for i in range(n_segments):
            start_time = i * self.change_interval
            end_time = min((i + 1) * self.change_interval, self.duration)
            value = self.rng.uniform(param_min, param_max)
            segments.append(
                SegmentParam(start_time=start_time, end_time=end_time, value=value)
            )

        return segments

    def interpolate(
        self, schedule: list[SegmentParam], timestamp: float
    ) -> float:
        """Get interpolated parameter value at a given timestamp.

        Uses linear interpolation over 500ms at boundaries (250ms before
        and 250ms after the boundary) to smoothly transition between values.

        For timestamps before the first segment or after the last, returns
        the nearest segment's value.

        Args:
            schedule: List of SegmentParam (must be sorted by start_time).
            timestamp: The time position to query.

        Returns:
            The interpolated parameter value at the given timestamp.
        """
        if not schedule:
            return 0.0

        # Clamp to schedule bounds - return nearest segment value
        if timestamp <= schedule[0].start_time:
            return schedule[0].value
        if timestamp >= schedule[-1].end_time:
            return schedule[-1].value

        # Find the segment containing this timestamp
        seg_index = self._find_segment(schedule, timestamp)
        segment = schedule[seg_index]

        half_window = self.INTERPOLATION_WINDOW / 2.0  # 250ms

        # Check if we're near the start boundary (transition from previous)
        if seg_index > 0:
            boundary = segment.start_time
            if boundary - half_window < timestamp < boundary + half_window:
                prev_segment = schedule[seg_index - 1]
                # Linear interpolation: t=0 at boundary-250ms, t=1 at boundary+250ms
                t = (timestamp - (boundary - half_window)) / self.INTERPOLATION_WINDOW
                return prev_segment.value + (segment.value - prev_segment.value) * t

        # Check if we're near the end boundary (transition to next)
        if seg_index < len(schedule) - 1:
            boundary = segment.end_time
            if boundary - half_window < timestamp < boundary + half_window:
                next_segment = schedule[seg_index + 1]
                # Linear interpolation: t=0 at boundary-250ms, t=1 at boundary+250ms
                t = (timestamp - (boundary - half_window)) / self.INTERPOLATION_WINDOW
                return segment.value + (next_segment.value - segment.value) * t

        # In the middle of a segment, return exact value
        return segment.value

    def _find_segment(
        self, schedule: list[SegmentParam], timestamp: float
    ) -> int:
        """Find the index of the segment containing the timestamp.

        Uses linear search (segments are typically few). Returns the index
        of the segment whose [start_time, end_time) contains the timestamp.
        For the last segment, end_time is inclusive.
        """
        for i, seg in enumerate(schedule):
            if i == len(schedule) - 1:
                # Last segment: inclusive end
                if seg.start_time <= timestamp <= seg.end_time:
                    return i
            else:
                if seg.start_time <= timestamp < seg.end_time:
                    return i

        # Fallback: return last segment
        return len(schedule) - 1
