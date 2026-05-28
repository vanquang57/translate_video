"""Parameter scheduling for time-varying perturbation values.

This module implements the ParameterScheduler class which generates
time-segmented parameter schedules and provides smooth interpolation
between segment values at boundaries.

Wave 2 additions:
- Sinusoidal mode: smooth sine-wave oscillation instead of random jumps
- Correlated mode: shared base oscillation with phase offsets for natural movement
- CorrelatedScheduler: generates multiple parameter schedules with correlation
"""

from __future__ import annotations

import math
import random

from .perturbation_config import SegmentParam


class ParameterScheduler:
    """Generates parameter schedules for the entire video duration.

    The scheduler divides the video into segments of `change_interval` length
    and assigns a parameter value to each segment. At segment boundaries,
    linear interpolation over 500ms provides smooth transitions.

    Supports three scheduling modes:
    - "random": independent random values per segment (original behavior)
    - "sinusoidal": smooth sinusoidal oscillation
    - "correlated": sinusoidal with a shared base phase for cross-parameter correlation
    """

    # Duration of the interpolation window at boundaries (in seconds).
    # Default 2.0s for smooth, natural-looking transitions.
    INTERPOLATION_WINDOW: float = 2.0  # 2000ms total (1000ms each side)

    def __init__(
        self, duration: float, change_interval: float, rng: random.Random,
        interpolation_window: float | None = None,
    ) -> None:
        """Initialize the scheduler.

        Args:
            duration: Total video duration in seconds (>= 0).
            change_interval: Time between parameter changes in seconds (> 0).
            rng: Seeded Random instance for reproducibility.
            interpolation_window: Override for transition duration in seconds.
                If None, uses the class default (2.0s).
        """
        self.duration = duration
        self.change_interval = change_interval
        self.rng = rng
        if interpolation_window is not None:
            self.INTERPOLATION_WINDOW = interpolation_window

    def schedule(
        self, param_min: float, param_max: float, mode: str = "random",
        base_frequency: float = 0.1, base_phase: float | None = None,
    ) -> list[SegmentParam]:
        """Generate a list of segments with values in [min, max].

        Args:
            param_min: Minimum parameter value (inclusive).
            param_max: Maximum parameter value (inclusive).
            mode: Scheduling mode - "random", "sinusoidal", or "correlated".
            base_frequency: Oscillation frequency in Hz (for sinusoidal/correlated).
            base_phase: Shared phase offset (for correlated mode). If None,
                a random phase is generated.

        Returns:
            List of SegmentParam with contiguous time coverage from 0 to duration.
        """
        if mode == "sinusoidal":
            return self._schedule_sinusoidal(param_min, param_max, base_frequency)
        elif mode == "correlated":
            return self._schedule_correlated(
                param_min, param_max, base_frequency, base_phase
            )
        else:
            return self._schedule_random(param_min, param_max)

    def _schedule_random(
        self, param_min: float, param_max: float
    ) -> list[SegmentParam]:
        """Generate segments with random values (original behavior).

        If change_interval > duration, produces exactly 1 segment covering
        the entire duration. Otherwise produces ceil(duration / change_interval)
        segments.
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

    def _schedule_sinusoidal(
        self, param_min: float, param_max: float, base_frequency: float,
    ) -> list[SegmentParam]:
        """Generate segments with sinusoidal oscillation values.

        value(t) = center + amplitude * sin(2π * frequency * t + phase)
        where:
        - center = (param_min + param_max) / 2
        - amplitude = (param_max - param_min) / 2
        - frequency = base_frequency with slight random variation
        - phase = random offset
        """
        if self.duration <= 0:
            return []

        center = (param_min + param_max) / 2.0
        amplitude = (param_max - param_min) / 2.0

        # Add slight random variation to frequency (±20%)
        freq = base_frequency * self.rng.uniform(0.8, 1.2)
        phase = self.rng.uniform(0, 2 * math.pi)

        if self.change_interval >= self.duration:
            # Single segment: evaluate at midpoint
            t_mid = self.duration / 2.0
            value = center + amplitude * math.sin(2 * math.pi * freq * t_mid + phase)
            return [SegmentParam(start_time=0.0, end_time=self.duration, value=value)]

        n_segments = math.ceil(self.duration / self.change_interval)
        segments: list[SegmentParam] = []

        for i in range(n_segments):
            start_time = i * self.change_interval
            end_time = min((i + 1) * self.change_interval, self.duration)
            # Evaluate sine at segment midpoint for smooth representation
            t_mid = (start_time + end_time) / 2.0
            value = center + amplitude * math.sin(2 * math.pi * freq * t_mid + phase)
            # Clamp to bounds (floating point safety)
            value = max(param_min, min(param_max, value))
            segments.append(
                SegmentParam(start_time=start_time, end_time=end_time, value=value)
            )

        return segments

    def _schedule_correlated(
        self, param_min: float, param_max: float,
        base_frequency: float, base_phase: float | None,
    ) -> list[SegmentParam]:
        """Generate segments with correlated sinusoidal oscillation.

        Like sinusoidal mode, but uses a shared base_phase so multiple
        parameters oscillate together. A small per-parameter phase offset
        is added for natural variation.

        Args:
            param_min: Minimum parameter value.
            param_max: Maximum parameter value.
            base_frequency: Shared oscillation frequency.
            base_phase: Shared base phase. If None, generates a random one.
        """
        if self.duration <= 0:
            return []

        center = (param_min + param_max) / 2.0
        amplitude = (param_max - param_min) / 2.0

        # Use shared base_phase with a small per-parameter offset
        if base_phase is None:
            base_phase = self.rng.uniform(0, 2 * math.pi)
        # Small phase offset for this specific parameter (±π/6 = ±30°)
        param_offset = self.rng.uniform(-math.pi / 6, math.pi / 6)
        phase = base_phase + param_offset

        # Slight frequency variation (±10% — less than sinusoidal to stay correlated)
        freq = base_frequency * self.rng.uniform(0.9, 1.1)

        if self.change_interval >= self.duration:
            t_mid = self.duration / 2.0
            value = center + amplitude * math.sin(2 * math.pi * freq * t_mid + phase)
            value = max(param_min, min(param_max, value))
            return [SegmentParam(start_time=0.0, end_time=self.duration, value=value)]

        n_segments = math.ceil(self.duration / self.change_interval)
        segments: list[SegmentParam] = []

        for i in range(n_segments):
            start_time = i * self.change_interval
            end_time = min((i + 1) * self.change_interval, self.duration)
            t_mid = (start_time + end_time) / 2.0
            value = center + amplitude * math.sin(2 * math.pi * freq * t_mid + phase)
            value = max(param_min, min(param_max, value))
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


class CorrelatedScheduler:
    """Generates correlated parameter schedules with shared oscillation.

    Wraps multiple ParameterSchedulers and ensures their values move
    together naturally by sharing a base phase and frequency.

    Correlation pairs:
    - zoom ↔ crop_drift: zoom increases → crop drift speed increases
    - rotation ↔ spatial_drift: rotation direction correlates with pan direction
    - audio_tempo ↔ speed: audio tempo follows video speed
    """

    def __init__(
        self,
        duration: float,
        change_interval: float,
        rng: random.Random,
        mode: str = "correlated",
        base_frequency: float = 0.1,
    ) -> None:
        """Initialize the correlated scheduler.

        Args:
            duration: Total video duration in seconds.
            change_interval: Time between parameter changes in seconds.
            rng: Seeded Random instance for reproducibility.
            mode: Scheduling mode - "random", "sinusoidal", or "correlated".
            base_frequency: Base oscillation frequency in Hz.
        """
        self.duration = duration
        self.change_interval = change_interval
        self.rng = rng
        self.mode = mode
        self.base_frequency = base_frequency

        # Generate a shared base phase for correlated mode
        self._base_phase = rng.uniform(0, 2 * math.pi)

    @property
    def base_phase(self) -> float:
        """The shared base phase for correlated scheduling."""
        return self._base_phase

    def generate_schedule(
        self, param_min: float, param_max: float,
    ) -> list[SegmentParam]:
        """Generate a single parameter schedule using the shared base phase.

        Args:
            param_min: Minimum parameter value.
            param_max: Maximum parameter value.

        Returns:
            List of SegmentParam for this parameter.
        """
        scheduler = ParameterScheduler(
            duration=self.duration,
            change_interval=self.change_interval,
            rng=random.Random(self.rng.randint(0, 2**32 - 1)),
        )
        return scheduler.schedule(
            param_min, param_max,
            mode=self.mode,
            base_frequency=self.base_frequency,
            base_phase=self._base_phase,
        )

    def generate_all_schedules(
        self, param_specs: dict[str, tuple[float, float]],
    ) -> dict[str, list[SegmentParam]]:
        """Generate correlated schedules for multiple parameters.

        In "correlated" mode, all parameters share the same base phase
        so they oscillate together with slight per-parameter offsets.
        In "sinusoidal" mode, each gets independent sinusoidal oscillation.
        In "random" mode, each gets independent random values.

        Args:
            param_specs: Dict mapping parameter name to (min, max) tuple.

        Returns:
            Dict mapping parameter name to its schedule.
        """
        schedules: dict[str, list[SegmentParam]] = {}

        for name, (p_min, p_max) in param_specs.items():
            schedules[name] = self.generate_schedule(p_min, p_max)

        return schedules
