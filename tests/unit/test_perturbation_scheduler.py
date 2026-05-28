"""Unit tests for perturbation_scheduler module."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

# Add src to path so we can import without the full package chain
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from video_text_translator.perturbation_config import SegmentParam
from video_text_translator.perturbation_scheduler import ParameterScheduler


# ---------------------------------------------------------------------------
# Schedule generation tests
# ---------------------------------------------------------------------------


class TestScheduleGeneration:
    """Tests for ParameterScheduler.schedule()."""

    def test_correct_number_of_segments(self) -> None:
        """Schedule produces ceil(duration / change_interval) segments."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=30.0, change_interval=10.0, rng=rng)
        segments = scheduler.schedule(0.99, 1.01)
        assert len(segments) == 3  # ceil(30/10) = 3

    def test_correct_number_of_segments_non_divisible(self) -> None:
        """When duration is not evenly divisible, ceil is used."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=25.0, change_interval=10.0, rng=rng)
        segments = scheduler.schedule(0.99, 1.01)
        assert len(segments) == 3  # ceil(25/10) = 3

    def test_correct_number_of_segments_small_interval(self) -> None:
        """Many segments for small change_interval."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=10.0, change_interval=2.0, rng=rng)
        segments = scheduler.schedule(0.0, 1.0)
        assert len(segments) == 5  # ceil(10/2) = 5

    def test_all_values_within_bounds(self) -> None:
        """All segment values must be within [param_min, param_max]."""
        rng = random.Random(123)
        scheduler = ParameterScheduler(duration=60.0, change_interval=5.0, rng=rng)
        segments = scheduler.schedule(0.5, 0.8)
        for seg in segments:
            assert 0.5 <= seg.value <= 0.8, f"Value {seg.value} out of bounds"

    def test_single_segment_when_interval_exceeds_duration(self) -> None:
        """If change_interval > duration, produce exactly 1 segment."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=5.0, change_interval=30.0, rng=rng)
        segments = scheduler.schedule(0.99, 1.01)
        assert len(segments) == 1
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 5.0

    def test_single_segment_when_interval_equals_duration(self) -> None:
        """If change_interval == duration, produce exactly 1 segment."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=10.0, change_interval=10.0, rng=rng)
        segments = scheduler.schedule(0.99, 1.01)
        assert len(segments) == 1
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 10.0

    def test_segments_cover_full_duration(self) -> None:
        """Segments should cover from 0 to duration without gaps."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=25.0, change_interval=10.0, rng=rng)
        segments = scheduler.schedule(0.0, 1.0)
        assert segments[0].start_time == 0.0
        assert segments[-1].end_time == 25.0
        # Check contiguity
        for i in range(len(segments) - 1):
            assert segments[i].end_time == segments[i + 1].start_time

    def test_last_segment_end_time_clamped_to_duration(self) -> None:
        """Last segment's end_time should not exceed duration."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=7.5, change_interval=5.0, rng=rng)
        segments = scheduler.schedule(0.0, 1.0)
        assert len(segments) == 2  # ceil(7.5/5) = 2
        assert segments[-1].end_time == 7.5

    def test_reproducibility_with_same_seed(self) -> None:
        """Same seed produces same schedule."""
        scheduler1 = ParameterScheduler(duration=30.0, change_interval=10.0, rng=random.Random(42))
        scheduler2 = ParameterScheduler(duration=30.0, change_interval=10.0, rng=random.Random(42))
        segments1 = scheduler1.schedule(0.0, 1.0)
        segments2 = scheduler2.schedule(0.0, 1.0)
        assert segments1 == segments2

    def test_different_seeds_produce_different_values(self) -> None:
        """Different seeds should (almost certainly) produce different values."""
        scheduler1 = ParameterScheduler(duration=30.0, change_interval=10.0, rng=random.Random(1))
        scheduler2 = ParameterScheduler(duration=30.0, change_interval=10.0, rng=random.Random(999))
        segments1 = scheduler1.schedule(0.0, 1.0)
        segments2 = scheduler2.schedule(0.0, 1.0)
        values1 = [s.value for s in segments1]
        values2 = [s.value for s in segments2]
        assert values1 != values2


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestScheduleEdgeCases:
    """Edge case tests for schedule generation."""

    def test_duration_zero(self) -> None:
        """Duration of 0 should produce empty schedule."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=0.0, change_interval=10.0, rng=rng)
        segments = scheduler.schedule(0.0, 1.0)
        assert segments == []

    def test_very_short_duration(self) -> None:
        """Very short duration (< change_interval) produces 1 segment."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=0.1, change_interval=10.0, rng=rng)
        segments = scheduler.schedule(0.0, 1.0)
        assert len(segments) == 1
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 0.1

    def test_param_min_equals_param_max(self) -> None:
        """When min == max, all values should be that value."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=30.0, change_interval=10.0, rng=rng)
        segments = scheduler.schedule(0.5, 0.5)
        for seg in segments:
            assert seg.value == 0.5


# ---------------------------------------------------------------------------
# Interpolation tests
# ---------------------------------------------------------------------------


class TestInterpolation:
    """Tests for ParameterScheduler.interpolate()."""

    def _make_schedule(self) -> list[SegmentParam]:
        """Create a simple 3-segment schedule for testing."""
        return [
            SegmentParam(start_time=0.0, end_time=10.0, value=1.0),
            SegmentParam(start_time=10.0, end_time=20.0, value=2.0),
            SegmentParam(start_time=20.0, end_time=30.0, value=3.0),
        ]

    def test_middle_of_segment_returns_exact_value(self) -> None:
        """Timestamp in the middle of a segment returns that segment's value."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=30.0, change_interval=10.0, rng=rng)
        schedule = self._make_schedule()

        assert scheduler.interpolate(schedule, 5.0) == 1.0
        assert scheduler.interpolate(schedule, 15.0) == 2.0
        assert scheduler.interpolate(schedule, 25.0) == 3.0

    def test_before_first_segment_returns_first_value(self) -> None:
        """Timestamp before schedule returns first segment's value."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=30.0, change_interval=10.0, rng=rng)
        schedule = self._make_schedule()

        assert scheduler.interpolate(schedule, -1.0) == 1.0
        assert scheduler.interpolate(schedule, 0.0) == 1.0

    def test_after_last_segment_returns_last_value(self) -> None:
        """Timestamp after schedule returns last segment's value."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=30.0, change_interval=10.0, rng=rng)
        schedule = self._make_schedule()

        assert scheduler.interpolate(schedule, 30.0) == 3.0
        assert scheduler.interpolate(schedule, 35.0) == 3.0

    def test_interpolation_at_boundary_midpoint(self) -> None:
        """At the exact boundary, interpolation should be at 50% between values."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=30.0, change_interval=10.0, rng=rng)
        schedule = self._make_schedule()

        # At boundary t=10.0, midpoint of interpolation window [9.75, 10.25]
        # t=10.0 is 0.25 into the 0.5s window → t_ratio = 0.25/0.5 = 0.5
        result = scheduler.interpolate(schedule, 10.0)
        expected = 1.0 + (2.0 - 1.0) * 0.5  # = 1.5
        assert abs(result - expected) < 1e-10

    def test_interpolation_at_boundary_start(self) -> None:
        """At 250ms before boundary, interpolation starts (t_ratio ≈ 0)."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=30.0, change_interval=10.0, rng=rng)
        schedule = self._make_schedule()

        # Just after the start of interpolation window (9.75 + epsilon)
        result = scheduler.interpolate(schedule, 9.76)
        # t_ratio = (9.76 - 9.75) / 0.5 = 0.02
        expected = 1.0 + (2.0 - 1.0) * 0.02
        assert abs(result - expected) < 1e-10

    def test_interpolation_at_boundary_end(self) -> None:
        """At 250ms after boundary, interpolation ends (t_ratio ≈ 1)."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=30.0, change_interval=10.0, rng=rng)
        schedule = self._make_schedule()

        # Just before the end of interpolation window (10.25 - epsilon)
        result = scheduler.interpolate(schedule, 10.24)
        # t_ratio = (10.24 - 9.75) / 0.5 = 0.98
        expected = 1.0 + (2.0 - 1.0) * 0.98
        assert abs(result - expected) < 1e-10

    def test_interpolation_smooth_transition(self) -> None:
        """Values should transition smoothly across the boundary window."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=30.0, change_interval=10.0, rng=rng)
        schedule = self._make_schedule()

        # Sample across the boundary at t=10.0
        timestamps = [9.75 + i * 0.05 for i in range(11)]  # 9.75 to 10.25
        values = [scheduler.interpolate(schedule, t) for t in timestamps]

        # Values should be monotonically increasing (1.0 → 2.0)
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-10

    def test_interpolation_outside_window_returns_exact(self) -> None:
        """Timestamps well inside a segment (far from boundaries) return exact value."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=30.0, change_interval=10.0, rng=rng)
        schedule = self._make_schedule()

        # Well inside first segment (far from boundary at 10.0)
        assert scheduler.interpolate(schedule, 2.0) == 1.0
        assert scheduler.interpolate(schedule, 8.0) == 1.0

        # Well inside second segment
        assert scheduler.interpolate(schedule, 12.0) == 2.0
        assert scheduler.interpolate(schedule, 18.0) == 2.0

    def test_empty_schedule(self) -> None:
        """Empty schedule returns 0.0."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=0.0, change_interval=10.0, rng=rng)
        assert scheduler.interpolate([], 5.0) == 0.0

    def test_single_segment_no_interpolation(self) -> None:
        """Single segment schedule always returns that segment's value."""
        rng = random.Random(42)
        scheduler = ParameterScheduler(duration=5.0, change_interval=30.0, rng=rng)
        schedule = [SegmentParam(start_time=0.0, end_time=5.0, value=0.75)]

        assert scheduler.interpolate(schedule, 0.0) == 0.75
        assert scheduler.interpolate(schedule, 2.5) == 0.75
        assert scheduler.interpolate(schedule, 5.0) == 0.75
