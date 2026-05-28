"""Temporal drift processing for video perturbation.

This module implements the TemporalDriftProcessor class which applies
speed variation, frame drop/duplicate, and micro time offsets to create
a frame mapping that subtly alters the temporal structure of a video.
"""

from __future__ import annotations

import random

from .perturbation_config import PerturbationConfig
from .perturbation_scheduler import ParameterScheduler


class TemporalDriftProcessor:
    """Applies speed variation and frame drop/duplicate.

    The processor computes an output-to-input frame index mapping that
    encodes speed changes per segment, random frame drops/duplicates,
    and micro time offsets at segment boundaries. The output frame count
    is constrained to preserve duration within 1%.
    """

    def __init__(self, config: PerturbationConfig, rng: random.Random) -> None:
        """Initialize the temporal drift processor.

        Args:
            config: Perturbation configuration with temporal parameters.
            rng: Seeded Random instance for reproducibility.
        """
        self.config = config
        self.rng = rng

    def compute_frame_map(self, n_frames: int, fps: float) -> list[int]:
        """Compute output-to-input frame index mapping.

        Returns a list where output_frame[i] = input_frame_index. The mapping
        encodes speed variation per segment, frame drops/duplicates, and micro
        time offsets. The output length is constrained to be within 1% of
        n_frames to preserve duration.

        Args:
            n_frames: Total number of input frames.
            fps: Frames per second of the video.

        Returns:
            List of input frame indices for each output frame position.
        """
        if n_frames <= 0 or fps <= 0:
            return []

        duration = n_frames / fps

        # Step 1: Create speed schedule using ParameterScheduler
        scheduler = ParameterScheduler(
            duration=duration,
            change_interval=self.config.change_interval,
            rng=self.rng,
            interpolation_window=self.config.transition_window,
        )
        speed_schedule = scheduler.schedule(
            self.config.speed_min, self.config.speed_max
        )

        if not speed_schedule:
            return list(range(n_frames))

        # Step 2: Apply micro time offsets to segment boundaries
        offset_schedule = self._apply_micro_offsets(speed_schedule, duration)

        # Step 3: Build frame map based on speed variation
        frame_map = self._build_speed_frame_map(offset_schedule, n_frames, fps)

        # Step 4: Apply frame drops/duplicates
        frame_map = self._apply_frame_drops_duplicates(frame_map, n_frames)

        # Step 5: Enforce duration preservation within 1%
        frame_map = self._enforce_duration_constraint(frame_map, n_frames)

        return frame_map

    def _apply_micro_offsets(
        self,
        schedule: list,
        duration: float,
    ) -> list[tuple[float, float, float]]:
        """Apply micro time offsets at segment boundaries.

        Shifts each segment boundary by a random amount within
        ±micro_offset_ms, keeping boundaries within [0, duration].

        Args:
            schedule: List of SegmentParam from the scheduler.
            duration: Total video duration in seconds.

        Returns:
            List of (start_time, end_time, speed_value) tuples with
            adjusted boundaries.
        """
        if not schedule:
            return []

        max_offset_s = self.config.micro_offset_ms / 1000.0

        # Collect original boundaries (internal ones only, not 0 and duration)
        boundaries = [0.0]
        for seg in schedule:
            if seg.end_time < duration:
                boundaries.append(seg.end_time)
        boundaries.append(duration)

        # Apply random offsets to internal boundaries
        adjusted_boundaries = [boundaries[0]]  # Keep start at 0
        for i in range(1, len(boundaries) - 1):
            offset = self.rng.uniform(-max_offset_s, max_offset_s)
            new_boundary = boundaries[i] + offset
            # Clamp to stay between previous and next boundary
            prev = adjusted_boundaries[-1]
            next_b = boundaries[i + 1]
            new_boundary = max(prev + 0.001, min(new_boundary, next_b - 0.001))
            adjusted_boundaries.append(new_boundary)
        adjusted_boundaries.append(boundaries[-1])  # Keep end at duration

        # Build adjusted segments with original speed values
        result = []
        for i, seg in enumerate(schedule):
            start = adjusted_boundaries[i]
            end = adjusted_boundaries[i + 1]
            result.append((start, end, seg.value))

        return result

    def _build_speed_frame_map(
        self,
        offset_schedule: list[tuple[float, float, float]],
        n_frames: int,
        fps: float,
    ) -> list[int]:
        """Build frame map based on speed variation per segment.

        For each output frame, compute which input frame it maps to based
        on accumulated speed. A speed > 1.0 means we advance through input
        frames faster (time compression), speed < 1.0 means slower (time
        expansion).

        Args:
            offset_schedule: List of (start_time, end_time, speed) tuples.
            n_frames: Total number of input frames.
            fps: Frames per second.

        Returns:
            List of input frame indices for each output frame.
        """
        if not offset_schedule:
            return list(range(n_frames))

        duration = n_frames / fps
        frame_map: list[int] = []

        # For each output frame, compute the corresponding input time
        # based on accumulated speed
        input_time = 0.0  # Accumulated input time
        frame_duration = 1.0 / fps

        for out_idx in range(n_frames):
            output_time = out_idx / fps

            # Find which segment this output time falls in
            speed = 1.0
            for start, end, seg_speed in offset_schedule:
                if start <= output_time < end:
                    speed = seg_speed
                    break
            else:
                # If past all segments, use last segment's speed
                if offset_schedule:
                    speed = offset_schedule[-1][2]

            # Advance input time by speed * frame_duration
            if out_idx == 0:
                input_time = 0.0
            else:
                input_time += speed * frame_duration

            # Map input time to input frame index
            input_frame = int(round(input_time * fps))
            input_frame = max(0, min(input_frame, n_frames - 1))
            frame_map.append(input_frame)

        return frame_map

    def _apply_frame_drops_duplicates(
        self, frame_map: list[int], n_frames: int
    ) -> list[int]:
        """Apply random frame drops and duplicates.

        Randomly marks frames for drop (skip) or duplicate (repeat) at a
        rate not exceeding max_frame_drop_percent. Ensures no two consecutive
        frames are both affected.

        Args:
            frame_map: Current frame mapping.
            n_frames: Original frame count.

        Returns:
            Modified frame map with drops/duplicates applied.
        """
        if not frame_map:
            return frame_map

        max_affected = int(
            len(frame_map) * self.config.max_frame_drop_percent / 100.0
        )
        if max_affected <= 0:
            return frame_map

        # Determine how many frames to affect (random up to max)
        num_affected = self.rng.randint(0, max_affected)
        if num_affected == 0:
            return frame_map

        # Select candidate positions (not first or last frame)
        # Ensure no two consecutive positions are selected
        candidates = list(range(1, len(frame_map) - 1))
        self.rng.shuffle(candidates)

        affected_positions: list[int] = []
        affected_set: set[int] = set()

        for pos in candidates:
            if len(affected_positions) >= num_affected:
                break
            # Check non-consecutiveness: neither neighbor should be affected
            if (pos - 1) in affected_set or (pos + 1) in affected_set:
                continue
            affected_positions.append(pos)
            affected_set.add(pos)

        # Apply drops and duplicates
        result = list(frame_map)
        for pos in affected_positions:
            if self.rng.random() < 0.5:
                # Drop: use the previous frame's mapping (effectively skipping)
                result[pos] = result[pos - 1]
            else:
                # Duplicate: repeat the current frame (same as keeping it,
                # but mark it as a duplicate by copying from previous)
                result[pos] = result[pos - 1]

        return result

    def _enforce_duration_constraint(
        self, frame_map: list[int], n_frames: int
    ) -> list[int]:
        """Enforce that output frame count is within 1% of input frame count.

        If the frame map length deviates by more than 1% from n_frames,
        trim or pad to stay within tolerance.

        Args:
            frame_map: Current frame mapping.
            n_frames: Original input frame count.

        Returns:
            Frame map constrained to within 1% of n_frames length.
        """
        if not frame_map or n_frames <= 0:
            return frame_map

        tolerance = 0.01
        min_frames = int(n_frames * (1.0 - tolerance))
        max_frames = int(n_frames * (1.0 + tolerance))

        current_len = len(frame_map)

        if current_len > max_frames:
            # Trim: remove frames from the end
            frame_map = frame_map[:max_frames]
        elif current_len < min_frames:
            # Pad: duplicate the last frame
            last_frame = frame_map[-1] if frame_map else 0
            while len(frame_map) < min_frames:
                frame_map.append(last_frame)

        return frame_map
