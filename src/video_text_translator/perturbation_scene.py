"""Scene recomposition processor for video perturbation.

This module implements the SceneRecompositionProcessor class which detects
scene boundaries using OpenCV histogram comparison, reorders micro-scenes
with random permutations, and inserts cross-fade transitions between scenes.
"""

from __future__ import annotations

import logging
import random

import cv2
import numpy as np

from .perturbation_config import PerturbationConfig, Scene, SceneBoundary

logger = logging.getLogger(__name__)


class SceneRecompositionProcessor:
    """Detects scenes and reorders micro-scenes.

    The processor uses frame-to-frame HSV histogram comparison to detect
    scene boundaries, classifies scenes as micro-scenes based on duration,
    reorders micro-scenes with a random permutation different from the
    original, and inserts cross-fade transitions between adjacent scenes.
    """

    def __init__(self, config: PerturbationConfig, rng: random.Random) -> None:
        """Initialize the scene recomposition processor.

        Args:
            config: Perturbation configuration with scene parameters.
            rng: Seeded Random instance for reproducibility.
        """
        self.config = config
        self.rng = rng

    def detect_scenes(self, video_path: str) -> list[SceneBoundary]:
        """Detect scene boundaries using histogram comparison.

        Compares consecutive frame HSV color histograms. When the difference
        between two consecutive frames exceeds scene_threshold, a boundary
        is marked at that frame.

        Args:
            video_path: Path to the input video file.

        Returns:
            List of SceneBoundary objects marking detected boundaries.
            Returns empty list if video cannot be opened or has < 2 frames.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Cannot open video for scene detection: %s", video_path)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Fallback

        boundaries: list[SceneBoundary] = []
        prev_hist: np.ndarray | None = None
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to HSV and compute histogram
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist(
                [hsv], [0, 1], None, [50, 60], [0, 180, 0, 256]
            )
            cv2.normalize(hist, hist)

            if prev_hist is not None:
                # Compare histograms using correlation
                # correlation returns 1.0 for identical, 0.0 for no correlation
                # We convert to a "difference" metric: diff = 1 - correlation
                correlation = cv2.compareHist(
                    prev_hist, hist, cv2.HISTCMP_CORREL
                )
                diff = 1.0 - correlation

                if diff > self.config.scene_threshold:
                    timestamp = frame_index / fps
                    boundaries.append(
                        SceneBoundary(
                            frame_index=frame_index,
                            timestamp=timestamp,
                            histogram_diff=diff,
                        )
                    )

            prev_hist = hist
            frame_index += 1

        cap.release()
        return boundaries

    def boundaries_to_scenes(
        self, boundaries: list[SceneBoundary], total_frames: int, fps: float
    ) -> list[Scene]:
        """Convert scene boundaries into Scene objects.

        Creates Scene objects from the gaps between boundaries, classifying
        each as a micro-scene if its duration is less than max_scene_duration.

        Args:
            boundaries: List of detected scene boundaries.
            total_frames: Total number of frames in the video.
            fps: Frames per second of the video.

        Returns:
            List of Scene objects covering the entire video.
        """
        if total_frames <= 0 or fps <= 0:
            return []

        scenes: list[Scene] = []
        # Build scene start/end points from boundaries
        scene_starts = [0] + [b.frame_index for b in boundaries]
        scene_ends = [b.frame_index for b in boundaries] + [total_frames]

        for start_frame, end_frame in zip(scene_starts, scene_ends):
            if end_frame <= start_frame:
                continue
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = end_time - start_time
            is_micro = duration < self.config.max_scene_duration
            scenes.append(
                Scene(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=start_time,
                    end_time=end_time,
                    is_micro=is_micro,
                )
            )

        return scenes

    def reorder_scenes(self, scenes: list[Scene]) -> list[Scene]:
        """Reorder micro-scenes with random permutation different from original.

        Only micro-scenes (is_micro=True) are reordered. Non-micro scenes
        remain in their original positions. The permutation of micro-scenes
        must differ from the original order.

        Edge cases:
        - If fewer than 2 micro-scenes exist, skip reordering with a warning.
        - If fewer than 2 total scenes, skip entirely with a warning.

        Args:
            scenes: List of Scene objects to reorder.

        Returns:
            New list of scenes with micro-scenes reordered. Returns the
            original list unchanged if reordering is not possible.
        """
        if len(scenes) < 2:
            logger.warning(
                "Fewer than 2 scenes detected (%d). Skipping scene recomposition.",
                len(scenes),
            )
            return list(scenes)

        # Separate micro-scenes and their positions
        micro_indices: list[int] = []
        micro_scenes: list[Scene] = []
        for i, scene in enumerate(scenes):
            if scene.is_micro:
                micro_indices.append(i)
                micro_scenes.append(scene)

        if len(micro_scenes) < 2:
            logger.warning(
                "Fewer than 2 micro-scenes found (%d). Skipping reordering.",
                len(micro_scenes),
            )
            return list(scenes)

        # Generate a permutation different from the original
        reordered = list(micro_scenes)
        max_attempts = 100
        for _ in range(max_attempts):
            self.rng.shuffle(reordered)
            if reordered != micro_scenes:
                break
        else:
            # Extremely unlikely with >= 2 scenes, but handle gracefully
            # Force a swap of first two elements
            reordered[0], reordered[1] = reordered[1], reordered[0]

        # Rebuild the full scene list with reordered micro-scenes
        result = list(scenes)
        for idx, micro_idx in enumerate(micro_indices):
            result[micro_idx] = reordered[idx]

        return result

    def insert_transitions(self, scenes: list[Scene]) -> list[Scene]:
        """Insert cross-fade transitions between adjacent scenes.

        The transition duration is clamped to min(configured_duration,
        scene1_duration, scene2_duration) to avoid transitions longer
        than either adjacent scene.

        Transitions are represented by adjusting scene boundaries to
        create overlap regions where cross-fading occurs.

        Args:
            scenes: List of Scene objects (already reordered if applicable).

        Returns:
            List of scenes with adjusted boundaries to accommodate transitions.
            Scene timing metadata is updated to reflect transition overlaps.
        """
        if len(scenes) < 2:
            return list(scenes)

        configured_duration_s = self.config.transition_duration_ms / 1000.0
        result: list[Scene] = []

        for i, scene in enumerate(scenes):
            if i == 0:
                result.append(scene)
                continue

            prev_scene = scenes[i - 1]
            prev_duration = prev_scene.end_time - prev_scene.start_time
            curr_duration = scene.end_time - scene.start_time

            # Clamp transition duration
            transition_duration = min(
                configured_duration_s, prev_duration, curr_duration
            )

            # Adjust the current scene to reflect the transition overlap.
            # The transition eats into the beginning of the current scene
            # and the end of the previous scene equally.
            # We represent this by shifting the start_time earlier by half
            # the transition duration (creating an overlap region).
            half_transition = transition_duration / 2.0

            # Calculate adjusted timing for the current scene
            # The scene effectively starts earlier (overlap with previous)
            adjusted_start_time = scene.start_time - half_transition
            adjusted_start_frame = max(
                0,
                int(round(adjusted_start_time * self._estimate_fps(scene))),
            )

            adjusted_scene = Scene(
                start_frame=adjusted_start_frame,
                end_frame=scene.end_frame,
                start_time=max(0.0, adjusted_start_time),
                end_time=scene.end_time,
                is_micro=scene.is_micro,
            )
            result.append(adjusted_scene)

        return result

    def _estimate_fps(self, scene: Scene) -> float:
        """Estimate FPS from a scene's frame and time data.

        Args:
            scene: A Scene object.

        Returns:
            Estimated FPS, or 30.0 as fallback.
        """
        duration = scene.end_time - scene.start_time
        if duration <= 0:
            return 30.0
        n_frames = scene.end_frame - scene.start_frame
        if n_frames <= 0:
            return 30.0
        return n_frames / duration
