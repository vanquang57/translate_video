"""Unit tests for perturbation_scene module."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path so we can import without the full package chain
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from video_text_translator.perturbation_config import (
    PerturbationConfig,
    Scene,
    SceneBoundary,
)
from video_text_translator.perturbation_scene import SceneRecompositionProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> PerturbationConfig:
    """Create a PerturbationConfig with sensible defaults and overrides."""
    defaults = {
        "input_path": "test.mp4",
        "output_path": "out.mp4",
        "preset": "medium",
        "scene_enabled": True,
        "scene_threshold": 0.3,
        "max_scene_duration": 5.0,
        "transition_duration_ms": 500.0,
        "change_interval": 10.0,
    }
    defaults.update(overrides)
    return PerturbationConfig(**defaults)


def _make_scene(
    start_frame: int,
    end_frame: int,
    fps: float = 30.0,
    is_micro: bool = True,
) -> Scene:
    """Create a Scene with computed timestamps."""
    return Scene(
        start_frame=start_frame,
        end_frame=end_frame,
        start_time=start_frame / fps,
        end_time=end_frame / fps,
        is_micro=is_micro,
    )


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for SceneRecompositionProcessor initialization."""

    def test_init_stores_config_and_rng(self) -> None:
        """Processor should store config and rng."""
        config = _make_config()
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        assert processor.config is config
        assert processor.rng is rng


# ---------------------------------------------------------------------------
# boundaries_to_scenes tests
# ---------------------------------------------------------------------------


class TestBoundariesToScenes:
    """Tests for converting boundaries to Scene objects."""

    def test_no_boundaries_single_scene(self) -> None:
        """No boundaries should produce a single scene covering the whole video."""
        config = _make_config()
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        scenes = processor.boundaries_to_scenes([], total_frames=300, fps=30.0)

        assert len(scenes) == 1
        assert scenes[0].start_frame == 0
        assert scenes[0].end_frame == 300
        assert scenes[0].start_time == 0.0
        assert scenes[0].end_time == 10.0

    def test_one_boundary_two_scenes(self) -> None:
        """One boundary should produce two scenes."""
        config = _make_config()
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        boundaries = [
            SceneBoundary(frame_index=150, timestamp=5.0, histogram_diff=0.5)
        ]
        scenes = processor.boundaries_to_scenes(boundaries, total_frames=300, fps=30.0)

        assert len(scenes) == 2
        assert scenes[0].start_frame == 0
        assert scenes[0].end_frame == 150
        assert scenes[1].start_frame == 150
        assert scenes[1].end_frame == 300

    def test_micro_scene_classification(self) -> None:
        """Scenes shorter than max_scene_duration should be marked as micro."""
        config = _make_config(max_scene_duration=5.0)
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        # Boundary at frame 90 (3 seconds) → first scene is 3s (micro),
        # second scene is 7s (not micro)
        boundaries = [
            SceneBoundary(frame_index=90, timestamp=3.0, histogram_diff=0.5)
        ]
        scenes = processor.boundaries_to_scenes(boundaries, total_frames=300, fps=30.0)

        assert scenes[0].is_micro is True  # 3 seconds < 5 seconds
        assert scenes[1].is_micro is False  # 7 seconds >= 5 seconds

    def test_empty_video(self) -> None:
        """Zero frames should return empty list."""
        config = _make_config()
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        scenes = processor.boundaries_to_scenes([], total_frames=0, fps=30.0)
        assert scenes == []

    def test_zero_fps(self) -> None:
        """Zero fps should return empty list."""
        config = _make_config()
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        scenes = processor.boundaries_to_scenes([], total_frames=300, fps=0.0)
        assert scenes == []


# ---------------------------------------------------------------------------
# reorder_scenes tests
# ---------------------------------------------------------------------------


class TestReorderScenes:
    """Tests for scene reordering."""

    def test_reorder_produces_different_order(self) -> None:
        """Reordered micro-scenes should differ from original order."""
        config = _make_config()
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        scenes = [
            _make_scene(0, 90, is_micro=True),
            _make_scene(90, 180, is_micro=True),
            _make_scene(180, 270, is_micro=True),
        ]

        result = processor.reorder_scenes(scenes)

        # Should contain the same scenes
        assert set(id(s) for s in scenes) != set(id(s) for s in result) or result != scenes
        # At least one position should differ
        assert result != scenes

    def test_reorder_preserves_all_scenes(self) -> None:
        """Reordering should preserve all scene elements."""
        config = _make_config()
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        scenes = [
            _make_scene(0, 90, is_micro=True),
            _make_scene(90, 180, is_micro=True),
            _make_scene(180, 270, is_micro=True),
        ]

        result = processor.reorder_scenes(scenes)

        # Same elements, possibly different order
        assert sorted(result, key=lambda s: s.start_frame) == sorted(
            scenes, key=lambda s: s.start_frame
        )

    def test_reorder_only_micro_scenes(self) -> None:
        """Only micro-scenes should be reordered; non-micro stay in place."""
        config = _make_config()
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        scenes = [
            _make_scene(0, 90, is_micro=True),
            _make_scene(90, 300, is_micro=False),  # Long scene, not micro
            _make_scene(300, 390, is_micro=True),
            _make_scene(390, 480, is_micro=True),
        ]

        result = processor.reorder_scenes(scenes)

        # Non-micro scene should remain at index 1
        assert result[1] == scenes[1]

    def test_fewer_than_2_scenes_skip(self) -> None:
        """Fewer than 2 total scenes should skip reordering."""
        config = _make_config()
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        scenes = [_make_scene(0, 300, is_micro=True)]
        result = processor.reorder_scenes(scenes)

        assert result == scenes

    def test_fewer_than_2_micro_scenes_skip(self) -> None:
        """Fewer than 2 micro-scenes should skip reordering."""
        config = _make_config()
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        scenes = [
            _make_scene(0, 90, is_micro=True),
            _make_scene(90, 300, is_micro=False),
            _make_scene(300, 600, is_micro=False),
        ]

        result = processor.reorder_scenes(scenes)

        # Only 1 micro-scene, so no reordering
        assert result == scenes

    def test_empty_scenes_list(self) -> None:
        """Empty scene list should return empty list."""
        config = _make_config()
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        result = processor.reorder_scenes([])
        assert result == []

    def test_two_micro_scenes_swap(self) -> None:
        """With exactly 2 micro-scenes, they must swap."""
        config = _make_config()
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        scenes = [
            _make_scene(0, 90, is_micro=True),
            _make_scene(90, 180, is_micro=True),
        ]

        result = processor.reorder_scenes(scenes)

        # With 2 elements, the only different permutation is a swap
        assert result[0] == scenes[1]
        assert result[1] == scenes[0]

    def test_reproducibility_with_same_seed(self) -> None:
        """Same seed should produce same reordering."""
        config = _make_config()

        scenes = [
            _make_scene(0, 90, is_micro=True),
            _make_scene(90, 180, is_micro=True),
            _make_scene(180, 270, is_micro=True),
            _make_scene(270, 360, is_micro=True),
        ]

        processor1 = SceneRecompositionProcessor(config, random.Random(42))
        processor2 = SceneRecompositionProcessor(config, random.Random(42))

        result1 = processor1.reorder_scenes(scenes)
        result2 = processor2.reorder_scenes(scenes)

        assert result1 == result2


# ---------------------------------------------------------------------------
# insert_transitions tests
# ---------------------------------------------------------------------------


class TestInsertTransitions:
    """Tests for transition insertion."""

    def test_transition_duration_clamped_to_scene_duration(self) -> None:
        """Transition should not exceed duration of either adjacent scene."""
        # Configure a long transition (2 seconds) but scenes are short (1 second)
        config = _make_config(transition_duration_ms=2000.0)
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        # Two 1-second scenes
        scenes = [
            _make_scene(0, 30, is_micro=True),   # 1 second
            _make_scene(30, 60, is_micro=True),  # 1 second
        ]

        result = processor.insert_transitions(scenes)

        # The transition should be clamped to 1.0 seconds (min of scene durations)
        # The second scene's start should be adjusted by half of 1.0 = 0.5 seconds
        assert result[1].start_time < scenes[1].start_time

    def test_transition_uses_configured_duration_when_smaller(self) -> None:
        """When configured duration is smaller than scenes, use configured."""
        config = _make_config(transition_duration_ms=500.0)  # 0.5 seconds
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        # Two 10-second scenes
        scenes = [
            _make_scene(0, 300, is_micro=True),    # 10 seconds
            _make_scene(300, 600, is_micro=True),  # 10 seconds
        ]

        result = processor.insert_transitions(scenes)

        # Transition is 0.5s, half = 0.25s
        # Second scene start should be adjusted by 0.25s
        expected_start = scenes[1].start_time - 0.25
        assert abs(result[1].start_time - expected_start) < 0.01

    def test_single_scene_no_transitions(self) -> None:
        """Single scene should have no transitions."""
        config = _make_config()
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        scenes = [_make_scene(0, 300, is_micro=True)]
        result = processor.insert_transitions(scenes)

        assert len(result) == 1
        assert result[0] == scenes[0]

    def test_empty_scenes_no_transitions(self) -> None:
        """Empty scene list should return empty list."""
        config = _make_config()
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        result = processor.insert_transitions([])
        assert result == []

    def test_first_scene_unchanged(self) -> None:
        """First scene should not be modified by transitions."""
        config = _make_config()
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        scenes = [
            _make_scene(0, 150, is_micro=True),
            _make_scene(150, 300, is_micro=True),
        ]

        result = processor.insert_transitions(scenes)

        assert result[0] == scenes[0]

    def test_multiple_transitions(self) -> None:
        """Multiple scenes should each get transitions with their neighbors."""
        config = _make_config(transition_duration_ms=500.0)
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        scenes = [
            _make_scene(0, 150, is_micro=True),    # 5 seconds
            _make_scene(150, 300, is_micro=True),  # 5 seconds
            _make_scene(300, 450, is_micro=True),  # 5 seconds
        ]

        result = processor.insert_transitions(scenes)

        assert len(result) == 3
        # First scene unchanged
        assert result[0] == scenes[0]
        # Second and third scenes have adjusted start times
        assert result[1].start_time < scenes[1].start_time
        assert result[2].start_time < scenes[2].start_time


# ---------------------------------------------------------------------------
# detect_scenes tests (with mocked cv2)
# ---------------------------------------------------------------------------


class TestDetectScenes:
    """Tests for scene detection using mocked OpenCV."""

    def test_detect_scenes_empty_video(self) -> None:
        """Video with no frames should return empty boundaries."""
        config = _make_config()
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        with patch("video_text_translator.perturbation_scene.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0
            mock_cap.read.return_value = (False, None)

            boundaries = processor.detect_scenes("test.mp4")

        assert boundaries == []

    def test_detect_scenes_cannot_open(self) -> None:
        """Unopenable video should return empty boundaries."""
        config = _make_config()
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        with patch("video_text_translator.perturbation_scene.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cap.isOpened.return_value = False

            boundaries = processor.detect_scenes("nonexistent.mp4")

        assert boundaries == []

    def test_detect_scenes_finds_boundary(self) -> None:
        """Should detect a boundary when histogram diff exceeds threshold."""
        config = _make_config(scene_threshold=0.3)
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        with patch("video_text_translator.perturbation_scene.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0

            # Simulate 3 frames: frame 0 and 1 are similar, frame 2 is different
            frame1 = MagicMock()
            frame2 = MagicMock()
            frame3 = MagicMock()
            mock_cap.read.side_effect = [
                (True, frame1),
                (True, frame2),
                (True, frame3),
                (False, None),
            ]

            # Mock HSV conversion and histogram
            mock_cv2.cvtColor.return_value = MagicMock()
            hist1 = MagicMock()
            hist2 = MagicMock()
            hist3 = MagicMock()
            mock_cv2.calcHist.side_effect = [hist1, hist2, hist3]
            mock_cv2.normalize.return_value = None

            # Frame 0→1: high correlation (no boundary)
            # Frame 1→2: low correlation (boundary)
            mock_cv2.compareHist.side_effect = [0.9, 0.2]
            mock_cv2.HISTCMP_CORREL = 0
            mock_cv2.COLOR_BGR2HSV = 0

            boundaries = processor.detect_scenes("test.mp4")

        # diff at frame 1: 1 - 0.9 = 0.1 (< 0.3, no boundary)
        # diff at frame 2: 1 - 0.2 = 0.8 (> 0.3, boundary!)
        assert len(boundaries) == 1
        assert boundaries[0].frame_index == 2
        assert abs(boundaries[0].histogram_diff - 0.8) < 0.01


# ---------------------------------------------------------------------------
# Edge case integration tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for the full workflow."""

    def test_all_scenes_are_micro(self) -> None:
        """When all scenes are micro, all should be reordered."""
        config = _make_config(max_scene_duration=10.0)
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        scenes = [
            _make_scene(0, 90, is_micro=True),
            _make_scene(90, 180, is_micro=True),
            _make_scene(180, 270, is_micro=True),
        ]

        result = processor.reorder_scenes(scenes)

        # All scenes should be present
        assert len(result) == 3
        assert sorted(result, key=lambda s: s.start_frame) == sorted(
            scenes, key=lambda s: s.start_frame
        )
        # Order should differ
        assert result != scenes

    def test_no_scenes_are_micro(self) -> None:
        """When no scenes are micro, skip reordering."""
        config = _make_config(max_scene_duration=1.0)
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        scenes = [
            _make_scene(0, 150, is_micro=False),   # 5 seconds > 1 second
            _make_scene(150, 300, is_micro=False),  # 5 seconds > 1 second
        ]

        result = processor.reorder_scenes(scenes)

        # No micro-scenes, so no reordering
        assert result == scenes

    def test_transition_with_very_short_scene(self) -> None:
        """Transition should be clamped when a scene is very short."""
        config = _make_config(transition_duration_ms=1000.0)  # 1 second
        rng = random.Random(42)
        processor = SceneRecompositionProcessor(config, rng)

        # First scene is 0.5 seconds, second is 5 seconds
        scenes = [
            _make_scene(0, 15, is_micro=True),     # 0.5 seconds at 30fps
            _make_scene(15, 165, is_micro=True),   # 5 seconds
        ]

        result = processor.insert_transitions(scenes)

        # Transition should be clamped to 0.5s (duration of first scene)
        # Half of 0.5 = 0.25s adjustment
        expected_start = scenes[1].start_time - 0.25
        assert abs(result[1].start_time - expected_start) < 0.02
