"""Unit tests for PerturbationPipeline.

Tests cover:
- Error handling: invalid input returns exit code 2, config error returns 1
- Skip behavior: no audio → skip audio perturbation, < 2 scenes → skip scene recomposition
- Seed reproducibility: same seed produces same output
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from video_text_translator.perturbation_config import PerturbationConfig
from video_text_translator.perturbation_pipeline import PerturbationPipeline


def _create_test_video(path: str, n_frames: int = 30, fps: float = 30.0,
                       width: int = 64, height: int = 64,
                       with_audio: bool = False) -> None:
    """Create a minimal test video file using OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(n_frames):
        # Create frames with varying content for scene detection
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 8) % 256  # Varying blue channel
        frame[:, :, 1] = (i * 4) % 256  # Varying green channel
        writer.write(frame)
    writer.release()


class TestPerturbationPipelineErrorHandling:
    """Test error handling and exit codes."""

    def test_config_error_returns_exit_code_1(self, tmp_path: Path) -> None:
        """Invalid config parameter returns exit code 1."""
        input_video = str(tmp_path / "input.mp4")
        output_video = str(tmp_path / "output.mp4")
        _create_test_video(input_video)

        # speed_min out of range (must be >= 0.90)
        config = PerturbationConfig(
            input_path=input_video,
            output_path=output_video,
            speed_min=0.5,  # Invalid: below 0.90
        )
        pipeline = PerturbationPipeline(config)
        result = pipeline.run()
        assert result == 1

    def test_missing_input_returns_exit_code_2(self, tmp_path: Path) -> None:
        """Non-existent input file returns exit code 2."""
        config = PerturbationConfig(
            input_path=str(tmp_path / "nonexistent.mp4"),
            output_path=str(tmp_path / "output.mp4"),
        )
        pipeline = PerturbationPipeline(config)
        result = pipeline.run()
        assert result == 2

    def test_empty_input_path_returns_exit_code_2(self, tmp_path: Path) -> None:
        """Empty input path returns exit code 2."""
        config = PerturbationConfig(
            input_path="",
            output_path=str(tmp_path / "output.mp4"),
        )
        pipeline = PerturbationPipeline(config)
        result = pipeline.run()
        assert result == 2

    def test_empty_output_path_returns_exit_code_2(self, tmp_path: Path) -> None:
        """Empty output path returns exit code 2."""
        input_video = str(tmp_path / "input.mp4")
        _create_test_video(input_video)

        config = PerturbationConfig(
            input_path=input_video,
            output_path="",
        )
        pipeline = PerturbationPipeline(config)
        result = pipeline.run()
        assert result == 2

    def test_unwritable_output_dir_returns_exit_code_2(self, tmp_path: Path) -> None:
        """Output directory that doesn't exist returns exit code 2."""
        input_video = str(tmp_path / "input.mp4")
        _create_test_video(input_video)

        config = PerturbationConfig(
            input_path=input_video,
            output_path=str(tmp_path / "nonexistent_dir" / "output.mp4"),
        )
        pipeline = PerturbationPipeline(config)
        result = pipeline.run()
        assert result == 2

    def test_corrupted_video_returns_exit_code_2(self, tmp_path: Path) -> None:
        """Corrupted/invalid video file returns exit code 2."""
        input_video = str(tmp_path / "corrupted.mp4")
        # Write garbage data
        Path(input_video).write_bytes(b"not a video file at all")

        config = PerturbationConfig(
            input_path=input_video,
            output_path=str(tmp_path / "output.mp4"),
        )
        pipeline = PerturbationPipeline(config)
        result = pipeline.run()
        assert result == 2


class TestPerturbationPipelineSkipBehavior:
    """Test graceful skip behavior for non-critical conditions."""

    @patch("video_text_translator.perturbation_pipeline._has_audio")
    def test_no_audio_skips_audio_perturbation(
        self, mock_has_audio: MagicMock, tmp_path: Path
    ) -> None:
        """When video has no audio, audio perturbation is skipped gracefully."""
        mock_has_audio.return_value = False

        input_video = str(tmp_path / "input.mp4")
        output_video = str(tmp_path / "output.mp4")
        _create_test_video(input_video, n_frames=10)

        config = PerturbationConfig(
            input_path=input_video,
            output_path=output_video,
            audio_enabled=True,
            temporal_enabled=False,
            spatial_enabled=False,
            scene_enabled=False,
            seed=42,
        )
        pipeline = PerturbationPipeline(config)
        result = pipeline.run()
        assert result == 0
        assert Path(output_video).exists()
        assert Path(output_video).stat().st_size > 0

    @patch("video_text_translator.perturbation_pipeline._has_audio")
    def test_fewer_than_2_scenes_skips_scene_recomposition(
        self, mock_has_audio: MagicMock, tmp_path: Path
    ) -> None:
        """When fewer than 2 scenes detected, scene recomposition is skipped."""
        mock_has_audio.return_value = False

        input_video = str(tmp_path / "input.mp4")
        output_video = str(tmp_path / "output.mp4")
        # Create a uniform video (no scene changes)
        _create_test_video(input_video, n_frames=30)

        config = PerturbationConfig(
            input_path=input_video,
            output_path=output_video,
            scene_enabled=True,
            temporal_enabled=False,
            spatial_enabled=False,
            audio_enabled=False,
            seed=42,
        )
        pipeline = PerturbationPipeline(config)
        result = pipeline.run()
        # Should succeed (scene recomposition skipped gracefully)
        assert result == 0
        assert Path(output_video).exists()


class TestPerturbationPipelineSeedReproducibility:
    """Test that same seed produces same output."""

    @patch("video_text_translator.perturbation_pipeline._has_audio")
    def test_same_seed_produces_same_output(
        self, mock_has_audio: MagicMock, tmp_path: Path
    ) -> None:
        """Running pipeline twice with same seed produces identical output."""
        mock_has_audio.return_value = False

        input_video = str(tmp_path / "input.mp4")
        output1 = str(tmp_path / "output1.mp4")
        output2 = str(tmp_path / "output2.mp4")
        _create_test_video(input_video, n_frames=15)

        config1 = PerturbationConfig(
            input_path=input_video,
            output_path=output1,
            seed=12345,
            temporal_enabled=True,
            spatial_enabled=True,
            scene_enabled=False,
            audio_enabled=False,
        )
        config2 = PerturbationConfig(
            input_path=input_video,
            output_path=output2,
            seed=12345,
            temporal_enabled=True,
            spatial_enabled=True,
            scene_enabled=False,
            audio_enabled=False,
        )

        pipeline1 = PerturbationPipeline(config1)
        result1 = pipeline1.run()

        pipeline2 = PerturbationPipeline(config2)
        result2 = pipeline2.run()

        assert result1 == 0
        assert result2 == 0

        # Both outputs should exist and have the same size
        size1 = Path(output1).stat().st_size
        size2 = Path(output2).stat().st_size
        assert size1 == size2

        # Read and compare frame content
        cap1 = cv2.VideoCapture(output1)
        cap2 = cv2.VideoCapture(output2)

        frames_match = True
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                # Both should end at the same time
                assert ret1 == ret2
                break
            if not np.array_equal(frame1, frame2):
                frames_match = False
                break

        cap1.release()
        cap2.release()
        assert frames_match

    @patch("video_text_translator.perturbation_pipeline._has_audio")
    def test_different_seed_produces_different_output(
        self, mock_has_audio: MagicMock, tmp_path: Path
    ) -> None:
        """Running pipeline with different seeds produces different output."""
        mock_has_audio.return_value = False

        input_video = str(tmp_path / "input.mp4")
        output1 = str(tmp_path / "output1.mp4")
        output2 = str(tmp_path / "output2.mp4")

        # Create a video with high-contrast varying content
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(input_video, fourcc, 30.0, (128, 128))
        rng = np.random.default_rng(0)
        for i in range(90):
            # Random noise frames ensure spatial transforms produce visible differences
            frame = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        # Use strong spatial parameters to ensure visible differences
        config1 = PerturbationConfig(
            input_path=input_video,
            output_path=output1,
            seed=11111,
            temporal_enabled=False,
            spatial_enabled=True,
            max_crop_percent=30.0,
            max_zoom=1.8,
            scene_enabled=False,
            audio_enabled=False,
            change_interval=0.5,
        )
        config2 = PerturbationConfig(
            input_path=input_video,
            output_path=output2,
            seed=99999,
            temporal_enabled=False,
            spatial_enabled=True,
            max_crop_percent=30.0,
            max_zoom=1.8,
            scene_enabled=False,
            audio_enabled=False,
            change_interval=0.5,
        )

        pipeline1 = PerturbationPipeline(config1)
        result1 = pipeline1.run()

        pipeline2 = PerturbationPipeline(config2)
        result2 = pipeline2.run()

        assert result1 == 0
        assert result2 == 0

        # With strong spatial transforms and random noise input,
        # the outputs should differ in file size or frame content
        size1 = Path(output1).stat().st_size
        size2 = Path(output2).stat().st_size

        # Read frames and check they differ
        cap1 = cv2.VideoCapture(output1)
        cap2 = cv2.VideoCapture(output2)

        total_diff = 0
        frame_count = 0
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                break
            frame_count += 1
            total_diff += np.sum(np.abs(frame1.astype(int) - frame2.astype(int)))

        cap1.release()
        cap2.release()

        # With random noise input and different crop/zoom parameters,
        # there should be significant pixel differences
        assert total_diff > 0 or size1 != size2


class TestPerturbationPipelineSuccessPath:
    """Test successful pipeline execution."""

    @patch("video_text_translator.perturbation_pipeline._has_audio")
    def test_minimal_pipeline_success(
        self, mock_has_audio: MagicMock, tmp_path: Path
    ) -> None:
        """Pipeline with all transforms disabled produces valid output."""
        mock_has_audio.return_value = False

        input_video = str(tmp_path / "input.mp4")
        output_video = str(tmp_path / "output.mp4")
        _create_test_video(input_video, n_frames=10)

        config = PerturbationConfig(
            input_path=input_video,
            output_path=output_video,
            temporal_enabled=False,
            spatial_enabled=False,
            scene_enabled=False,
            audio_enabled=False,
            seed=42,
        )
        pipeline = PerturbationPipeline(config)
        result = pipeline.run()
        assert result == 0
        assert Path(output_video).exists()
        assert Path(output_video).stat().st_size > 0

    @patch("video_text_translator.perturbation_pipeline._has_audio")
    def test_temporal_only_pipeline(
        self, mock_has_audio: MagicMock, tmp_path: Path
    ) -> None:
        """Pipeline with only temporal drift enabled succeeds."""
        mock_has_audio.return_value = False

        input_video = str(tmp_path / "input.mp4")
        output_video = str(tmp_path / "output.mp4")
        _create_test_video(input_video, n_frames=30)

        config = PerturbationConfig(
            input_path=input_video,
            output_path=output_video,
            temporal_enabled=True,
            spatial_enabled=False,
            scene_enabled=False,
            audio_enabled=False,
            seed=42,
        )
        pipeline = PerturbationPipeline(config)
        result = pipeline.run()
        assert result == 0
        assert Path(output_video).exists()

    @patch("video_text_translator.perturbation_pipeline._has_audio")
    def test_spatial_only_pipeline(
        self, mock_has_audio: MagicMock, tmp_path: Path
    ) -> None:
        """Pipeline with only spatial transform enabled succeeds."""
        mock_has_audio.return_value = False

        input_video = str(tmp_path / "input.mp4")
        output_video = str(tmp_path / "output.mp4")
        _create_test_video(input_video, n_frames=15)

        config = PerturbationConfig(
            input_path=input_video,
            output_path=output_video,
            temporal_enabled=False,
            spatial_enabled=True,
            scene_enabled=False,
            audio_enabled=False,
            seed=42,
        )
        pipeline = PerturbationPipeline(config)
        result = pipeline.run()
        assert result == 0
        assert Path(output_video).exists()
