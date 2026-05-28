"""Perturbation pipeline orchestrator.

This module implements the PerturbationPipeline class which orchestrates
the full video perturbation process: input validation, scene detection,
temporal drift, spatial transform, audio perturbation, and final muxing.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
import subprocess
import traceback
from pathlib import Path

import cv2
import numpy as np

from .encoder import FFmpegEncoder
from .errors import (
    InvalidConfigError,
    InvalidInputError,
    OutputWriteError,
    PipelineError,
)
from .perturbation_audio import AudioPerturbationProcessor
from .perturbation_color import ColorDriftProcessor
from .perturbation_combo import MultiTransformCombo
from .perturbation_config import PerturbationConfig, validate_config
from .perturbation_overlay import OverlayProcessor
from .perturbation_parallel import ParallelFrameProcessor
from .perturbation_rotation import RotationDriftProcessor
from .perturbation_scene import SceneRecompositionProcessor
from .perturbation_scheduler import CorrelatedScheduler
from .perturbation_spatial import SpatialTransformProcessor
from .perturbation_temporal import TemporalDriftProcessor
from .perturbation_warp import LocalizedWarpProcessor
from .progress import ProgressReporter

logger = logging.getLogger(__name__)


def _has_audio(input_path: str) -> bool:
    """Probe the input file for an audio stream using ffprobe."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                input_path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return bool(proc.stdout.strip())
    except FileNotFoundError:
        logger.warning("ffprobe not found on PATH; assuming no audio.")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("ffprobe timed out; assuming no audio.")
        return False


class PerturbationPipeline:
    """Main orchestrator for the perturbation process.

    Coordinates all perturbation processors (temporal, spatial, scene,
    audio, combo) and manages the full pipeline lifecycle from input
    validation through to final output muxing.
    """

    def __init__(
        self,
        config: PerturbationConfig,
        progress: ProgressReporter | None = None,
    ) -> None:
        """Initialize the perturbation pipeline.

        Args:
            config: Perturbation configuration with all parameters.
            progress: Optional progress reporter for UI updates.
        """
        self.config = config
        self.progress = progress or ProgressReporter()

    def run(self) -> int:
        """Execute full perturbation pipeline. Returns exit code.

        Exit codes:
            0: success
            1: config error
            2: input error
            3: processing error
            4: unexpected crash
        """
        try:
            # Step 1: Validate config
            validate_config(self.config)
        except InvalidConfigError as exc:
            logger.error("config error: %s", exc)
            return 1

        try:
            # Step 2: Validate inputs and open video
            self._validate_inputs()
            cap, meta = self._open_video()
        except InvalidInputError as exc:
            logger.error("input error: %s", exc)
            return 2

        width, height, fps, n_frames = meta
        try:
            self._process(cap, width, height, fps, n_frames)
            return self._verify_output()
        except PipelineError as exc:
            logger.error("processing error: %s", exc)
            return 3
        except Exception:  # noqa: BLE001 - top-level safety net
            logger.error("unexpected crash: %s", traceback.format_exc())
            return 4
        finally:
            self.progress.close()

    def _process(
        self,
        cap: cv2.VideoCapture,
        width: int,
        height: int,
        fps: float,
        n_frames: int,
    ) -> None:
        """Run the core processing pipeline.

        Args:
            cap: Opened video capture.
            width: Frame width.
            height: Frame height.
            fps: Frames per second.
            n_frames: Total frame count.
        """
        duration = n_frames / fps

        # Step 3: Initialize RNG
        if self.config.seed is not None:
            rng = random.Random(self.config.seed)
        else:
            rng = random.Random()

        # Step 4: If combo_enabled, create MultiTransformCombo and get scaled config
        config = self.config
        if config.combo_enabled:
            combo = MultiTransformCombo(config, rng)
            config = combo.create_scaled_config()

        # Step 5: Scene detection and reordering (if enabled)
        scene_frame_order: list[tuple[int, int]] | None = None
        if config.scene_enabled:
            scene_frame_order = self._process_scenes(config, rng, cap, n_frames, fps)

        # Step 6: Compute temporal drift frame map
        frame_map: list[int] | None = None
        if config.temporal_enabled:
            temporal = TemporalDriftProcessor(config, rng)
            frame_map = temporal.compute_frame_map(n_frames, fps)
            logger.info(
                "temporal drift: %d output frames from %d input frames",
                len(frame_map), n_frames,
            )

        # Step 7: Create spatial transform processor
        spatial: SpatialTransformProcessor | None = None
        if config.spatial_enabled:
            spatial = SpatialTransformProcessor(config, width, height, rng)

        # Step 7a: Create correlated scheduler for shared oscillation (Wave 2)
        correlated_sched: CorrelatedScheduler | None = None
        if config.scheduling_mode != "random":
            correlated_sched = CorrelatedScheduler(
                duration=duration,
                change_interval=config.change_interval,
                rng=random.Random(rng.randint(0, 2**32 - 1)),
                mode=config.scheduling_mode,
                base_frequency=config.base_frequency,
            )
            logger.info(
                "correlated scheduling: mode=%s, freq=%.3f Hz",
                config.scheduling_mode, config.base_frequency,
            )

        # Step 7b: Create rotation drift processor (Wave 1)
        rotation: RotationDriftProcessor | None = None
        if config.rotation_enabled:
            rotation = RotationDriftProcessor(config, width, height, rng)

        # Step 7c: Create color drift processor (Wave 1)
        color: ColorDriftProcessor | None = None
        if config.color_drift_enabled:
            color = ColorDriftProcessor(config, rng)

        # Step 7d: Create overlay processor (Wave 1)
        overlay: OverlayProcessor | None = None
        if config.overlay_enabled:
            overlay = OverlayProcessor(config, width, height, rng)

        # Step 7e: Determine GOP size (Wave 1)
        gop_size: int | None = None
        if config.gop_perturbation_enabled:
            gop_size = rng.randint(config.gop_min, config.gop_max)
            logger.info("GOP perturbation: using gop_size=%d", gop_size)

        # Step 7f: Create localized warp processor (Wave 2)
        warp: LocalizedWarpProcessor | None = None
        if config.warp_enabled:
            base_phase = correlated_sched.base_phase if correlated_sched else None
            warp = LocalizedWarpProcessor(config, width, height, rng, base_phase)
            logger.info(
                "warp: grid=%dx%d, max_disp=%.1fpx",
                config.warp_grid_size, config.warp_grid_size,
                config.warp_max_displacement,
            )

        # Step 8: Frame-by-frame processing
        output_frame_count = len(frame_map) if frame_map else n_frames
        tmp_video_path = self._tmp_video_path()

        encoder = FFmpegEncoder(
            output_path=tmp_video_path,
            width=width,
            height=height,
            fps=fps,
            encoder_mode=config.encoder,
            encoder_preset=config.encoder_preset,
            gop_size=gop_size,
        )
        encoder.open()

        self.progress.start(output_frame_count, "perturbation: processing frames")

        try:
            if scene_frame_order is not None:
                # Scene recomposition changes the reading order
                self._process_frames_with_scenes(
                    cap, encoder, scene_frame_order, frame_map,
                    spatial, warp, rotation, color, overlay, fps, width, height
                )
            else:
                # Determine worker count: force single-thread when seed is
                # set to guarantee bit-exact reproducibility.
                workers = config.parallel_workers
                if self.config.seed is not None:
                    workers = 1

                parallel = ParallelFrameProcessor(
                    num_workers=workers,
                    buffer_size=32,
                )
                parallel.process_frames(
                    cap, encoder, frame_map,
                    spatial, warp, rotation, color, overlay,
                    fps, n_frames, output_frame_count,
                    progress=self.progress,
                )
        finally:
            cap.release()
            encoder.close()
            self.progress.close()

        # Step 9-10: Audio extraction and perturbation
        self._process_audio(config, rng)

        # Step 11: Mux video + audio
        self._mux_audio()

        # Step 12: Clean up temp files
        self._cleanup_temp_files()

    def _process_scenes(
        self,
        config: PerturbationConfig,
        rng: random.Random,
        cap: cv2.VideoCapture,
        n_frames: int,
        fps: float,
    ) -> list[tuple[int, int]] | None:
        """Detect scenes, reorder, and return frame ranges.

        Returns:
            List of (start_frame, end_frame) tuples in the new order,
            or None if scene recomposition was skipped.
        """
        scene_proc = SceneRecompositionProcessor(config, rng)
        boundaries = scene_proc.detect_scenes(self.config.input_path)

        scenes = scene_proc.boundaries_to_scenes(boundaries, n_frames, fps)

        if len(scenes) < 2:
            logger.warning(
                "Fewer than 2 scenes detected (%d). Skipping scene recomposition.",
                len(scenes),
            )
            return None

        # Reorder scenes
        reordered = scene_proc.reorder_scenes(scenes)

        # Insert transitions
        reordered = scene_proc.insert_transitions(reordered)

        # Build frame order from reordered scenes
        frame_order = [
            (scene.start_frame, scene.end_frame) for scene in reordered
        ]
        return frame_order

    def _process_frames(
        self,
        cap: cv2.VideoCapture,
        encoder: FFmpegEncoder,
        frame_map: list[int] | None,
        spatial: SpatialTransformProcessor | None,
        warp: LocalizedWarpProcessor | None,
        rotation: RotationDriftProcessor | None,
        color: ColorDriftProcessor | None,
        overlay: OverlayProcessor | None,
        fps: float,
        n_frames: int,
        output_frame_count: int,
    ) -> None:
        """Process frames sequentially using frame_map and transforms.

        Transform order: Spatial → Warp → Rotation → Color → Overlay

        Uses streaming approach with a small sliding buffer to avoid
        loading all frames into memory. This keeps RAM usage constant
        regardless of video length.
        """
        if frame_map is not None:
            # Streaming approach: read frames sequentially, keep a small
            # sliding buffer. frame_map is nearly monotonic (each entry
            # is >= previous or at most a few frames back due to
            # drops/duplicates), so we only need to buffer a small window.
            #
            # Buffer strategy: keep frames from min_needed to current read
            # position, evict frames that are no longer referenced.

            read_pos = -1  # Last frame index read from cap
            frame_buffer: dict[int, np.ndarray] = {}
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            # Precompute the minimum future reference for each position
            # to know when we can evict a frame from buffer.
            # For efficiency, compute the last output index that needs each input frame.
            last_use: dict[int, int] = {}
            for out_idx, in_idx in enumerate(frame_map):
                last_use[in_idx] = out_idx

            for out_idx, in_idx in enumerate(frame_map):
                # Read forward until we have the frame we need
                while read_pos < in_idx:
                    ret, frame = cap.read()
                    read_pos += 1
                    if not ret:
                        break
                    # Only buffer if this frame is still needed
                    if read_pos in last_use and last_use[read_pos] >= out_idx:
                        frame_buffer[read_pos] = frame

                frame = frame_buffer.get(in_idx)
                if frame is None:
                    # Fallback: black frame if unavailable
                    frame = np.zeros(
                        (frame_height, frame_width, 3), dtype=np.uint8
                    )

                # Apply transforms in order: Spatial → Warp → Rotation → Color → Overlay
                timestamp = out_idx / fps
                if spatial is not None:
                    frame = spatial.transform_frame(frame, timestamp)
                if warp is not None:
                    frame = warp.transform_frame(frame, timestamp)
                if rotation is not None:
                    frame = rotation.transform_frame(frame, timestamp)
                if color is not None:
                    frame = color.transform_frame(frame, timestamp)
                if overlay is not None:
                    frame = overlay.transform_frame(frame, timestamp)

                encoder.write(frame)
                self.progress.update(1)

                # Evict frames no longer needed
                if in_idx in last_use and last_use[in_idx] <= out_idx:
                    frame_buffer.pop(in_idx, None)
        else:
            # No temporal drift: read and process frames sequentially
            for out_idx in range(n_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # Apply transforms in order: Spatial → Warp → Rotation → Color → Overlay
                timestamp = out_idx / fps
                if spatial is not None:
                    frame = spatial.transform_frame(frame, timestamp)
                if warp is not None:
                    frame = warp.transform_frame(frame, timestamp)
                if rotation is not None:
                    frame = rotation.transform_frame(frame, timestamp)
                if color is not None:
                    frame = color.transform_frame(frame, timestamp)
                if overlay is not None:
                    frame = overlay.transform_frame(frame, timestamp)

                encoder.write(frame)
                self.progress.update(1)

    def _process_frames_with_scenes(
        self,
        cap: cv2.VideoCapture,
        encoder: FFmpegEncoder,
        scene_frame_order: list[tuple[int, int]],
        frame_map: list[int] | None,
        spatial: SpatialTransformProcessor | None,
        warp: LocalizedWarpProcessor | None,
        rotation: RotationDriftProcessor | None,
        color: ColorDriftProcessor | None,
        overlay: OverlayProcessor | None,
        fps: float,
        width: int,
        height: int,
    ) -> None:
        """Process frames with scene recomposition ordering.

        Reads frames according to the scene order, applies temporal drift
        and all transforms (Spatial → Warp → Rotation → Color → Overlay).
        """
        # Read all frames into memory for scene reordering
        all_frames: list[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)

        # Build output frames from scene order
        output_frames: list[np.ndarray] = []
        for start_frame, end_frame in scene_frame_order:
            for idx in range(start_frame, min(end_frame, len(all_frames))):
                output_frames.append(all_frames[idx])

        # Apply temporal drift frame_map if present
        if frame_map is not None:
            mapped_frames: list[np.ndarray] = []
            for in_idx in frame_map:
                if in_idx < len(output_frames):
                    mapped_frames.append(output_frames[in_idx])
                elif output_frames:
                    mapped_frames.append(output_frames[-1])
                else:
                    mapped_frames.append(
                        np.zeros((height, width, 3), dtype=np.uint8)
                    )
            output_frames = mapped_frames

        # Write frames with all transforms
        for out_idx, frame in enumerate(output_frames):
            timestamp = out_idx / fps
            if spatial is not None:
                frame = spatial.transform_frame(frame, timestamp)
            if warp is not None:
                frame = warp.transform_frame(frame, timestamp)
            if rotation is not None:
                frame = rotation.transform_frame(frame, timestamp)
            if color is not None:
                frame = color.transform_frame(frame, timestamp)
            if overlay is not None:
                frame = overlay.transform_frame(frame, timestamp)
            encoder.write(frame)
            self.progress.update(1)

    def _process_audio(
        self,
        config: PerturbationConfig,
        rng: random.Random,
    ) -> None:
        """Extract audio, apply perturbation if enabled, save result."""
        input_path = self.config.input_path
        has_audio = _has_audio(input_path)

        if not has_audio:
            logger.warning("No audio track found. Skipping audio perturbation.")
            return

        # Extract audio to temp WAV
        audio_tmp = self._tmp_audio_path()
        self._extract_audio(input_path, audio_tmp)

        if not Path(audio_tmp).exists() or Path(audio_tmp).stat().st_size == 0:
            logger.warning("Audio extraction produced empty file. Skipping audio perturbation.")
            return

        if config.audio_enabled:
            self.progress.start(1, "perturbation: audio processing")
            try:
                audio_proc = AudioPerturbationProcessor(config, rng)
                audio_output = self._tmp_audio_output_path()
                audio_proc.process(audio_tmp, audio_output)
                # Replace the extracted audio with the processed version
                if Path(audio_output).exists():
                    shutil.move(audio_output, audio_tmp)
            except Exception as exc:
                logger.warning(
                    "Audio perturbation failed: %s. Using original audio.", exc
                )
            finally:
                self.progress.update(1)
                self.progress.close()

    def _extract_audio(self, input_path: str, output_path: str) -> None:
        """Extract audio from video using ffmpeg."""
        try:
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", input_path,
                "-vn",
                "-acodec", "pcm_s16le",
                output_path,
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
            if proc.returncode != 0:
                logger.warning(
                    "Audio extraction failed (exit %d): %s",
                    proc.returncode, proc.stderr.strip(),
                )
        except FileNotFoundError:
            logger.warning("ffmpeg not found on PATH; cannot extract audio.")
        except subprocess.TimeoutExpired:
            logger.warning("Audio extraction timed out.")

    def _mux_audio(self) -> None:
        """Mux video and audio into final output."""
        tmp_video = self._tmp_video_path()
        tmp_audio = self._tmp_audio_path()
        output = self.config.output_path

        has_audio_file = Path(tmp_audio).exists() and Path(tmp_audio).stat().st_size > 0

        try:
            if has_audio_file:
                logger.info("Muxing video + audio into final output.")
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", tmp_video,
                    "-i", tmp_audio,
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    output,
                ]
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300
                )
                if proc.returncode != 0:
                    raise OutputWriteError(
                        f"ffmpeg mux failed (exit {proc.returncode}): "
                        f"{proc.stderr.strip()}"
                    )
            else:
                # No audio: just rename/copy the temp video to output
                logger.info("No audio to mux; copying video to output.")
                if Path(output).exists():
                    Path(output).unlink()
                shutil.move(tmp_video, output)
        except OutputWriteError:
            raise
        except FileNotFoundError as exc:
            raise OutputWriteError(
                "ffmpeg not found on PATH; cannot mux audio"
            ) from exc

    def _cleanup_temp_files(self) -> None:
        """Remove temporary files created during processing."""
        for tmp_path in [
            self._tmp_video_path(),
            self._tmp_audio_path(),
            self._tmp_audio_output_path(),
        ]:
            try:
                if Path(tmp_path).exists():
                    Path(tmp_path).unlink()
            except OSError as exc:
                logger.warning("Failed to remove temp file %s: %s", tmp_path, exc)

    def _validate_inputs(self) -> None:
        """Validate input file and output directory."""
        in_path = self.config.input_path
        out_path = self.config.output_path

        if not in_path:
            raise InvalidInputError("Input file path is empty.")
        if not out_path:
            raise InvalidInputError("Output file path is empty.")

        in_p = Path(in_path)
        if not in_p.exists():
            raise InvalidInputError(
                f"Input file not found or unreadable: {in_path}"
            )
        if not in_p.is_file():
            raise InvalidInputError(
                f"Input file not found or unreadable: {in_path}"
            )
        if not os.access(in_p, os.R_OK):
            raise InvalidInputError(
                f"Input file not found or unreadable: {in_path}"
            )

        out_p = Path(out_path)
        out_dir = out_p.parent if out_p.parent != Path("") else Path(".")
        if not out_dir.exists():
            raise InvalidInputError(
                f"Output directory not writable: {out_dir}"
            )
        if not os.access(out_dir, os.W_OK):
            raise InvalidInputError(
                f"Output directory not writable: {out_dir}"
            )

    def _open_video(self) -> tuple[cv2.VideoCapture, tuple[int, int, float, int]]:
        """Open video and extract metadata."""
        cap = cv2.VideoCapture(self.config.input_path)
        if not cap.isOpened():
            raise InvalidInputError(
                f"Cannot open video (corrupted or invalid): {self.config.input_path}"
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0 or fps <= 0 or n_frames <= 0:
            cap.release()
            raise InvalidInputError(
                f"Video metadata invalid (w={width}, h={height}, "
                f"fps={fps}, n={n_frames})"
            )

        logger.info(
            "Video opened: %dx%d @ %.2f fps, %d frames (%.1fs)",
            width, height, fps, n_frames, n_frames / fps,
        )
        return cap, (width, height, fps, n_frames)

    def _verify_output(self) -> int:
        """Verify the output file exists and is non-empty."""
        out = Path(self.config.output_path)
        if not out.exists() or out.stat().st_size == 0:
            logger.error("Output file missing or empty: %s", out)
            return 3
        logger.info("Output written: %s", out.resolve())
        return 0

    def _tmp_video_path(self) -> str:
        """Get path for temporary video file."""
        out = Path(self.config.output_path)
        return str(out.with_suffix(".tmp_video.mp4"))

    def _tmp_audio_path(self) -> str:
        """Get path for temporary extracted audio file."""
        out = Path(self.config.output_path)
        return str(out.with_suffix(".tmp_audio.wav"))

    def _tmp_audio_output_path(self) -> str:
        """Get path for temporary processed audio file."""
        out = Path(self.config.output_path)
        return str(out.with_suffix(".tmp_audio_out.wav"))
