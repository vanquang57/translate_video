"""FFmpeg-based video encoder with hardware acceleration support.

Replaces cv2.VideoWriter with an FFmpeg subprocess pipe for:
- Hardware-accelerated encoding (NVENC, QSV, AMF)
- Better compression and speed via libx264 presets
- Automatic fallback when hardware encoder is unavailable

The encoder writes raw BGR frames to FFmpeg's stdin pipe.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Literal

import numpy as np

from .errors import OutputWriteError

logger = logging.getLogger(__name__)

Encoder_Mode = Literal["auto", "cpu", "nvenc", "qsv", "amf"]


def _ffmpeg_path() -> str:
    """Return the ffmpeg binary path or raise if not found."""
    path = shutil.which("ffmpeg")
    if not path:
        raise OutputWriteError("ffmpeg not found on PATH")
    return path


def _encoder_available(encoder_name: str) -> bool:
    """Check if a specific encoder actually works (not just listed).

    Simply checking ``ffmpeg -encoders`` is insufficient because ffmpeg
    may list hardware encoders (e.g. h264_nvenc) even when the required
    GPU hardware is not present. We must actually attempt a tiny encode
    to confirm the encoder initializes successfully.
    """
    try:
        # First quick check: is it even listed?
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if encoder_name not in proc.stdout:
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

    # For hardware encoders, actually test encoding a single frame.
    if encoder_name in ("h264_nvenc", "h264_qsv", "h264_amf"):
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=black:s=64x64:d=0.04:r=25",
                    "-c:v", encoder_name,
                    "-frames:v", "1",
                    "-f", "null", "-",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return proc.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    # Software encoders (libx264) — listing is sufficient.
    return True


def detect_best_encoder(preferred: Encoder_Mode = "auto") -> tuple[str, str]:
    """Detect the best available encoder.

    Returns (encoder_name, description) tuple.
    Priority: nvenc > qsv > amf > libx264
    """
    if preferred == "nvenc":
        if _encoder_available("h264_nvenc"):
            return "h264_nvenc", "NVIDIA NVENC"
        raise OutputWriteError("h264_nvenc requested but not available")

    if preferred == "qsv":
        if _encoder_available("h264_qsv"):
            return "h264_qsv", "Intel QuickSync"
        raise OutputWriteError("h264_qsv requested but not available")

    if preferred == "amf":
        if _encoder_available("h264_amf"):
            return "h264_amf", "AMD AMF"
        raise OutputWriteError("h264_amf requested but not available")

    if preferred == "cpu":
        return "libx264", "CPU (libx264)"

    # Auto-detect
    if _encoder_available("h264_nvenc"):
        return "h264_nvenc", "NVIDIA NVENC (auto-detected)"
    if _encoder_available("h264_qsv"):
        return "h264_qsv", "Intel QuickSync (auto-detected)"
    if _encoder_available("h264_amf"):
        return "h264_amf", "AMD AMF (auto-detected)"
    return "libx264", "CPU libx264 (fallback)"


def _build_encoder_args(
    encoder_name: str,
    preset: str,
) -> list[str]:
    """Build encoder-specific FFmpeg arguments."""
    if encoder_name == "h264_nvenc":
        # NVENC presets: p1 (fastest) to p7 (slowest/best quality)
        nvenc_preset_map = {
            "ultrafast": "p1",
            "fast": "p4",
            "medium": "p5",
        }
        nvenc_p = nvenc_preset_map.get(preset, "p4")
        return ["-c:v", "h264_nvenc", "-preset", nvenc_p, "-rc", "vbr",
                "-cq", "23", "-b:v", "0"]

    if encoder_name == "h264_qsv":
        qsv_preset_map = {
            "ultrafast": "veryfast",
            "fast": "fast",
            "medium": "medium",
        }
        qsv_p = qsv_preset_map.get(preset, "fast")
        return ["-c:v", "h264_qsv", "-preset", qsv_p, "-global_quality", "23"]

    if encoder_name == "h264_amf":
        # AMF doesn't have traditional presets; use quality/speed balance
        amf_quality_map = {
            "ultrafast": "speed",
            "fast": "balanced",
            "medium": "quality",
        }
        amf_q = amf_quality_map.get(preset, "balanced")
        return ["-c:v", "h264_amf", "-quality", amf_q, "-rc", "vbr_latency",
                "-qp_i", "23", "-qp_p", "23"]

    # libx264 fallback
    x264_preset_map = {
        "ultrafast": "ultrafast",
        "fast": "veryfast",
        "medium": "medium",
    }
    x264_p = x264_preset_map.get(preset, "veryfast")
    cpu_count = os.cpu_count() or 4
    threads = min(cpu_count, 8)
    return ["-c:v", "libx264", "-preset", x264_p, "-crf", "23",
            "-threads", str(threads)]


class FFmpegEncoder:
    """Write video frames via FFmpeg subprocess pipe.

    Usage:
        encoder = FFmpegEncoder(output_path, width, height, fps, ...)
        encoder.open()
        for frame in frames:
            encoder.write(frame)
        encoder.close()
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        encoder_mode: Encoder_Mode = "auto",
        encoder_preset: str = "fast",
        gop_size: int | None = None,
    ) -> None:
        self._output_path = output_path
        self._width = width
        self._height = height
        self._fps = fps
        self._encoder_mode = encoder_mode
        self._encoder_preset = encoder_preset
        self._gop_size = gop_size
        self._process: subprocess.Popen | None = None
        self._encoder_name: str = ""
        self._frames_written: int = 0

    @property
    def encoder_name(self) -> str:
        return self._encoder_name

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def open(self) -> None:
        """Start the FFmpeg process and prepare for frame writing."""
        _ffmpeg_path()  # Validate ffmpeg exists

        encoder_name, desc = detect_best_encoder(self._encoder_mode)
        self._encoder_name = encoder_name
        logger.info("encoder: using %s (%s)", encoder_name, desc)

        encoder_args = _build_encoder_args(encoder_name, self._encoder_preset)

        # GOP perturbation: add -g flag for custom keyframe interval
        gop_args: list[str] = []
        if self._gop_size is not None:
            gop_args = ["-g", str(self._gop_size)]
            logger.info("encoder: GOP size set to %d", self._gop_size)

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            # Input: raw video from pipe
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self._width}x{self._height}",
            "-r", str(self._fps),
            "-i", "-",
            # Output encoding
            *encoder_args,
            *gop_args,
            "-pix_fmt", "yuv420p",
            # Output file
            self._output_path,
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise OutputWriteError(
                "ffmpeg not found on PATH; cannot encode video"
            ) from exc
        except OSError as exc:
            raise OutputWriteError(
                f"failed to start ffmpeg encoder: {exc}"
            ) from exc

    def write(self, frame: np.ndarray) -> None:
        """Write a single BGR frame to the encoder."""
        if self._process is None or self._process.stdin is None:
            raise OutputWriteError("encoder not opened; call open() first")
        try:
            self._process.stdin.write(frame.tobytes())
            self._frames_written += 1
        except BrokenPipeError as exc:
            stderr = ""
            if self._process.stderr:
                stderr = self._process.stderr.read().decode(errors="replace")
            raise OutputWriteError(
                f"ffmpeg encoder pipe broken after {self._frames_written} frames: {stderr}"
            ) from exc

    def close(self) -> None:
        """Flush and close the encoder, waiting for FFmpeg to finish."""
        if self._process is None:
            return
        try:
            if self._process.stdin:
                self._process.stdin.close()
            self._process.wait(timeout=60)
            if self._process.returncode != 0:
                stderr = ""
                if self._process.stderr:
                    stderr = self._process.stderr.read().decode(errors="replace")
                raise OutputWriteError(
                    f"ffmpeg encoder failed (exit {self._process.returncode}): "
                    f"{stderr.strip()}"
                )
        except subprocess.TimeoutExpired:
            self._process.kill()
            raise OutputWriteError("ffmpeg encoder timed out during close")
        finally:
            self._process = None

    def __enter__(self) -> "FFmpegEncoder":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()
