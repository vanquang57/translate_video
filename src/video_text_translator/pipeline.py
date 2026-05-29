"""End-to-end pipeline orchestrator.

The pipeline runs in two passes:

1. Pass 1 - streaming detection + tracking. Frames are read once; OCR is
   invoked every ``ocr_stride`` frames. Active segments are matched and
   maintained by the tracker, then finalized.
2. Translation - all canonical segment texts are translated in one phase
   (the translator caches duplicates).
3. Pass 2 - streaming inpainting + rendering + writing. Frames are read
   again. For frames covered by a segment, the inpainter wipes the
   original Chinese text and the renderer composites the Vietnamese
   translation. Frames are written to a temporary MP4.
4. Audio mux - ffmpeg copies the temporary video together with the
   original audio (if any) into the final output without re-encoding.

Pre-flight validation runs before any frame is processed.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from .detector import IDetector
from .encoder import FFmpegEncoder
from .errors import InvalidInputError, OutputWriteError, PipelineError
from .inpainter import IInpainter
from .models import Bounding_Box, Config, Frame_Region_Entry, Subtitle_Region, Text_Segment
from .parallel_pass2 import ParallelPass2, _should_use_parallel
from .progress import ProgressReporter
from .renderer import IRenderer
from .tracker import ITracker
from .translator import ITranslator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    def __init__(
        self,
        config: Config,
        detector: IDetector,
        tracker: ITracker,
        inpainter: IInpainter,
        translator: ITranslator,
        renderer: IRenderer,
        progress: ProgressReporter | None = None,
        export_subtitles: bool = False,
        subtitle_region: Subtitle_Region | None = None,
        remove_text_in_region: bool = False,
    ) -> None:
        self.config = config
        self.detector = detector
        self.tracker = tracker
        self.inpainter = inpainter
        self.translator = translator
        self.renderer = renderer
        self.progress = progress or ProgressReporter()
        self._export_subtitles = export_subtitles
        self._subtitle_region = subtitle_region
        self._remove_text_in_region = remove_text_in_region

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Execute the full pipeline; returns a process exit code."""
        try:
            self._validate_inputs()
            cap, meta = self._open_video()
        except PipelineError as exc:
            logger.error("pre-flight: %s", exc)
            return 2

        width, height, fps, n_frames = meta
        try:
            segments = self._pass1(cap, width, height, fps, n_frames)
            cap.release()

            if not segments:
                logger.warning(
                    "no Chinese text was detected in the video; "
                    "copying input to output"
                )
                self._copy_input_to_output()
                return self._verify_output()

            translations = self._translate_segments(segments)

            # --- SRT subtitle export ---
            if self._export_subtitles:
                self._export_srt(segments, translations)

            self._pass2(width, height, fps, n_frames, segments, translations)
            self._mux_audio()
            return self._verify_output()
        except PipelineError as exc:
            logger.error("pipeline failed: %s", exc)
            return 3
        except Exception:  # noqa: BLE001 - top-level safety net
            logger.error("pipeline crashed: %s", traceback.format_exc())
            return 4
        finally:
            self.progress.close()

    # ------------------------------------------------------------------
    # Subtitle export
    # ------------------------------------------------------------------

    def _export_srt(
        self,
        segments: Sequence[Text_Segment],
        translations: dict[str, str],
    ) -> None:
        """Generate and write the SRT file. Non-critical — logs on failure."""
        from .subtitle_exporter import export_srt

        try:
            srt_path = export_srt(
                segments=segments,
                translations=translations,
                video_output_path=self.config.output_path,
                region=self._subtitle_region,
            )
            logger.info("SRT exported: %s", srt_path)
        except Exception as exc:
            logger.warning("SRT export failed (non-critical): %s", exc)

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _validate_inputs(self) -> None:
        in_path = self.config.input_path
        out_path = self.config.output_path
        if not in_path:
            raise InvalidInputError("input_path is empty (use --input)")
        if not out_path:
            raise InvalidInputError("output_path is empty (use --output)")
        in_p = Path(in_path)
        if not in_p.is_file():
            raise InvalidInputError(f"input file not found: {in_p}")
        if not os.access(in_p, os.R_OK):
            raise InvalidInputError(f"input file is not readable: {in_p}")

        size = in_p.stat().st_size
        if size == 0:
            raise InvalidInputError(f"input file is empty: {in_p}")
        if size > self.config.performance.max_file_size_bytes:
            raise InvalidInputError(
                f"input file size ({size} bytes) exceeds limit "
                f"({self.config.performance.max_file_size_bytes} bytes)"
            )

        out_p = Path(out_path)
        out_dir = out_p.parent if out_p.parent != Path("") else Path(".")
        if not out_dir.exists():
            raise InvalidInputError(
                f"output directory does not exist: {out_dir}"
            )
        if not os.access(out_dir, os.W_OK):
            raise InvalidInputError(
                f"output directory is not writable: {out_dir}"
            )

    def _open_video(self) -> tuple[cv2.VideoCapture, tuple[int, int, float, int]]:
        cap = cv2.VideoCapture(self.config.input_path)
        if not cap.isOpened():
            raise InvalidInputError(
                f"cannot open video (unsupported format or corrupted file): "
                f"{self.config.input_path}"
            )
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or fps <= 0 or n_frames <= 0:
            raise InvalidInputError(
                f"video metadata invalid (w={width}, h={height}, fps={fps}, n={n_frames})"
            )
        duration = n_frames / fps if fps > 0 else 0.0
        if duration > self.config.performance.max_duration_seconds:
            raise InvalidInputError(
                f"video duration {duration:.1f}s exceeds limit "
                f"({self.config.performance.max_duration_seconds}s)"
            )
        logger.info(
            "video opened: %dx%d @ %.2f fps, %d frames (%.1fs)",
            width, height, fps, n_frames, duration,
        )
        return cap, (width, height, fps, n_frames)

    # ------------------------------------------------------------------
    # Pass 1 - detection + tracking
    # ------------------------------------------------------------------

    def _pass1(
        self,
        cap: cv2.VideoCapture,
        width: int,
        height: int,
        fps: float,
        n_frames: int,
    ) -> tuple[Text_Segment, ...]:
        stride = self.config.performance.ocr_stride
        self.progress.start(n_frames, "pass1 detect+track")

        t0 = time.time()
        frame_index = 0
        last_processed = -1
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            should_ocr = (
                frame_index % stride == 0
                or frame_index == n_frames - 1
            )
            if should_ocr:
                ts = frame_index / fps
                regions = list(
                    self.detector.detect(frame, frame_index, ts)
                )
                self.tracker.update(frame_index, ts, regions)
                last_processed = frame_index
            self.progress.update(1)
            frame_index += 1

        # Tracker handles fill_missing inside finalize().
        segments = self.tracker.finalize()
        elapsed = time.time() - t0
        logger.info(
            "pass1 done in %.1fs: %d segments (last OCR'd frame=%d)",
            elapsed, len(segments), last_processed,
        )
        self.progress.close()
        return segments

    # ------------------------------------------------------------------
    # Translation phase
    # ------------------------------------------------------------------

    def _translate_segments(
        self, segments: Sequence[Text_Segment]
    ) -> dict[str, str]:
        logger.info("translating %d segments...", len(segments))
        results = self.translator.translate_segments(segments)
        translated = sum(1 for r in results.values() if r.status == "translated")
        passthrough = sum(1 for r in results.values() if r.status == "passthrough")
        failed = sum(1 for r in results.values() if r.status == "untranslated")
        logger.info(
            "translation: %d ok, %d passthrough, %d failed",
            translated, passthrough, failed,
        )
        return {sid: r.translated_text for sid, r in results.items()}

    # ------------------------------------------------------------------
    # Pass 2 - inpaint + render + write
    # ------------------------------------------------------------------

    def _pass2(
        self,
        width: int,
        height: int,
        fps: float,
        n_frames: int,
        segments: Sequence[Text_Segment],
        translations: dict[str, str],
    ) -> None:
        # Pre-compute one stable font size per segment.
        # Strategy: use the MEDIAN box (by area) for a representative size.
        # This avoids the problem where the smallest box is too small
        # (causing None) and prevents font size from being recalculated
        # per-frame which causes jitter.
        fixed_font_size: dict[str, int | None] = {}
        for seg in segments:
            text_vi = translations.get(seg.segment_id, seg.canonical_text)
            if not seg.entries:
                fixed_font_size[seg.segment_id] = None
                continue
            # Sort entries by area and pick the median box as representative.
            sorted_entries = sorted(seg.entries, key=lambda e: e.box.area)
            median_idx = len(sorted_entries) // 2
            median_box = sorted_entries[median_idx].box
            min_box = sorted_entries[0].box
            # Try min_box first (guarantees text fits all frames).
            try:
                size = self.renderer.auto_fit_font_size(
                    text_vi, min_box, self.config.renderer
                )
            except AttributeError:
                size = None
            # If min_box is too small, try median_box as fallback.
            if size is None:
                try:
                    size = self.renderer.auto_fit_font_size(
                        text_vi, median_box, self.config.renderer
                    )
                except AttributeError:
                    size = None
            fixed_font_size[seg.segment_id] = size

        use_parallel = _should_use_parallel(self.config.performance)

        if use_parallel:
            self._pass2_parallel(
                width, height, fps, n_frames, segments, translations,
                fixed_font_size,
            )
        else:
            self._pass2_sequential(
                width, height, fps, n_frames, segments, translations,
                fixed_font_size,
            )

    def _pass2_parallel(
        self,
        width: int,
        height: int,
        fps: float,
        n_frames: int,
        segments: Sequence[Text_Segment],
        translations: dict[str, str],
        fixed_font_size: dict[str, int | None],
    ) -> None:
        """Pass2 using multi-threaded pipeline with FFmpeg encoder."""
        from .encoder import detect_best_encoder

        # Report encoder info before starting.
        perf = self.config.performance
        encoder_name, encoder_desc = detect_best_encoder(perf.encoder)
        self.progress.set_info("Encoder", encoder_desc)
        workers = self._resolve_pass2_workers()
        self.progress.set_info("Workers", f"{workers} threads (CPU inpaint+render)")

        self.progress.start(n_frames, "pass2 inpaint+render (parallel)")
        try:
            executor = ParallelPass2(
                config=self.config,
                inpainter=self.inpainter,
                renderer=self.renderer,
                segments=segments,
                translations=translations,
                width=width,
                height=height,
                fps=fps,
                n_frames=n_frames,
                fixed_font_size=fixed_font_size,
                progress=self.progress,
                remove_text_in_region=self._remove_text_in_region,
                subtitle_region=self._subtitle_region,
            )
            tmp_path = executor.run()
            self._tmp_video = tmp_path
        finally:
            self.progress.close()

    def _resolve_pass2_workers(self) -> int:
        """Determine number of pass2 workers (mirrors parallel_pass2 logic)."""
        perf = self.config.performance
        if perf.pass2_mode == "sequential":
            return 1
        if perf.parallel_workers > 0:
            return perf.parallel_workers
        import os as _os
        cores = _os.cpu_count() or 4
        return max(1, min(cores - 2, 8))

    def _pass2_sequential(
        self,
        width: int,
        height: int,
        fps: float,
        n_frames: int,
        segments: Sequence[Text_Segment],
        translations: dict[str, str],
        fixed_font_size: dict[str, int | None],
    ) -> None:
        """Pass2 using single-threaded loop with FFmpeg encoder."""
        from .encoder import detect_best_encoder

        # Index entries by frame_index for O(1) lookup per frame.
        per_frame: dict[int, list[tuple[Text_Segment, Frame_Region_Entry]]] = {}
        for seg in segments:
            for entry in seg.entries:
                per_frame.setdefault(entry.frame_index, []).append((seg, entry))

        tmp_path = self._tmp_video_path()
        perf = self.config.performance

        # Report encoder info before starting.
        encoder_name, encoder_desc = detect_best_encoder(perf.encoder)
        self.progress.set_info("Encoder", encoder_desc)
        self.progress.set_info("Workers", "1 thread (sequential)")

        # Use FFmpeg encoder instead of cv2.VideoWriter for better
        # performance and hardware acceleration support.
        encoder = FFmpegEncoder(
            output_path=tmp_path,
            width=width,
            height=height,
            fps=fps,
            encoder_mode=perf.encoder,
            encoder_preset=perf.encoder_preset,
        )
        encoder.open()

        cap = cv2.VideoCapture(self.config.input_path)
        if not cap.isOpened():
            encoder.close()
            raise OutputWriteError(
                f"cannot reopen input for pass 2: {self.config.input_path}"
            )

        self.progress.start(n_frames, "pass2 inpaint+render (sequential)")
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                hits = per_frame.get(frame_index, [])
                if hits:
                    boxes: list[Bounding_Box] = [e.box for _seg, e in hits]
                    frame = self.inpainter.inpaint_frame(frame, boxes)
                    for seg, entry in hits:
                        # Skip rendering if remove_text_in_region is active
                        # and segment center falls inside the region
                        if self._remove_text_in_region and self._subtitle_region is not None:
                            if seg.entries:
                                cx, cy = seg.entries[0].box.center
                                if self._subtitle_region.contains_point(cx, cy):
                                    continue
                        text_vi = translations.get(seg.segment_id, seg.canonical_text)
                        frame = self.renderer.render(
                            frame,
                            text_vi,
                            entry.box,
                            self.config.renderer,
                            segment_id=seg.segment_id,
                            frame_index=frame_index,
                            fixed_font_size=fixed_font_size.get(seg.segment_id),
                            frame_size=(width, height),
                        )
                encoder.write(frame)
                self.progress.update(1)
                frame_index += 1
        finally:
            cap.release()
            encoder.close()
            self.progress.close()

        self._tmp_video = tmp_path
        logger.info("pass2 wrote %d frames to %s", frame_index, tmp_path)

    # ------------------------------------------------------------------
    # Audio mux
    # ------------------------------------------------------------------

    def _tmp_video_path(self) -> str:
        out = Path(self.config.output_path)
        return str(out.with_suffix(".tmp.mp4"))

    def _mux_audio(self) -> None:
        tmp = self._tmp_video_path()
        out = self.config.output_path
        has_audio = _has_audio(self.config.input_path)
        try:
            if has_audio:
                logger.info("muxing audio from input into output via ffmpeg")
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", tmp,
                    "-i", self.config.input_path,
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c", "copy",
                    out,
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise OutputWriteError(
                        f"ffmpeg mux failed (exit {proc.returncode}): "
                        f"{proc.stderr.strip()}"
                    )
            else:
                logger.info("no audio in input; renaming tmp video to output")
                if Path(out).exists():
                    Path(out).unlink()
                shutil.move(tmp, out)
                return
        except OutputWriteError:
            raise
        except FileNotFoundError as exc:
            raise OutputWriteError(
                "ffmpeg not found on PATH; cannot mux audio"
            ) from exc

        # Clean up tmp on success.
        try:
            if Path(tmp).exists():
                Path(tmp).unlink()
        except OSError as exc:
            logger.warning("failed to remove tmp file %s: %s", tmp, exc)

    # ------------------------------------------------------------------
    # No-text fallback / final verification
    # ------------------------------------------------------------------

    def _copy_input_to_output(self) -> None:
        shutil.copyfile(self.config.input_path, self.config.output_path)

    def _verify_output(self) -> int:
        out = Path(self.config.output_path)
        if not out.exists() or out.stat().st_size == 0:
            logger.error(
                "output file missing or empty: %s",
                out.resolve() if out.parent.exists() else out,
            )
            return 5
        abs_path = str(out.resolve())
        logger.info("output written: %s", abs_path)
        print(abs_path)
        return 0
