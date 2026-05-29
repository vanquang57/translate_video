"""Parallel pass2 pipeline: multi-threaded read → process → write.

Splits pass2 into three concurrent stages connected by queues:
  1. Reader thread: reads frames from the input video
  2. Worker thread(s): inpaint + render (CPU-bound, multiple workers)
  3. Writer thread: encodes frames to output via FFmpeg pipe

This achieves overlap between I/O and computation, and parallelizes
the CPU-bound inpaint+render across multiple cores.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Sequence

import cv2
import numpy as np

from .encoder import FFmpegEncoder
from .errors import OutputWriteError
from .inpainter import IInpainter
from .models import (
    Bounding_Box,
    Config,
    Frame_Region_Entry,
    Performance_Config,
    Subtitle_Region,
    Text_Segment,
)
from .progress import ProgressReporter
from .renderer import IRenderer

logger = logging.getLogger(__name__)

# Sentinel value to signal end of stream.
_SENTINEL = None


def _resolve_workers(perf: Performance_Config) -> int:
    """Determine the number of parallel workers for pass2."""
    if perf.pass2_mode == "sequential":
        return 1
    if perf.parallel_workers > 0:
        return perf.parallel_workers
    # Auto: use available cores minus 2 (for reader + writer threads).
    cores = os.cpu_count() or 4
    return max(1, min(cores - 2, 8))


def _should_use_parallel(perf: Performance_Config) -> bool:
    """Decide whether to use parallel mode."""
    if perf.pass2_mode == "sequential":
        return False
    if perf.pass2_mode == "parallel":
        return True
    # Auto: use parallel if >= 4 cores available.
    cores = os.cpu_count() or 4
    return cores >= 4


class ParallelPass2:
    """Multi-threaded pass2 executor."""

    def __init__(
        self,
        config: Config,
        inpainter: IInpainter,
        renderer: IRenderer,
        segments: Sequence[Text_Segment],
        translations: dict[str, str],
        width: int,
        height: int,
        fps: float,
        n_frames: int,
        fixed_font_size: dict[str, int | None],
        progress: ProgressReporter | None = None,
        remove_text_in_region: bool = False,
        subtitle_region: Subtitle_Region | None = None,
    ) -> None:
        self._config = config
        self._inpainter = inpainter
        self._renderer = renderer
        self._segments = segments
        self._translations = translations
        self._width = width
        self._height = height
        self._fps = fps
        self._n_frames = n_frames
        self._fixed_font_size = fixed_font_size
        self._progress = progress
        self._remove_text_in_region = remove_text_in_region
        self._subtitle_region = subtitle_region

        # Build per-frame index for O(1) lookup.
        self._per_frame: dict[int, list[tuple[Text_Segment, Frame_Region_Entry]]] = {}
        for seg in segments:
            for entry in seg.entries:
                self._per_frame.setdefault(entry.frame_index, []).append(
                    (seg, entry)
                )

        self._error: Exception | None = None
        self._error_lock = threading.Lock()

    def run(self) -> str:
        """Execute parallel pass2 and return the output tmp path."""
        perf = self._config.performance
        n_workers = _resolve_workers(perf)
        logger.info(
            "pass2 parallel: %d workers, encoder=%s, preset=%s",
            n_workers,
            perf.encoder,
            perf.encoder_preset,
        )

        tmp_path = self._tmp_video_path()

        # Queue sizes: limit memory usage. Each frame is ~6MB (1080p BGR).
        # read_q holds (frame_index, frame) tuples.
        # write_q holds (frame_index, processed_frame) tuples.
        buffer_size = max(n_workers * 2, perf.io_buffer_frames)
        read_q: queue.Queue = queue.Queue(maxsize=buffer_size)
        write_q: queue.Queue = queue.Queue(maxsize=buffer_size)

        # Start threads.
        reader = threading.Thread(
            target=self._reader_loop,
            args=(read_q,),
            name="pass2-reader",
            daemon=True,
        )
        workers = []
        for i in range(n_workers):
            w = threading.Thread(
                target=self._worker_loop,
                args=(read_q, write_q),
                name=f"pass2-worker-{i}",
                daemon=True,
            )
            workers.append(w)
        writer = threading.Thread(
            target=self._writer_loop,
            args=(write_q, tmp_path, n_workers),
            name="pass2-writer",
            daemon=True,
        )

        reader.start()
        for w in workers:
            w.start()
        writer.start()

        # Wait for completion.
        reader.join()
        for w in workers:
            w.join()
        writer.join()

        if self._error is not None:
            raise self._error

        return tmp_path

    def _set_error(self, exc: Exception) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = exc

    def _reader_loop(self, read_q: queue.Queue) -> None:
        """Read frames from input video and put them on the read queue."""
        try:
            cap = cv2.VideoCapture(self._config.input_path)
            if not cap.isOpened():
                raise OutputWriteError(
                    f"cannot reopen input for pass2: {self._config.input_path}"
                )
            frame_index = 0
            while True:
                if self._error is not None:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                # Use timeout to avoid blocking forever if workers are stuck.
                while True:
                    if self._error is not None:
                        break
                    try:
                        read_q.put((frame_index, frame), timeout=2)
                        break
                    except queue.Full:
                        continue
                frame_index += 1
            cap.release()
        except Exception as exc:
            self._set_error(exc)
        finally:
            # Signal all workers that reading is done.
            n_workers = _resolve_workers(self._config.performance)
            for _ in range(n_workers):
                try:
                    read_q.put(_SENTINEL, timeout=2)
                except queue.Full:
                    pass

    def _worker_loop(
        self, read_q: queue.Queue, write_q: queue.Queue
    ) -> None:
        """Process frames: inpaint + render, then put on write queue."""
        try:
            while True:
                if self._error is not None:
                    break
                try:
                    item = read_q.get(timeout=2)
                except queue.Empty:
                    if self._error is not None:
                        break
                    continue
                if item is _SENTINEL:
                    break
                frame_index, frame = item
                processed = self._process_frame(frame_index, frame)
                # Use timeout to avoid blocking forever if writer is stuck.
                while True:
                    if self._error is not None:
                        break
                    try:
                        write_q.put((frame_index, processed), timeout=2)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:
            self._set_error(exc)
        finally:
            write_q.put(_SENTINEL)

    def _writer_loop(
        self, write_q: queue.Queue, tmp_path: str, n_workers: int
    ) -> None:
        """Collect processed frames in order and write to encoder."""
        try:
            perf = self._config.performance
            encoder = FFmpegEncoder(
                output_path=tmp_path,
                width=self._width,
                height=self._height,
                fps=self._fps,
                encoder_mode=perf.encoder,
                encoder_preset=perf.encoder_preset,
            )
            encoder.open()

            # We need to write frames in order. Buffer out-of-order frames.
            next_frame = 0
            buffer: dict[int, np.ndarray] = {}
            sentinels_received = 0

            while sentinels_received < n_workers:
                if self._error is not None:
                    break
                try:
                    item = write_q.get(timeout=5)
                except queue.Empty:
                    # Check if all workers are done (error or finished).
                    if self._error is not None:
                        break
                    continue
                if item is _SENTINEL:
                    sentinels_received += 1
                    continue
                frame_index, frame = item
                buffer[frame_index] = frame

                # Flush all consecutive frames from the buffer.
                while next_frame in buffer:
                    encoder.write(buffer.pop(next_frame))
                    if self._progress:
                        self._progress.update(1)
                    next_frame += 1

            # Flush any remaining buffered frames.
            while next_frame in buffer:
                encoder.write(buffer.pop(next_frame))
                if self._progress:
                    self._progress.update(1)
                next_frame += 1

            encoder.close()
            logger.info("pass2 parallel wrote %d frames to %s", next_frame, tmp_path)
        except Exception as exc:
            self._set_error(exc)

    def _process_frame(
        self, frame_index: int, frame: np.ndarray
    ) -> np.ndarray:
        """Inpaint and render a single frame."""
        hits = self._per_frame.get(frame_index, [])
        if not hits:
            return frame

        boxes: list[Bounding_Box] = [e.box for _seg, e in hits]
        frame = self._inpainter.inpaint_frame(frame, boxes)

        for seg, entry in hits:
            # Skip rendering if remove_text_in_region is active
            # and segment center falls inside the region
            if self._remove_text_in_region and self._subtitle_region is not None:
                if seg.entries:
                    cx, cy = seg.entries[0].box.center
                    if self._subtitle_region.contains_point(cx, cy):
                        continue
            text_vi = self._translations.get(seg.segment_id, seg.canonical_text)
            frame = self._renderer.render(
                frame,
                text_vi,
                entry.box,
                self._config.renderer,
                segment_id=seg.segment_id,
                frame_index=frame_index,
                fixed_font_size=self._fixed_font_size.get(seg.segment_id),
                frame_size=(self._width, self._height),
            )
        return frame

    def _tmp_video_path(self) -> str:
        from pathlib import Path

        out = Path(self._config.output_path)
        return str(out.with_suffix(".tmp.mp4"))
