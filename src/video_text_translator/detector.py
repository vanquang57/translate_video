"""Detector module: PaddleOCR 3.x wrapper with optional CPU/GPU mode + downscale.

Pipeline of :meth:`PaddleOCRDetector.detect`:

1. Resize the frame by ``OCR_Downscale`` for speed.
2. Run ``PaddleOCR.predict``.
3. Filter recognized text by confidence threshold.
4. Drop boxes whose recognized text contains no CJK code point.
5. Scale the surviving bounding boxes back to original-frame coordinates.

Failures from PaddleOCR on a single frame are caught, logged, and
returned as an empty list so the rest of the pipeline can continue
(Req 2.7).

Notes on PaddleOCR 3 result format:
    The ``predict`` method returns a list of result objects (one per input
    image). Each result behaves like a mapping with keys including:
        - ``rec_texts``  : ``list[str]`` recognised strings
        - ``rec_scores`` : ``list[float]`` confidence scores in [0, 1]
        - ``rec_polys``  : ``list[numpy.ndarray]`` 4-vertex polygons (in the
          coordinate space of the (possibly downscaled) input image)
"""

from __future__ import annotations

import logging
import os
from typing import Protocol, Sequence

import cv2
import numpy as np

from .errors import ComputeInitError
from .models import Bounding_Box, Text_Region
from .text_utils import has_cjk

logger = logging.getLogger(__name__)


class IDetector(Protocol):
    def detect(
        self, frame: np.ndarray, frame_index: int, timestamp: float
    ) -> Sequence[Text_Region]:
        ...

    def detect_batch(
        self,
        frames: Sequence[np.ndarray],
        frame_indices: Sequence[int],
        timestamps: Sequence[float],
    ) -> Sequence[Sequence[Text_Region]]:
        ...

    def warmup(self) -> None:
        ...


class PaddleOCRDetector:
    """PaddleOCR 3.x backed detector with downscale + CJK filtering."""

    def __init__(
        self,
        compute_mode: str = "cpu",
        confidence_threshold: float = 0.5,
        downscale: float = 1.5,
        lang: str = "ch",
        model_variant: str = "mobile",
        cpu_threads: int = 0,
    ) -> None:
        if compute_mode not in ("cpu", "gpu"):
            raise ComputeInitError(
                f'compute_mode must be "cpu" or "gpu" (got "{compute_mode}")'
            )
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0] "
                f"(got {confidence_threshold})"
            )
        if not (1.0 <= downscale <= 4.0):
            raise ValueError(f"downscale must be in [1.0, 4.0] (got {downscale})")
        if model_variant not in ("mobile", "server"):
            raise ValueError(
                f'model_variant must be "mobile" or "server" (got "{model_variant}")'
            )
        if not (0 <= cpu_threads <= 64):
            raise ValueError(f"cpu_threads must be in [0, 64] (got {cpu_threads})")

        self._confidence_threshold = confidence_threshold
        self._downscale = downscale
        self._lang = lang
        self._model_variant = model_variant
        self._cpu_threads = cpu_threads or os.cpu_count() or 4
        self._effective_mode = compute_mode
        self._ocr = self._build_ocr(compute_mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def effective_compute_mode(self) -> str:
        return self._effective_mode

    def warmup(self) -> None:
        """Run OCR once on a tiny black frame to trigger lazy weight loading."""
        try:
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self._ocr.predict(dummy)
        except Exception as exc:  # pragma: no cover - non-fatal
            logger.debug("detector warmup ignored error: %s", exc)

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
    ) -> list[Text_Region]:
        try:
            small, scale_x, scale_y = self._maybe_downscale(frame)
            raw = self._ocr.predict(small)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "detector: OCR failed at frame %d: %s", frame_index, exc
            )
            return []
        return self._postprocess(
            raw, scale_x, scale_y, frame_index, timestamp, frame.shape
        )

    def detect_batch(
        self,
        frames: Sequence[np.ndarray],
        frame_indices: Sequence[int],
        timestamps: Sequence[float],
    ) -> list[list[Text_Region]]:
        if len(frames) != len(frame_indices) or len(frames) != len(timestamps):
            raise ValueError("frames/frame_indices/timestamps length mismatch")
        return [
            self.detect(f, idx, ts)
            for f, idx, ts in zip(frames, frame_indices, timestamps)
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_ocr(self, compute_mode: str):
        """Instantiate PaddleOCR, falling back from GPU to CPU if needed."""
        from paddleocr import PaddleOCR  # type: ignore

        wanted_gpu = compute_mode == "gpu"
        det_model = f"PP-OCRv5_{self._model_variant}_det"
        rec_model = f"PP-OCRv5_{self._model_variant}_rec"
        kwargs = dict(
            lang=self._lang,
            text_detection_model_name=det_model,
            text_recognition_model_name=rec_model,
            # We don't need any of these in our use case (subtitles, not docs).
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # oneDNN (MKL-DNN) acceleration for Intel CPUs.
            # Can provide 2-4x speedup on inference. If PaddlePaddle raises
            # "ConvertPirAttribute2RuntimeAttribute not support ..." errors,
            # set enable_mkldnn=False as a workaround.
            enable_mkldnn=True,
            cpu_threads=self._cpu_threads,
        )
        logger.info(
            "PaddleOCR init: model=%s, device=%s, cpu_threads=%d",
            det_model.replace("_det", ""),
            "gpu" if wanted_gpu else "cpu",
            self._cpu_threads,
        )

        try:
            ocr = PaddleOCR(device=("gpu" if wanted_gpu else "cpu"), **kwargs)
            self._effective_mode = "gpu" if wanted_gpu else "cpu"
            return ocr
        except Exception as exc:  # noqa: BLE001
            if not wanted_gpu:
                raise ComputeInitError(
                    f"failed to initialise PaddleOCR on CPU: {exc}"
                ) from exc
            logger.warning(
                "PaddleOCR GPU initialisation failed, falling back to CPU: %s",
                exc,
            )

        # Fallback path: CPU
        try:
            ocr = PaddleOCR(device="cpu", **kwargs)
            self._effective_mode = "cpu"
            return ocr
        except Exception as exc:  # noqa: BLE001
            raise ComputeInitError(
                f"failed to initialise PaddleOCR on both GPU and CPU: {exc}"
            ) from exc

    def _maybe_downscale(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        if self._downscale == 1.0:
            return frame, 1.0, 1.0
        h, w = frame.shape[:2]
        new_w = max(1, int(round(w / self._downscale)))
        new_h = max(1, int(round(h / self._downscale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        scale_x = w / new_w
        scale_y = h / new_h
        return resized, scale_x, scale_y

    def _postprocess(
        self,
        raw,
        scale_x: float,
        scale_y: float,
        frame_index: int,
        timestamp: float,
        frame_shape,
    ) -> list[Text_Region]:
        """Convert PaddleOCR 3.x ``predict`` output to ``Text_Region`` objects."""
        if not raw:
            return []
        # ``predict`` always returns a list; we feed a single image so the
        # first element is the result for that image.
        result = raw[0]
        try:
            texts = list(result["rec_texts"])
            scores = list(result["rec_scores"])
            polys = list(result["rec_polys"])
        except (KeyError, TypeError):
            # Defensive: if the API shape changes, treat as no detections.
            logger.debug(
                "detector: unexpected predict() result shape at frame %d",
                frame_index,
            )
            return []

        h_full, w_full = frame_shape[:2]
        out: list[Text_Region] = []
        for text, conf, poly in zip(texts, scores, polys):
            try:
                conf_f = float(conf)
            except (TypeError, ValueError):
                continue
            if conf_f < self._confidence_threshold:
                continue
            text_str = "" if text is None else str(text)
            if not text_str or not has_cjk(text_str):
                continue
            box = self._poly_to_box(poly, scale_x, scale_y, w_full, h_full)
            if box is None:
                continue
            out.append(
                Text_Region(
                    box=box,
                    text=text_str,
                    confidence=conf_f,
                    frame_index=frame_index,
                    timestamp=timestamp,
                )
            )
        return out

    @staticmethod
    def _poly_to_box(
        poly,
        scale_x: float,
        scale_y: float,
        frame_w: int,
        frame_h: int,
    ) -> Bounding_Box | None:
        """Convert a (4,2) polygon to an axis-aligned Bounding_Box.

        ``poly`` may be a numpy array, list of points, etc. Coordinates are
        in OCR-input space; we scale them by ``(scale_x, scale_y)`` to
        recover original-frame coordinates and clip to the frame.
        """
        try:
            points = np.asarray(poly, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] < 2:
            return None
        xs = points[:, 0] * scale_x
        ys = points[:, 1] * scale_y
        x_min = max(0, int(round(float(xs.min()))))
        y_min = max(0, int(round(float(ys.min()))))
        x_max = min(frame_w, int(round(float(xs.max()))))
        y_max = min(frame_h, int(round(float(ys.max()))))
        w = x_max - x_min
        h = y_max - y_min
        if w <= 0 or h <= 0:
            return None
        return Bounding_Box(x_min, y_min, w, h)
