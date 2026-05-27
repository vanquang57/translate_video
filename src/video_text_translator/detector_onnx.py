"""ONNX Runtime detector with OpenVINO Execution Provider.

This module provides an alternative OCR backend using ONNX Runtime
(optionally accelerated by Intel OpenVINO EP) instead of PaddlePaddle.
It runs the same PP-OCRv5 models (converted to ONNX format) with
identical pre/post-processing, so detection quality is the same.

Supported devices via OpenVINO EP:
  - cpu   : Intel CPU (optimized with oneDNN kernels)
  - npu   : Intel NPU (Core Ultra series)
  - auto  : Automatic device selection (best available)

Requirements:
  - pip install onnxruntime-openvino
  - ONNX models in the configured model directory
    (use scripts/convert_models.py to convert from PaddlePaddle)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .errors import ComputeInitError
from .models import Bounding_Box, Text_Region
from .text_utils import has_cjk

logger = logging.getLogger(__name__)


class OnnxDetector:
    """PP-OCRv5 detector using ONNX Runtime + OpenVINO EP."""

    def __init__(
        self,
        model_dir: str = "models/onnx",
        model_variant: str = "mobile",
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        downscale: float = 1.5,
        cpu_threads: int = 0,
    ) -> None:
        if model_variant not in ("mobile", "server"):
            raise ValueError(
                f'model_variant must be "mobile" or "server" (got "{model_variant}")'
            )
        if device not in ("cpu", "npu", "auto"):
            raise ValueError(
                f'device must be "cpu", "npu", or "auto" (got "{device}")'
            )
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0] "
                f"(got {confidence_threshold})"
            )
        if not (1.0 <= downscale <= 4.0):
            raise ValueError(f"downscale must be in [1.0, 4.0] (got {downscale})")

        self._model_dir = Path(model_dir)
        self._model_variant = model_variant
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._downscale = downscale
        self._cpu_threads = cpu_threads or os.cpu_count() or 4

        self._det_session = None
        self._rec_session = None
        self._rec_chars: list[str] = []

        self._build_sessions()

    # ------------------------------------------------------------------
    # Public API (matches IDetector protocol)
    # ------------------------------------------------------------------

    @property
    def effective_compute_mode(self) -> str:
        return f"onnx-{self._device}"

    def warmup(self) -> None:
        """Run inference once on a dummy image to warm up the sessions."""
        try:
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self.detect(dummy, 0, 0.0)
        except Exception as exc:  # pragma: no cover
            logger.debug("onnx detector warmup ignored error: %s", exc)

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
    ) -> list[Text_Region]:
        try:
            small, scale_x, scale_y = self._maybe_downscale(frame)
            boxes, scores_det = self._run_detection(small)
            if not boxes:
                return []
            results = self._run_recognition(small, boxes)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "onnx detector: OCR failed at frame %d: %s", frame_index, exc
            )
            return []

        h_full, w_full = frame.shape[:2]
        out: list[Text_Region] = []
        for (poly, text, conf) in results:
            if conf < self._confidence_threshold:
                continue
            if not text or not has_cjk(text):
                continue
            box = self._poly_to_box(poly, scale_x, scale_y, w_full, h_full)
            if box is None:
                continue
            out.append(
                Text_Region(
                    box=box,
                    text=text,
                    confidence=conf,
                    frame_index=frame_index,
                    timestamp=timestamp,
                )
            )
        return out

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
    # Session building
    # ------------------------------------------------------------------

    @staticmethod
    def _setup_openvino_dlls() -> None:
        """Add OpenVINO DLL directories to search path (Windows only)."""
        if os.name != "nt":
            return
        import sys

        ov_libs = os.path.join(
            sys.prefix, "Lib", "site-packages", "openvino", "libs"
        )
        if os.path.isdir(ov_libs):
            try:
                os.add_dll_directory(ov_libs)
            except (OSError, AttributeError):
                pass

    def _build_sessions(self) -> None:
        """Build ONNX Runtime sessions for detection and recognition."""
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ComputeInitError(
                "onnxruntime-openvino is not installed. "
                "Install with: pip install onnxruntime-openvino"
            ) from exc

        # On Windows, OpenVINO DLLs need to be discoverable.
        # Add openvino/libs to DLL search path if available.
        self._setup_openvino_dlls()

        det_path = self._model_dir / f"PP-OCRv5_{self._model_variant}_det_infer.onnx"
        rec_path = self._model_dir / f"PP-OCRv5_{self._model_variant}_rec_infer.onnx"
        char_path = self._model_dir / "ppocr_keys_v1.txt"

        # Also check without _infer suffix (backward compat)
        if not det_path.is_file():
            det_path = self._model_dir / f"PP-OCRv5_{self._model_variant}_det.onnx"
        if not rec_path.is_file():
            rec_path = self._model_dir / f"PP-OCRv5_{self._model_variant}_rec.onnx"

        if not det_path.is_file():
            raise ComputeInitError(
                f"Detection model not found: {det_path}\n"
                f"Run: python scripts/convert_models.py --variant {self._model_variant}"
            )
        if not rec_path.is_file():
            raise ComputeInitError(
                f"Recognition model not found: {rec_path}\n"
                f"Run: python scripts/convert_models.py --variant {self._model_variant}"
            )
        if not char_path.is_file():
            raise ComputeInitError(
                f"Character dictionary not found: {char_path}\n"
                f"Run: python scripts/convert_models.py --variant {self._model_variant}"
            )

        # Load character dictionary
        self._rec_chars = ["blank"]  # CTC blank at index 0
        with char_path.open("r", encoding="utf-8") as f:
            for line in f:
                ch = line.rstrip("\n")
                if ch:
                    self._rec_chars.append(ch)
        self._rec_chars.append(" ")  # space token at end

        # Build session options
        providers = self._get_providers()
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        logger.info(
            "ONNX detector init: variant=%s, device=%s, providers=%s",
            self._model_variant,
            self._device,
            [p if isinstance(p, str) else p[0] for p in providers],
        )

        try:
            self._det_session = ort.InferenceSession(
                str(det_path), sess_options=sess_opts, providers=providers
            )
            self._rec_session = ort.InferenceSession(
                str(rec_path), sess_options=sess_opts, providers=providers
            )
        except Exception as exc:
            raise ComputeInitError(
                f"Failed to create ONNX sessions: {exc}"
            ) from exc

    def _get_providers(self) -> list:
        """Get ONNX Runtime execution providers based on device config."""
        device_map = {
            "cpu": "CPU",
            "npu": "NPU",
            "auto": "AUTO:CPU,NPU",
        }
        ov_device = device_map.get(self._device, "CPU")

        config = {
            ov_device.split(":")[0] if ":" in ov_device else ov_device: {
                "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": "1",
            }
        }
        if self._device == "cpu":
            config["CPU"]["INFERENCE_NUM_THREADS"] = str(self._cpu_threads)

        ov_options = {
            "device_type": ov_device,
            "load_config": json.dumps(config),
        }

        # Cache compiled models for faster subsequent loads
        cache_dir = str(self._model_dir / ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        ov_options["cache_dir"] = cache_dir

        return [
            ("OpenVINOExecutionProvider", ov_options),
            "CPUExecutionProvider",  # fallback
        ]

    # ------------------------------------------------------------------
    # Detection inference
    # ------------------------------------------------------------------

    def _run_detection(
        self, img: np.ndarray
    ) -> tuple[list[np.ndarray], list[float]]:
        """Run text detection model, return list of polygons and scores."""
        # Preprocess: resize to multiple of 32, normalize
        h, w = img.shape[:2]
        input_tensor = self._det_preprocess(img)

        # Run inference
        det_input_name = self._det_session.get_inputs()[0].name
        outputs = self._det_session.run(None, {det_input_name: input_tensor})

        # Postprocess: threshold + find contours → polygons
        pred = outputs[0]  # shape: (1, 1, H, W) or (1, H, W)
        if pred.ndim == 4:
            pred = pred[0, 0]
        elif pred.ndim == 3:
            pred = pred[0]

        return self._det_postprocess(pred, h, w)

    def _det_preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for detection model."""
        h, w = img.shape[:2]
        # Resize to multiple of 32 (limit max side to 960 for speed)
        max_side = 960
        ratio = 1.0
        if max(h, w) > max_side:
            ratio = max_side / max(h, w)
        new_h = max(32, int(round(h * ratio / 32)) * 32)
        new_w = max(32, int(round(w * ratio / 32)) * 32)

        resized = cv2.resize(img, (new_w, new_h))
        # Normalize: (img / 255 - mean) / std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (resized.astype(np.float32) / 255.0 - mean) / std
        # HWC -> CHW, add batch dim
        transposed = normalized.transpose(2, 0, 1)
        return transposed[np.newaxis, :, :, :]

    def _det_postprocess(
        self, pred: np.ndarray, orig_h: int, orig_w: int
    ) -> tuple[list[np.ndarray], list[float]]:
        """Convert detection heatmap to polygons."""
        # Binarize
        thresh = 0.3
        bitmap = (pred > thresh).astype(np.uint8)

        # Resize bitmap back to original image size
        bitmap_resized = cv2.resize(bitmap, (orig_w, orig_h))

        # Find contours
        contours, _ = cv2.findContours(
            bitmap_resized, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes: list[np.ndarray] = []
        scores: list[float] = []
        for contour in contours:
            if contour.shape[0] < 4:
                continue
            # Get minimum area rectangle
            rect = cv2.minAreaRect(contour)
            box_points = cv2.boxPoints(rect)
            box_points = np.array(box_points, dtype=np.float32)

            # Filter by size
            w_box = rect[1][0]
            h_box = rect[1][1]
            if min(w_box, h_box) < 3 or max(w_box, h_box) < 10:
                continue

            # Compute score from the prediction map
            mask = np.zeros(bitmap_resized.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [box_points.astype(np.int32)], 1)
            # Use the resized pred for scoring
            pred_resized = cv2.resize(pred, (orig_w, orig_h))
            score = float(cv2.mean(pred_resized, mask=mask)[0])
            if score < thresh:
                continue

            # Order points: top-left, top-right, bottom-right, bottom-left
            box_points = self._order_points(box_points)
            boxes.append(box_points)
            scores.append(score)

        return boxes, scores

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect

    # ------------------------------------------------------------------
    # Recognition inference
    # ------------------------------------------------------------------

    def _run_recognition(
        self, img: np.ndarray, boxes: list[np.ndarray]
    ) -> list[tuple[np.ndarray, str, float]]:
        """Run text recognition on cropped regions."""
        results: list[tuple[np.ndarray, str, float]] = []
        for box in boxes:
            # Crop and rectify the text region
            cropped = self._crop_text_region(img, box)
            if cropped is None or cropped.size == 0:
                continue

            # Preprocess for recognition
            input_tensor = self._rec_preprocess(cropped)

            # Run inference
            rec_input_name = self._rec_session.get_inputs()[0].name
            outputs = self._rec_session.run(None, {rec_input_name: input_tensor})

            # Decode CTC output
            text, conf = self._ctc_decode(outputs[0])
            results.append((box, text, conf))

        return results

    def _crop_text_region(
        self, img: np.ndarray, box: np.ndarray
    ) -> np.ndarray | None:
        """Perspective-transform a rotated box into a horizontal strip."""
        # Compute width and height of the destination rectangle
        w1 = np.linalg.norm(box[1] - box[0])
        w2 = np.linalg.norm(box[2] - box[3])
        h1 = np.linalg.norm(box[3] - box[0])
        h2 = np.linalg.norm(box[2] - box[1])
        width = max(1, int(round(max(w1, w2))))
        height = max(1, int(round(max(h1, h2))))

        dst = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(box.astype(np.float32), dst)
        cropped = cv2.warpPerspective(img, M, (width, height))

        # If height > width, it's likely vertical text — rotate
        if height > width * 1.5:
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return cropped

    def _rec_preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess cropped text image for recognition model."""
        # Resize to fixed height 48, variable width (max 320)
        target_h = 48
        h, w = img.shape[:2]
        ratio = target_h / h
        target_w = min(320, max(1, int(round(w * ratio))))
        resized = cv2.resize(img, (target_w, target_h))

        # Pad to width 320
        padded = np.zeros((target_h, 320, 3), dtype=np.uint8)
        padded[:, :target_w, :] = resized

        # Normalize
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        normalized = (padded.astype(np.float32) / 255.0 - mean) / std
        transposed = normalized.transpose(2, 0, 1)
        return transposed[np.newaxis, :, :, :]

    def _ctc_decode(self, output: np.ndarray) -> tuple[str, float]:
        """Decode CTC output to text string and confidence."""
        # output shape: (1, seq_len, num_classes)
        if output.ndim == 3:
            output = output[0]  # (seq_len, num_classes)

        # Greedy decode
        indices = np.argmax(output, axis=1)
        probs = np.max(output, axis=1)

        # Remove duplicates and blanks (index 0 = blank)
        chars: list[str] = []
        confs: list[float] = []
        prev_idx = -1
        for i, idx in enumerate(indices):
            if idx == 0:  # blank
                prev_idx = idx
                continue
            if idx == prev_idx:  # duplicate
                continue
            if 0 < idx < len(self._rec_chars):
                chars.append(self._rec_chars[idx])
                confs.append(float(probs[i]))
            prev_idx = idx

        text = "".join(chars)
        conf = float(np.mean(confs)) if confs else 0.0
        return text, conf

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

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

    @staticmethod
    def _poly_to_box(
        poly: np.ndarray,
        scale_x: float,
        scale_y: float,
        frame_w: int,
        frame_h: int,
    ) -> Bounding_Box | None:
        """Convert a (4,2) polygon to an axis-aligned Bounding_Box."""
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
