"""Remove the original Chinese text from each frame via OpenCV inpainting.

The mask is a binary image (uint8, values 0 or 255) covering all bounding
boxes of detected text, expanded by ``padding`` to absorb stroke and shadow,
then clipped to the frame. ``cv2.inpaint`` is then applied with TELEA or NS.
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

import cv2
import numpy as np

from .errors import InvalidConfigError
from .models import Bounding_Box

logger = logging.getLogger(__name__)


_ALGO_FLAGS: dict[str, int] = {
    "telea": cv2.INPAINT_TELEA,
    "ns": cv2.INPAINT_NS,
}


class IInpainter(Protocol):
    def inpaint_frame(
        self, frame: np.ndarray, boxes: Sequence[Bounding_Box]
    ) -> np.ndarray:
        ...


def make_mask(
    frame_shape: tuple[int, int],
    boxes: Sequence[Bounding_Box],
    padding: int = 4,
) -> np.ndarray:
    """Build a binary uint8 mask covering all (padded, clipped) boxes.

    ``frame_shape`` is ``(height, width)``. The mask uses 255 for pixels
    that should be inpainted, 0 elsewhere.
    """
    if padding < 0:
        raise ValueError(f"padding must be >= 0 (got {padding})")
    height, width = frame_shape
    if height <= 0 or width <= 0:
        raise ValueError(f"frame_shape must be positive (got {frame_shape})")
    mask = np.zeros((height, width), dtype=np.uint8)
    for b in boxes:
        x1 = max(0, b.x - padding)
        y1 = max(0, b.y - padding)
        x2 = min(width, b.x + b.width + padding)
        y2 = min(height, b.y + b.height + padding)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


class OpenCVInpainter:
    """Inpainter implementation backed by ``cv2.inpaint``."""

    def __init__(
        self,
        algorithm: str = "telea",
        radius: int = 3,
        padding: int = 4,
    ) -> None:
        algo_norm = (algorithm or "").strip().lower()
        if algo_norm not in _ALGO_FLAGS:
            raise InvalidConfigError(
                f'inpainter.algorithm must be "telea" or "ns" '
                f'(got "{algorithm}")'
            )
        if not (1 <= radius <= 20):
            raise InvalidConfigError(
                f"inpainter.radius must be in [1, 20] (got {radius})"
            )
        if not (0 <= padding <= 20):
            raise InvalidConfigError(
                f"inpainter.padding must be in [0, 20] (got {padding})"
            )
        self._algo_flag = _ALGO_FLAGS[algo_norm]
        self._algo_name = algo_norm
        self._radius = radius
        self._padding = padding

    @property
    def algorithm(self) -> str:
        return self._algo_name

    def inpaint_frame(
        self, frame: np.ndarray, boxes: Sequence[Bounding_Box]
    ) -> np.ndarray:
        """Return a new frame with text regions inpainted.

        If ``boxes`` is empty, the original frame is returned unchanged.
        On unexpected failure the original frame is returned and the
        error is logged with the relevant context (Req 4 / design Error
        Handling Strategy).
        """
        if not boxes:
            return frame
        try:
            h, w = frame.shape[:2]
            mask = make_mask((h, w), boxes, padding=self._padding)
            return cv2.inpaint(frame, mask, self._radius, self._algo_flag)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "inpaint_frame failed (algo=%s, radius=%d, padding=%d, boxes=%d): %s",
                self._algo_name,
                self._radius,
                self._padding,
                len(boxes),
                exc,
            )
            return frame
