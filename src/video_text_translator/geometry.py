"""Pure geometry helpers used by Tracker, Inpainter, and Renderer.

All functions are deterministic and have no side effects, which makes
them straightforward to verify with property-based testing.
"""

from __future__ import annotations

import math

from .models import Bounding_Box


def iou(a: Bounding_Box, b: Bounding_Box) -> float:
    """Intersection-over-Union of two bounding boxes.

    Returns a value in [0.0, 1.0]; 0.0 if the boxes do not overlap, 1.0 if
    they are identical.
    """
    inter_x1 = max(a.x, b.x)
    inter_y1 = max(a.y, b.y)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def center_distance(a: Bounding_Box, b: Bounding_Box) -> float:
    """Euclidean distance between the centers of two boxes (in pixels)."""
    ax, ay = a.center
    bx, by = b.center
    return math.hypot(ax - bx, ay - by)


def frame_diagonal(width: int, height: int) -> float:
    """Length of the frame diagonal in pixels."""
    if width <= 0 or height <= 0:
        raise ValueError(f"frame size must be positive (got {width}x{height})")
    return math.hypot(float(width), float(height))


def scale_box(box: Bounding_Box, factor: float) -> Bounding_Box:
    """Scale a box's coordinates and dimensions by a scalar factor.

    Used to convert OCR-coordinate boxes (after downscaling) back into
    original-frame coordinates. The result is rounded to integers.
    """
    if factor <= 0:
        raise ValueError(f"scale factor must be > 0 (got {factor})")
    new_x = int(round(box.x * factor))
    new_y = int(round(box.y * factor))
    new_w = max(1, int(round(box.width * factor)))
    new_h = max(1, int(round(box.height * factor)))
    return Bounding_Box(new_x, new_y, new_w, new_h)


def clip_box_to_frame(box: Bounding_Box, width: int, height: int) -> Bounding_Box | None:
    """Clip a box so that it lies entirely within the frame.

    Returns ``None`` when the clipped box would have zero area.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"frame size must be positive (got {width}x{height})")
    x1 = max(0, box.x)
    y1 = max(0, box.y)
    x2 = min(width, box.x2)
    y2 = min(height, box.y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return Bounding_Box(x1, y1, x2 - x1, y2 - y1)
