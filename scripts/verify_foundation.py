"""Smoke test for the foundation layer (Task 2.7).

Run with: python scripts/verify_foundation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `from video_text_translator import ...` when running this script
# directly without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_text_translator import errors, models  # noqa: E402
from video_text_translator.geometry import (  # noqa: E402
    center_distance,
    clip_box_to_frame,
    frame_diagonal,
    iou,
    scale_box,
)
from video_text_translator.text_utils import (  # noqa: E402
    content_similarity,
    has_cjk,
    normalize_text,
)


def main() -> int:
    # 1. Bounding_Box construction + validation
    a = models.Bounding_Box(10, 20, 100, 50)
    b = models.Bounding_Box(60, 20, 100, 50)
    assert a.area == 5000
    assert a.center == (60.0, 45.0)

    # 2. Geometry
    overlap = iou(a, b)
    assert 0.0 < overlap < 1.0, f"expected partial overlap, got {overlap}"
    assert iou(a, a) == 1.0
    assert center_distance(a, b) == 50.0
    assert frame_diagonal(1920, 1080) > 0
    scaled = scale_box(a, 1.5)
    assert scaled.width == 150 and scaled.height == 75
    clipped = clip_box_to_frame(models.Bounding_Box(0, 0, 100, 100), 50, 50)
    assert clipped is not None and clipped.width == 50 and clipped.height == 50

    # 3. Text utils
    assert has_cjk("你好") is True
    assert has_cjk("hello") is False
    assert has_cjk("hello 你好 world") is True
    assert content_similarity("hello", "hello") == 1.0
    assert content_similarity("hello", "world") < 0.5
    assert content_similarity("", "") == 1.0
    assert content_similarity("a", "") == 0.0
    assert normalize_text("  hi  ") == "hi"

    # 4. Models with validation
    region = models.Text_Region(
        box=a, text="你好", confidence=0.95, frame_index=0, timestamp=0.0
    )
    assert region.text == "你好"

    style = models.Style_Preset(font_path="fonts/NotoSans-Regular.ttf")
    assert style.font_size_max == 64

    # 5. Error hierarchy
    assert issubclass(errors.InvalidConfigError, errors.PipelineError)

    # 6. Validation rejects bad input
    rejected = 0
    for ctor in (
        lambda: models.Bounding_Box(0, 0, 0, 10),  # zero width
        lambda: models.Bounding_Box(-1, 0, 10, 10),  # negative origin
        lambda: models.Text_Region(a, "x", 1.5, 0, 0.0),  # confidence > 1
        lambda: models.Style_Preset(font_path=""),  # empty font_path
    ):
        try:
            ctor()
        except (ValueError, errors.PipelineError):
            rejected += 1
    assert rejected == 4, f"expected 4 validation rejections, got {rejected}"

    print("Foundation layer OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
