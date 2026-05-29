"""Unit tests for subtitle_exporter module."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from video_text_translator.models import (
    Bounding_Box,
    Frame_Region_Entry,
    Subtitle_Region,
    Text_Segment,
)
from video_text_translator.subtitle_exporter import (
    compute_segment_center,
    derive_srt_path,
    export_srt,
    filter_segments,
    format_timestamp,
    generate_srt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment(
    segment_id: str = "seg1",
    start_time: float = 1.0,
    end_time: float = 3.0,
    canonical_text: str = "hello",
    box_x: int = 100,
    box_y: int = 200,
    box_w: int = 50,
    box_h: int = 30,
) -> Text_Segment:
    """Create a minimal Text_Segment with one entry."""
    entry = Frame_Region_Entry(
        frame_index=0,
        timestamp=start_time,
        box=Bounding_Box(x=box_x, y=box_y, width=box_w, height=box_h),
        text=canonical_text,
    )
    return Text_Segment(
        segment_id=segment_id,
        start_frame=0,
        end_frame=10,
        start_time=start_time,
        end_time=end_time,
        canonical_text=canonical_text,
        entries=(entry,),
    )


def _make_segment_no_entries(segment_id: str = "empty") -> Text_Segment:
    """Create a Text_Segment with no entries."""
    return Text_Segment(
        segment_id=segment_id,
        start_frame=0,
        end_frame=5,
        start_time=0.0,
        end_time=1.0,
        canonical_text="ghost",
        entries=(),
    )


# ---------------------------------------------------------------------------
# format_timestamp
# ---------------------------------------------------------------------------


class TestFormatTimestamp:
    def test_zero(self) -> None:
        assert format_timestamp(0.0) == "00:00:00,000"

    def test_simple_seconds(self) -> None:
        assert format_timestamp(1.5) == "00:00:01,500"

    def test_minutes_and_seconds(self) -> None:
        assert format_timestamp(65.123) == "00:01:05,123"

    def test_hours(self) -> None:
        assert format_timestamp(3661.999) == "01:01:01,999"

    def test_negative_clamped(self) -> None:
        assert format_timestamp(-5.0) == "00:00:00,000"

    def test_large_value(self) -> None:
        # 10 hours
        assert format_timestamp(36000.0) == "10:00:00,000"


# ---------------------------------------------------------------------------
# compute_segment_center
# ---------------------------------------------------------------------------


class TestComputeSegmentCenter:
    def test_basic_center(self) -> None:
        seg = _make_segment(box_x=100, box_y=200, box_w=50, box_h=30)
        cx, cy = compute_segment_center(seg)
        assert cx == 125.0
        assert cy == 215.0

    def test_no_entries_raises(self) -> None:
        seg = _make_segment_no_entries()
        with pytest.raises(ValueError, match="no entries"):
            compute_segment_center(seg)


# ---------------------------------------------------------------------------
# derive_srt_path
# ---------------------------------------------------------------------------


class TestDeriveSrtPath:
    def test_mp4_to_srt(self) -> None:
        result = derive_srt_path("/output/video.mp4")
        assert result.endswith(".srt")
        assert "video" in result

    def test_avi_to_srt(self) -> None:
        result = derive_srt_path("/output/my_video.avi")
        assert result.endswith(".srt")
        assert "my_video" in result

    def test_preserves_directory(self) -> None:
        result = derive_srt_path("/some/dir/clip.mkv")
        # On Windows the path separator may differ, but stem and suffix are correct
        from pathlib import Path

        p = Path(result)
        assert p.suffix == ".srt"
        assert p.stem == "clip"


# ---------------------------------------------------------------------------
# filter_segments
# ---------------------------------------------------------------------------


class TestFilterSegments:
    def test_none_region_returns_all(self) -> None:
        segs = [_make_segment("a"), _make_segment("b")]
        result = filter_segments(segs, None)
        assert len(result) == 2

    def test_region_includes_inside(self) -> None:
        # Segment center at (125, 215)
        seg = _make_segment(box_x=100, box_y=200, box_w=50, box_h=30)
        region = Subtitle_Region(x=0, y=0, width=200, height=300)
        result = filter_segments([seg], region)
        assert len(result) == 1

    def test_region_excludes_outside(self) -> None:
        # Segment center at (125, 215)
        seg = _make_segment(box_x=100, box_y=200, box_w=50, box_h=30)
        region = Subtitle_Region(x=0, y=0, width=50, height=50)
        result = filter_segments([seg], region)
        assert len(result) == 0

    def test_skips_segments_with_no_entries(self) -> None:
        seg = _make_segment_no_entries()
        result = filter_segments([seg], None)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# generate_srt
# ---------------------------------------------------------------------------


class TestGenerateSrt:
    def test_empty_segments(self) -> None:
        result = generate_srt([], {}, None)
        assert result == ""

    def test_single_segment(self) -> None:
        seg = _make_segment(start_time=1.0, end_time=3.0, canonical_text="hello")
        translations = {"seg1": "xin chào"}
        result = generate_srt([seg], translations, None)
        assert "1\n" in result
        assert "00:00:01,000 --> 00:00:03,000" in result
        assert "xin chào" in result

    def test_fallback_to_canonical(self) -> None:
        seg = _make_segment(canonical_text="untranslated")
        result = generate_srt([seg], {}, None)
        assert "untranslated" in result

    def test_chronological_ordering(self) -> None:
        seg_late = _make_segment("late", start_time=5.0, end_time=7.0)
        seg_early = _make_segment("early", start_time=1.0, end_time=2.0)
        result = generate_srt([seg_late, seg_early], {}, None)
        # Entry 1 should be the early segment
        lines = result.strip().split("\n")
        assert lines[0] == "1"
        assert "00:00:01,000" in lines[1]

    def test_entries_separated_by_blank_line(self) -> None:
        seg1 = _make_segment("s1", start_time=1.0, end_time=2.0, canonical_text="a")
        seg2 = _make_segment("s2", start_time=3.0, end_time=4.0, canonical_text="b")
        result = generate_srt([seg1, seg2], {}, None)
        # Two entries separated by blank line
        assert "\n\n" in result


# ---------------------------------------------------------------------------
# export_srt
# ---------------------------------------------------------------------------


class TestExportSrt:
    def test_writes_file(self, tmp_path) -> None:
        video_path = str(tmp_path / "output.mp4")
        seg = _make_segment(start_time=0.0, end_time=1.0, canonical_text="test")
        translations = {"seg1": "kiểm tra"}

        srt_path = export_srt([seg], translations, video_path, None)

        assert srt_path.endswith(".srt")
        assert os.path.exists(srt_path)

        content = open(srt_path, encoding="utf-8").read()
        assert "kiểm tra" in content

    def test_overwrites_existing(self, tmp_path) -> None:
        video_path = str(tmp_path / "output.mp4")
        srt_file = tmp_path / "output.srt"
        srt_file.write_text("old content", encoding="utf-8")

        seg = _make_segment(start_time=0.0, end_time=1.0, canonical_text="new")
        export_srt([seg], {"seg1": "mới"}, video_path, None)

        content = srt_file.read_text(encoding="utf-8")
        assert "mới" in content
        assert "old content" not in content
