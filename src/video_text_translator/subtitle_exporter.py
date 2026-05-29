"""SRT subtitle file generation from translated Text_Segments.

This module is a pure-logic component with no dependencies on GUI code or
video frame data.  It operates solely on in-memory segment data and the
translations dictionary.

Public API
----------
- format_timestamp(seconds) -> str
- compute_segment_center(segment) -> tuple[float, float]
- derive_srt_path(video_output_path) -> str
- filter_segments(segments, region) -> list[Text_Segment]
- generate_srt(segments, translations, region) -> str
- export_srt(segments, translations, video_output_path, region) -> str
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from .models import Subtitle_Region, Text_Segment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_timestamp(seconds: float) -> str:
    """Convert *seconds* to SRT timestamp format ``HH:MM:SS,mmm``.

    Negative values are clamped to 0.0.
    """
    if seconds < 0.0:
        seconds = 0.0

    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def compute_segment_center(segment: Text_Segment) -> tuple[float, float]:
    """Return the center point of the first entry's bounding box.

    Raises
    ------
    ValueError
        If the segment has no entries.
    """
    if not segment.entries:
        raise ValueError(
            f"Segment '{segment.segment_id}' has no entries; cannot compute center."
        )
    return segment.entries[0].box.center


def derive_srt_path(video_output_path: str) -> str:
    """Replace the video file extension with ``.srt``.

    Uses :mod:`pathlib` for robust cross-platform path handling.
    """
    return str(Path(video_output_path).with_suffix(".srt"))


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_segments(
    segments: Sequence[Text_Segment],
    region: Subtitle_Region | None,
) -> list[Text_Segment]:
    """Filter segments whose center falls inside *region*.

    If *region* is ``None`` (full-frame mode), all segments are returned.
    Segments with no entries are skipped with a warning.
    """
    if region is None:
        # Full-frame mode: return all segments (skip those with no entries)
        result: list[Text_Segment] = []
        for seg in segments:
            if not seg.entries:
                logger.warning(
                    "Segment '%s' has no entries; skipping.", seg.segment_id
                )
                continue
            result.append(seg)
        return result

    filtered: list[Text_Segment] = []
    for seg in segments:
        if not seg.entries:
            logger.warning(
                "Segment '%s' has no entries; skipping.", seg.segment_id
            )
            continue
        cx, cy = compute_segment_center(seg)
        if region.contains_point(cx, cy):
            filtered.append(seg)
    return filtered


# ---------------------------------------------------------------------------
# SRT generation
# ---------------------------------------------------------------------------


def generate_srt(
    segments: Sequence[Text_Segment],
    translations: dict[str, str],
    region: Subtitle_Region | None = None,
) -> str:
    """Generate complete SRT file content from translated segments.

    Steps:
    1. Filter segments by region (or include all if region is None).
    2. Sort filtered segments chronologically by start_time.
    3. Format each as a numbered SRT entry.
    4. Join with blank line separators.
    """
    filtered = filter_segments(segments, region)
    sorted_segments = sorted(filtered, key=lambda s: s.start_time)

    entries: list[str] = []
    for idx, seg in enumerate(sorted_segments, start=1):
        translated_text = translations.get(seg.segment_id, seg.canonical_text)
        start_ts = format_timestamp(seg.start_time)
        end_ts = format_timestamp(seg.end_time)
        entry = f"{idx}\n{start_ts} --> {end_ts}\n{translated_text}"
        entries.append(entry)

    return "\n\n".join(entries) + ("\n" if entries else "")


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def export_srt(
    segments: Sequence[Text_Segment],
    translations: dict[str, str],
    video_output_path: str,
    region: Subtitle_Region | None = None,
) -> str:
    """Generate SRT content and write it to disk.

    The output path is derived from *video_output_path* by replacing its
    extension with ``.srt``.

    Returns
    -------
    str
        The absolute path of the written SRT file.
    """
    srt_content = generate_srt(segments, translations, region)
    srt_path = derive_srt_path(video_output_path)
    Path(srt_path).write_text(srt_content, encoding="utf-8")
    return srt_path
