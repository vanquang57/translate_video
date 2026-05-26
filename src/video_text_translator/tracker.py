"""Tracking module: group Text_Region across frames into Text_Segment.

The tracker matches an incoming region to active segments using:
  (a) IoU >= iou_threshold, OR
  (b) content_similarity >= sim_threshold AND
      center distance <= center_distance_ratio * frame_diagonal.

Segments that go unmatched for ``n_inactive_effective`` frames are
closed. ``n_inactive_effective = max(3, ceil(n_inactive * ocr_stride))``
to compensate for frame skipping.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from .geometry import center_distance, frame_diagonal, iou
from .models import (
    Bounding_Box,
    Frame_Region_Entry,
    Text_Region,
    Text_Segment,
)
from .text_utils import content_similarity, normalize_text

logger = logging.getLogger(__name__)


class ITracker(Protocol):
    def update(
        self,
        frame_index: int,
        timestamp: float,
        regions: Sequence[Text_Region],
    ) -> None:
        ...

    def finalize(self) -> tuple[Text_Segment, ...]:
        ...


# ---------------------------------------------------------------------------
# Internal mutable state for an in-flight segment.
# ---------------------------------------------------------------------------


@dataclass
class _ActiveSegment:
    segment_id: str
    start_frame: int
    start_time: float
    last_match_frame: int
    last_match_time: float
    canonical_text: str
    entries: list[Frame_Region_Entry] = field(default_factory=list)
    closed: bool = False

    def append(self, entry: Frame_Region_Entry, *, observed: bool) -> None:
        self.entries.append(entry)
        if observed:
            self.last_match_frame = entry.frame_index
            self.last_match_time = entry.timestamp


def _lerp_box(a: Bounding_Box, b: Bounding_Box, t: float) -> Bounding_Box:
    """Linearly interpolate two boxes at ratio ``t`` in [0, 1]."""
    x = int(round(a.x + (b.x - a.x) * t))
    y = int(round(a.y + (b.y - a.y) * t))
    w = max(1, int(round(a.width + (b.width - a.width) * t)))
    h = max(1, int(round(a.height + (b.height - a.height) * t)))
    return Bounding_Box(x, y, w, h)


# ---------------------------------------------------------------------------
# Public Tracker implementation.
# ---------------------------------------------------------------------------


class IoUContentTracker:
    """Stateful Tracker keyed on IoU + content + center distance."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        iou_threshold: float = 0.5,
        content_similarity_threshold: float = 0.7,
        center_distance_ratio: float = 0.10,
        n_inactive: int = 3,
        ocr_stride: int = 1,
        max_active_segments: int = 100,
        smooth_lock_threshold: int = 3,
        smooth_ema_alpha: float = 0.3,
    ) -> None:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(
                f"frame size must be positive (got {frame_width}x{frame_height})"
            )
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._diag = frame_diagonal(frame_width, frame_height)
        self._iou_threshold = iou_threshold
        self._sim_threshold = content_similarity_threshold
        self._dist_threshold = center_distance_ratio * self._diag
        # Req 10.9: scale n_inactive by stride to avoid premature closure.
        self._n_inactive_effective = max(3, math.ceil(n_inactive * ocr_stride))
        self._max_active = max_active_segments
        self._smooth_lock_threshold = smooth_lock_threshold
        self._smooth_ema_alpha = smooth_ema_alpha

        self._active: list[_ActiveSegment] = []
        self._closed: list[_ActiveSegment] = []
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_inactive_effective(self) -> int:
        return self._n_inactive_effective

    def update(
        self,
        frame_index: int,
        timestamp: float,
        regions: Sequence[Text_Region],
    ) -> None:
        """Match ``regions`` to active segments and update tracker state."""
        # Pre-filter regions that have no usable text content (Req 3.8).
        usable: list[Text_Region] = []
        for r in regions:
            if not normalize_text(r.text):
                logger.debug(
                    "tracker: dropping empty region at frame %d", frame_index
                )
                continue
            usable.append(r)

        # Score every (region, active_segment) pair that satisfies the
        # matching rule. Build a list (region_idx, segment, score).
        candidates: list[tuple[int, _ActiveSegment, float]] = []
        for idx, r in enumerate(usable):
            for seg in self._active:
                if seg.closed:
                    continue
                if frame_index - seg.last_match_frame > self._n_inactive_effective:
                    continue
                last = seg.entries[-1]
                iou_val = iou(last.box, r.box)
                sim = content_similarity(last.text, r.text)
                cdist = center_distance(last.box, r.box)
                cond_a = iou_val >= self._iou_threshold
                cond_b = (
                    sim >= self._sim_threshold and cdist <= self._dist_threshold
                )
                if cond_a or cond_b:
                    score = iou_val + sim - (cdist / self._diag if self._diag > 0 else 0.0)
                    candidates.append((idx, seg, score))

        # Greedy resolve: highest score wins, each region and each segment
        # may participate in at most one match per frame.
        candidates.sort(key=lambda c: c[2], reverse=True)
        used_regions: set[int] = set()
        used_segments: set[str] = set()
        for idx, seg, _score in candidates:
            if idx in used_regions or seg.segment_id in used_segments:
                continue
            r = usable[idx]
            seg.append(
                Frame_Region_Entry(
                    frame_index=r.frame_index,
                    timestamp=r.timestamp,
                    box=r.box,
                    text=r.text,
                    interpolated=False,
                ),
                observed=True,
            )
            seg.canonical_text = self._pick_canonical(seg.canonical_text, r.text)
            used_regions.add(idx)
            used_segments.add(seg.segment_id)

        # Unmatched regions create new segments.
        for idx, r in enumerate(usable):
            if idx in used_regions:
                continue
            self._spawn_segment(r)

        # Close segments whose last match is too old.
        for seg in list(self._active):
            if frame_index - seg.last_match_frame > self._n_inactive_effective:
                self._close(seg)

        # Enforce max_active by closing the oldest segments first.
        if len(self._active) > self._max_active:
            self._active.sort(key=lambda s: s.start_frame)
            while len(self._active) > self._max_active:
                victim = self._active[0]
                self._close(victim)

    def finalize(self) -> tuple[Text_Segment, ...]:
        """Close all remaining segments, fill missing frames, and return all."""
        for seg in list(self._active):
            self._close(seg)

        for seg in self._closed:
            self._fill_missing(seg)
            self._smooth_boxes(
                seg,
                lock_threshold=self._smooth_lock_threshold,
                ema_alpha=self._smooth_ema_alpha,
            )

        result: list[Text_Segment] = []
        for seg in self._closed:
            entries_sorted = sorted(seg.entries, key=lambda e: e.frame_index)
            result.append(
                Text_Segment(
                    segment_id=seg.segment_id,
                    start_frame=seg.start_frame,
                    end_frame=seg.last_match_frame,
                    start_time=seg.start_time,
                    end_time=seg.last_match_time,
                    canonical_text=seg.canonical_text,
                    entries=tuple(entries_sorted),
                )
            )
        return tuple(result)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spawn_segment(self, region: Text_Region) -> None:
        seg = _ActiveSegment(
            segment_id=self._next_segment_id(),
            start_frame=region.frame_index,
            start_time=region.timestamp,
            last_match_frame=region.frame_index,
            last_match_time=region.timestamp,
            canonical_text=region.text,
        )
        seg.append(
            Frame_Region_Entry(
                frame_index=region.frame_index,
                timestamp=region.timestamp,
                box=region.box,
                text=region.text,
                interpolated=False,
            ),
            observed=True,
        )
        self._active.append(seg)

    def _close(self, seg: _ActiveSegment) -> None:
        if seg.closed:
            return
        seg.closed = True
        if seg in self._active:
            self._active.remove(seg)
        self._closed.append(seg)

    def _fill_missing(self, seg: _ActiveSegment) -> None:
        """Insert interpolated entries for every frame in [start, end].

        For frames between two observed entries we linearly interpolate the
        bounding box (and timestamp) so the text moves smoothly instead of
        snapping every ``ocr_stride`` frames. For frames before the first
        or after the last observation we fall back to nearest-neighbour
        (hold-previous semantics) since interpolation is undefined there.
        """
        if not seg.entries:
            return
        observed = sorted(seg.entries, key=lambda e: e.frame_index)
        seen: set[int] = {e.frame_index for e in observed}
        first_frame = observed[0].frame_index
        last_frame = observed[-1].frame_index
        if last_frame - first_frame + 1 == len(observed):
            return  # already dense

        # Walk between consecutive observed pairs and fill the gap with
        # linearly-interpolated entries.
        for prev, nxt in zip(observed, observed[1:]):
            if nxt.frame_index - prev.frame_index <= 1:
                continue
            for f in range(prev.frame_index + 1, nxt.frame_index):
                if f in seen:
                    continue
                ratio = (f - prev.frame_index) / (nxt.frame_index - prev.frame_index)
                box = _lerp_box(prev.box, nxt.box, ratio)
                ts = prev.timestamp + ratio * (nxt.timestamp - prev.timestamp)
                # Choose the text from whichever observation has the longer
                # canonical content (animations often reveal more glyphs over
                # time, so the longer string is usually the more complete
                # one).
                text = prev.text if len(prev.text) >= len(nxt.text) else nxt.text
                seg.entries.append(
                    Frame_Region_Entry(
                        frame_index=f,
                        timestamp=ts,
                        box=box,
                        text=text,
                        interpolated=True,
                    )
                )

    @staticmethod
    def _nearest_observed(
        observed: list[Frame_Region_Entry],
        observed_frames: list[int],
        target: int,
    ) -> Frame_Region_Entry:
        # Linear scan: small lists in practice (<= ~10 OCR samples).
        best = observed[0]
        best_dist = abs(observed_frames[0] - target)
        # Prefer earlier frame on ties (i.e. <=).
        for entry, frame in zip(observed[1:], observed_frames[1:]):
            d = abs(frame - target)
            if d < best_dist or (d == best_dist and frame < target):
                best = entry
                best_dist = d
        return best

    @staticmethod
    def _interpolate_timestamp(
        observed: list[Frame_Region_Entry], target: int
    ) -> float:
        # If we can find two surrounding observations, linearly interpolate
        # the timestamp; otherwise reuse the nearest one's timestamp.
        before: Frame_Region_Entry | None = None
        after: Frame_Region_Entry | None = None
        for e in observed:
            if e.frame_index <= target and (before is None or e.frame_index > before.frame_index):
                before = e
            if e.frame_index >= target and (after is None or e.frame_index < after.frame_index):
                after = e
        if before is not None and after is not None and after.frame_index != before.frame_index:
            ratio = (target - before.frame_index) / (
                after.frame_index - before.frame_index
            )
            return before.timestamp + ratio * (after.timestamp - before.timestamp)
        if before is not None:
            return before.timestamp
        if after is not None:
            return after.timestamp
        return 0.0

    @staticmethod
    def _smooth_boxes(
        seg: _ActiveSegment,
        lock_threshold: int = 3,
        ema_alpha: float = 0.3,
    ) -> None:
        """Stabilize box positions using position locking + EMA smoothing.

        Two-phase approach to eliminate subtitle jitter:

        1. **Position locking**: If the incoming box differs from the
           current smoothed position by less than ``lock_threshold`` pixels
           on every axis (x, y, w, h), the position is held (locked). This
           eliminates micro-jitter for static text where OCR noise causes
           1-2 px fluctuations between frames.

        2. **EMA (Exponential Moving Average)**: When the delta exceeds
           the lock threshold (genuine motion), the smoothed position is
           updated via EMA: ``smoothed = alpha * new + (1 - alpha) * prev``.
           This produces a gradual transition instead of a hard snap,
           making real movement look fluid rather than jerky.

        Parameters
        ----------
        seg : _ActiveSegment
            The segment whose entries will be smoothed in-place.
        lock_threshold : int
            Maximum per-axis pixel difference that is treated as noise and
            locked out. Default 3 px covers typical OCR jitter.
        ema_alpha : float
            EMA responsiveness in (0, 1]. Lower values = smoother but
            laggier; higher = more responsive. Default 0.3 balances
            smoothness with tracking speed.
        """
        if len(seg.entries) < 2:
            return
        ordered = sorted(seg.entries, key=lambda e: e.frame_index)

        # Initialize EMA state from the first entry.
        ema_x = float(ordered[0].box.x)
        ema_y = float(ordered[0].box.y)
        ema_w = float(ordered[0].box.width)
        ema_h = float(ordered[0].box.height)

        smoothed: list[Frame_Region_Entry] = [ordered[0]]

        for entry in ordered[1:]:
            raw_x = float(entry.box.x)
            raw_y = float(entry.box.y)
            raw_w = float(entry.box.width)
            raw_h = float(entry.box.height)

            # Check if the movement is within the lock threshold (noise).
            dx = abs(raw_x - ema_x)
            dy = abs(raw_y - ema_y)
            dw = abs(raw_w - ema_w)
            dh = abs(raw_h - ema_h)

            if dx <= lock_threshold and dy <= lock_threshold and dw <= lock_threshold and dh <= lock_threshold:
                # Position locked — keep the previous smoothed values.
                pass
            else:
                # Genuine motion — update EMA.
                ema_x = ema_alpha * raw_x + (1.0 - ema_alpha) * ema_x
                ema_y = ema_alpha * raw_y + (1.0 - ema_alpha) * ema_y
                ema_w = ema_alpha * raw_w + (1.0 - ema_alpha) * ema_w
                ema_h = ema_alpha * raw_h + (1.0 - ema_alpha) * ema_h

            new_box = Bounding_Box(
                max(0, int(round(ema_x))),
                max(0, int(round(ema_y))),
                max(1, int(round(ema_w))),
                max(1, int(round(ema_h))),
            )
            smoothed.append(
                Frame_Region_Entry(
                    frame_index=entry.frame_index,
                    timestamp=entry.timestamp,
                    box=new_box,
                    text=entry.text,
                    interpolated=entry.interpolated,
                )
            )
        seg.entries = smoothed

    @staticmethod
    def _pick_canonical(current: str, candidate: str) -> str:
        """Prefer the longer of the two strings as the canonical text.

        Rationale: OCR results may grow as the text grows on screen
        (animation effects revealing more glyphs), so a longer string is
        usually the more complete sample.
        """
        c_norm = normalize_text(current)
        cand_norm = normalize_text(candidate)
        if len(cand_norm) > len(c_norm):
            return candidate
        return current

    def _next_segment_id(self) -> str:
        sid = f"seg-{self._next_id:06d}"
        self._next_id += 1
        return sid
