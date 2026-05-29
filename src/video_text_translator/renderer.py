"""Render Vietnamese text on top of inpainted frames using Pillow.

The renderer auto-fits the font size so the text (including stroke and
shadow offset) sits inside the original Bounding_Box, and centers it
within a 2-pixel tolerance.

When the translated text would not fit even at the configured minimum
font size, the renderer follows a cascade of fallbacks (controlled by
``Overflow_Config``):

  1. **Expand bbox** - allow the box to grow by up to ``expand_bbox_max``
     while staying inside frame bounds (centered on the original).
  2. **Word wrap** - break the translation onto multiple lines (up to
     ``word_wrap_max_lines``).
  3. **Condensed font** - retry with a narrower font (e.g. NotoSans
     Condensed) which packs more glyphs in the same width.

If all fallbacks still cannot fit the text the frame is left untouched
and a warning is logged.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .errors import InvalidConfigError
from .models import Bounding_Box, Overflow_Config, Style_Preset

logger = logging.getLogger(__name__)


class IRenderer(Protocol):
    def render(
        self,
        frame: np.ndarray,
        text_vi: str,
        box: Bounding_Box,
        style: Style_Preset,
        *,
        segment_id: str | None = None,
        frame_index: int | None = None,
        fixed_font_size: int | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class _LayoutPlan:
    """The result of fitting a translated string into (a possibly expanded) box."""

    box: Bounding_Box
    font_path: str
    font_size: int
    lines: tuple[str, ...]
    line_height: int


class PillowRenderer:
    """Pillow-based renderer for Vietnamese text overlays."""

    def __init__(self, default_font_path: str) -> None:
        if not default_font_path or not os.path.isfile(default_font_path):
            raise InvalidConfigError(
                f"renderer: font file not found: {default_font_path!r}"
            )
        self._default_font_path = default_font_path
        # Cache of (font_path, size) -> ImageFont so we don't reload the
        # font binary every frame.
        self._font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
        # Track segments that already logged "does not fit" to avoid spam.
        self._nofit_logged: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        text_vi: str,
        box: Bounding_Box,
        style: Style_Preset,
        *,
        segment_id: str | None = None,
        frame_index: int | None = None,
        fixed_font_size: int | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Render ``text_vi`` into ``box`` over ``frame``."""
        if not text_vi.strip():
            return frame

        h, w = frame.shape[:2]
        plan = self._plan_layout(
            text_vi,
            box,
            style,
            frame_w=w,
            frame_h=h,
            fixed_font_size=fixed_font_size,
        )
        if plan is None:
            seg_key = segment_id or ""
            if seg_key not in self._nofit_logged:
                self._nofit_logged.add(seg_key)
                logger.warning(
                    "renderer: text does not fit even after fallbacks "
                    "(segment=%s, first_frame=%s, box=%dx%d)",
                    segment_id,
                    frame_index,
                    box.width,
                    box.height,
                )
            return frame
        return self._draw(frame, plan, style)

    # ------------------------------------------------------------------
    # Layout planning
    # ------------------------------------------------------------------

    def _plan_layout(
        self,
        text: str,
        box: Bounding_Box,
        style: Style_Preset,
        *,
        frame_w: int,
        frame_h: int,
        fixed_font_size: int | None,
    ) -> _LayoutPlan | None:
        """Try the fallback cascade and return the first plan that fits.

        When ``fixed_font_size`` is provided, the font size is NEVER changed.
        Instead, the box is expanded as needed to fit the text at that size.
        This eliminates font-size jitter for moving/zooming text.

        Cascade (with fixed_font_size):
          0. Try original box at fixed size
          1. Try expanded box at fixed size
          2. Try wrapped (multi-line) at fixed size with original/expanded box
          3. Force-expand box to exactly fit text at fixed size (clipped to frame)

        Cascade (without fixed_font_size — auto-fit):
          0. Original box, single line, regular font
          1. Expanded box, single line, regular font
          2. Original box, wrapped, regular font
          3. Expanded box, wrapped, regular font
          4. Steps 0-3 again with the condensed font.
        """
        regular = style.font_path or self._default_font_path
        ovf = style.overflow or Overflow_Config()
        condensed = (
            ovf.condensed_font_path
            if ovf.condensed_enabled and ovf.condensed_font_path
            else None
        )
        font_paths: list[str] = [regular]
        if condensed and condensed != regular and os.path.isfile(condensed):
            font_paths.append(condensed)
        elif condensed and ovf.condensed_enabled and not os.path.isfile(condensed):
            logger.debug(
                "renderer: condensed font not found at %s, skipping fallback",
                condensed,
            )

        expanded_box = (
            self._expand_box(
                box, ovf.effective_max_w, frame_w, frame_h,
                factor_h=ovf.effective_max_h,
            )
            if ovf.expand_bbox_enabled
            else None
        )
        max_lines = ovf.word_wrap_max_lines if ovf.word_wrap_enabled else 1

        # --- Standard cascade ---
        for path in font_paths:
            for candidate_box in (box, expanded_box):
                if candidate_box is None:
                    continue
                plan = self._fit_single_line(
                    text, candidate_box, style, path, fixed_font_size
                )
                if plan is not None:
                    return plan
            if max_lines >= 2:
                for candidate_box in (box, expanded_box):
                    if candidate_box is None:
                        continue
                    plan = self._fit_wrapped(
                        text, candidate_box, style, path, max_lines, fixed_font_size
                    )
                    if plan is not None:
                        return plan

        # --- Force-fit: when fixed_font_size is set, create a box that
        # exactly fits the text rather than changing font size. ---
        if fixed_font_size is not None:
            path = regular
            font = self._get_font(path, fixed_font_size)
            # Try single line first.
            text_w, text_h = self._block_size((text,), font, style)
            force_box = self._force_expand_box(box, text_w, text_h, frame_w, frame_h)
            if force_box is not None and self._fits((text_w, text_h), force_box):
                line_h = self._line_height(font, style)
                return _LayoutPlan(
                    box=force_box,
                    font_path=path,
                    font_size=fixed_font_size,
                    lines=(text,),
                    line_height=line_h,
                )
            # Try wrapped with force-expand.
            if max_lines >= 2:
                for n_lines in range(2, max_lines + 1):
                    lines = self._wrap_into_lines(text, n_lines)
                    if lines is None:
                        continue
                    block_w, block_h = self._block_size(lines, font, style)
                    force_box = self._force_expand_box(
                        box, block_w, block_h, frame_w, frame_h
                    )
                    if force_box is not None and self._fits(
                        (block_w, block_h), force_box
                    ):
                        line_h = self._line_height(font, style)
                        return _LayoutPlan(
                            box=force_box,
                            font_path=path,
                            font_size=fixed_font_size,
                            lines=tuple(lines),
                            line_height=line_h,
                        )

        # --- Last resort: retry without fixed constraint so text still
        # renders rather than disappearing (prevents flickering). ---
        if fixed_font_size is not None:
            for path in font_paths:
                for candidate_box in (box, expanded_box):
                    if candidate_box is None:
                        continue
                    plan = self._fit_single_line(
                        text, candidate_box, style, path, None
                    )
                    if plan is not None:
                        return plan
                if max_lines >= 2:
                    for candidate_box in (box, expanded_box):
                        if candidate_box is None:
                            continue
                        plan = self._fit_wrapped(
                            text, candidate_box, style, path, max_lines, None
                        )
                        if plan is not None:
                            return plan

        return None

    @staticmethod
    def _force_expand_box(
        box: Bounding_Box,
        need_w: int,
        need_h: int,
        frame_w: int,
        frame_h: int,
    ) -> Bounding_Box | None:
        """Expand box centered on its original center to fit (need_w, need_h).

        The result is clipped to frame bounds. Returns None if the needed
        size exceeds the frame entirely.
        """
        # Use at least the original box size.
        new_w = max(box.width, need_w)
        new_h = max(box.height, need_h)
        cx, cy = box.center
        new_x = int(round(cx - new_w / 2))
        new_y = int(round(cy - new_h / 2))
        # Clip to frame.
        x1 = max(0, new_x)
        y1 = max(0, new_y)
        x2 = min(frame_w, new_x + new_w)
        y2 = min(frame_h, new_y + new_h)
        clipped_w = x2 - x1
        clipped_h = y2 - y1
        if clipped_w <= 0 or clipped_h <= 0:
            return None
        return Bounding_Box(x1, y1, clipped_w, clipped_h)

    def _fit_single_line(
        self,
        text: str,
        box: Bounding_Box,
        style: Style_Preset,
        font_path: str,
        fixed_font_size: int | None,
    ) -> _LayoutPlan | None:
        size = self._auto_fit(
            (text,), box, style, font_path, fixed_font_size=fixed_font_size
        )
        if size is None:
            return None
        font = self._get_font(font_path, size)
        line_h = self._line_height(font, style)
        return _LayoutPlan(
            box=box,
            font_path=font_path,
            font_size=size,
            lines=(text,),
            line_height=line_h,
        )

    def _fit_wrapped(
        self,
        text: str,
        box: Bounding_Box,
        style: Style_Preset,
        font_path: str,
        max_lines: int,
        fixed_font_size: int | None,
    ) -> _LayoutPlan | None:
        """Try every line count from 2..max_lines and pick the largest fitting size."""
        best: _LayoutPlan | None = None
        for n_lines in range(2, max_lines + 1):
            lines = self._wrap_into_lines(text, n_lines)
            if lines is None:
                continue
            size = self._auto_fit(lines, box, style, font_path, fixed_font_size=fixed_font_size)
            if size is None:
                continue
            font = self._get_font(font_path, size)
            line_h = self._line_height(font, style)
            plan = _LayoutPlan(
                box=box,
                font_path=font_path,
                font_size=size,
                lines=tuple(lines),
                line_height=line_h,
            )
            # Larger font is preferred. Single line was already tried earlier
            # so we just return the first wrapped plan that fits — they only
            # get smaller as n_lines grows.
            if best is None or plan.font_size > best.font_size:
                best = plan
            else:
                break
        return best

    @staticmethod
    def _wrap_into_lines(text: str, n_lines: int) -> list[str] | None:
        """Greedy near-balanced split into ``n_lines`` lines on whitespace.

        Returns ``None`` if the text cannot be split into that many lines
        (e.g. fewer than ``n_lines`` whitespace-separated tokens).
        """
        words = text.split()
        if len(words) < n_lines:
            return None
        target = max(1, len(text) // n_lines)
        lines: list[str] = []
        current: list[str] = []
        current_len = 0
        for w in words:
            if current and current_len + 1 + len(w) > target and len(lines) < n_lines - 1:
                lines.append(" ".join(current))
                current = [w]
                current_len = len(w)
            else:
                if current:
                    current_len += 1 + len(w)
                else:
                    current_len = len(w)
                current.append(w)
        if current:
            lines.append(" ".join(current))
        # Edge case: if the greedy pass produced fewer lines than requested
        # (e.g. very short tokens), pad with empties so the caller can still
        # try again with a smaller n_lines.
        if len(lines) != n_lines:
            return None
        return lines

    def _auto_fit(
        self,
        lines: tuple[str, ...] | list[str],
        box: Bounding_Box,
        style: Style_Preset,
        font_path: str,
        *,
        fixed_font_size: int | None = None,
    ) -> int | None:
        """Largest font size in [min, max] for which ``lines`` fit ``box``."""
        if fixed_font_size is not None:
            font = self._get_font(font_path, fixed_font_size)
            if self._fits(self._block_size(lines, font, style), box):
                return fixed_font_size
            return None
        lo, hi = style.font_size_min, style.font_size_max
        font_min = self._get_font(font_path, lo)
        if not self._fits(self._block_size(lines, font_min, style), box):
            return None
        while lo < hi:
            mid = (lo + hi + 1) // 2
            font = self._get_font(font_path, mid)
            if self._fits(self._block_size(lines, font, style), box):
                lo = mid
            else:
                hi = mid - 1
        return lo

    # Public helpers used by Pipeline (font-size pre-computation).

    def auto_fit_font_size(
        self,
        text: str,
        box: Bounding_Box,
        style: Style_Preset,
        *,
        font_path: str | None = None,
    ) -> int | None:
        """Single-line auto-fit using the regular font (legacy API)."""
        path = font_path or style.font_path or self._default_font_path
        return self._auto_fit((text,), box, style, path)

    @staticmethod
    def place_text(
        text_size: tuple[int, int],
        box: Bounding_Box,
    ) -> tuple[int, int] | None:
        """Top-left position with ≤2 px tolerance from box center."""
        tw, th = text_size
        cx, cy = box.center
        x = int(round(cx - tw / 2))
        y = int(round(cy - th / 2))
        x = max(box.x, min(x, box.x2 - tw))
        y = max(box.y, min(y, box.y2 - th))
        if abs(x + tw / 2 - cx) > 2 or abs(y + th / 2 - cy) > 2:
            return None
        return x, y

    # ------------------------------------------------------------------
    # Text geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_box(
        box: Bounding_Box, factor: float, frame_w: int, frame_h: int,
        factor_h: float | None = None,
    ) -> Bounding_Box | None:
        """Expand box with separate horizontal/vertical factors.

        If factor_h is None, factor is used for both dimensions.
        """
        factor_w = factor
        if factor_h is None:
            factor_h = factor
        if factor_w <= 1.0 and factor_h <= 1.0:
            return None
        cx, cy = box.center
        new_w = int(round(box.width * factor_w))
        new_h = int(round(box.height * factor_h))
        # Center the expanded box on the original center, then clip to frame.
        new_x = int(round(cx - new_w / 2))
        new_y = int(round(cy - new_h / 2))
        x1 = max(0, new_x)
        y1 = max(0, new_y)
        x2 = min(frame_w, new_x + new_w)
        y2 = min(frame_h, new_y + new_h)
        clipped_w = x2 - x1
        clipped_h = y2 - y1
        if clipped_w <= box.width and clipped_h <= box.height:
            return None
        if clipped_w <= 0 or clipped_h <= 0:
            return None
        return Bounding_Box(x1, y1, clipped_w, clipped_h)

    def _get_font(self, path: str, size: int) -> ImageFont.FreeTypeFont:
        key = (path, size)
        cached = self._font_cache.get(key)
        if cached is not None:
            return cached
        try:
            font = ImageFont.truetype(path, size)
        except Exception as exc:
            raise InvalidConfigError(
                f"renderer: cannot load font {path!r} at size {size}: {exc}"
            ) from exc
        self._font_cache[key] = font
        return font

    @staticmethod
    def _line_height(font: ImageFont.FreeTypeFont, style: Style_Preset) -> int:
        ascent, descent = font.getmetrics()
        h = ascent + descent
        if style.stroke_enabled:
            h += 2 * style.stroke_width
        if style.shadow_enabled:
            h += max(0, style.shadow_offset[1])
        return int(h)

    @staticmethod
    def _line_width(
        text: str, font: ImageFont.FreeTypeFont, style: Style_Preset
    ) -> int:
        stroke = style.stroke_width if style.stroke_enabled else 0
        bbox = font.getbbox(text, stroke_width=stroke)
        w = bbox[2] - bbox[0]
        if style.shadow_enabled:
            w += max(0, style.shadow_offset[0])
        return int(w)

    @classmethod
    def _block_size(
        cls,
        lines: tuple[str, ...] | list[str],
        font: ImageFont.FreeTypeFont,
        style: Style_Preset,
    ) -> tuple[int, int]:
        max_w = 0
        for line in lines:
            max_w = max(max_w, cls._line_width(line, font, style))
        line_h = cls._line_height(font, style)
        return max_w, line_h * len(lines)

    @staticmethod
    def _fits(text_size: tuple[int, int], box: Bounding_Box) -> bool:
        return text_size[0] <= box.width and text_size[1] <= box.height

    # Kept for backward compatibility with callers / tests.

    def _text_bbox(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        style: Style_Preset,
    ) -> tuple[int, int]:
        return self._block_size((text,), font, style)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(
        self,
        frame: np.ndarray,
        plan: _LayoutPlan,
        style: Style_Preset,
    ) -> np.ndarray:
        # Optimized crop-only compositing: instead of converting the entire
        # frame (e.g. 1920x1080 = 2M pixels) to PIL and back, we only
        # convert the small region around the text box. This reduces memory
        # copies by ~95% for typical subtitle boxes (200x40 px).
        h_frame, w_frame = frame.shape[:2]

        # Compute the drawing region with padding for shadow/stroke overflow.
        pad = max(
            style.stroke_width if style.stroke_enabled else 0,
            max(abs(style.shadow_offset[0]), abs(style.shadow_offset[1]))
            if style.shadow_enabled else 0,
        ) + 2  # extra safety margin

        crop_x1 = max(0, plan.box.x - pad)
        crop_y1 = max(0, plan.box.y - pad)
        crop_x2 = min(w_frame, plan.box.x2 + pad)
        crop_y2 = min(h_frame, plan.box.y2 + pad)
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        if crop_w <= 0 or crop_h <= 0:
            return frame

        # Extract the crop region and convert only that to PIL.
        crop_bgr = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        crop_rgb = crop_bgr[..., ::-1]
        img = Image.fromarray(crop_rgb).convert("RGBA")
        overlay = Image.new("RGBA", (crop_w, crop_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        font = self._get_font(plan.font_path, plan.font_size)
        block_w, block_h = self._block_size(plan.lines, font, style)

        # Offset: translate absolute frame coords to crop-local coords.
        off_x = crop_x1
        off_y = crop_y1

        # Draw background covering the box used for layout.
        if style.background_enabled and style.background_alpha > 0:
            bg = (*style.background_rgb, style.background_alpha)
            draw.rectangle(
                (
                    (plan.box.x - off_x, plan.box.y - off_y),
                    (plan.box.x2 - 1 - off_x, plan.box.y2 - 1 - off_y),
                ),
                fill=bg,
            )

        # Center the whole text block inside plan.box (in crop-local coords).
        cx, cy = plan.box.center
        block_x = int(round(cx - block_w / 2))
        block_y = int(round(cy - block_h / 2))
        block_x = max(plan.box.x, min(block_x, plan.box.x2 - block_w))
        block_y = max(plan.box.y, min(block_y, plan.box.y2 - block_h))

        for i, line in enumerate(plan.lines):
            line_w = self._line_width(line, font, style)
            line_x = int(round(cx - line_w / 2))
            line_x = max(plan.box.x, min(line_x, plan.box.x2 - line_w))
            line_y = block_y + i * plan.line_height

            # Convert to crop-local coordinates.
            local_x = line_x - off_x
            local_y = line_y - off_y

            if style.shadow_enabled and style.shadow_offset != (0, 0):
                sx = local_x + style.shadow_offset[0]
                sy = local_y + style.shadow_offset[1]
                draw.text(
                    (sx, sy),
                    line,
                    font=font,
                    fill=(*style.shadow_rgb, 255),
                )
            stroke_w = style.stroke_width if style.stroke_enabled else 0
            draw.text(
                (local_x, local_y),
                line,
                font=font,
                fill=(*style.text_rgb, 255),
                stroke_width=stroke_w,
                stroke_fill=(*style.stroke_rgb, 255),
            )

        # Composite only the crop region and paste back.
        composed = Image.alpha_composite(img, overlay).convert("RGB")
        crop_result = np.asarray(composed)[..., ::-1].copy()

        # Write the rendered crop back into the frame (in-place for speed).
        out = frame.copy()
        out[crop_y1:crop_y2, crop_x1:crop_x2] = crop_result
        return out
