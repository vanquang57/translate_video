# Design Document: Subtitle Export

## Overview

The subtitle export feature adds SRT file generation to the Video Text Translator pipeline. When enabled, the system generates an SRT file containing Vietnamese translated text alongside the translated video output. An optional subtitle region selector filters which detected text regions are exported as subtitles.

The design prioritizes zero performance impact on the existing pipeline by operating only on in-memory segment data (no video frame re-processing) and writing the SRT file as a single I/O operation.

## Architecture

The feature consists of three components integrated into the existing pipeline:

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐
│  Pass 1     │────▶│  Translator  │────▶│ Subtitle Exporter │──▶ .srt file
│ (segments)  │     │(translations)│     │  (pure logic)     │
└─────────────┘     └──────────────┘     └───────────────────┘
                                                   ▲
                                                   │
                                          ┌────────┴────────┐
                                          │ Subtitle_Region  │
                                          │ (from GUI/config)│
                                          └─────────────────┘
```

1. **Subtitle Exporter Module** (`src/video_text_translator/subtitle_exporter.py`) — pure-logic module that generates SRT content from translated segments with optional region filtering.
2. **Region Selector Dialog** (GUI component in `gui.py`) — tkinter modal popup for drawing a subtitle region on a video frame preview.
3. **Pipeline Hook** — lightweight invocation in `Pipeline.run()` after translation completes, before Pass 2.

The exporter is invoked between the translation phase and Pass 2. It reads only the in-memory `Text_Segment` list and `translations` dict — no video frames are touched. SRT export failure is non-critical: the pipeline logs a warning and continues.

## Components and Interfaces

### Subtitle Exporter Module

**Location:** `src/video_text_translator/subtitle_exporter.py`

**Dependencies:** Only `models.Text_Segment`, `models.Bounding_Box` — no GUI, no video frame data, no external libraries beyond the standard library.

**Public API:**

```python
def generate_srt(
    segments: Sequence[Text_Segment],
    translations: dict[str, str],
    region: Subtitle_Region | None = None,
) -> str:
    """Generate SRT file content from translated segments.
    
    Steps:
    1. Filter segments by region (or include all if region is None)
    2. Sort filtered segments chronologically by start_time
    3. Format each as a numbered SRT entry
    4. Join with blank line separators
    
    Returns the complete SRT file content as a string.
    """


def export_srt(
    segments: Sequence[Text_Segment],
    translations: dict[str, str],
    video_output_path: str,
    region: Subtitle_Region | None = None,
) -> str:
    """Top-level export function: generate SRT and write to disk.
    
    Returns the path of the written SRT file.
    """


def derive_srt_path(video_output_path: str) -> str:
    """Derive the SRT file path from the video output path.
    
    Replaces the video file extension with .srt, keeping the same
    directory and stem.
    """


def filter_segments(
    segments: Sequence[Text_Segment],
    region: Subtitle_Region | None,
) -> list[Text_Segment]:
    """Filter segments to only those whose center falls inside the region.
    
    If region is None (full-frame mode), all segments are returned.
    """


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""


def compute_segment_center(segment: Text_Segment) -> tuple[float, float]:
    """Compute the representative center point of a segment.
    
    Uses the center of the first entry's bounding box.
    """
```

### Subtitle_Region Data Type

```python
@dataclass(frozen=True, slots=True)
class Subtitle_Region:
    """A rectangular region on the video frame for filtering subtitle text."""
    x: int
    y: int
    width: int
    height: int

    def contains_point(self, px: float, py: float) -> bool:
        """Return True if the point (px, py) falls inside this region."""
        return (
            self.x <= px <= self.x + self.width
            and self.y <= py <= self.y + self.height
        )
```

### Region Selector Dialog

**Location:** Inline in `gui.py` as class `RegionSelectorDialog`.

```python
class RegionSelectorDialog:
    """Modal dialog for selecting a subtitle region on a video frame."""
    
    def __init__(self, parent: tk.Tk, video_path: str) -> None:
        self.result: Subtitle_Region | None = None

    def show(self) -> Subtitle_Region | None:
        """Show the dialog modally and return the result."""
        return self.result
```

The dialog:
- Reads a single frame from the video using `cv2.VideoCapture`
- Displays it on a `tk.Canvas` scaled to fit (max 800×600)
- Provides a `ttk.Scale` slider to seek through frames
- Allows rectangle drawing via mouse drag
- Converts canvas coordinates back to original video pixel space on OK
- Returns `None` if OK is clicked without drawing a rectangle

### Pipeline Integration Interface

```python
class Pipeline:
    def __init__(
        self,
        config: Config,
        # ... existing params ...
        export_subtitles: bool = False,
        subtitle_region: Subtitle_Region | None = None,
    ) -> None:
        self._export_subtitles = export_subtitles
        self._subtitle_region = subtitle_region

    def _export_srt(
        self,
        segments: Sequence[Text_Segment],
        translations: dict[str, str],
    ) -> None:
        """Generate and write the SRT file. Non-critical — logs on failure."""
        from .subtitle_exporter import export_srt
        try:
            srt_path = export_srt(
                segments=segments,
                translations=translations,
                video_output_path=self.config.output_path,
                region=self._subtitle_region,
            )
            logger.info("SRT exported: %s", srt_path)
        except Exception as exc:
            logger.warning("SRT export failed (non-critical): %s", exc)
```

### GUI Controls

Added to `TranslatorApp._build_ui()`:

```python
# --- Subtitle export controls ---
subtitle_frame = ttk.LabelFrame(self.root, text="Subtitle Export", padding=10)

self._export_srt_var = tk.BooleanVar(value=False)  # default unchecked
self._export_cb = ttk.Checkbutton(
    subtitle_frame, text="Export subtitles",
    variable=self._export_srt_var,
    command=self._on_export_toggle,
)

self._region_btn = ttk.Button(
    subtitle_frame, text="Select Region",
    command=self._open_region_selector,
    state=tk.DISABLED,  # enabled when export checked + video selected
)

self._subtitle_region: Subtitle_Region | None = None
```

## Data Models

### Subtitle_Region

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| x | int | >= 0 | Left edge in video pixel coordinates |
| y | int | >= 0 | Top edge in video pixel coordinates |
| width | int | > 0 | Width in pixels |
| height | int | > 0 | Height in pixels |

### SRT Entry Format

Each subtitle entry in the generated file follows the SRT specification:

```
{index}\n
{HH:MM:SS,mmm} --> {HH:MM:SS,mmm}\n
{translated_text}\n
\n
```

Where:
- `index`: Sequential integer starting at 1
- Timestamps: Derived from `Text_Segment.start_time` and `Text_Segment.end_time`
- `translated_text`: Vietnamese translation from the translations dict, falling back to `canonical_text` if missing

### Existing Models Used

- **Text_Segment**: `segment_id`, `start_time`, `end_time`, `canonical_text`, `entries` (tuple of `Frame_Region_Entry`)
- **Frame_Region_Entry**: `frame_index`, `timestamp`, `box` (Bounding_Box), `text`
- **Bounding_Box**: `x`, `y`, `width`, `height` with `.center` property returning `(float, float)`

## Error Handling

| Scenario | Behavior |
|----------|----------|
| No segments after filtering | Write an empty SRT file (valid, zero entries) |
| Segment has no entries | `compute_segment_center` raises `ValueError`; segment skipped with warning |
| Output directory not writable | `export_srt` raises `OSError`; pipeline logs warning, continues |
| Translation missing for segment | Falls back to `canonical_text` (original text) |
| Invalid timestamp (negative) | Clamped to 0.0 in `format_timestamp` |
| SRT export fails for any reason | Pipeline logs warning, continues to produce video (non-critical) |

## Testing Strategy

**Dual approach:**
- **Property-based tests** for the pure logic functions (`filter_segments`, `generate_srt`, `format_timestamp`, `derive_srt_path`) — these have clear universal properties across all inputs.
- **Example-based unit tests** for GUI interactions, pipeline integration, and specific edge cases (empty input, overwrite behavior).

Property tests use `hypothesis` with minimum 100 iterations per property. Unit tests cover specific scenarios and integration points.

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Region filtering correctness

*For any* list of Text_Segments and *for any* Subtitle_Region, a segment is included in the filtered output if and only if the center point of its first entry's bounding box falls inside the region (i.e., `region.x <= cx <= region.x + region.width` and `region.y <= cy <= region.y + region.height`).

**Validates: Requirements 3.1, 3.2**

### Property 2: Full-frame mode includes all segments

*For any* list of Text_Segments, when no Subtitle_Region is defined (region is None), the filtered output contains exactly the same segments as the input (same count, same identities).

**Validates: Requirements 3.3**

### Property 3: SRT format correctness

*For any* list of Text_Segments with corresponding translations, the generated SRT content consists of entries where each entry has: (a) a sequential numeric index starting at 1, (b) a timestamp line in the format `HH:MM:SS,mmm --> HH:MM:SS,mmm` where the start timestamp equals the segment's `start_time` and the end timestamp equals the segment's `end_time`, (c) the translated text on the following line, and (d) entries are separated by exactly one blank line.

**Validates: Requirements 4.2, 4.3, 4.6**

### Property 4: Entry count invariant

*For any* list of N Text_Segments (after region filtering), the generated SRT content contains exactly N subtitle entries.

**Validates: Requirements 4.4**

### Property 5: Chronological ordering

*For any* list of Text_Segments (possibly unordered by start_time), the generated SRT entries are ordered such that each entry's start timestamp is less than or equal to the next entry's start timestamp.

**Validates: Requirements 4.5**

### Property 6: SRT path derivation

*For any* valid video output path with any file extension, the derived SRT path has the same parent directory and the same filename stem as the video path, with the extension replaced by `.srt`.

**Validates: Requirements 5.1, 5.2**
