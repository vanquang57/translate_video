# Implementation Plan: Subtitle Export

## Overview

Implement SRT subtitle file export for the Video Text Translator. The feature adds a pure-logic exporter module, a GUI region selector dialog, and pipeline integration — all operating on existing in-memory segment data without re-processing video frames.

## Tasks

- [x] 1. Create Subtitle_Region data model and exporter module
  - [x] 1.1 Add Subtitle_Region dataclass to models.py
    - Add `Subtitle_Region` frozen dataclass with fields `x`, `y`, `width`, `height` and `contains_point` method
    - Validate constraints: x >= 0, y >= 0, width > 0, height > 0
    - _Requirements: 3.1, 3.2_

  - [x] 1.2 Create subtitle_exporter.py with core functions
    - Create `src/video_text_translator/subtitle_exporter.py`
    - Implement `format_timestamp(seconds: float) -> str` — converts seconds to `HH:MM:SS,mmm` format, clamps negatives to 0.0
    - Implement `compute_segment_center(segment: Text_Segment) -> tuple[float, float]` — returns center of first entry's bounding box, raises ValueError if no entries
    - Implement `derive_srt_path(video_output_path: str) -> str` — replaces extension with `.srt`
    - Implement `filter_segments(segments, region) -> list[Text_Segment]` — filters by center point inside region, returns all if region is None
    - Implement `generate_srt(segments, translations, region) -> str` — filters, sorts chronologically, formats as SRT string
    - Implement `export_srt(segments, translations, video_output_path, region) -> str` — calls generate_srt, writes to disk, returns path
    - _Requirements: 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1, 5.2, 5.3, 7.1, 7.2, 7.3_

  - [ ]* 1.3 Write property test: Region filtering correctness (Property 1)
    - **Property 1: Region filtering correctness**
    - Use hypothesis to generate arbitrary Text_Segments and Subtitle_Regions
    - Assert segment included iff center point falls inside region bounds
    - **Validates: Requirements 3.1, 3.2**

  - [ ]* 1.4 Write property test: Full-frame mode includes all segments (Property 2)
    - **Property 2: Full-frame mode includes all segments**
    - Generate arbitrary segment lists, call filter_segments with region=None
    - Assert output count equals input count and same identities
    - **Validates: Requirements 3.3**

  - [ ]* 1.5 Write property test: SRT format correctness (Property 3)
    - **Property 3: SRT format correctness**
    - Generate segments with translations, verify each entry has sequential index, correct timestamp format, translated text, and blank-line separators
    - **Validates: Requirements 4.2, 4.3, 4.6**

  - [ ]* 1.6 Write property test: Entry count invariant (Property 4)
    - **Property 4: Entry count invariant**
    - Generate N segments, verify generated SRT has exactly N entries after filtering
    - **Validates: Requirements 4.4**

  - [ ]* 1.7 Write property test: Chronological ordering (Property 5)
    - **Property 5: Chronological ordering**
    - Generate unordered segments, verify SRT entries are sorted by start timestamp
    - **Validates: Requirements 4.5**

  - [ ]* 1.8 Write property test: SRT path derivation (Property 6)
    - **Property 6: SRT path derivation**
    - Generate arbitrary video paths with various extensions, verify derived path has same parent/stem with .srt extension
    - **Validates: Requirements 5.1, 5.2**

- [x] 2. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Integrate subtitle export into Pipeline
  - [x] 3.1 Add export_subtitles and subtitle_region parameters to Pipeline.__init__
    - Add `export_subtitles: bool = False` and `subtitle_region: Subtitle_Region | None = None` parameters
    - Store as `self._export_subtitles` and `self._subtitle_region`
    - _Requirements: 6.3, 6.4_

  - [x] 3.2 Add _export_srt helper method to Pipeline
    - Implement `_export_srt(self, segments, translations)` method
    - Import and call `export_srt` from subtitle_exporter module
    - Wrap in try/except: log warning on failure, never raise (non-critical)
    - _Requirements: 6.1, 6.2_

  - [x] 3.3 Invoke _export_srt in Pipeline.run() after translation
    - Call `self._export_srt(segments, translations)` after `_translate_segments` returns, before `_pass2`
    - Only invoke when `self._export_subtitles` is True
    - _Requirements: 6.1, 6.4_

  - [ ]* 3.4 Write unit tests for pipeline integration
    - Test that SRT is generated when export_subtitles=True
    - Test that SRT is skipped when export_subtitles=False
    - Test that pipeline continues when SRT export raises an exception
    - _Requirements: 6.1, 6.4_

- [x] 4. Implement Region Selector Dialog in GUI
  - [x] 4.1 Create RegionSelectorDialog class in gui.py
    - Implement modal dialog that reads a frame from video using cv2.VideoCapture
    - Display frame on tk.Canvas scaled to fit (max 800×600)
    - Add ttk.Scale slider to seek through video frames
    - Allow rectangle drawing via mouse drag on canvas
    - Convert canvas coordinates back to original video pixel space on OK
    - Return Subtitle_Region or None (if no rectangle drawn)
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x] 4.2 Add subtitle export controls to TranslatorApp._build_ui
    - Add "Subtitle Export" LabelFrame with "Export subtitles" checkbox (default unchecked)
    - Add "Select Region" button (disabled by default)
    - Add region display label and "Clear" button
    - Enable "Select Region" when checkbox is checked AND video file is selected
    - Wire _on_export_toggle and _open_region_selector handlers
    - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.7, 2.8_

  - [x] 4.3 Pass subtitle settings from GUI to Pipeline
    - Update `_run_pipeline` to pass `export_subtitles` and `subtitle_region` to Pipeline constructor
    - Read values from `self._export_srt_var` and `self._subtitle_region`
    - _Requirements: 6.3_

  - [ ]* 4.4 Write unit tests for GUI subtitle controls
    - Test checkbox state toggling enables/disables region button
    - Test region clear resets stored region to None
    - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.7, 2.8_

- [x] 5. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The subtitle exporter module has zero dependencies on GUI or video frame data (Requirement 7.2)
- SRT export failure is non-critical — pipeline always continues to produce video output

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["1.3", "1.4", "1.5", "1.6", "1.7", "1.8"] },
    { "id": 3, "tasks": ["3.1", "4.1"] },
    { "id": 4, "tasks": ["3.2", "4.2"] },
    { "id": 5, "tasks": ["3.3", "4.3"] },
    { "id": 6, "tasks": ["3.4", "4.4"] }
  ]
}
```
