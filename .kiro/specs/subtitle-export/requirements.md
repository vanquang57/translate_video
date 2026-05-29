# Requirements Document

## Introduction

This feature adds SRT subtitle file export to the Video Text Translator. When enabled, the system generates an SRT file containing the Vietnamese translated text alongside the translated video output. The exported subtitles match the hardcoded text rendered on the video frame-for-frame, enabling downstream TTS workflows where audio must align precisely with on-screen text. An optional subtitle region selector allows users to filter which detected text regions are exported as subtitles versus ignored (e.g., watermarks, titles).

## Glossary

- **Subtitle_Exporter**: The module responsible for generating SRT subtitle files from translated text segments.
- **Region_Selector**: The GUI component (modal popup dialog) that allows users to define a rectangular subtitle region on a video frame preview.
- **Subtitle_Region**: A user-defined rectangular area on the video frame; only text whose bounding box center falls inside this region is classified as subtitle text for export.
- **SRT_File**: A SubRip Text file (.srt) containing sequentially numbered subtitle entries with timestamps and Vietnamese translated text.
- **Text_Segment**: An existing domain model representing a contiguous run of frames where the same text appears, with start_time, end_time, and canonical_text.
- **Pipeline**: The existing end-to-end orchestrator that runs detection, tracking, translation, inpainting, rendering, and audio muxing.
- **GUI**: The existing tkinter-based graphical user interface (gui.py).
- **Center_Point**: The geometric center of a Text_Segment bounding box, computed from the segment entries.

## Requirements

### Requirement 1: Export Subtitles Checkbox

**User Story:** As a user, I want a checkbox in the GUI to enable subtitle export, so that I can choose whether an SRT file is generated alongside the translated video.

#### Acceptance Criteria

1. THE GUI SHALL display an "Export subtitles" checkbox in the main window controls area.
2. THE GUI SHALL default the "Export subtitles" checkbox to unchecked state on application launch.
3. WHEN the user checks the "Export subtitles" checkbox, THE GUI SHALL enable the subtitle export feature for the next pipeline run.

### Requirement 2: Subtitle Region Selection Dialog

**User Story:** As a user, I want to visually define a subtitle region on a video frame, so that only text in that area is exported to the SRT file.

#### Acceptance Criteria

1. WHEN the user checks the "Export subtitles" checkbox and a video file is selected, THE GUI SHALL enable a "Select Region" button.
2. WHEN the user clicks the "Select Region" button, THE Region_Selector SHALL open a modal popup dialog displaying a frame from the selected video.
3. THE Region_Selector SHALL provide a slider control to seek through video frames for preview.
4. THE Region_Selector SHALL allow the user to draw a rectangle on the displayed video frame to define the Subtitle_Region.
5. WHEN the user clicks OK in the Region_Selector dialog, THE Region_Selector SHALL close and return the selected Subtitle_Region coordinates to the main GUI.
6. WHEN the user clicks OK without drawing a rectangle, THE Region_Selector SHALL close and indicate that no region was defined (full-frame mode).
7. WHEN the Region_Selector returns coordinates to the main GUI, THE GUI SHALL display the selected region coordinates and a "Clear" button.
8. WHEN the user clicks the "Clear" button, THE GUI SHALL remove the stored Subtitle_Region and revert to full-frame mode.

### Requirement 3: Region-Based Text Filtering

**User Story:** As a user, I want only subtitle-area text exported to SRT, so that watermarks and non-subtitle text are excluded from the subtitle file.

#### Acceptance Criteria

1. WHILE a Subtitle_Region is defined, THE Subtitle_Exporter SHALL classify a Text_Segment as subtitle text only when the Center_Point of the segment bounding box falls inside the Subtitle_Region.
2. WHILE a Subtitle_Region is defined, THE Subtitle_Exporter SHALL exclude Text_Segments whose Center_Point falls outside the Subtitle_Region from the SRT_File.
3. WHILE no Subtitle_Region is defined (full-frame mode), THE Subtitle_Exporter SHALL include all translated Text_Segments in the SRT_File.

### Requirement 4: SRT File Generation

**User Story:** As a user, I want a properly formatted SRT file generated from the translated text segments, so that I can use it for TTS or external subtitle players.

#### Acceptance Criteria

1. WHEN subtitle export is enabled and the translation phase completes, THE Subtitle_Exporter SHALL generate a single SRT_File containing Vietnamese translated text.
2. THE Subtitle_Exporter SHALL format each subtitle entry with a sequential numeric index, a timestamp line (HH:MM:SS,mmm --> HH:MM:SS,mmm), and the Vietnamese translated text.
3. THE Subtitle_Exporter SHALL derive subtitle entry timestamps from the start_time and end_time of each Text_Segment.
4. THE Subtitle_Exporter SHALL produce one subtitle entry per Text_Segment, preserving the deduplication already performed by the tracker.
5. THE Subtitle_Exporter SHALL order subtitle entries chronologically by start_time.
6. THE Subtitle_Exporter SHALL separate subtitle entries with a blank line as required by the SRT format specification.

### Requirement 5: SRT File Output Location and Naming

**User Story:** As a user, I want the SRT file saved alongside the translated video with a matching filename, so that media players auto-detect the subtitle file.

#### Acceptance Criteria

1. THE Subtitle_Exporter SHALL save the SRT_File in the same output directory as the translated video file.
2. THE Subtitle_Exporter SHALL name the SRT_File using the translated video filename with the extension replaced by .srt.
3. IF the SRT_File path already exists, THEN THE Subtitle_Exporter SHALL overwrite the existing file.

### Requirement 6: Pipeline Integration

**User Story:** As a developer, I want the subtitle export to integrate cleanly into the existing pipeline without degrading performance, so that the translation workflow remains fast.

#### Acceptance Criteria

1. WHEN subtitle export is enabled, THE Pipeline SHALL invoke the Subtitle_Exporter after the translation phase completes and before or during Pass 2.
2. THE Subtitle_Exporter SHALL execute SRT generation using the existing Text_Segment data and translations dictionary without re-processing video frames.
3. THE Pipeline SHALL pass the Subtitle_Region coordinates (or absence thereof) from the GUI configuration to the Subtitle_Exporter.
4. IF subtitle export is disabled, THEN THE Pipeline SHALL skip all subtitle export processing entirely.

### Requirement 7: Module Organization

**User Story:** As a developer, I want the subtitle export code organized in its own module, so that the codebase remains maintainable and testable.

#### Acceptance Criteria

1. THE Subtitle_Exporter SHALL be implemented in a dedicated module within the src/video_text_translator package.
2. THE Subtitle_Exporter module SHALL have no dependencies on GUI code or video frame data.
3. THE Subtitle_Exporter module SHALL depend only on the Text_Segment model, translations dictionary, and optional Subtitle_Region coordinates.
