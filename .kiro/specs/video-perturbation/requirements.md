# Requirements Document

## Introduction

Video Perturbation là module độc lập tích hợp vào Video Text Translator, cho phép áp dụng các biến đổi động (dynamic transformations) lên video đã xử lý hoặc bất kỳ video nào. Mục đích là tạo ra các phiên bản video đã bị biến đổi tinh vi để kiểm thử hệ thống phát hiện reup video của TikTok. Điểm mấu chốt: tất cả biến đổi phải thay đổi theo thời gian (dynamic), không tĩnh (static), khiến việc phát hiện khó hơn.

## Glossary

- **Perturbation_Engine**: Module xử lý chính, nhận video đầu vào và áp dụng các biến đổi động theo cấu hình
- **Perturbation_Config**: File cấu hình YAML chứa tham số điều khiển các biến đổi (max_crop, speed_range, zoom_level, change_interval, presets)
- **Temporal_Drift**: Nhóm biến đổi thời gian — thay đổi tốc độ phát (0.99-1.01x) theo từng segment, drop/duplicate frame, micro offset
- **Spatial_Transform**: Nhóm biến đổi không gian động — animated crop/zoom thay đổi liên tục theo thời gian
- **Scene_Recomposition**: Nhóm biến đổi cấu trúc — sắp xếp lại micro-scene, cắt/chèn transition
- **Audio_Perturbation**: Nhóm biến đổi âm thanh — micro tempo drift, EQ thay đổi theo segment, thêm lớp ambience nhẹ
- **Multi_Transform_Combo**: Chế độ kết hợp nhiều biến đổi nhỏ trực giao (orthogonal) đồng thời
- **Preset**: Cấu hình sẵn định nghĩa mức độ biến đổi (light, medium, heavy)
- **GUI**: Giao diện tkinter hiện tại của Video Text Translator
- **CLI**: Giao diện dòng lệnh hiện tại qua main.py
- **Segment**: Đoạn video ngắn mà trong đó một tham số biến đổi được áp dụng trước khi chuyển sang giá trị khác
- **Change_Interval**: Khoảng thời gian (giây) giữa các lần thay đổi tham số biến đổi

## Requirements

### Requirement 1: Perturbation Engine Core

**User Story:** As a QA engineer, I want to apply dynamic perturbations to any video file, so that I can test the reup detection system with realistically transformed videos.

#### Acceptance Criteria

1. WHEN a video file in a supported container format (MP4, AVI, MOV, MKV, or WebM) and a valid Perturbation_Config are provided, THE Perturbation_Engine SHALL produce an output video in the same container format with all specified perturbations applied in sequence
2. WHEN the Perturbation_Engine processes a video, THE Perturbation_Engine SHALL randomly select new values within each parameter's configured min/max range at every Change_Interval boundary (1 to 300 seconds) and apply them to subsequent frames until the next boundary
3. IF the Change_Interval exceeds the total video duration, THEN THE Perturbation_Engine SHALL apply a single randomly selected parameter set for the entire video
4. THE Perturbation_Engine SHALL preserve the original video duration within 1% tolerance after applying Temporal_Drift
5. IF the input video file is unreadable, corrupted, or in an unsupported format, THEN THE Perturbation_Engine SHALL return an error message indicating the failure reason (unreadable, corrupted, or unsupported format) and the file path, without crashing or producing partial output
6. IF the output directory is not writable, THEN THE Perturbation_Engine SHALL return an error message indicating the permission issue and the target path before any video processing begins
7. IF the Perturbation_Config contains an unrecognized perturbation type or a parameter value outside its valid range, THEN THE Perturbation_Engine SHALL return an error message identifying the invalid field and its accepted values without processing the video

### Requirement 2: Temporal Drift Transforms

**User Story:** As a QA engineer, I want to apply subtle speed variations and frame manipulations that change over time, so that temporal fingerprinting is disrupted.

#### Acceptance Criteria

1. WHEN Temporal_Drift is enabled, THE Perturbation_Engine SHALL vary playback speed between the configured minimum and maximum (default 0.99x to 1.01x) per Segment, selecting a new random speed value for each Segment from a uniform distribution within the configured range
2. WHEN Temporal_Drift is enabled, THE Perturbation_Engine SHALL randomly drop or duplicate individual frames at a rate not exceeding the configured maximum percentage (default 1%), with no more than 1 consecutive frame affected per occurrence
3. WHEN Temporal_Drift is enabled, THE Perturbation_Engine SHALL apply micro time offsets by shifting Segment boundaries by a random duration within the configured range (default ±50 milliseconds)
4. WHILE Temporal_Drift is enabled, THE Perturbation_Engine SHALL change Temporal_Drift parameters at each Change_Interval boundary, using linear interpolation over a duration of 500 milliseconds to transition between the old and new parameter values
5. IF applying Temporal_Drift would cause the output video duration to deviate by more than 1% from the original, THEN THE Perturbation_Engine SHALL constrain speed and frame manipulation values to maintain duration within 1% tolerance

### Requirement 3: Dynamic Spatial Transforms

**User Story:** As a QA engineer, I want animated crop and zoom effects that continuously change, so that static spatial fingerprinting is defeated.

#### Acceptance Criteria

1. WHEN Spatial_Transform is enabled, THE Perturbation_Engine SHALL apply a crop that animates over time by interpolating the crop percentage between 0% and the configured max_crop_percent, transitioning to a new random target value at each Change_Interval boundary using linear interpolation between the previous and next target values
2. WHEN Spatial_Transform is enabled, THE Perturbation_Engine SHALL apply a zoom level that animates over time by interpolating between 1.0x and the configured max_zoom, transitioning to a new random target value at each Change_Interval boundary using linear interpolation between the previous and next target values
3. WHEN Spatial_Transform is enabled, THE Perturbation_Engine SHALL animate the crop region position by drifting the crop center at a rate not exceeding 0.5% of frame width per second horizontally and 0.5% of frame height per second vertically, selecting a new random drift direction at each Change_Interval boundary
4. THE Perturbation_Engine SHALL output video at the original resolution after applying Spatial_Transform, scaling the cropped/zoomed region back to the original dimensions
5. IF the computed crop region extends beyond the frame boundaries, THEN THE Perturbation_Engine SHALL clamp the crop region to remain fully within the frame dimensions without reducing the crop size
6. IF max_crop_percent is configured as 0 and max_zoom is configured as 1.0, THEN THE Perturbation_Engine SHALL skip Spatial_Transform processing and pass frames through unmodified

### Requirement 4: Scene Recomposition

**User Story:** As a QA engineer, I want to reorder micro-scenes and insert transitions, so that scene-based fingerprinting is disrupted.

#### Acceptance Criteria

1. WHEN Scene_Recomposition is enabled, THE Perturbation_Engine SHALL detect scene boundaries in the input video by computing frame-to-frame histogram difference and marking a boundary where the difference exceeds the configured scene_threshold (range: 0.1 to 1.0, default: 0.3)
2. WHEN Scene_Recomposition is enabled, THE Perturbation_Engine SHALL reorder detected micro-scenes (scenes with duration less than the configured max_scene_duration, range: 1 to 30 seconds, default: 5 seconds) using a random permutation that produces an ordering different from the original sequence
3. WHEN Scene_Recomposition is enabled, THE Perturbation_Engine SHALL insert cross-fade transitions between reordered scenes with a configured transition_duration (range: 100 to 2000 milliseconds, default: 500 milliseconds) that does not exceed the duration of either adjacent scene
4. IF the video contains fewer than 2 detectable scenes, THEN THE Perturbation_Engine SHALL skip Scene_Recomposition and log a warning indicating insufficient scenes were detected
5. IF the video contains 2 or more scenes but fewer than 2 micro-scenes (scenes shorter than max_scene_duration), THEN THE Perturbation_Engine SHALL skip reordering, log a warning indicating no micro-scenes were found, and output the video unchanged

### Requirement 5: Audio Perturbation

**User Story:** As a QA engineer, I want subtle audio modifications that change over time, so that audio fingerprinting is disrupted.

#### Acceptance Criteria

1. WHEN Audio_Perturbation is enabled, THE Perturbation_Engine SHALL apply tempo drift to the audio track by selecting a random speed factor within the configured range (default 0.99x to 1.01x) independently for each Change_Interval
2. WHEN Audio_Perturbation is enabled, THE Perturbation_Engine SHALL apply EQ adjustments that change per Change_Interval, shifting each of three frequency bands (bass: 20–250 Hz, mid: 250–4000 Hz, treble: 4000–20000 Hz) by a random offset within the configured dB range (default ±2 dB)
3. WHEN Audio_Perturbation is enabled, THE Perturbation_Engine SHALL overlay an ambient noise layer at a volume level randomly selected within the configured range (default -40 dB to -30 dB relative to main audio RMS) per Change_Interval
4. THE Perturbation_Engine SHALL change Audio_Perturbation parameters (tempo factor, EQ offsets, noise level) at each Change_Interval boundary, where Change_Interval defaults to 30 seconds and is configurable between 5 and 120 seconds
5. IF the input video has no audio track, THEN THE Perturbation_Engine SHALL skip Audio_Perturbation and log a warning message indicating that no audio track was found
6. IF the audio track duration is shorter than one Change_Interval, THEN THE Perturbation_Engine SHALL apply a single set of randomly selected perturbation parameters to the entire audio track

### Requirement 6: Multi-Transform Combo

**User Story:** As a QA engineer, I want to combine multiple small perturbations simultaneously, so that the combined effect is harder to detect than any single transform.

#### Acceptance Criteria

1. WHEN Multi_Transform_Combo is enabled, THE Perturbation_Engine SHALL apply perturbations from at least 2 and at most 3 different groups (Temporal_Drift, Spatial_Transform, Audio_Perturbation) to the same processing segment
2. WHEN Multi_Transform_Combo is enabled, THE Perturbation_Engine SHALL multiply each individual perturbation's standalone intensity by the configured combo_intensity_factor (valid range: 0.1 to 1.0, default 0.5)
3. WHILE Multi_Transform_Combo is active, THE Perturbation_Engine SHALL apply selected perturbations in the fixed order: Temporal_Drift first, then Spatial_Transform, then Audio_Perturbation, skipping any group not selected for the current combo
4. IF a perturbation group fails during combo application, THEN THE Perturbation_Engine SHALL skip the failed group, continue applying the remaining selected groups in order, and report which group was skipped

### Requirement 7: YAML Configuration

**User Story:** As a QA engineer, I want a YAML configuration file with presets and tunable parameters, so that I can quickly switch between perturbation intensities without editing code.

#### Acceptance Criteria

1. THE Perturbation_Config SHALL be stored at configs/perturbation.yaml using YAML structure with commented section headers, inline range annotations in comments, and UTF-8 encoding consistent with configs/default.yaml
2. THE Perturbation_Config SHALL define three Presets named "light", "medium", and "heavy", where "light" applies minimal perturbation (e.g., max_crop_percent ≤ 5, speed_drift_range ≤ 0.02, max_zoom ≤ 1.05), "medium" applies moderate perturbation (e.g., max_crop_percent ≤ 15, speed_drift_range ≤ 0.05, max_zoom ≤ 1.15), and "heavy" applies aggressive perturbation (e.g., max_crop_percent ≤ 30, speed_drift_range ≤ 0.10, max_zoom ≤ 1.30)
3. THE Perturbation_Config SHALL include configurable parameters with the following valid ranges: max_crop_percent [0, 50] (percentage of frame), speed_drift_range [0.0, 0.10] (deviation from 1.0x speed, yielding playback between 0.9x and 1.1x), max_zoom [1.0, 2.0] (zoom multiplier), change_interval [0.5, 30.0] (seconds between perturbation changes), combo_intensity_factor [0.0, 1.0] (scaling factor when multiple perturbations combine), max_frame_drop_percent [0, 20] (percentage of frames to drop), eq_range_db [-12.0, 12.0] (audio equalization adjustment in decibels), ambience_volume_db [-40.0, 0.0] (ambient noise volume in decibels), transition_duration [0.1, 5.0] (seconds for perturbation transitions), and max_scene_duration [1.0, 60.0] (seconds before forcing a scene change)
4. WHEN a Preset is selected via the "preset" key in the configuration, THE Perturbation_Engine SHALL load all parameter values defined in that Preset as defaults, then apply any individual parameter overrides specified alongside the preset key, with overrides taking precedence over Preset values
5. IF any configuration parameter value falls outside its defined valid range, THEN THE Perturbation_Engine SHALL reject the entire configuration before processing begins and report a validation error message indicating the parameter name, the provided value, and the acceptable range
6. IF the "preset" key specifies a value other than "light", "medium", or "heavy", THEN THE Perturbation_Engine SHALL reject the configuration and report a validation error message indicating the invalid preset name and the list of valid preset names
7. IF the perturbation configuration file is missing or contains invalid YAML syntax, THEN THE Perturbation_Engine SHALL halt startup and report an error message indicating the file path and the nature of the parsing failure

### Requirement 8: GUI Integration

**User Story:** As a QA engineer, I want a button in the GUI to trigger perturbation on a video, so that I can use the feature without command-line knowledge.

#### Acceptance Criteria

1. THE GUI SHALL display a "Perturb Video" button in the button frame alongside the existing "Start" button
2. IF no video file is selected, THEN THE GUI SHALL disable the "Perturb Video" button
3. WHEN the "Perturb Video" button is clicked, THE GUI SHALL display a modal dialog containing a selection control (radio buttons or dropdown) with preset options "light", "medium", and "heavy", a "Custom" option that exposes numeric fields for the perturbation parameters defined in the configuration, and "OK" and "Cancel" buttons
4. WHEN the user clicks "Cancel" in the preset dialog, THE GUI SHALL close the dialog without starting perturbation
5. WHEN the user clicks "OK" in the preset dialog with a valid selection, THE GUI SHALL disable both the "Perturb Video" button and the "Start" button, clear the log area, and begin the perturbation operation in a background thread
6. WHILE perturbation is running, THE GUI SHALL display the current stage name in the stage label and update the progress bar to reflect percentage complete, polling the message queue at intervals no greater than 200 milliseconds
7. WHEN perturbation completes successfully, THE GUI SHALL display the output file path in the log area, re-enable the "Perturb Video" and "Start" buttons, and set the stage label to "Done!"
8. IF perturbation fails, THEN THE GUI SHALL display an error message indicating the failure reason in the log area, re-enable the "Perturb Video" and "Start" buttons, and set the stage label to "Error"

### Requirement 9: CLI Integration

**User Story:** As a QA engineer, I want to trigger perturbation from the command line, so that I can automate batch testing of the reup detection system.

#### Acceptance Criteria

1. THE CLI SHALL accept a --perturb flag that is mutually exclusive with translation mode, such that specifying --perturb disables translation processing and enables perturbation processing
2. THE CLI SHALL accept a --perturb-preset argument with values limited to light, medium, or heavy, defaulting to medium when --perturb is specified without --perturb-preset
3. THE CLI SHALL accept a --perturb-config argument that takes a file path string (maximum 4096 characters) pointing to a custom Perturbation_Config YAML file
4. WHEN --perturb is specified with --input and --output, THE CLI SHALL run the Perturbation_Engine on the input video and write the perturbed result to the output path, returning exit code 0 on success
5. IF --perturb is specified without --input, THEN THE CLI SHALL display an error message indicating that input path is required and return a non-zero exit code
6. IF --perturb is specified without --output, THEN THE CLI SHALL display an error message indicating that output path is required and return a non-zero exit code
7. IF --perturb-config is specified with a path that does not exist or is not a valid YAML file, THEN THE CLI SHALL display an error message indicating the configuration file problem and return a non-zero exit code
8. IF --perturb-preset is specified with a value other than light, medium, or heavy, THEN THE CLI SHALL display an error message indicating the valid preset options and return a non-zero exit code
