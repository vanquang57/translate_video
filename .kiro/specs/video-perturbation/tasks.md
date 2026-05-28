# Implementation Plan: Video Perturbation

## Overview

Implement the Video Perturbation module as an independent processing pipeline that applies dynamic (time-varying) transformations to video files. The implementation follows the architecture defined in the design: core data models and config first, then individual processors (Temporal, Spatial, Scene, Audio, Combo), then the pipeline orchestrator, and finally CLI/GUI integration. All new code lives in `src/video_text_translator/` and reuses existing infrastructure (FFmpegEncoder, ProgressReporter, YAML config loading).

## Tasks

- [x] 1. Set up project structure, dependencies, and core data models
  - [x] 1.1 Add dependencies to requirements.txt and create perturbation config YAML
    - Add `pydub>=0.25.1`, `scipy>=1.11.0`, `hypothesis>=6.90.0` to `requirements.txt`
    - Create `configs/perturbation.yaml` with full YAML structure (presets, all parameter sections with comments and range annotations) matching the design's YAML Config File Structure
    - _Requirements: 7.1, 7.2, 7.3_

  - [x] 1.2 Create PerturbationConfig dataclass and validation logic
    - Create `src/video_text_translator/perturbation_config.py`
    - Implement `PerturbationConfig` frozen dataclass with all fields from design (temporal, spatial, scene, audio, combo, timing, performance)
    - Implement `PRESETS` dict with light/medium/heavy values
    - Implement `SegmentParam`, `SceneBoundary`, `Scene` dataclasses
    - Implement `validate_config()` that checks all parameter ranges and raises `InvalidConfigError` with parameter name and valid range
    - Implement `load_perturbation_config(yaml_path, preset_override, param_overrides)` that loads YAML, applies preset defaults, then applies overrides
    - _Requirements: 1.7, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

  - [ ]* 1.3 Write property test for config validation (Property 3)
    - **Property 3: Configuration validation rejects out-of-range values**
    - **Validates: Requirements 1.5, 1.7, 7.3, 7.5**

  - [ ]* 1.4 Write property test for preset override precedence (Property 14)
    - **Property 14: Preset override precedence**
    - **Validates: Requirements 7.4**

- [x] 2. Implement ParameterScheduler
  - [x] 2.1 Create ParameterScheduler class
    - Create `src/video_text_translator/perturbation_scheduler.py`
    - Implement `ParameterScheduler.__init__(duration, change_interval, rng)`
    - Implement `schedule(param_min, param_max) -> list[SegmentParam]` generating ⌈duration / change_interval⌉ segments with random values in [min, max]
    - Implement `interpolate(schedule, timestamp) -> float` with 500ms linear interpolation at boundaries
    - Handle edge case: change_interval > duration → single segment
    - _Requirements: 1.2, 1.3, 2.4_

  - [ ]* 2.2 Write property test for parameter scheduling bounds (Property 1)
    - **Property 1: Parameter scheduling produces values within configured bounds**
    - **Validates: Requirements 1.2, 1.3, 2.1, 2.3, 3.1, 3.2, 5.1, 5.2, 5.3, 5.4**

  - [ ]* 2.3 Write property test for interpolation correctness (Property 5)
    - **Property 5: Linear interpolation correctness at boundaries**
    - **Validates: Requirements 2.4**

- [x] 3. Implement TemporalDriftProcessor
  - [x] 3.1 Create TemporalDriftProcessor class
    - Create `src/video_text_translator/perturbation_temporal.py`
    - Implement `TemporalDriftProcessor.__init__(config, rng)`
    - Implement `compute_frame_map(n_frames, fps) -> list[int]` that:
      - Uses ParameterScheduler for speed variation per segment
      - Applies frame drop/duplicate at configured max percentage
      - Ensures no two consecutive frames are both affected
      - Applies micro time offsets at segment boundaries
      - Constrains output to preserve duration within 1%
    - _Requirements: 2.1, 2.2, 2.3, 2.5_

  - [ ]* 3.2 Write property test for duration preservation (Property 2)
    - **Property 2: Duration preservation invariant**
    - **Validates: Requirements 1.4, 2.5**

  - [ ]* 3.3 Write property test for frame drop constraints (Property 4)
    - **Property 4: Frame drop rate and non-consecutiveness**
    - **Validates: Requirements 2.2**

- [x] 4. Implement SpatialTransformProcessor
  - [x] 4.1 Create SpatialTransformProcessor class
    - Create `src/video_text_translator/perturbation_spatial.py`
    - Implement `SpatialTransformProcessor.__init__(config, width, height, rng)`
    - Implement `compute_crop_region(timestamp) -> tuple[int, int, int, int]` with animated crop using ParameterScheduler, drift rate ≤ 0.5% per second, clamped to frame bounds
    - Implement `transform_frame(frame, timestamp) -> np.ndarray` that crops, zooms, and scales back to original resolution
    - Handle skip condition: max_crop_percent == 0 and max_zoom == 1.0
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ]* 4.2 Write property test for spatial drift rate limit (Property 6)
    - **Property 6: Spatial drift rate limit**
    - **Validates: Requirements 3.3**

  - [ ]* 4.3 Write property test for output resolution invariant (Property 7)
    - **Property 7: Output resolution invariant**
    - **Validates: Requirements 3.4**

  - [ ]* 4.4 Write property test for crop region clamping (Property 8)
    - **Property 8: Crop region clamping**
    - **Validates: Requirements 3.5**

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement SceneRecompositionProcessor
  - [x] 6.1 Create SceneRecompositionProcessor class
    - Create `src/video_text_translator/perturbation_scene.py`
    - Implement `SceneRecompositionProcessor.__init__(config, rng)`
    - Implement `detect_scenes(video_path) -> list[SceneBoundary]` using OpenCV histogram comparison (frame-to-frame diff > threshold)
    - Implement `reorder_scenes(scenes) -> list[Scene]` with random permutation different from original, filtering micro-scenes by max_scene_duration
    - Implement `insert_transitions(scenes) -> list[Scene]` with cross-fade transitions clamped to min(transition_duration, scene1_duration, scene2_duration)
    - Handle edge cases: < 2 scenes → skip with warning; < 2 micro-scenes → skip reordering with warning
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ]* 6.2 Write property test for scene boundary detection (Property 9)
    - **Property 9: Scene boundary detection correctness**
    - **Validates: Requirements 4.1**

  - [ ]* 6.3 Write property test for scene reorder permutation (Property 10)
    - **Property 10: Scene reorder produces valid permutation**
    - **Validates: Requirements 4.2**

  - [ ]* 6.4 Write property test for transition duration clamping (Property 11)
    - **Property 11: Transition duration clamping**
    - **Validates: Requirements 4.3**

- [x] 7. Implement AudioPerturbationProcessor
  - [x] 7.1 Create AudioPerturbationProcessor class
    - Create `src/video_text_translator/perturbation_audio.py`
    - Implement `AudioPerturbationProcessor.__init__(config, rng)`
    - Implement `process(audio_path, output_path)` that processes entire audio track with time-varying perturbations per Change_Interval
    - Implement `apply_tempo(segment, factor) -> AudioSegment` using pydub + scipy for tempo change without pitch shift
    - Implement `apply_eq(segment, bass_db, mid_db, treble_db) -> AudioSegment` using scipy bandpass filters (bass: 20-250Hz, mid: 250-4000Hz, treble: 4000-20000Hz)
    - Implement ambient noise overlay at configured volume range
    - Handle edge cases: audio shorter than one Change_Interval → single parameter set; no audio track → skip with warning
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [x] 8. Implement MultiTransformCombo
  - [x] 8.1 Create MultiTransformCombo class
    - Create `src/video_text_translator/perturbation_combo.py`
    - Implement `MultiTransformCombo.__init__(config, rng)`
    - Implement `select_groups() -> list[str]` selecting 2-3 groups from available (Temporal_Drift, Spatial_Transform, Audio_Perturbation)
    - Implement `scale_intensity(base_value, param_min, param_max) -> float` applying combo_intensity_factor: `min + (value - min) * factor`
    - Ensure fixed application order: Temporal → Spatial → Audio
    - Handle graceful degradation: if a group fails, skip it and continue with remaining
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [ ]* 8.2 Write property test for combo group selection count (Property 12)
    - **Property 12: Combo group selection count**
    - **Validates: Requirements 6.1**

  - [ ]* 8.3 Write property test for combo intensity scaling (Property 13)
    - **Property 13: Combo intensity scaling**
    - **Validates: Requirements 6.2**

- [x] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement PerturbationPipeline orchestrator
  - [x] 10.1 Create PerturbationPipeline class
    - Create `src/video_text_translator/perturbation_pipeline.py`
    - Implement `PerturbationPipeline.__init__(config, progress)` initializing all processors based on config flags
    - Implement `run() -> int` orchestrating the full pipeline:
      1. Input validation + video probe (reuse patterns from existing Pipeline)
      2. Scene detection (if scene_enabled)
      3. Scene reordering + transition insertion (if applicable)
      4. Frame-by-frame processing: TemporalDrift frame map → SpatialTransform per frame
      5. Audio extraction + AudioPerturbation (parallel when possible)
      6. Mux video + audio via FFmpegEncoder → final output
    - Implement error handling: exit codes 0/1/2/3/4, graceful skip on non-critical failures
    - Use existing FFmpegEncoder for video output
    - Use existing ProgressReporter for progress updates
    - Handle seed: if config.seed is None, use random; otherwise use specified seed for reproducibility
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.4, 2.5_

  - [ ]* 10.2 Write unit tests for PerturbationPipeline
    - Test error handling: invalid input returns exit code 2, config error returns 1
    - Test skip behavior: no audio → skip audio perturbation, < 2 scenes → skip scene recomposition
    - Test seed reproducibility: same seed produces same output
    - _Requirements: 1.5, 1.6, 4.4, 4.5, 5.5_

- [x] 11. Implement CLI integration
  - [x] 11.1 Add perturbation CLI arguments and entry point
    - Modify `src/video_text_translator/config.py`: add `--perturb`, `--perturb-preset`, `--perturb-config` arguments to `build_argparser()`
    - Make `--perturb` mutually exclusive with translation mode
    - Modify `main.py`: detect `--perturb` flag, load perturbation config, instantiate and run PerturbationPipeline instead of translation Pipeline
    - Default preset to "medium" when `--perturb` specified without `--perturb-preset`
    - Validate `--perturb-config` path exists and is valid YAML
    - Return appropriate exit codes
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_

  - [ ]* 11.2 Write unit tests for CLI perturbation arguments
    - Test `--perturb` flag parsing and mutual exclusivity
    - Test `--perturb-preset` validation (light/medium/heavy/invalid)
    - Test `--perturb-config` with valid and invalid paths
    - Test error messages for missing --input and --output
    - _Requirements: 9.1, 9.2, 9.3, 9.5, 9.6, 9.7, 9.8_

- [x] 12. Implement GUI integration
  - [x] 12.1 Add "Perturb Video" button and preset dialog to GUI
    - Modify `gui.py`: add "Perturb Video" button in button frame next to "Start"
    - Implement preset selection dialog (modal) with radio buttons for light/medium/heavy + Custom option with numeric fields
    - Implement button state management: disabled when no file selected, disabled during processing
    - Implement background thread execution using existing GuiProgressReporter pattern
    - Implement progress display: stage label + progress bar polling at ≤ 200ms
    - Implement completion handling: show output path on success, show error on failure, re-enable buttons
    - Output naming: `{name}_perturbed_{preset}.mp4`
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

- [x] 13. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- All new processor files are created in `src/video_text_translator/` following existing project conventions
- The implementation reuses FFmpegEncoder, ProgressReporter, and YAML config loading patterns already in the project
- Python 3.12 on Windows; FFmpeg must be on PATH (already a project dependency)

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["1.3", "1.4", "2.1"] },
    { "id": 3, "tasks": ["2.2", "2.3", "3.1", "4.1"] },
    { "id": 4, "tasks": ["3.2", "3.3", "4.2", "4.3", "4.4", "6.1", "7.1", "8.1"] },
    { "id": 5, "tasks": ["6.2", "6.3", "6.4", "8.2", "8.3"] },
    { "id": 6, "tasks": ["10.1"] },
    { "id": 7, "tasks": ["10.2", "11.1"] },
    { "id": 8, "tasks": ["11.2", "12.1"] }
  ]
}
```
