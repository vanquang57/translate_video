# Design Document: Video Perturbation

## Overview

The Video Perturbation module is an independent processing pipeline that applies dynamic (time-varying) transformations to video files. Its purpose is to generate subtly altered video variants that defeat reup detection systems by modifying temporal, spatial, and audio fingerprints.

The module integrates into the existing Video Text Translator project as a peer to the translation pipeline — sharing infrastructure (FFmpegEncoder, ProgressReporter, YAML config loading) but operating independently with its own config, pipeline class, and entry points.

**Key Design Decisions:**
- Audio processing via **pydub + scipy** (lightweight, sufficient for EQ/tempo manipulation)
- Scene detection via **self-implemented OpenCV histogram comparison** (avoids extra dependency)
- Reproducibility via **optional seed in config** (default: random)
- Output naming: GUI uses `{name}_perturbed_{preset}.mp4`, CLI uses `--output`
- Runs independently on any video (not tied to translation pipeline)

## Architecture

```mermaid
graph TD
    subgraph Entry Points
        CLI[main.py --perturb]
        GUI[gui.py "Perturb Video" button]
    end

    subgraph Perturbation Module
        PC[PerturbationConfig]
        PP[PerturbationPipeline]
        PS[ParameterScheduler]
        TD[TemporalDriftProcessor]
        ST[SpatialTransformProcessor]
        SR[SceneRecompositionProcessor]
        AP[AudioPerturbationProcessor]
        MC[MultiTransformCombo]
    end

    subgraph Shared Infrastructure
        ENC[FFmpegEncoder]
        PROG[ProgressReporter]
        YAML[YAML Config Loader]
    end

    CLI --> PC
    GUI --> PC
    PC --> PP
    PP --> PS
    PP --> TD
    PP --> ST
    PP --> SR
    PP --> AP
    PP --> MC
    PP --> ENC
    PP --> PROG
    PC --> YAML
```

The pipeline processes video in a single pass for Temporal_Drift and Spatial_Transform (frame-by-frame), with Scene_Recomposition as a pre-processing step (requires full scene analysis first) and Audio_Perturbation as a post-processing step (operates on the extracted audio track independently).

**Processing Order:**
1. Input validation + video probe
2. Scene detection (if Scene_Recomposition enabled)
3. Scene reordering + transition insertion (if applicable)
4. Frame-by-frame processing: Temporal_Drift → Spatial_Transform
5. Audio extraction + Audio_Perturbation (parallel with step 4 when possible)
6. Mux video + audio → final output

## Components and Interfaces

### PerturbationConfig (dataclass)

```python
@dataclass(frozen=True, slots=True)
class PerturbationConfig:
    """Immutable configuration for the perturbation pipeline."""
    input_path: str
    output_path: str
    preset: Literal["light", "medium", "heavy"] = "medium"
    seed: int | None = None  # None = random

    # Temporal Drift
    temporal_enabled: bool = True
    speed_min: float = 0.99        # [0.90, 1.0]
    speed_max: float = 1.01        # [1.0, 1.10]
    max_frame_drop_percent: float = 1.0  # [0, 20]
    micro_offset_ms: float = 50.0  # [0, 500]

    # Spatial Transform
    spatial_enabled: bool = True
    max_crop_percent: float = 5.0  # [0, 50]
    max_zoom: float = 1.05         # [1.0, 2.0]

    # Scene Recomposition
    scene_enabled: bool = False
    scene_threshold: float = 0.3   # [0.1, 1.0]
    max_scene_duration: float = 5.0  # [1.0, 60.0]
    transition_duration_ms: float = 500.0  # [100, 2000]

    # Audio Perturbation
    audio_enabled: bool = True
    audio_tempo_min: float = 0.99  # [0.90, 1.0]
    audio_tempo_max: float = 1.01  # [1.0, 1.10]
    eq_range_db: float = 2.0       # [0, 12.0]
    ambience_volume_min_db: float = -40.0  # [-60, 0]
    ambience_volume_max_db: float = -30.0  # [-60, 0]

    # Multi-Transform Combo
    combo_enabled: bool = False
    combo_intensity_factor: float = 0.5  # [0.1, 1.0]

    # Timing
    change_interval: float = 10.0  # [0.5, 300.0] seconds

    # Performance
    encoder: str = "auto"
    encoder_preset: str = "fast"
```

### ParameterScheduler

Generates time-varying parameter values for each Change_Interval segment.

```python
class ParameterScheduler:
    """Generates parameter schedules for the entire video duration."""

    def __init__(self, duration: float, change_interval: float,
                 rng: random.Random) -> None: ...

    def schedule(self, param_min: float, param_max: float) -> list[SegmentParam]:
        """Generate a list of (start_time, end_time, value) for each segment."""
        ...

    def interpolate(self, schedule: list[SegmentParam], timestamp: float) -> float:
        """Get interpolated parameter value at a given timestamp.
        Uses linear interpolation over 500ms at boundaries."""
        ...
```

### TemporalDriftProcessor

```python
class TemporalDriftProcessor:
    """Applies speed variation and frame drop/duplicate."""

    def __init__(self, config: PerturbationConfig, rng: random.Random) -> None: ...

    def compute_frame_map(self, n_frames: int, fps: float) -> list[int]:
        """Compute output-to-input frame index mapping.
        Returns list where output_frame[i] = input_frame_index.
        Ensures duration preservation within 1%."""
        ...
```

### SpatialTransformProcessor

```python
class SpatialTransformProcessor:
    """Applies animated crop/zoom per frame."""

    def __init__(self, config: PerturbationConfig, width: int, height: int,
                 rng: random.Random) -> None: ...

    def transform_frame(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """Apply crop + zoom for the given timestamp, return frame at original resolution."""
        ...

    def compute_crop_region(self, timestamp: float) -> tuple[int, int, int, int]:
        """Compute (x, y, w, h) crop region for timestamp, clamped to frame bounds."""
        ...
```

### SceneRecompositionProcessor

```python
class SceneRecompositionProcessor:
    """Detects scenes and reorders micro-scenes."""

    def __init__(self, config: PerturbationConfig, rng: random.Random) -> None: ...

    def detect_scenes(self, video_path: str) -> list[SceneBoundary]:
        """Detect scene boundaries using histogram comparison."""
        ...

    def reorder_scenes(self, scenes: list[Scene]) -> list[Scene]:
        """Reorder micro-scenes with random permutation different from original."""
        ...

    def insert_transitions(self, scenes: list[Scene]) -> list[Scene]:
        """Insert cross-fade transitions between scenes."""
        ...
```

### AudioPerturbationProcessor

```python
class AudioPerturbationProcessor:
    """Applies tempo drift, EQ changes, and ambient noise overlay."""

    def __init__(self, config: PerturbationConfig, rng: random.Random) -> None: ...

    def process(self, audio_path: str, output_path: str) -> None:
        """Process entire audio track with time-varying perturbations."""
        ...

    def apply_tempo(self, segment: AudioSegment, factor: float) -> AudioSegment:
        """Apply tempo change without pitch shift using pydub + scipy."""
        ...

    def apply_eq(self, segment: AudioSegment, bass_db: float,
                 mid_db: float, treble_db: float) -> AudioSegment:
        """Apply 3-band EQ using scipy filters."""
        ...
```

### MultiTransformCombo

```python
class MultiTransformCombo:
    """Orchestrates combining multiple perturbation groups."""

    def __init__(self, config: PerturbationConfig, rng: random.Random) -> None: ...

    def select_groups(self) -> list[str]:
        """Select 2-3 groups randomly from available groups."""
        ...

    def scale_intensity(self, base_value: float, param_min: float,
                       param_max: float) -> float:
        """Scale parameter intensity by combo_intensity_factor."""
        ...
```

### PerturbationPipeline

```python
class PerturbationPipeline:
    """Main orchestrator for the perturbation process."""

    def __init__(self, config: PerturbationConfig,
                 progress: ProgressReporter | None = None) -> None: ...

    def run(self) -> int:
        """Execute full perturbation pipeline. Returns exit code."""
        ...
```

## Data Models

### Configuration Presets

```python
PRESETS: dict[str, dict[str, Any]] = {
    "light": {
        "speed_min": 0.995, "speed_max": 1.005,
        "max_frame_drop_percent": 0.5,
        "micro_offset_ms": 20.0,
        "max_crop_percent": 3.0,
        "max_zoom": 1.03,
        "audio_tempo_min": 0.995, "audio_tempo_max": 1.005,
        "eq_range_db": 1.0,
        "ambience_volume_min_db": -40.0, "ambience_volume_max_db": -35.0,
        "change_interval": 15.0,
    },
    "medium": {
        "speed_min": 0.99, "speed_max": 1.01,
        "max_frame_drop_percent": 1.0,
        "micro_offset_ms": 50.0,
        "max_crop_percent": 5.0,
        "max_zoom": 1.05,
        "audio_tempo_min": 0.99, "audio_tempo_max": 1.01,
        "eq_range_db": 2.0,
        "ambience_volume_min_db": -38.0, "ambience_volume_max_db": -30.0,
        "change_interval": 10.0,
    },
    "heavy": {
        "speed_min": 0.97, "speed_max": 1.03,
        "max_frame_drop_percent": 3.0,
        "micro_offset_ms": 100.0,
        "max_crop_percent": 10.0,
        "max_zoom": 1.15,
        "audio_tempo_min": 0.97, "audio_tempo_max": 1.03,
        "eq_range_db": 4.0,
        "ambience_volume_min_db": -35.0, "ambience_volume_max_db": -25.0,
        "change_interval": 5.0,
    },
}
```

### SegmentParam

```python
@dataclass(frozen=True, slots=True)
class SegmentParam:
    """A parameter value for one time segment."""
    start_time: float
    end_time: float
    value: float
```

### SceneBoundary / Scene

```python
@dataclass(frozen=True, slots=True)
class SceneBoundary:
    """A detected scene boundary."""
    frame_index: int
    timestamp: float
    histogram_diff: float

@dataclass(frozen=True, slots=True)
class Scene:
    """A video scene between two boundaries."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    is_micro: bool  # duration < max_scene_duration
```

### YAML Config File Structure (configs/perturbation.yaml)

```yaml
# Perturbation configuration
preset: medium          # light | medium | heavy
seed: null              # null = random, integer = reproducible

# Override individual parameters (takes precedence over preset)
temporal:
  enabled: true
  speed_min: 0.99
  speed_max: 1.01
  max_frame_drop_percent: 1.0
  micro_offset_ms: 50.0

spatial:
  enabled: true
  max_crop_percent: 5.0
  max_zoom: 1.05

scene:
  enabled: false
  scene_threshold: 0.3
  max_scene_duration: 5.0
  transition_duration_ms: 500.0

audio:
  enabled: true
  tempo_min: 0.99
  tempo_max: 1.01
  eq_range_db: 2.0
  ambience_volume_min_db: -40.0
  ambience_volume_max_db: -30.0

combo:
  enabled: false
  intensity_factor: 0.5

timing:
  change_interval: 10.0

performance:
  encoder: auto
  encoder_preset: fast
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Parameter scheduling produces values within configured bounds

*For any* video duration, change_interval, and parameter min/max range, all values generated by the ParameterScheduler SHALL fall within [param_min, param_max] inclusive, and the schedule SHALL contain exactly ⌈duration / change_interval⌉ segments (or 1 segment if change_interval > duration).

**Validates: Requirements 1.2, 1.3, 2.1, 2.3, 3.1, 3.2, 5.1, 5.2, 5.3, 5.4**

### Property 2: Duration preservation invariant

*For any* input video with N frames at F fps, and any valid Temporal_Drift configuration (speed_min, speed_max, max_frame_drop_percent), the frame map produced by TemporalDriftProcessor SHALL yield an output frame count whose implied duration is within 1% of the original duration (N/F).

**Validates: Requirements 1.4, 2.5**

### Property 3: Configuration validation rejects out-of-range values

*For any* PerturbationConfig where at least one parameter value falls outside its defined valid range, the config validator SHALL raise a validation error whose message contains the parameter name and the acceptable range. Conversely, for any config where all values are within valid ranges, validation SHALL succeed.

**Validates: Requirements 1.5, 1.7, 7.3, 7.5**

### Property 4: Frame drop rate and non-consecutiveness

*For any* frame sequence of length N and max_frame_drop_percent P, the TemporalDriftProcessor SHALL drop/duplicate at most ⌊N × P / 100⌋ frames, and no two consecutively indexed frames in the output SHALL both be affected (dropped or duplicated).

**Validates: Requirements 2.2**

### Property 5: Linear interpolation correctness at boundaries

*For any* two consecutive SegmentParam values (v1, v2) and any timestamp t within the 500ms transition window at their boundary, the interpolated value SHALL equal v1 + (v2 - v1) × ((t - boundary_start) / 500ms), and for timestamps outside the transition window, the value SHALL equal the current segment's value exactly.

**Validates: Requirements 2.4**

### Property 6: Spatial drift rate limit

*For any* two consecutive frames at timestamps t1 and t2 in a video of dimensions W×H, the horizontal displacement of the crop center SHALL not exceed 0.005 × W × (t2 - t1) pixels, and the vertical displacement SHALL not exceed 0.005 × H × (t2 - t1) pixels.

**Validates: Requirements 3.3**

### Property 7: Output resolution invariant

*For any* input frame of dimensions W×H and any valid Spatial_Transform parameters (max_crop_percent, max_zoom), the output frame from SpatialTransformProcessor.transform_frame() SHALL have dimensions exactly W×H.

**Validates: Requirements 3.4**

### Property 8: Crop region clamping

*For any* frame of dimensions W×H and any computed crop region (x, y, w, h), after clamping the region SHALL satisfy: x >= 0, y >= 0, x + w <= W, y + h <= H, and w > 0, h > 0.

**Validates: Requirements 3.5**

### Property 9: Scene boundary detection correctness

*For any* sequence of frame histogram differences and a threshold T, the scene detector SHALL mark a boundary at frame i if and only if diff[i] > T.

**Validates: Requirements 4.1**

### Property 10: Scene reorder produces valid permutation

*For any* list of micro-scenes with length >= 2, the reordered list SHALL contain exactly the same set of scenes (same elements, possibly different order), and the ordering SHALL differ from the original in at least one position.

**Validates: Requirements 4.2**

### Property 11: Transition duration clamping

*For any* two adjacent scenes with durations d1 and d2, and a configured transition_duration T, the actual applied transition duration SHALL be min(T, d1, d2).

**Validates: Requirements 4.3**

### Property 12: Combo group selection count

*For any* Multi_Transform_Combo configuration with at least 2 available perturbation groups, the selected group count SHALL be >= 2 and <= 3.

**Validates: Requirements 6.1**

### Property 13: Combo intensity scaling

*For any* standalone parameter value V within range [min, max] and combo_intensity_factor F, the effective parameter value in combo mode SHALL equal min + (V - min) × F, keeping the result within [min, max].

**Validates: Requirements 6.2**

### Property 14: Preset override precedence

*For any* preset name and any set of parameter overrides, the final PerturbationConfig SHALL have the override value for each overridden parameter, and the preset's default value for all non-overridden parameters.

**Validates: Requirements 7.4**

## Error Handling

### Input Validation Errors

| Condition | Error Type | Message Format |
|-----------|-----------|----------------|
| File not found / unreadable | `InvalidInputError` | `"Input file not found or unreadable: {path}"` |
| Unsupported format | `InvalidInputError` | `"Unsupported video format: {path} (supported: MP4, AVI, MOV, MKV, WebM)"` |
| Corrupted video | `InvalidInputError` | `"Cannot open video (corrupted or invalid): {path}"` |
| Output dir not writable | `InvalidInputError` | `"Output directory not writable: {dir_path}"` |
| Invalid config parameter | `InvalidConfigError` | `"Invalid {param_name}: {value} (valid range: {min}-{max})"` |
| Invalid preset name | `InvalidConfigError` | `"Invalid preset '{name}' (valid: light, medium, heavy)"` |
| Missing/malformed YAML | `InvalidConfigError` | `"Failed to parse perturbation config {path}: {reason}"` |

### Runtime Errors

| Condition | Behavior |
|-----------|----------|
| No audio track | Skip Audio_Perturbation, log warning, continue |
| < 2 scenes detected | Skip Scene_Recomposition, log warning, continue |
| < 2 micro-scenes | Skip reordering, log warning, output unchanged |
| Perturbation group fails in combo | Skip failed group, continue with remaining, log which group was skipped |
| FFmpeg encoder failure | Raise `OutputWriteError`, abort pipeline |
| Disk full during write | Raise `OutputWriteError`, clean up partial output |

### Error Propagation

- Validation errors are raised **before** any processing begins
- Runtime errors in individual processors are caught by the pipeline orchestrator
- The pipeline returns exit codes: 0 (success), 1 (config error), 2 (input error), 3 (processing error), 4 (unexpected crash)
- GUI receives errors via the message queue and displays them in the log area
- CLI prints errors to stderr and returns non-zero exit code

## Testing Strategy

### Property-Based Tests (Hypothesis)

The project will use **Hypothesis** for property-based testing. Each property from the Correctness Properties section maps to one property-based test with minimum 100 iterations.

**Test file:** `tests/property/test_perturbation_properties.py`

| Property | Test Function | Key Generators |
|----------|--------------|----------------|
| 1: Parameter scheduling bounds | `test_parameter_schedule_within_bounds` | `st.floats` for duration/interval/min/max |
| 2: Duration preservation | `test_duration_preservation` | `st.integers` for frame count, `st.floats` for fps/speed range |
| 3: Config validation | `test_config_validation_rejects_invalid` | Custom strategy generating configs with one invalid field |
| 4: Frame drop constraints | `test_frame_drop_constraints` | `st.integers` for frame count, `st.floats` for drop percent |
| 5: Interpolation correctness | `test_interpolation_at_boundaries` | `st.floats` for values and timestamps |
| 6: Drift rate limit | `test_spatial_drift_rate_limit` | `st.integers` for dimensions, `st.floats` for timestamps |
| 7: Output resolution | `test_output_resolution_invariant` | `st.integers` for W/H, `st.floats` for crop/zoom params |
| 8: Crop clamping | `test_crop_region_clamping` | `st.integers` for dimensions, `st.floats` for crop params |
| 9: Scene detection | `test_scene_boundary_detection` | `st.lists(st.floats)` for histogram diffs, `st.floats` for threshold |
| 10: Scene reorder | `test_scene_reorder_permutation` | `st.lists` of scenes (min length 2) |
| 11: Transition clamping | `test_transition_duration_clamping` | `st.floats` for scene durations and transition config |
| 12: Combo group count | `test_combo_group_selection` | `st.booleans` for group availability |
| 13: Intensity scaling | `test_combo_intensity_scaling` | `st.floats` for values, ranges, and factor |
| 14: Preset override | `test_preset_override_precedence` | Custom strategy for preset + overrides |

Each test is tagged: `# Feature: video-perturbation, Property {N}: {title}`

Configuration: `@settings(max_examples=200, deadline=None)`

### Unit Tests

**Test file:** `tests/unit/test_perturbation.py`

- Preset loading (light/medium/heavy produce expected values)
- CLI argument parsing (--perturb, --perturb-preset, --perturb-config)
- Error messages contain required information (file path, param name, valid range)
- Scene detection with known histogram sequences
- Audio skip when no audio track present
- Combo fixed ordering (Temporal → Spatial → Audio)
- Combo graceful degradation when a group fails

### Integration Tests

**Test file:** `tests/integration/test_perturbation_integration.py`

- End-to-end pipeline with a short test video (verify output is valid video)
- GUI button state management (enabled/disabled based on file selection)
- CLI exit codes for success and various error conditions

### Dependencies

Add to `requirements.txt`:
```
pydub>=0.25.1
scipy>=1.11.0
hypothesis>=6.90.0
```

Note: `pydub` requires FFmpeg on PATH (already a project dependency). `scipy` is used for audio filtering (bandpass/EQ). `hypothesis` is dev-only for property tests.
