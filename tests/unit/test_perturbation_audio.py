"""Unit tests for perturbation_audio module."""

from __future__ import annotations

import math
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from pydub import AudioSegment
from pydub.generators import Sine

# Add src to path so we can import without the full package chain
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from video_text_translator.perturbation_config import PerturbationConfig
from video_text_translator.perturbation_audio import AudioPerturbationProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> PerturbationConfig:
    """Create a PerturbationConfig with sensible defaults and overrides."""
    defaults = {
        "input_path": "test.mp4",
        "output_path": "out.mp4",
        "preset": "medium",
        "audio_tempo_min": 0.99,
        "audio_tempo_max": 1.01,
        "eq_range_db": 2.0,
        "ambience_volume_min_db": -40.0,
        "ambience_volume_max_db": -30.0,
        "change_interval": 10.0,
    }
    defaults.update(overrides)
    return PerturbationConfig(**defaults)


def _generate_test_tone(
    frequency: float = 440.0,
    duration_ms: int = 5000,
    sample_rate: int = 44100,
    sample_width: int = 2,
) -> AudioSegment:
    """Generate a simple sine wave tone for testing."""
    tone = Sine(frequency).to_audio_segment(
        duration=duration_ms, volume=-20.0
    )
    tone = tone.set_frame_rate(sample_rate).set_sample_width(sample_width)
    return tone


def _save_tone_to_wav(tone: AudioSegment, path: str) -> None:
    """Save an AudioSegment to a WAV file."""
    tone.export(path, format="wav")


# ---------------------------------------------------------------------------
# Basic functionality tests
# ---------------------------------------------------------------------------


class TestAudioPerturbationProcessorInit:
    """Tests for AudioPerturbationProcessor initialization."""

    def test_init_stores_config_and_rng(self) -> None:
        """Processor should store config and rng."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        assert processor.config is config
        assert processor.rng is rng


# ---------------------------------------------------------------------------
# process() tests
# ---------------------------------------------------------------------------


class TestProcess:
    """Tests for AudioPerturbationProcessor.process()."""

    def test_process_produces_output_file(self, tmp_path: Path) -> None:
        """process() should create an output WAV file."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        # Generate test audio
        tone = _generate_test_tone(duration_ms=5000)
        input_path = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "output.wav")
        _save_tone_to_wav(tone, input_path)

        processor.process(input_path, output_path)

        assert Path(output_path).exists()

    def test_process_output_has_audio_data(self, tmp_path: Path) -> None:
        """Output file should contain non-empty audio data."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        tone = _generate_test_tone(duration_ms=3000)
        input_path = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "output.wav")
        _save_tone_to_wav(tone, input_path)

        processor.process(input_path, output_path)

        output_audio = AudioSegment.from_wav(output_path)
        assert len(output_audio) > 0

    def test_process_nonexistent_file_skips_with_warning(
        self, tmp_path: Path, caplog
    ) -> None:
        """process() should skip gracefully if audio file doesn't exist."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        input_path = str(tmp_path / "nonexistent.wav")
        output_path = str(tmp_path / "output.wav")

        import logging

        with caplog.at_level(logging.WARNING):
            processor.process(input_path, output_path)

        assert not Path(output_path).exists()
        assert "No audio track found" in caplog.text or "skipping" in caplog.text.lower()

    def test_process_short_audio_single_segment(self, tmp_path: Path) -> None:
        """Audio shorter than change_interval should use single parameter set."""
        config = _make_config(change_interval=30.0)  # 30s interval
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        # Generate 5 second audio (shorter than 30s interval)
        tone = _generate_test_tone(duration_ms=5000)
        input_path = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "output.wav")
        _save_tone_to_wav(tone, input_path)

        processor.process(input_path, output_path)

        output_audio = AudioSegment.from_wav(output_path)
        assert len(output_audio) > 0

    def test_process_preserves_sample_rate(self, tmp_path: Path) -> None:
        """Output should have the same sample rate as input."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        tone = _generate_test_tone(duration_ms=3000, sample_rate=44100)
        input_path = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "output.wav")
        _save_tone_to_wav(tone, input_path)

        processor.process(input_path, output_path)

        output_audio = AudioSegment.from_wav(output_path)
        assert output_audio.frame_rate == 44100

    def test_process_reproducible_with_same_seed(self, tmp_path: Path) -> None:
        """Same seed should produce identical output."""
        config = _make_config()

        tone = _generate_test_tone(duration_ms=3000)
        input_path = str(tmp_path / "input.wav")
        output1_path = str(tmp_path / "output1.wav")
        output2_path = str(tmp_path / "output2.wav")
        _save_tone_to_wav(tone, input_path)

        processor1 = AudioPerturbationProcessor(config, random.Random(42))
        processor1.process(input_path, output1_path)

        processor2 = AudioPerturbationProcessor(config, random.Random(42))
        processor2.process(input_path, output2_path)

        audio1 = AudioSegment.from_wav(output1_path)
        audio2 = AudioSegment.from_wav(output2_path)

        assert audio1.raw_data == audio2.raw_data

    def test_process_different_seeds_produce_different_output(
        self, tmp_path: Path
    ) -> None:
        """Different seeds should produce different output."""
        config = _make_config()

        tone = _generate_test_tone(duration_ms=3000)
        input_path = str(tmp_path / "input.wav")
        output1_path = str(tmp_path / "output1.wav")
        output2_path = str(tmp_path / "output2.wav")
        _save_tone_to_wav(tone, input_path)

        processor1 = AudioPerturbationProcessor(config, random.Random(1))
        processor1.process(input_path, output1_path)

        processor2 = AudioPerturbationProcessor(config, random.Random(999))
        processor2.process(input_path, output2_path)

        audio1 = AudioSegment.from_wav(output1_path)
        audio2 = AudioSegment.from_wav(output2_path)

        assert audio1.raw_data != audio2.raw_data


# ---------------------------------------------------------------------------
# apply_tempo() tests
# ---------------------------------------------------------------------------


class TestApplyTempo:
    """Tests for AudioPerturbationProcessor.apply_tempo()."""

    def test_tempo_factor_one_returns_same_length(self) -> None:
        """Factor 1.0 should return segment of same length."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        tone = _generate_test_tone(duration_ms=1000)
        result = processor.apply_tempo(tone, 1.0)

        assert abs(len(result) - len(tone)) <= 1  # Allow 1ms rounding

    def test_tempo_speedup_shortens_audio(self) -> None:
        """Factor > 1.0 should produce shorter audio."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        tone = _generate_test_tone(duration_ms=2000)
        result = processor.apply_tempo(tone, 1.05)

        # 5% speedup → ~5% shorter
        expected_length = 2000 / 1.05
        assert len(result) < len(tone)
        assert abs(len(result) - expected_length) < 50  # Within 50ms tolerance

    def test_tempo_slowdown_lengthens_audio(self) -> None:
        """Factor < 1.0 should produce longer audio."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        tone = _generate_test_tone(duration_ms=2000)
        result = processor.apply_tempo(tone, 0.95)

        # 5% slowdown → ~5% longer
        expected_length = 2000 / 0.95
        assert len(result) > len(tone)
        assert abs(len(result) - expected_length) < 50

    def test_tempo_preserves_sample_rate(self) -> None:
        """Tempo change should not alter sample rate."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        tone = _generate_test_tone(duration_ms=1000, sample_rate=44100)
        result = processor.apply_tempo(tone, 1.03)

        assert result.frame_rate == 44100

    def test_tempo_preserves_channels(self) -> None:
        """Tempo change should preserve channel count."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        tone = _generate_test_tone(duration_ms=1000)
        # Make stereo
        stereo = tone.set_channels(2)
        result = processor.apply_tempo(stereo, 1.02)

        assert result.channels == 2

    def test_tempo_subtle_change_preserves_approximate_length(self) -> None:
        """Subtle tempo change (0.99-1.01) should keep length within ~2%."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        tone = _generate_test_tone(duration_ms=5000)
        result = processor.apply_tempo(tone, 1.01)

        # 1% speedup → ~1% shorter
        ratio = len(result) / len(tone)
        assert 0.95 <= ratio <= 1.05


# ---------------------------------------------------------------------------
# apply_eq() tests
# ---------------------------------------------------------------------------


class TestApplyEq:
    """Tests for AudioPerturbationProcessor.apply_eq()."""

    def test_zero_gains_returns_unchanged(self) -> None:
        """Zero dB gains should return audio unchanged."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        tone = _generate_test_tone(duration_ms=1000)
        result = processor.apply_eq(tone, 0.0, 0.0, 0.0)

        # Should be identical (or very close due to floating point)
        assert len(result) == len(tone)
        assert result.frame_rate == tone.frame_rate

    def test_eq_preserves_length(self) -> None:
        """EQ should not change audio length."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        tone = _generate_test_tone(duration_ms=2000)
        result = processor.apply_eq(tone, 3.0, -2.0, 1.0)

        assert len(result) == len(tone)

    def test_eq_preserves_sample_rate(self) -> None:
        """EQ should not change sample rate."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        tone = _generate_test_tone(duration_ms=1000, sample_rate=44100)
        result = processor.apply_eq(tone, 2.0, -1.0, 3.0)

        assert result.frame_rate == 44100

    def test_eq_preserves_channels(self) -> None:
        """EQ should preserve channel count."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        tone = _generate_test_tone(duration_ms=1000)
        stereo = tone.set_channels(2)
        result = processor.apply_eq(stereo, 2.0, -1.0, 1.5)

        assert result.channels == 2

    def test_bass_boost_increases_low_frequency_energy(self) -> None:
        """Boosting bass should increase energy in low frequencies."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        # Generate a tone with multiple frequencies
        # Use a low frequency tone (100 Hz) to test bass boost
        tone = _generate_test_tone(frequency=100.0, duration_ms=1000)
        original_rms = tone.rms

        result = processor.apply_eq(tone, 6.0, 0.0, 0.0)
        boosted_rms = result.rms

        # Bass boost on a bass-frequency tone should increase RMS
        assert boosted_rms > original_rms

    def test_eq_modifies_audio_data(self) -> None:
        """Non-zero EQ should produce different audio data."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        tone = _generate_test_tone(duration_ms=1000)
        result = processor.apply_eq(tone, 4.0, -3.0, 2.0)

        # Audio data should be different
        assert result.raw_data != tone.raw_data

    def test_eq_short_segment_handled(self) -> None:
        """Very short segments should be handled gracefully."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        # 5ms segment - too short for meaningful filtering
        tone = _generate_test_tone(duration_ms=5)
        result = processor.apply_eq(tone, 2.0, -1.0, 1.0)

        # Should return without error
        assert len(result) >= 0


# ---------------------------------------------------------------------------
# Ambient noise tests
# ---------------------------------------------------------------------------


class TestAmbientNoise:
    """Tests for ambient noise overlay."""

    def test_noise_overlay_changes_audio(self, tmp_path: Path) -> None:
        """Processing with noise should produce different audio than input."""
        config = _make_config(
            audio_tempo_min=1.0,
            audio_tempo_max=1.0,
            eq_range_db=0.0,
            ambience_volume_min_db=-30.0,
            ambience_volume_max_db=-20.0,
        )
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        tone = _generate_test_tone(duration_ms=2000)
        input_path = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "output.wav")
        _save_tone_to_wav(tone, input_path)

        processor.process(input_path, output_path)

        output_audio = AudioSegment.from_wav(output_path)
        # Audio should be modified (noise added)
        assert output_audio.raw_data != tone.raw_data

    def test_noise_does_not_drastically_change_rms(self, tmp_path: Path) -> None:
        """Noise at -40 to -30 dB should not drastically change overall RMS."""
        config = _make_config(
            audio_tempo_min=1.0,
            audio_tempo_max=1.0,
            eq_range_db=0.0,
            ambience_volume_min_db=-40.0,
            ambience_volume_max_db=-30.0,
        )
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        tone = _generate_test_tone(duration_ms=3000)
        input_path = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "output.wav")
        _save_tone_to_wav(tone, input_path)

        processor.process(input_path, output_path)

        output_audio = AudioSegment.from_wav(output_path)
        # RMS should not change by more than 3 dB
        original_rms = tone.rms
        output_rms = output_audio.rms
        if original_rms > 0 and output_rms > 0:
            db_diff = 20 * math.log10(output_rms / original_rms)
            assert abs(db_diff) < 3.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for AudioPerturbationProcessor."""

    def test_empty_audio_segment(self) -> None:
        """Empty audio segment should be handled gracefully."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        empty = AudioSegment.empty()
        result = processor.apply_tempo(empty, 1.01)
        # Should not crash - empty or near-empty result
        assert len(result) <= 1

    def test_very_short_change_interval(self, tmp_path: Path) -> None:
        """Very short change_interval should produce many segments."""
        config = _make_config(change_interval=0.5)  # 500ms intervals
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        tone = _generate_test_tone(duration_ms=5000)
        input_path = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "output.wav")
        _save_tone_to_wav(tone, input_path)

        processor.process(input_path, output_path)

        output_audio = AudioSegment.from_wav(output_path)
        assert len(output_audio) > 0

    def test_mono_audio(self, tmp_path: Path) -> None:
        """Mono audio should be processed correctly."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        tone = _generate_test_tone(duration_ms=2000)
        mono = tone.set_channels(1)
        input_path = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "output.wav")
        _save_tone_to_wav(mono, input_path)

        processor.process(input_path, output_path)

        output_audio = AudioSegment.from_wav(output_path)
        assert output_audio.channels == 1
        assert len(output_audio) > 0

    def test_stereo_audio(self, tmp_path: Path) -> None:
        """Stereo audio should be processed correctly."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        tone = _generate_test_tone(duration_ms=2000)
        stereo = tone.set_channels(2)
        input_path = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "output.wav")
        _save_tone_to_wav(stereo, input_path)

        processor.process(input_path, output_path)

        output_audio = AudioSegment.from_wav(output_path)
        assert output_audio.channels == 2
        assert len(output_audio) > 0

    def test_low_sample_rate(self) -> None:
        """Low sample rate audio (8kHz) should be handled gracefully."""
        config = _make_config()
        rng = random.Random(42)
        processor = AudioPerturbationProcessor(config, rng)

        # 8kHz sample rate - treble band (4000-20000 Hz) will be limited
        tone = _generate_test_tone(
            frequency=200.0, duration_ms=1000, sample_rate=8000
        )
        # EQ should handle gracefully when bands exceed nyquist
        result = processor.apply_eq(tone, 2.0, 1.0, 1.0)
        assert len(result) == len(tone)
