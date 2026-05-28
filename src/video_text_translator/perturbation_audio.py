"""Audio perturbation processing for video perturbation.

This module implements the AudioPerturbationProcessor class which applies
time-varying tempo drift, 3-band EQ adjustments, and ambient noise overlay
to an audio track. Parameters change at each Change_Interval boundary.
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path

import numpy as np
from pydub import AudioSegment
from scipy.signal import butter, sosfilt, resample

from .perturbation_config import PerturbationConfig
from .perturbation_scheduler import ParameterScheduler

logger = logging.getLogger(__name__)


class AudioPerturbationProcessor:
    """Applies tempo drift, EQ changes, and ambient noise overlay.

    The processor divides the audio into segments of Change_Interval length,
    generates random perturbation parameters for each segment, and applies
    tempo change (without pitch shift), 3-band EQ, and pink noise overlay.
    """

    def __init__(self, config: PerturbationConfig, rng: random.Random) -> None:
        """Initialize the audio perturbation processor.

        Args:
            config: Perturbation configuration with audio parameters.
            rng: Seeded Random instance for reproducibility.
        """
        self.config = config
        self.rng = rng

    def process(self, audio_path: str, output_path: str) -> None:
        """Process entire audio track with time-varying perturbations.

        Divides the audio into Change_Interval segments, applies tempo drift,
        EQ adjustments, and ambient noise overlay per segment, then concatenates
        and exports the result.

        Args:
            audio_path: Path to input WAV file.
            output_path: Path to write the processed WAV file.

        Raises:
            FileNotFoundError: If audio_path does not exist.
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            logger.warning("No audio track found at %s, skipping audio perturbation.", audio_path)
            return

        # Load audio
        audio = AudioSegment.from_wav(audio_path)
        if len(audio) == 0:
            logger.warning("Audio track is empty, skipping audio perturbation.")
            return

        duration_s = len(audio) / 1000.0  # pydub uses milliseconds
        sample_rate = audio.frame_rate
        change_interval = self.config.change_interval

        # Create parameter schedules
        scheduler = ParameterScheduler(
            duration=duration_s,
            change_interval=change_interval,
            rng=self.rng,
        )

        tempo_schedule = scheduler.schedule(
            self.config.audio_tempo_min, self.config.audio_tempo_max
        )
        # EQ: 3 independent schedules for bass, mid, treble (±eq_range_db)
        bass_schedule = scheduler.schedule(
            -self.config.eq_range_db, self.config.eq_range_db
        )
        mid_schedule = scheduler.schedule(
            -self.config.eq_range_db, self.config.eq_range_db
        )
        treble_schedule = scheduler.schedule(
            -self.config.eq_range_db, self.config.eq_range_db
        )
        # Ambient noise volume schedule
        noise_schedule = scheduler.schedule(
            self.config.ambience_volume_min_db, self.config.ambience_volume_max_db
        )

        # Process audio in segments
        processed_segments: list[AudioSegment] = []

        if not tempo_schedule:
            # No segments (zero duration) - just export as-is
            audio.export(output_path, format="wav")
            return

        for i, seg_param in enumerate(tempo_schedule):
            start_ms = int(seg_param.start_time * 1000)
            end_ms = int(seg_param.end_time * 1000)

            # Clamp to audio length
            end_ms = min(end_ms, len(audio))
            if start_ms >= len(audio):
                break

            segment = audio[start_ms:end_ms]
            if len(segment) == 0:
                continue

            # Apply tempo drift
            tempo_factor = seg_param.value
            segment = self.apply_tempo(segment, tempo_factor)

            # Apply EQ
            bass_db = bass_schedule[i].value if i < len(bass_schedule) else 0.0
            mid_db = mid_schedule[i].value if i < len(mid_schedule) else 0.0
            treble_db = treble_schedule[i].value if i < len(treble_schedule) else 0.0
            segment = self.apply_eq(segment, bass_db, mid_db, treble_db)

            # Apply ambient noise overlay
            noise_db = noise_schedule[i].value if i < len(noise_schedule) else -35.0
            segment = self._apply_ambient_noise(segment, noise_db)

            processed_segments.append(segment)

        if not processed_segments:
            audio.export(output_path, format="wav")
            return

        # Concatenate all processed segments
        result = processed_segments[0]
        for seg in processed_segments[1:]:
            result = result + seg

        # Export
        result.export(output_path, format="wav")

    def apply_tempo(self, segment: AudioSegment, factor: float) -> AudioSegment:
        """Apply tempo change without pitch shift using scipy resample.

        Changes playback speed by resampling the audio to a different length,
        then setting the sample rate back to the original. This stretches/
        compresses time without altering pitch.

        Args:
            segment: Audio segment to process.
            factor: Tempo factor (>1.0 = faster, <1.0 = slower).

        Returns:
            AudioSegment with tempo adjusted.
        """
        if abs(factor - 1.0) < 1e-6:
            return segment

        # Handle empty or very short segments
        if len(segment) == 0:
            return segment

        # Convert to numpy array
        samples = np.array(segment.get_array_of_samples(), dtype=np.float64)
        sample_rate = segment.frame_rate
        channels = segment.channels
        sample_width = segment.sample_width

        if len(samples) == 0:
            return segment

        # For stereo, reshape to (n_samples, channels)
        if channels > 1:
            samples = samples.reshape(-1, channels)

        # Resample: to speed up by factor, we need fewer samples
        # New length = original_length / factor
        original_length = samples.shape[0]
        new_length = int(round(original_length / factor))

        if new_length <= 0:
            new_length = 1

        if channels > 1:
            # Resample each channel independently
            resampled = np.zeros((new_length, channels), dtype=np.float64)
            for ch in range(channels):
                resampled[:, ch] = resample(samples[:, ch], new_length)
            resampled = resampled.flatten()
        else:
            resampled = resample(samples, new_length)

        # Clip to valid range for the sample width
        max_val = (2 ** (sample_width * 8 - 1)) - 1
        min_val = -(2 ** (sample_width * 8 - 1))
        resampled = np.clip(resampled, min_val, max_val).astype(np.int16)

        # Create new AudioSegment from resampled data
        result = AudioSegment(
            data=resampled.tobytes(),
            sample_width=sample_width,
            frame_rate=sample_rate,
            channels=channels,
        )

        return result

    def apply_eq(
        self, segment: AudioSegment, bass_db: float, mid_db: float, treble_db: float
    ) -> AudioSegment:
        """Apply 3-band EQ using scipy butterworth bandpass filters.

        Frequency bands:
        - Bass: 20-250 Hz
        - Mid: 250-4000 Hz
        - Treble: 4000-20000 Hz

        Each band is boosted or cut by the specified dB amount.

        Args:
            segment: Audio segment to process.
            bass_db: Bass band gain in dB (positive = boost, negative = cut).
            mid_db: Mid band gain in dB.
            treble_db: Treble band gain in dB.

        Returns:
            AudioSegment with EQ applied.
        """
        # Skip if all gains are effectively zero
        if abs(bass_db) < 0.01 and abs(mid_db) < 0.01 and abs(treble_db) < 0.01:
            return segment

        sample_rate = segment.frame_rate
        channels = segment.channels
        sample_width = segment.sample_width
        nyquist = sample_rate / 2.0

        # Need at least some samples to filter
        if len(segment) < 10:  # less than 10ms
            return segment

        # Convert to float64 numpy array
        samples = np.array(segment.get_array_of_samples(), dtype=np.float64)

        if channels > 1:
            samples = samples.reshape(-1, channels)

        # Process each band
        # We use second-order sections (sos) for numerical stability
        result = samples.copy()

        # Bass: 20-250 Hz (bandpass)
        if abs(bass_db) >= 0.01:
            low_freq = 20.0 / nyquist
            high_freq = min(250.0 / nyquist, 0.99)
            if low_freq < high_freq and low_freq > 0:
                try:
                    sos = butter(2, [low_freq, high_freq], btype='band', output='sos')
                    gain = 10 ** (bass_db / 20.0) - 1.0
                    if channels > 1:
                        for ch in range(channels):
                            band = sosfilt(sos, samples[:, ch])
                            result[:, ch] += gain * band
                    else:
                        band = sosfilt(sos, samples)
                        result += gain * band
                except ValueError:
                    pass  # Skip if filter design fails (e.g., very low sample rate)

        # Mid: 250-4000 Hz (bandpass)
        if abs(mid_db) >= 0.01:
            low_freq = 250.0 / nyquist
            high_freq = min(4000.0 / nyquist, 0.99)
            if low_freq < high_freq and low_freq > 0:
                try:
                    sos = butter(2, [low_freq, high_freq], btype='band', output='sos')
                    gain = 10 ** (mid_db / 20.0) - 1.0
                    if channels > 1:
                        for ch in range(channels):
                            band = sosfilt(sos, samples[:, ch])
                            result[:, ch] += gain * band
                    else:
                        band = sosfilt(sos, samples)
                        result += gain * band
                except ValueError:
                    pass

        # Treble: 4000-20000 Hz (bandpass, capped at nyquist)
        if abs(treble_db) >= 0.01:
            low_freq = 4000.0 / nyquist
            high_freq = min(20000.0 / nyquist, 0.99)
            if low_freq < high_freq and low_freq > 0:
                try:
                    sos = butter(2, [low_freq, high_freq], btype='band', output='sos')
                    gain = 10 ** (treble_db / 20.0) - 1.0
                    if channels > 1:
                        for ch in range(channels):
                            band = sosfilt(sos, samples[:, ch])
                            result[:, ch] += gain * band
                    else:
                        band = sosfilt(sos, samples)
                        result += gain * band
                except ValueError:
                    pass

        # Flatten if stereo
        if channels > 1:
            result = result.flatten()

        # Clip and convert back
        max_val = (2 ** (sample_width * 8 - 1)) - 1
        min_val = -(2 ** (sample_width * 8 - 1))
        result = np.clip(result, min_val, max_val).astype(np.int16)

        # Create new AudioSegment
        eq_segment = AudioSegment(
            data=result.tobytes(),
            sample_width=sample_width,
            frame_rate=sample_rate,
            channels=channels,
        )

        return eq_segment

    def _apply_ambient_noise(
        self, segment: AudioSegment, noise_volume_db: float
    ) -> AudioSegment:
        """Overlay ambient pink noise at the specified volume level.

        Generates pink noise matching the segment length and overlays it
        at the specified dB level relative to the main audio RMS.

        Args:
            segment: Audio segment to add noise to.
            noise_volume_db: Noise volume in dB relative to main audio RMS.

        Returns:
            AudioSegment with noise overlaid.
        """
        if len(segment) == 0:
            return segment

        sample_rate = segment.frame_rate
        channels = segment.channels
        sample_width = segment.sample_width
        n_samples = int(len(segment) * sample_rate / 1000)

        # Generate pink noise using the Voss-McCartney algorithm (simplified)
        noise = self._generate_pink_noise(n_samples)

        # Normalize noise to unit RMS
        noise_rms = np.sqrt(np.mean(noise ** 2))
        if noise_rms > 0:
            noise = noise / noise_rms

        # Calculate target noise amplitude from dB relative to audio RMS
        audio_rms = segment.rms
        if audio_rms <= 0:
            audio_rms = 1  # Avoid log(0)

        # noise_volume_db is relative to main audio RMS
        # target_amplitude = audio_rms * 10^(noise_volume_db / 20)
        target_amplitude = audio_rms * (10 ** (noise_volume_db / 20.0))

        # Scale noise
        noise = noise * target_amplitude

        # If stereo, duplicate noise to both channels
        if channels > 1:
            noise_stereo = np.column_stack([noise[:n_samples]] * channels).flatten()
            noise_samples = noise_stereo
        else:
            noise_samples = noise[:n_samples]

        # Clip to valid range
        max_val = (2 ** (sample_width * 8 - 1)) - 1
        min_val = -(2 ** (sample_width * 8 - 1))
        noise_samples = np.clip(noise_samples, min_val, max_val).astype(np.int16)

        # Create noise AudioSegment
        noise_segment = AudioSegment(
            data=noise_samples.tobytes(),
            sample_width=sample_width,
            frame_rate=sample_rate,
            channels=channels,
        )

        # Overlay noise onto segment
        # Trim or pad noise to match segment length exactly
        if len(noise_segment) > len(segment):
            noise_segment = noise_segment[:len(segment)]
        elif len(noise_segment) < len(segment):
            # Pad with silence
            silence = AudioSegment.silent(
                duration=len(segment) - len(noise_segment),
                frame_rate=sample_rate,
            )
            noise_segment = noise_segment + silence

        return segment.overlay(noise_segment)

    def _generate_pink_noise(self, n_samples: int) -> np.ndarray:
        """Generate pink noise (1/f spectrum) using spectral shaping.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            Numpy array of pink noise samples (float64).
        """
        # Generate white noise
        white = np.array(
            [self.rng.gauss(0, 1) for _ in range(n_samples)], dtype=np.float64
        )

        # Apply 1/f spectral shaping via FFT
        fft = np.fft.rfft(white)
        frequencies = np.fft.rfftfreq(n_samples)

        # Avoid division by zero at DC
        frequencies[0] = 1.0

        # Pink noise: amplitude proportional to 1/sqrt(f)
        pink_filter = 1.0 / np.sqrt(frequencies)
        pink_filter[0] = 0.0  # Remove DC component

        fft_pink = fft * pink_filter
        pink = np.fft.irfft(fft_pink, n=n_samples)

        return pink
