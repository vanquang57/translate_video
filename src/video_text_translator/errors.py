"""Custom exception hierarchy for the Video Text Translator pipeline."""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for all pipeline-related errors."""


class InvalidConfigError(PipelineError):
    """Raised when a configuration value is missing, of the wrong type, or out of range."""


class InvalidInputError(PipelineError):
    """Raised when the input video path / format / size / duration is invalid."""


class ComputeInitError(PipelineError):
    """Raised when the compute backend (CPU/GPU) cannot be initialized."""


class MemoryLimitExceeded(PipelineError):
    """Raised when the process exceeds the configured RAM/VRAM ceiling."""


class OutputWriteError(PipelineError):
    """Raised when the output video cannot be written / muxed."""
