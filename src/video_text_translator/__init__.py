"""Video Text Translator package."""

__version__ = "0.1.0"

from .encoder import FFmpegEncoder, detect_best_encoder  # noqa: F401
from .parallel_pass2 import ParallelPass2  # noqa: F401
