"""CLI entry point for the Video Text Translator."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from ``path`` into ``os.environ``.

    Minimal parser (no external dependency):
      * Ignores blank lines and lines starting with ``#``.
      * Strips a single pair of surrounding quotes from the value.
      * Does NOT override variables that are already set in the
        environment, so real shell exports always win.
    """
    if not path.is_file():
        return
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            os.environ[key] = value
    except OSError:
        # .env is best-effort; fall back to existing environment.
        pass


_PROJECT_ROOT = Path(__file__).resolve().parent
_load_dotenv(_PROJECT_ROOT / ".env")

# Allow `from video_text_translator import ...` when running from a clone
# without installing the package.
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from video_text_translator.config import build_argparser, load_config  # noqa: E402
from video_text_translator.detector import PaddleOCRDetector  # noqa: E402
from video_text_translator.errors import InvalidConfigError  # noqa: E402
from video_text_translator.inpainter import OpenCVInpainter  # noqa: E402
from video_text_translator.logging_config import setup_logging  # noqa: E402
from video_text_translator.pipeline import Pipeline  # noqa: E402
from video_text_translator.progress import ProgressReporter  # noqa: E402
from video_text_translator.renderer import PillowRenderer  # noqa: E402
from video_text_translator.tracker import IoUContentTracker  # noqa: E402
from video_text_translator.translator import GoogleTranslator  # noqa: E402
from video_text_translator.translator_llm import LlmTranslator  # noqa: E402

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    try:
        config = load_config(args)
    except InvalidConfigError as exc:
        logger.error("config error: %s", exc)
        return 1

    if not config.input_path or not config.output_path:
        logger.error("both --input and --output are required")
        return 1

    # Probe video size up-front so the tracker knows the diagonal.
    import cv2  # local import keeps cold-start fast
    cap = cv2.VideoCapture(config.input_path)
    if not cap.isOpened():
        logger.error("cannot open input video: %s", config.input_path)
        return 1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    detector = PaddleOCRDetector(
        compute_mode=config.compute_mode,
        confidence_threshold=config.detector.confidence_threshold,
        downscale=config.performance.ocr_downscale,
        model_variant=config.detector.model_variant,
        cpu_threads=config.detector.cpu_threads,
    )
    tracker = IoUContentTracker(
        frame_width=width,
        frame_height=height,
        iou_threshold=config.tracker.iou_threshold,
        content_similarity_threshold=config.tracker.content_similarity_threshold,
        center_distance_ratio=config.tracker.center_distance_ratio,
        n_inactive=config.tracker.n_inactive,
        ocr_stride=config.performance.ocr_stride,
        max_active_segments=config.tracker.max_active_segments,
        smooth_lock_threshold=config.tracker.smooth_lock_threshold,
        smooth_ema_alpha=config.tracker.smooth_ema_alpha,
    )
    inpainter = OpenCVInpainter(
        algorithm=config.inpainter.algorithm,
        radius=config.inpainter.radius,
        padding=config.inpainter.padding,
    )
    translator: object
    if config.translator.backend == "llm":
        logger.info("translator backend: LLM (%s via %s)", config.translator.llm.model, config.translator.llm.base_url)
        translator = LlmTranslator(
            config=config.translator.llm,
            max_retries=config.translator.max_retries,
        )
    else:
        logger.info("translator backend: Google Translate")
        translator = GoogleTranslator(
            timeout_seconds=config.translator.timeout_seconds,
            max_chars=config.translator.max_chars,
            max_retries=config.translator.max_retries,
        )
    renderer = PillowRenderer(default_font_path=config.renderer.font_path)
    progress = ProgressReporter()

    pipeline = Pipeline(
        config=config,
        detector=detector,
        tracker=tracker,
        inpainter=inpainter,
        translator=translator,
        renderer=renderer,
        progress=progress,
    )
    return pipeline.run()


if __name__ == "__main__":
    sys.exit(main())
