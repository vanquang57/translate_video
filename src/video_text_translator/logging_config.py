"""Centralized logging setup for the pipeline."""

from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure the root logger with stderr output and a stable format.

    Precedence: ``quiet`` wins over ``verbose`` if both are set.
    """
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any handlers that may have been attached by 3rd-party code on
    # import (e.g. paddleocr installs its own handlers); we want a clean
    # configuration owned by this app.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt=_FMT, datefmt=_DATE_FMT))
    root.addHandler(handler)

    # Quiet down noisy third-party loggers unless verbose is requested.
    if not verbose:
        for noisy in ("PIL", "paddle", "ppocr"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
