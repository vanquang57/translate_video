"""Smoke test for the configuration layer (Task 3.3).

Run with: python scripts/verify_config.py
Requires: pyyaml (pip install pyyaml)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_text_translator.config import (  # noqa: E402
    build_argparser,
    build_config,
    cli_overrides,
    deep_merge,
    load_yaml,
)
from video_text_translator.errors import InvalidConfigError  # noqa: E402


def main() -> int:
    # 1. Load default.yaml.
    yaml_dict = load_yaml("configs/default.yaml")
    assert yaml_dict["compute_mode"] == "cpu"
    assert yaml_dict["performance"]["ocr_stride"] == 3

    # 2. Build a Config object.
    base = dict(yaml_dict)
    base["input_path"] = "fake-in.mp4"
    base["output_path"] = "fake-out.mp4"
    cfg = build_config(base)
    assert cfg.compute_mode == "cpu"
    assert cfg.detector.confidence_threshold == 0.5
    assert cfg.performance.ocr_stride == 3
    assert cfg.renderer.font_path.endswith("NotoSans-Regular.ttf")

    # 3. CLI override path.
    parser = build_argparser()
    args = parser.parse_args(
        [
            "-i", "in.mp4", "-o", "out.mp4",
            "--compute-mode", "GPU",
            "--ocr-stride", "5",
            "--inpaint-algo", "ns",
            "--confidence", "0.7",
        ]
    )
    overrides = cli_overrides(args)
    merged = deep_merge(yaml_dict, overrides)
    cfg2 = build_config(merged)
    assert cfg2.compute_mode == "gpu"   # CLI normalised to lowercase
    assert cfg2.performance.ocr_stride == 5
    assert cfg2.inpainter.algorithm == "ns"
    assert cfg2.detector.confidence_threshold == 0.7

    # 4. Validation rejects bad values.
    rejected = 0
    for bad_key, bad_val in (
        ("compute_mode", "auto"),
        ("performance", {"ocr_stride": 99}),
        ("inpainter", {"algorithm": "magic"}),
        ("detector", {"confidence_threshold": 2.5}),
    ):
        bad_dict = dict(base)
        if isinstance(bad_val, dict):
            bad_dict[bad_key] = deep_merge(bad_dict.get(bad_key, {}), bad_val)
        else:
            bad_dict[bad_key] = bad_val
        try:
            build_config(bad_dict)
        except InvalidConfigError:
            rejected += 1
    assert rejected == 4, f"expected 4 rejections, got {rejected}"

    print("Config layer OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
