"""Convert or download PP-OCRv5 ONNX models for the ONNX Runtime backend.

This script attempts to convert PaddlePaddle models to ONNX format using
paddle2onnx. If conversion fails (common on Windows due to DLL issues),
it falls back to downloading pre-converted ONNX models.

Usage:
    python scripts/convert_models.py
    python scripts/convert_models.py --variant server
    python scripts/convert_models.py --variant mobile --output-dir models/onnx

Requirements:
    - For conversion: pip install paddle2onnx (may not work on Windows)
    - For download: internet connection
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


# --- HuggingFace URLs for pre-converted ONNX models ---
# These are community-maintained conversions of PP-OCRv5 models.
ONNX_DOWNLOAD_URLS = {
    "mobile": {
        "det": "https://huggingface.co/marsena/paddleocr-onnx-models/resolve/main/PP-OCRv5_mobile_det_infer.onnx",
        "rec": "https://huggingface.co/marsena/paddleocr-onnx-models/resolve/main/PP-OCRv5_mobile_rec_infer.onnx",
    },
    "server": {
        "det": "https://huggingface.co/marsena/paddleocr-onnx-models/resolve/main/PP-OCRv5_server_det_infer.onnx",
        "rec": "https://huggingface.co/marsena/paddleocr-onnx-models/resolve/main/PP-OCRv5_server_rec_infer.onnx",
    },
}


def find_paddle_model_dir(variant: str, model_type: str) -> Path | None:
    """Find the cached PaddlePaddle model directory."""
    home = Path.home()
    model_name = f"PP-OCRv5_{variant}_{model_type}"

    direct = home / ".paddlex" / "official_models" / model_name
    if direct.is_dir():
        has_model = (
            (direct / "inference.json").is_file()
            or (direct / "inference.pdmodel").is_file()
        )
        has_params = (direct / "inference.pdiparams").is_file()
        if has_model and has_params:
            return direct

    search_dirs = [
        home / ".paddlex" / "official_models",
        home / ".paddleocr",
    ]
    for base in search_dirs:
        if not base.exists():
            continue
        for p in base.iterdir():
            if not p.is_dir():
                continue
            if model_name.lower() in p.name.lower():
                has_model = (
                    (p / "inference.json").is_file()
                    or (p / "inference.pdmodel").is_file()
                )
                has_params = (p / "inference.pdiparams").is_file()
                if has_model and has_params:
                    return p
    return None


def try_convert_paddle_to_onnx(
    paddle_dir: Path, output_path: Path
) -> bool:
    """Try to convert using paddle2onnx. Returns False if it fails."""
    model_file = None
    params_file = None

    if (paddle_dir / "inference.json").is_file():
        model_file = "inference.json"
    elif (paddle_dir / "inference.pdmodel").is_file():
        model_file = "inference.pdmodel"

    if (paddle_dir / "inference.pdiparams").is_file():
        params_file = "inference.pdiparams"

    if model_file is None or params_file is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try Python API first
    try:
        import paddle2onnx
        from paddle2onnx import export

        model_path = str(paddle_dir / model_file)
        params_path = str(paddle_dir / params_file)
        onnx_bytes = export(model_path, params_path, opset_version=11)
        if onnx_bytes and len(onnx_bytes) > 0:
            output_path.write_bytes(onnx_bytes)
            print(f"  Converted via paddle2onnx API: {output_path}")
            return True
    except (ImportError, Exception) as exc:
        print(f"  paddle2onnx API failed: {exc}")

    # Try CLI fallback
    cmd = [
        sys.executable, "-m", "paddle2onnx",
        "--model_dir", str(paddle_dir),
        "--model_filename", model_file,
        "--params_filename", params_file,
        "--save_file", str(output_path),
        "--opset_version", "11",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0:
            print(f"  Converted via paddle2onnx CLI: {output_path}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False


def download_onnx_model(url: str, output_path: Path) -> bool:
    """Download a pre-converted ONNX model from URL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {url}")
    print(f"  -> {output_path}")
    try:
        urllib.request.urlretrieve(url, str(output_path))
        if output_path.is_file() and output_path.stat().st_size > 0:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  OK ({size_mb:.1f} MB)")
            return True
        else:
            print("  ERROR: Downloaded file is empty")
            return False
    except Exception as exc:
        print(f"  ERROR: Download failed: {exc}")
        return False


def extract_char_dict(output_dir: Path, variant: str) -> bool:
    """Extract character dictionary from the recognition model's inference.yml."""
    home = Path.home()
    rec_dir = home / ".paddlex" / "official_models" / f"PP-OCRv5_{variant}_rec"
    yml_path = rec_dir / "inference.yml"

    if yml_path.is_file():
        try:
            import yaml
            data = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
            label_list = None
            if isinstance(data, dict):
                pp = data.get("PostProcess", {})
                if isinstance(pp, dict):
                    label_list = pp.get("character_dict")
                if not label_list:
                    label_list = data.get("label_list") or data.get("character")

            if label_list and isinstance(label_list, list):
                dest = output_dir / "ppocr_keys_v1.txt"
                with dest.open("w", encoding="utf-8") as f:
                    for ch in label_list:
                        f.write(f"{ch}\n")
                print(f"  Extracted {len(label_list)} chars -> {dest}")
                return True
        except Exception as exc:
            print(f"  WARNING: Failed to parse inference.yml: {exc}")

    # Fallback: try to find ppocr_keys_v1.txt
    search_paths = [home / ".paddlex", home / ".paddleocr"]
    for base in search_paths:
        if not base.exists():
            continue
        for p in base.rglob("ppocr_keys_v1.txt"):
            dest = output_dir / "ppocr_keys_v1.txt"
            shutil.copy2(p, dest)
            print(f"  Copied: {dest}")
            return True

    # Try paddleocr package
    try:
        import paddleocr
        pkg_dir = Path(paddleocr.__file__).parent
        for p in pkg_dir.rglob("ppocr_keys_v1.txt"):
            dest = output_dir / "ppocr_keys_v1.txt"
            shutil.copy2(p, dest)
            print(f"  Copied from package: {dest}")
            return True
    except (ImportError, Exception):
        pass

    print("  WARNING: Character dictionary not found.")
    return False


def ensure_models_downloaded(variant: str) -> None:
    """Ensure PaddlePaddle models are downloaded (needed for char dict)."""
    det_dir = find_paddle_model_dir(variant, "det")
    if det_dir is not None:
        return  # Already downloaded

    print(f"  Triggering model download for PP-OCRv5_{variant}...")
    try:
        import numpy as np
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(
            lang="ch",
            text_detection_model_name=f"PP-OCRv5_{variant}_det",
            text_recognition_model_name=f"PP-OCRv5_{variant}_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        ocr.predict(dummy)
    except Exception as exc:
        print(f"  Warning: model download may have failed: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Get PP-OCRv5 ONNX models (convert or download)"
    )
    parser.add_argument(
        "--variant", choices=["mobile", "server"], default="mobile",
        help="Model variant (default: mobile)",
    )
    parser.add_argument(
        "--output-dir", default="models/onnx",
        help="Output directory for ONNX models (default: models/onnx)",
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Skip conversion attempt, download directly",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variant = args.variant

    print(f"Getting PP-OCRv5_{variant} ONNX models...")
    print(f"Output directory: {output_dir.resolve()}")
    print()

    # Ensure PaddlePaddle models are downloaded (for char dict extraction)
    print("[0/3] Ensuring PaddlePaddle models are cached...")
    ensure_models_downloaded(variant)

    det_path = output_dir / f"PP-OCRv5_{variant}_det.onnx"
    rec_path = output_dir / f"PP-OCRv5_{variant}_rec.onnx"

    # Step 1: Detection model
    print(f"\n[1/3] Detection model: {det_path.name}")
    det_ok = False
    if not args.force_download:
        paddle_dir = find_paddle_model_dir(variant, "det")
        if paddle_dir:
            det_ok = try_convert_paddle_to_onnx(paddle_dir, det_path)

    if not det_ok:
        urls = ONNX_DOWNLOAD_URLS.get(variant, {})
        det_url = urls.get("det")
        if det_url:
            print("  Conversion failed/skipped. Downloading pre-converted model...")
            det_ok = download_onnx_model(det_url, det_path)
        else:
            print(f"  ERROR: No download URL for {variant} det model")

    # Step 2: Recognition model
    print(f"\n[2/3] Recognition model: {rec_path.name}")
    rec_ok = False
    if not args.force_download:
        paddle_dir = find_paddle_model_dir(variant, "rec")
        if paddle_dir:
            rec_ok = try_convert_paddle_to_onnx(paddle_dir, rec_path)

    if not rec_ok:
        urls = ONNX_DOWNLOAD_URLS.get(variant, {})
        rec_url = urls.get("rec")
        if rec_url:
            print("  Conversion failed/skipped. Downloading pre-converted model...")
            rec_ok = download_onnx_model(rec_url, rec_path)
        else:
            print(f"  ERROR: No download URL for {variant} rec model")

    # Step 3: Character dictionary
    print("\n[3/3] Character dictionary:")
    dict_ok = extract_char_dict(output_dir, variant)

    # Summary
    print("\n" + "=" * 50)
    if det_ok and rec_ok:
        print("Done! ONNX models ready.")
        print(f"  {det_path}")
        print(f"  {rec_path}")
        if dict_ok:
            print(f"  {output_dir / 'ppocr_keys_v1.txt'}")
        else:
            print("\n  WARNING: Character dictionary missing.")
            print("  Recognition may not work without it.")
        print(f"\nTo use: set detector.backend=onnx in configs/default.yaml")
        return 0
    else:
        print("FAILED. Check errors above.")
        if not det_ok:
            print(f"  Missing: {det_path}")
        if not rec_ok:
            print(f"  Missing: {rec_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
