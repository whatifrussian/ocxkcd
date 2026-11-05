from __future__ import annotations

import argparse
import importlib.resources
import json
import os
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

from . import __version__

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def safe_key(path: Path, used: set[str]) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in path.stem).strip("._") or "image"
    key = stem
    index = 2
    while key in used:
        key = f"{stem}-{index}"
        index += 1
    used.add(key)
    return key


def collect_images(inputs: list[Path], recursive: bool, exclude_globs: list[str] | None = None) -> list[Path]:
    exclude_globs = exclude_globs or []
    images: list[Path] = []
    for item in inputs:
        path = item.expanduser().resolve()
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            images.extend(sorted(candidate.resolve() for candidate in iterator if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS))
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
        else:
            raise FileNotFoundError(f"Input is not a supported image or directory: {item}")
    deduped: list[Path] = []
    seen: set[Path] = set()
    for image in images:
        if any(fnmatch(image.name, pattern) or fnmatch(str(image), pattern) for pattern in exclude_globs):
            continue
        if image not in seen:
            seen.add(image)
            deduped.append(image)
    if not deduped:
        raise RuntimeError("No input images found")
    return deduped


def write_manifest(images: list[Path], output: Path) -> Path:
    used: set[str] = set()
    comics = []
    for image in images:
        key = safe_key(image, used)
        comics.append(
            {
                "key": key,
                "title": key,
                "page": "",
                "source": str(image),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "input-manifest.json"
    manifest.write_text(json.dumps({"comics": comics, "items": []}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_command(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def bundled_checkpoint(filename: str) -> Path | None:
    package_path = Path(__file__).resolve().parent / "models" / "ocxkcd" / filename
    if package_path.exists():
        return package_path
    try:
        with importlib.resources.as_file(importlib.resources.files("ocxkcd") / "models" / "ocxkcd" / filename) as path:
            return path if path.exists() else None
    except Exception:
        return None


def default_checkpoint(model_root: Path | None, filename: str) -> Path | None:
    candidates: list[Path] = []
    if model_root is not None:
        candidates.append(model_root.expanduser().resolve() / "ocxkcd" / filename)
    candidates.append(Path("models") / "ocxkcd" / filename)
    bundled = bundled_checkpoint(filename)
    if bundled is not None:
        candidates.append(bundled)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def process(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    work = args.work_dir.expanduser().resolve() if args.work_dir else output / "_work"
    images = collect_images(args.inputs, recursive=args.recursive, exclude_globs=args.exclude_glob)
    manifest = write_manifest(images, work)
    ocr_dir = args.ocr_report.expanduser().resolve() if args.ocr_report else work / "ocr"
    export_dir = output

    env = os.environ.copy()
    model_root = args.model_root.expanduser().resolve() if args.model_root else None
    if model_root:
        env["OCXKCD_MODEL_ROOT"] = str(model_root)

    ocr_cmd = [
        sys.executable,
        "-m",
        "ocxkcd._paddle_report",
        "--manifest",
        str(manifest),
        "--output",
        str(ocr_dir),
        "--det-model",
        args.det_model,
        "--rec-model",
        args.rec_model,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--upscale",
        str(args.upscale),
        "--det-limit-side-len",
        str(args.det_limit_side_len),
        "--det-limit-type",
        args.det_limit_type,
        "--det-thresh",
        str(args.det_thresh),
        "--det-box-thresh",
        str(args.det_box_thresh),
        "--det-unclip-ratio",
        str(args.det_unclip_ratio),
        "--cpu-threads",
        str(args.cpu_threads),
    ]
    ocr_cmd.append("--enable-mkldnn" if args.enable_mkldnn else "--no-enable-mkldnn")
    run_command(ocr_cmd, env=env)

    export_cmd = [
        sys.executable,
        "-m",
        "ocxkcd._comic_export",
        "--report",
        str(ocr_dir / "report.json"),
        "--output",
        str(export_dir),
        "--min-score",
        str(args.min_score),
        "--min-chars",
        str(args.min_chars),
        "--min-single-char-score",
        str(args.min_single_char_score),
        "--min-width",
        str(args.min_width),
        "--min-height",
        str(args.min_height),
        "--crop-refiner-combine",
        args.crop_refiner_combine,
        "--crop-refiner-threshold",
        str(args.crop_refiner_threshold),
        "--crop-refiner-device",
        args.crop_refiner_device,
        "--crop-refiner-dilate",
        str(args.crop_refiner_dilate),
        "--refine-fill-from-interior",
        "--refine-fill-local-ring",
        str(args.refine_fill_local_ring),
        "--fill-prior-device",
        args.fill_prior_device,
        "--fill-prior-min-confidence",
        str(args.fill_prior_min_confidence),
        "--fill-prior-black-min-confidence",
        str(args.fill_prior_black_min_confidence),
        "--no-fill-prior-allow-transparent",
        "--dark-panel-fill",
        "--rotated-padding",
        str(args.rotated_padding),
        "--rotated-text-box-padding",
        str(args.rotated_text_box_padding),
        "--ink-dilate",
        str(args.ink_dilate),
        "--visible-transparent-text-box-height-ratio",
        str(args.visible_transparent_text_box_height_ratio),
    ]
    if args.keep_single_char:
        export_cmd.append("--keep-single-char")
    crop_refiner_checkpoint = args.crop_refiner_checkpoint.expanduser().resolve() if args.crop_refiner_checkpoint else default_checkpoint(
        model_root, "refiner-smoke-seed1337.pt"
    )
    fill_prior_checkpoint = args.fill_prior_checkpoint.expanduser().resolve() if args.fill_prior_checkpoint else default_checkpoint(
        model_root, "fill-classifier-stratified-seed2026.pt"
    )
    if crop_refiner_checkpoint:
        export_cmd += ["--crop-refiner-checkpoint", str(crop_refiner_checkpoint)]
    if fill_prior_checkpoint:
        export_cmd += ["--fill-prior-checkpoint", str(fill_prior_checkpoint)]
    if args.cleaner == "conditioned":
        checkpoint = args.cleaner_checkpoint or default_checkpoint(model_root, "conditioned-cleaner-v1.pt")
        if checkpoint is None:
            raise RuntimeError("Conditioned cleaner checkpoint missing; reinstall ocxkcd or use --cleaner legacy")
        export_cmd += ["--cleaner-checkpoint", str(checkpoint), "--cleaner-device", args.cleaner_device]
    elif args.cleaner_checkpoint:
        raise ValueError("--cleaner-checkpoint cannot be used with --cleaner legacy")
    run_command(export_cmd, env=env)

    print(
        json.dumps(
            {
                "output": str(export_dir),
                "ocrReport": str(ocr_dir / "index.html"),
                "images": len(images),
                "cropRefinerCheckpoint": str(crop_refiner_checkpoint) if crop_refiner_checkpoint else None,
                "fillPriorCheckpoint": str(fill_prior_checkpoint) if fill_prior_checkpoint else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def download_models(args: argparse.Namespace) -> int:
    env = os.environ.copy()
    if args.model_root:
        model_root = args.model_root.expanduser().resolve()
        env["OCXKCD_MODEL_ROOT"] = str(model_root)
        env["PADDLE_PDX_CACHE_HOME"] = str(model_root / "paddlex")

    temporary = args.output.expanduser().resolve() / "_download_check"
    temporary.mkdir(parents=True, exist_ok=True)
    image = temporary / "blank.png"
    if not image.exists():
        from PIL import Image

        Image.new("RGB", (160, 80), "white").save(image)
    manifest = write_manifest([image], temporary)
    command = [
        sys.executable,
        "-m",
        "ocxkcd._paddle_report",
        "--manifest",
        str(manifest),
        "--output",
        str(temporary / "ocr"),
        "--det-model",
        args.det_model,
        "--rec-model",
        args.rec_model,
        "--device",
        args.device,
        "--limit",
        "1",
        "--cpu-threads",
        str(args.cpu_threads),
        "--no-enable-mkldnn",
    ]
    run_command(command, env=env)
    print(json.dumps({"modelRoot": env.get("OCXKCD_MODEL_ROOT", "models"), "status": "ready"}, indent=2))
    return 0


def doctor(_args: argparse.Namespace) -> int:
    checks = {}
    for module in ("PIL", "cv2", "numpy"):
        try:
            __import__(module)
            checks[module] = "ok"
        except Exception as exc:
            checks[module] = f"missing: {exc}"
    try:
        __import__("paddleocr")
        checks["paddleocr"] = "ok"
    except Exception as exc:
        checks["paddleocr"] = f"missing: {exc}"
    try:
        __import__("torch")
        checks["torch"] = "ok"
    except Exception as exc:
        checks["torch"] = f"optional missing: {exc}"
    print(json.dumps({"version": __version__, "checks": checks}, indent=2))
    return 0 if all(not value.startswith("missing:") for value in checks.values()) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocxkcd", description="OCR xkcd/what-if images and export comic_editor-compatible cleanup files.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process", help="Run OCR, transcription, cleanup, and comic_editor export.")
    process_parser.add_argument("inputs", nargs="+", type=Path, help="Image files or directories.")
    process_parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory.")
    process_parser.add_argument("--recursive", action="store_true", help="Recurse into input directories.")
    process_parser.add_argument("--exclude-glob", action="append", default=[], help="Exclude input images by filename/path glob. Can be repeated.")
    process_parser.add_argument("--work-dir", type=Path, help="Intermediate manifest/OCR directory. Defaults to OUTPUT/_work.")
    process_parser.add_argument("--ocr-report", type=Path, help="Directory for OCR report. Defaults to WORK_DIR/ocr.")
    process_parser.add_argument("--model-root", type=Path, help="Local Paddle model root. Also readable as OCXKCD_MODEL_ROOT.")
    process_parser.add_argument("--det-model", default="PP-OCRv5_server_det")
    process_parser.add_argument("--rec-model", default="PP-OCRv5_server_rec")
    process_parser.add_argument("--device", default="cpu", help="Paddle device, e.g. cpu or gpu:0.")
    process_parser.add_argument("--batch-size", type=int, default=8)
    process_parser.add_argument("--upscale", type=int, default=2)
    process_parser.add_argument("--det-limit-side-len", type=int, default=960)
    process_parser.add_argument("--det-limit-type", default="max")
    process_parser.add_argument("--det-thresh", type=float, default=0.2)
    process_parser.add_argument("--det-box-thresh", type=float, default=0.3)
    process_parser.add_argument("--det-unclip-ratio", type=float, default=2.0)
    process_parser.add_argument("--enable-mkldnn", action=argparse.BooleanOptionalAction, default=False)
    process_parser.add_argument("--cpu-threads", type=int, default=8)
    process_parser.add_argument("--min-score", type=float, default=0.0)
    process_parser.add_argument("--min-width", type=float, default=8.0)
    process_parser.add_argument("--min-height", type=float, default=6.0)
    process_parser.add_argument("--min-chars", type=int, default=1)
    process_parser.add_argument("--min-single-char-score", type=float, default=0.5)
    process_parser.add_argument("--keep-single-char", action=argparse.BooleanOptionalAction, default=True)
    process_parser.add_argument("--crop-refiner-checkpoint", type=Path)
    process_parser.add_argument("--cleaner", default="conditioned", choices=("conditioned", "legacy"), help="Pixel cleaner; legacy disables the new postprocessing stage")
    process_parser.add_argument("--cleaner-checkpoint", type=Path)
    process_parser.add_argument("--cleaner-device", default="cpu", choices=("cpu", "cuda"))
    process_parser.add_argument("--crop-refiner-combine", default="prefer", choices=("off", "intersect", "union", "prefer", "replace"))
    process_parser.add_argument("--crop-refiner-threshold", type=float, default=0.7)
    process_parser.add_argument("--crop-refiner-device", default="auto", choices=("auto", "cpu", "cuda"))
    process_parser.add_argument("--crop-refiner-dilate", type=int, default=0)
    process_parser.add_argument("--fill-prior-checkpoint", type=Path)
    process_parser.add_argument("--fill-prior-device", default="auto", choices=("auto", "cpu", "cuda"))
    process_parser.add_argument("--fill-prior-min-confidence", type=float, default=0.45)
    process_parser.add_argument("--fill-prior-black-min-confidence", type=float, default=0.45)
    process_parser.add_argument("--refine-fill-local-ring", type=int, default=3)
    process_parser.add_argument("--rotated-padding", type=int, default=0)
    process_parser.add_argument("--rotated-text-box-padding", type=float, default=0.35)
    process_parser.add_argument("--ink-dilate", type=int, default=0)
    process_parser.add_argument("--visible-transparent-text-box-height-ratio", type=float, default=0.78)
    process_parser.set_defaults(func=process)

    doctor_parser = subparsers.add_parser("doctor", help="Check runtime dependencies.")
    doctor_parser.set_defaults(func=doctor)

    download_parser = subparsers.add_parser("download-models", help="Download/cache PaddleOCR models without processing user images.")
    download_parser.add_argument("--output", type=Path, default=Path("/tmp/ocxkcd-model-download"))
    download_parser.add_argument("--model-root", type=Path, default=Path(os.environ.get("OCXKCD_MODEL_ROOT", "models")))
    download_parser.add_argument("--det-model", default="PP-OCRv5_server_det")
    download_parser.add_argument("--rec-model", default="PP-OCRv5_server_rec")
    download_parser.add_argument("--device", default="cpu")
    download_parser.add_argument("--cpu-threads", type=int, default=8)
    download_parser.set_defaults(func=download_models)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
