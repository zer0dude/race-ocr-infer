from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raceocr",
        description="Race OCR inference CLI (YOLO + PaddleOCR).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="raceocr 0.0.1",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- setup ----
    p_setup = subparsers.add_parser(
        "setup",
        help="Download weights and warm caches (placeholder in v0).",
    )
    p_setup.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory for model weights (default: ~/.cache/raceocr).",
    )
    p_setup.add_argument(
        "--yolo-url",
        default=None,
        help="URL to YOLO weights (default will be configured later).",
    )

    # ---- infer ----
    p_infer = subparsers.add_parser(
        "infer",
        help="Run single-image inference (placeholder in v0).",
    )
    p_infer.add_argument(
        "--img",
        required=True,
        help="Path to a single input image.",
    )
    p_infer.add_argument(
        "--out-dir",
        default="artifacts",
        help="Directory to store run artifacts (default: ./artifacts).",
    )
    p_infer.add_argument(
        "--ocr-conf",
        type=float,
        default=0.75,
        help="Minimum OCR confidence to keep (default: 0.75).",
    )
    p_infer.add_argument(
        "--filter-words",
        default="",
        help="Comma-separated list of words to filter from OCR results.",
    )
    p_infer.add_argument(
        "--filter-words-file",
        default=None,
        help="File containing filter words (one per line).",
    )
    p_infer.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    p_infer.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (save extra artifacts).",
    )

    # ---- album ----
    p_album = subparsers.add_parser(
        "album",
        help="Run album/folder inference and aggregate results (placeholder in v0).",
    )
    p_album.add_argument(
        "--dir",
        required=True,
        help="Path to a folder of images (one athlete album).",
    )
    p_album.add_argument(
        "--out-dir",
        default="artifacts",
        help="Directory to store run artifacts (default: ./artifacts).",
    )
    p_album.add_argument(
        "--ocr-conf",
        type=float,
        default=0.75,
        help="Minimum OCR confidence to keep (default: 0.75).",
    )
    p_album.add_argument(
        "--filter-words",
        default="",
        help="Comma-separated list of words to filter from OCR results.",
    )
    p_album.add_argument(
        "--filter-words-file",
        default=None,
        help="File containing filter words (one per line).",
    )
    p_album.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    p_album.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (save extra artifacts).",
    )

    return parser


def _cmd_setup(args: argparse.Namespace) -> int:
    print("[setup] placeholder")
    print(f"  cache_dir={args.cache_dir}")
    print(f"  yolo_url={args.yolo_url}")
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .io import make_run_dir, write_params, write_results
    from .util import get_env_info

    img_path = Path(args.img)
    input_label = img_path.stem if img_path.name else "image"

    run_dir = make_run_dir("infer", input_label, args.out_dir)

    params = {
        "command": "infer",
        "img": str(img_path),
        "out_dir": str(args.out_dir),
        "ocr_conf": args.ocr_conf,
        "filter_words": args.filter_words,
        "filter_words_file": args.filter_words_file,
        "verbose": bool(args.verbose),
        "debug": bool(args.debug),
        "env": get_env_info(),
        "artifact_dir": str(run_dir),
    }
    write_params(run_dir, params)

    # Placeholder results schema (v0)
    results = {
        "mode": "infer",
        "input_image_path": str(img_path),
        "artifact_dir": str(run_dir),
        "detections": [],
        "ocr_candidates": [],
        "notes": "placeholder results (Step 2).",
    }
    write_results(run_dir, results)

    print(f"[infer] wrote artifacts to: {run_dir}")
    return 0


def _cmd_album(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .io import make_run_dir, write_params, write_results
    from .util import get_env_info

    dir_path = Path(args.dir)
    input_label = dir_path.name if dir_path.name else "album"

    run_dir = make_run_dir("album", input_label, args.out_dir)

    params = {
        "command": "album",
        "dir": str(dir_path),
        "out_dir": str(args.out_dir),
        "ocr_conf": args.ocr_conf,
        "filter_words": args.filter_words,
        "filter_words_file": args.filter_words_file,
        "verbose": bool(args.verbose),
        "debug": bool(args.debug),
        "env": get_env_info(),
        "artifact_dir": str(run_dir),
    }
    write_params(run_dir, params)

    results = {
        "mode": "album",
        "input_folder_path": str(dir_path),
        "artifact_dir": str(run_dir),
        "num_images": 0,
        "ranked_counts": [],
        "notes": "placeholder results (Step 2).",
    }
    write_results(run_dir, results)

    print(f"[album] wrote artifacts to: {run_dir}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "setup":
        return _cmd_setup(args)
    if args.command == "infer":
        return _cmd_infer(args)
    if args.command == "album":
        return _cmd_album(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())