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
    p_setup.add_argument(
        "--yolo-sha256",
        default=None,
        help="Expected sha256 for YOLO weights (optional, recommended).",
    )
    p_setup.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of YOLO weights even if present.",
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
    from pathlib import Path

    from .config import DEFAULT_YOLO_URL, default_cache_dir, yolo_cache_path
    from .setup import ensure_yolo_weights

    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache_dir()
    yolo_url = args.yolo_url or DEFAULT_YOLO_URL
    yolo_dest = yolo_cache_path(cache_dir)

    print("[setup] preparing cache")
    print(f"  cache_dir: {cache_dir}")
    print(f"  yolo_url:  {yolo_url}")
    print(f"  yolo_dest: {yolo_dest}")

    path = ensure_yolo_weights(
        url=yolo_url,
        dest=yolo_dest,
        sha256=args.yolo_sha256,
        force=bool(args.force),
    )

    print(f"[setup] YOLO weights ready: {path} ({path.stat().st_size} bytes)")
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    print("[infer] placeholder")
    print(f"  img={args.img}")
    print(f"  out_dir={args.out_dir}")
    print(f"  ocr_conf={args.ocr_conf}")
    print(f"  filter_words={args.filter_words}")
    print(f"  filter_words_file={args.filter_words_file}")
    print(f"  verbose={args.verbose} debug={args.debug}")
    return 0


def _cmd_album(args: argparse.Namespace) -> int:
    print("[album] placeholder")
    print(f"  dir={args.dir}")
    print(f"  out_dir={args.out_dir}")
    print(f"  ocr_conf={args.ocr_conf}")
    print(f"  filter_words={args.filter_words}")
    print(f"  filter_words_file={args.filter_words_file}")
    print(f"  verbose={args.verbose} debug={args.debug}")
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