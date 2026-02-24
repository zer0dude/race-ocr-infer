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
    p_infer.add_argument(
        "--yolo-weights",
        default=None,
        help="Path to YOLO weights. Default: cached weights from `raceocr setup`.",
    )
    p_infer.add_argument(
        "--yolo-conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold (default: 0.25).",
    )
    p_infer.add_argument(
        "--yolo-iou",
        type=float,
        default=0.45,
        help="YOLO IoU threshold (default: 0.45).",
    )
    p_infer.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO inference image size (default: 1280).",
    )
    p_infer.add_argument(
        "--device",
        default=None,
        help='Device for YOLO (e.g. "cpu", "0", "cuda:0"). Default: Ultralytics auto.',
    )
    p_infer.add_argument(
        "--save-vis",
        action="store_true",
        help="Save a visualization image with YOLO boxes.",
    )
    p_infer.add_argument(
    "--save-crops",
    action="store_true",
    help="Save cropped detection regions to artifacts/crops/.",
    )
    p_infer.add_argument(
        "--pad",
        type=float,
        default=0.08,
        help="Crop padding as fraction of box size (default: 0.08).",
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
    from pathlib import Path

    from .config import default_cache_dir, yolo_cache_path
    from .infer import detections_to_dict, load_yolo, render_detections, run_yolo_detect
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
        "yolo_weights": args.yolo_weights,
        "yolo_conf": float(args.yolo_conf),
        "yolo_iou": float(args.yolo_iou),
        "imgsz": int(args.imgsz),
        "device": args.device,
        "save_vis": bool(args.save_vis),
        "env": get_env_info(),
        "artifact_dir": str(run_dir),
    }
    write_params(run_dir, params)

    # Resolve weights
    weights_path = Path(args.yolo_weights) if args.yolo_weights else yolo_cache_path(default_cache_dir())
    if not weights_path.exists():
        raise SystemExit(
            f"YOLO weights not found at {weights_path}. Run `raceocr setup` or pass --yolo-weights."
        )

    # Run YOLO
    model = load_yolo(weights_path)
    dets = run_yolo_detect(
        model=model,
        img_path=img_path,
        conf=float(args.yolo_conf),
        iou=float(args.yolo_iou),
        imgsz=int(args.imgsz),
        device=args.device,
    )

    vis_path = None
    if args.save_vis:
        vis_dir = run_dir / "vis"
        vis_path = vis_dir / f"{img_path.stem}_yolo.jpg"
        render_detections(img_path, dets, vis_path)

    crop_meta = None
    if args.save_crops or args.debug:
        crops_dir = run_dir / "crops"
        from .infer import save_crops

        crop_meta = save_crops(
            img_path=img_path,
            detections=dets,
            crops_dir=crops_dir,
            pad_frac=float(args.pad),
        )

    results = {
        "mode": "infer",
        "input_image_path": str(img_path),
        "artifact_dir": str(run_dir),
        "yolo": {
            "weights": str(weights_path),
            "conf": float(args.yolo_conf),
            "iou": float(args.yolo_iou),
            "imgsz": int(args.imgsz),
            "device": args.device,
        },
        "detections": detections_to_dict(dets),
        "vis_path": str(vis_path) if vis_path else None,
        "ocr_candidates": [],
        "notes": "YOLO detections implemented (Step 4). OCR comes next.",
        "crops": crop_meta if crop_meta is not None else [],
        "crop_settings": {
            "pad_frac": float(args.pad),
            "saved": bool(args.save_crops or args.debug)
        },
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
        "notes": "Album placeholder (Step 4). Aggregation comes later.",
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