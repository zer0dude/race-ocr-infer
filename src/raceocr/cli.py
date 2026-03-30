from __future__ import annotations

import argparse
from typing import List, Optional


YOLO_CLASS_NAME_TO_ID = {
    "race_bibs": 0,
    "hedbands": 1,
    "bike-labels": 2,
}


def parse_yolo_classes(value: str) -> Optional[List[int]]:
    """
    Parse a YOLO class selection string into a list of class IDs.

    Supported forms:
      - "all" -> None
      - "race_bibs"
      - "race_bibs,hedbands"
      - "0"
      - "0,2"

    Returns:
        None if all classes should be used, otherwise a sorted list of class IDs.

    Raises:
        argparse.ArgumentTypeError on invalid class names or IDs.
    """
    if value is None:
        return [0]

    raw = value.strip()
    if not raw:
        return [0]

    if raw.lower() == "all":
        return None

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return [0]

    class_ids = set()
    valid_ids = set(YOLO_CLASS_NAME_TO_ID.values())

    for part in parts:
        if part.isdigit():
            cls_id = int(part)
            if cls_id not in valid_ids:
                raise argparse.ArgumentTypeError(
                    f"Invalid YOLO class id '{part}'. Valid ids: {sorted(valid_ids)}."
                )
            class_ids.add(cls_id)
            continue

        if part not in YOLO_CLASS_NAME_TO_ID:
            raise argparse.ArgumentTypeError(
                f"Invalid YOLO class name '{part}'. "
                f"Valid names: {sorted(YOLO_CLASS_NAME_TO_ID.keys())}, or 'all'."
            )
        class_ids.add(YOLO_CLASS_NAME_TO_ID[part])

    return sorted(class_ids)


def parse_allowed_ids(value: str) -> Optional[List[str]]:
    """
    Parse a comma-separated allowed ID list.

    Examples:
      - "123,456,789" -> ["123", "456", "789"]
      - "" -> None

    Returns:
        None if no restriction should be applied, otherwise a deduplicated list of strings.
    """
    if value is None:
        return None

    raw = value.strip()
    if not raw:
        return None

    out: List[str] = []
    seen = set()
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        if s not in seen:
            out.append(s)
            seen.add(s)

    return out or None


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
        help="Download weights and warm caches.",
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
    p_setup.add_argument(
        "--warm-ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm PaddleOCR models (default: on). Use --no-warm-ocr to skip.",
    )

    # ---- infer ----
    p_infer = subparsers.add_parser(
        "infer",
        help="Run single-image inference.",
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
        default=0.95,
        help="Minimum OCR confidence to keep (default: 0.95).",
    )
    p_infer.add_argument(
        "--allowed-ids",
        type=parse_allowed_ids,
        default=None,
        help="Optional comma-separated whitelist of valid OCR results, e.g. 123,456,789",
    )
    p_infer.add_argument(
        "--ocr-char-set",
        default="numeric",
        choices=["numeric", "alnum", "any"],
        help=(
            "Allowed OCR character set. "
            "numeric = digits only, alnum = letters+digits only, any = any non-empty text. "
            "Default: numeric."
        ),
    )
    p_infer.add_argument(
        "--min-box-area",
        type=float,
        default=10000.0,
        help="Minimum YOLO bounding box area in px^2 to keep in production JSON (default: 10000).",
    )
    p_infer.add_argument(
        "--yolo-weights",
        default=None,
        help="Path to YOLO weights. Default: cached weights from `raceocr setup`.",
    )
    p_infer.add_argument(
        "--yolo-conf",
        type=float,
        default=0.86,
        help="YOLO confidence threshold (default: 0.86).",
    )
    p_infer.add_argument(
        "--yolo-iou",
        type=float,
        default=0.45,
        help="YOLO IoU threshold (default: 0.45).",
    )
    p_infer.add_argument(
        "--yolo-classes",
        type=parse_yolo_classes,
        default=parse_yolo_classes("race_bibs"),
        help=(
            "YOLO classes to detect. Default: race_bibs. "
            "Examples: race_bibs | race_bibs,hedbands | 0,2 | all"
        ),
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
        "--create-vis",
        action="store_true",
        help="Create YOLO visualization image (default: off).",
    )
    p_infer.add_argument(
        "--delete-crops",
        action="store_true",
        help="Delete crop image files after OCR to save disk space (default: off).",
    )
    p_infer.add_argument(
        "--pad",
        type=float,
        default=0.01,
        help="Crop padding as fraction of box size (default: 0.01).",
    )
    p_infer.add_argument(
        "--ocr-device",
        default="cpu",
        choices=["gpu", "cpu"],
        help='Device for PaddleOCR (default: "cpu").',
    )
    p_infer.add_argument(
        "--runs-dir",
        default="runs",
        help='Directory to write production JSON (default: ./runs).',
    )
    p_infer.add_argument(
        "--output-name",
        default=None,
        help='Optional output JSON name (e.g. "infer_400.json"). Default uses infer_<imgstem>_<timestamp>.json',
    )

    # ---- album ----
    p_album = subparsers.add_parser(
        "album",
        help="Run album/folder inference and aggregate results.",
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
        default=0.95,
        help="Minimum OCR confidence to keep (default: 0.95).",
    )
    p_album.add_argument(
        "--allowed-ids",
        type=parse_allowed_ids,
        default=None,
        help="Optional comma-separated whitelist of valid OCR results, e.g. 123,456,789",
    )
    p_album.add_argument(
        "--ocr-char-set",
        default="numeric",
        choices=["numeric", "alnum", "any"],
        help=(
            "Allowed OCR character set. "
            "numeric = digits only, alnum = letters+digits only, any = any non-empty text. "
            "Default: numeric."
        ),
    )
    p_album.add_argument(
        "--min-box-area",
        type=float,
        default=10000.0,
        help="Minimum YOLO bounding box area in px^2 to keep in production JSON (default: 10000).",
    )
    p_album.add_argument(
        "--create-vis",
        action="store_true",
        help="Create YOLO visualization per image (default: off).",
    )
    p_album.add_argument(
        "--delete-crops",
        action="store_true",
        help="Delete crop image files after OCR to save disk space (default: off).",
    )
    p_album.add_argument(
        "--pad",
        type=float,
        default=0.01,
        help="Crop padding as fraction of box size (default: 0.01).",
    )
    p_album.add_argument(
        "--yolo-weights",
        default=None,
        help="Path to YOLO weights. Default: cached weights from `raceocr setup`.",
    )
    p_album.add_argument(
        "--yolo-conf",
        type=float,
        default=0.86,
        help="YOLO confidence threshold (default: 0.86).",
    )
    p_album.add_argument(
        "--yolo-iou",
        type=float,
        default=0.45,
        help="YOLO IoU threshold (default: 0.45).",
    )
    p_album.add_argument(
        "--yolo-classes",
        type=parse_yolo_classes,
        default=parse_yolo_classes("race_bibs"),
        help=(
            "YOLO classes to detect. Default: race_bibs. "
            "Examples: race_bibs | race_bibs,hedbands | 0,2 | all"
        ),
    )
    p_album.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO inference image size (default: 1280).",
    )
    p_album.add_argument(
        "--device",
        default=None,
        help='Device for YOLO (e.g. "cpu", "0", "cuda:0"). Default: Ultralytics auto.',
    )
    p_album.add_argument(
        "--ocr-device",
        default="cpu",
        choices=["gpu", "cpu"],
        help='Device for PaddleOCR (default: "cpu").',
    )
    p_album.add_argument(
        "--runs-dir",
        default="runs",
        help='Directory to write production JSON (default: ./runs).',
    )
    p_album.add_argument(
        "--output-name",
        default=None,
        help='Optional output JSON name (e.g. "album_400.json"). Default uses album_<folder>_<timestamp>.json',
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

    if args.warm_ocr:
        print("[setup] warming PaddleOCR (cpu)...")
        from .setup import warm_paddleocr_cpu

        warm_paddleocr_cpu()
        print("[setup] PaddleOCR warm complete.")
    else:
        print("[setup] skipping PaddleOCR warm (per --no-warm-ocr).")

    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .config import default_cache_dir, yolo_cache_path
    from .infer import (
        detections_to_dict,
        init_paddle_ocr,
        load_yolo,
        render_detections,
        run_ocr_on_crop_paths,
        run_yolo_detect,
        save_crops,
    )
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
        "allowed_ids": args.allowed_ids,
        "ocr_char_set": args.ocr_char_set,
        "min_box_area": args.min_box_area,
        "yolo_weights": args.yolo_weights,
        "yolo_conf": float(args.yolo_conf),
        "yolo_iou": float(args.yolo_iou),
        "yolo_classes": args.yolo_classes,
        "create_vis": bool(args.create_vis),
        "delete_crops": bool(args.delete_crops),
        "imgsz": int(args.imgsz),
        "device": args.device,
        "ocr_device": args.ocr_device,
        "env": get_env_info(),
        "artifact_dir": str(run_dir),
        "runs_dir": str(args.runs_dir),
        "output_name": args.output_name,
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
        classes=args.yolo_classes,
    )

    vis_path = None
    if args.create_vis:
        vis_dir = run_dir / "vis"
        vis_path = vis_dir / f"{img_path.stem}_yolo.jpg"
        render_detections(img_path, dets, vis_path)

    crops_dir = run_dir / "crops"
    crop_meta = save_crops(
        img_path=img_path,
        detections=dets,
        crops_dir=crops_dir,
        pad_frac=float(args.pad),
    )

    ocr = init_paddle_ocr(ocr_device=args.ocr_device)

    ocr_candidates = run_ocr_on_crop_paths(
        ocr=ocr,
        crop_meta=crop_meta if crop_meta is not None else [],
        ocr_conf=float(args.ocr_conf),
    )

    if args.delete_crops:
        for cm in crop_meta:
            p = cm.get("crop_path")
            if p:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass
            cm["crop_path"] = None

    results = {
        "mode": "infer",
        "input_image_path": str(img_path),
        "artifact_dir": str(run_dir),
        "yolo": {
            "weights": str(weights_path),
            "conf": float(args.yolo_conf),
            "iou": float(args.yolo_iou),
            "classes": args.yolo_classes,
            "imgsz": int(args.imgsz),
            "device": args.device,
            "runs_dir": str(args.runs_dir),
            "output_name": args.output_name,
        },
        "detections": detections_to_dict(dets),
        "vis_path": str(vis_path) if vis_path else None,
        "crops": crop_meta,
        "crop_settings": {
            "pad_frac": float(args.pad),
            "deleted_after_ocr": bool(args.delete_crops),
        },
        "candidate_filter": {
            "allowed_ids": args.allowed_ids,
            "ocr_char_set": args.ocr_char_set,
            "min_box_area": args.min_box_area,
        },
        "ocr": {
            "conf": float(args.ocr_conf),
            "device": args.ocr_device,
        },
        "ocr_candidates": ocr_candidates,
    }
    write_results(run_dir, results)

    # --- Production JSON output ---
    from .production import infer_to_production_json, make_production_out_path, write_production_json

    prod = infer_to_production_json(results)

    runs_dir = Path(args.runs_dir)
    out_path = make_production_out_path(
        mode="infer",
        input_label=img_path.stem,
        runs_dir=runs_dir,
        output_name=args.output_name,
        include_ts_default=True,
    )
    write_production_json(prod, out_path)
    print(f"[infer] production json: {out_path}")

    print(f"[infer] wrote artifacts to: {run_dir}")
    return 0


def _cmd_album(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .album import list_images
    from .config import default_cache_dir, yolo_cache_path
    from .infer import (
        init_paddle_ocr,
        load_yolo,
        render_detections,
        run_ocr_on_crop_paths,
        run_yolo_detect,
        save_crops,
        detections_to_dict,
    )
    from .io import make_run_dir, write_params, write_results
    from .util import get_env_info, write_json

    dir_path = Path(args.dir)
    if not dir_path.exists() or not dir_path.is_dir():
        raise SystemExit(f"Album dir not found or not a directory: {dir_path}")

    images = list_images(dir_path)
    num_total = len(images)
    if num_total == 0:
        raise SystemExit(f"No images found in {dir_path}")

    run_dir = make_run_dir("album", dir_path.name or "album", args.out_dir)

    params = {
        "command": "album",
        "dir": str(dir_path),
        "out_dir": str(args.out_dir),
        "ocr_conf": float(args.ocr_conf),
        "allowed_ids": args.allowed_ids,
        "ocr_char_set": args.ocr_char_set,
        "min_box_area": args.min_box_area,
        "pad": float(args.pad),
        "yolo_weights": args.yolo_weights,
        "yolo_conf": float(args.yolo_conf),
        "yolo_iou": float(args.yolo_iou),
        "yolo_classes": args.yolo_classes,
        "imgsz": int(args.imgsz),
        "create_vis": bool(args.create_vis),
        "delete_crops": bool(args.delete_crops),
        "device": args.device,
        "ocr_device": args.ocr_device,
        "env": get_env_info(),
        "artifact_dir": str(run_dir),
        "runs_dir": str(args.runs_dir),
        "output_name": args.output_name,
    }
    write_params(run_dir, params)

    # Resolve YOLO weights
    weights_path = Path(args.yolo_weights) if args.yolo_weights else yolo_cache_path(default_cache_dir())
    if not weights_path.exists():
        raise SystemExit(
            f"YOLO weights not found at {weights_path}. Run `raceocr setup` or pass --yolo-weights."
        )

    # Init models ONCE
    yolo = load_yolo(weights_path)
    ocr = init_paddle_ocr(ocr_device=args.ocr_device)

    per_image_results = []
    failed_images = []

    # optional debug folder
    per_image_dir = run_dir / "per_image"
    per_image_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        try:
            image_label = img_path.stem
            img_run_dir = per_image_dir / image_label
            img_run_dir.mkdir(parents=True, exist_ok=True)

            # YOLO
            dets = run_yolo_detect(
                model=yolo,
                img_path=img_path,
                conf=float(args.yolo_conf),
                iou=float(args.yolo_iou),
                imgsz=int(args.imgsz),
                device=args.device,
                classes=args.yolo_classes,
            )

            # optional per-image vis
            vis_path = None
            if args.create_vis:
                vis_path = img_run_dir / f"{img_path.stem}_yolo.jpg"
                render_detections(img_path, dets, vis_path)

            # crops + OCR
            crops_meta = save_crops(
                img_path=img_path,
                detections=dets,
                crops_dir=(img_run_dir / "crops"),
                pad_frac=float(args.pad),
            )

            ocr_candidates = run_ocr_on_crop_paths(
                ocr=ocr,
                crop_meta=crops_meta,
                ocr_conf=float(args.ocr_conf),
            )

            if args.delete_crops:
                for cm in crops_meta:
                    p = cm.get("crop_path")
                    if p:
                        try:
                            Path(p).unlink(missing_ok=True)
                        except Exception:
                            pass
                    cm["crop_path"] = None

            # Build an INTERNAL infer-style result dict (enough for infer_to_production_json)
            img_results = {
                "mode": "infer",
                "input_image_path": str(img_path),
                "artifact_dir": str(img_run_dir),
                "yolo": {
                    "weights": str(weights_path),
                    "conf": float(args.yolo_conf),
                    "iou": float(args.yolo_iou),
                    "classes": args.yolo_classes,
                    "imgsz": int(args.imgsz),
                    "device": args.device,
                },
                "detections": detections_to_dict(dets),
                "vis_path": str(vis_path) if vis_path else None,
                "crops": crops_meta,
                "crop_settings": {
                    "pad_frac": float(args.pad),
                    "deleted_after_ocr": bool(args.delete_crops),
                },
                "candidate_filter": {
                    "allowed_ids": args.allowed_ids,
                    "ocr_char_set": args.ocr_char_set,
                    "min_box_area": args.min_box_area,
                },
                "ocr": {
                    "conf": float(args.ocr_conf),
                    "device": args.ocr_device,
                    "runs_dir": str(args.runs_dir),
                    "output_name": args.output_name,
                },
                "ocr_candidates": ocr_candidates,
            }

            summary = {
                "orig_img": str(img_path),
                "num_detections": len(img_results["detections"]),
                "num_ocr_candidates": len(ocr_candidates),
                "vis_path": img_results["vis_path"],
            }
            write_json(img_run_dir / "summary.json", summary)

            per_image_results.append(img_results)

        except Exception as e:
            failed_images.append({"image": str(img_path), "error": repr(e)})

    results = {
        "mode": "album",
        "input_folder_path": str(dir_path),
        "artifact_dir": str(run_dir),
        "num_images_total": int(num_total),
        "num_images_processed": int(len(per_image_results)),
        "num_images_failed": int(len(failed_images)),
        "failed_images": failed_images,
        "candidate_filter": {
            "allowed_ids": args.allowed_ids,
            "ocr_char_set": args.ocr_char_set,
            "min_box_area": args.min_box_area,
        },
        "yolo": {
            "weights": str(weights_path),
            "conf": float(args.yolo_conf),
            "iou": float(args.yolo_iou),
            "classes": args.yolo_classes,
            "imgsz": int(args.imgsz),
            "device": args.device,
        },
        "ocr": {
            "conf": float(args.ocr_conf),
            "device": args.ocr_device,
        },
        "per_image_results": per_image_results,
    }
    write_results(run_dir, results)

    # --- Production JSON output ---
    from .production import album_to_production_json, make_production_out_path, write_production_json

    prod = album_to_production_json(results)

    runs_dir = Path(args.runs_dir)
    out_path = make_production_out_path(
        mode="album",
        input_label=dir_path.name,
        runs_dir=runs_dir,
        output_name=args.output_name,
        include_ts_default=True,
    )
    write_production_json(prod, out_path)
    print(f"[album] production json: {out_path}")

    print(f"[album] wrote artifacts to: {run_dir}")
    if failed_images:
        print(f"[album] WARNING: {len(failed_images)} images failed (see artifacts + production meta.failed_images).")
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