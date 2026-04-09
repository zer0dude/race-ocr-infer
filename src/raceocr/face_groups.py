from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_groups(groups_json_path: Path) -> Dict[str, List[str]]:
    """Load refined_groups.json and return the groups dict (noise excluded)."""
    with open(groups_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("groups") or {}


def load_face_meta(embeddings_dir: Path, entry_id: str) -> Dict[str, Any]:
    """Load <entry_id>_meta.json from embeddings_dir."""
    meta_path = embeddings_dir / f"{entry_id}_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_image_index(images_dir: Path) -> Dict[str, Path]:
    """
    Walk images_dir recursively and build a {filename -> full_path} index.
    If the same filename appears in multiple subdirectories, last one wins.
    """
    index: Dict[str, Path] = {}
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff",
                "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.WEBP", "*.TIF", "*.TIFF"):
        for p in images_dir.rglob(ext):
            index[p.name] = p
    return index


def spatial_affinity(
    face_bbox: List[float],
    bib_xyxy: List[float],
    sigma: float = 1.5,
) -> float:
    """
    Compute spatial affinity between a face bounding box and a bib bounding box.

    Rewards bibs that are horizontally aligned with the face (gaussian falloff)
    and vertically positioned below the face top (runner's own bib is below their face).

    face_bbox: [x1, y1, x2, y2]
    bib_xyxy:  [bx1, by1, bx2, by2]
    sigma:     gaussian sigma as a multiple of face_width (default 1.5)

    Returns a value in (0, ~1.2].
    """
    fx1, fy1, fx2, _ = face_bbox
    bx1, by1, bx2, _ = bib_xyxy

    face_cx = (fx1 + fx2) / 2.0
    face_width = fx2 - fx1

    if face_width <= 0:
        return 0.0

    bib_cx = (bx1 + bx2) / 2.0

    # Gaussian horizontal alignment
    sigma_px = face_width * sigma
    horiz = math.exp(-0.5 * ((bib_cx - face_cx) / sigma_px) ** 2)

    # Vertical bonus: bib starting below face top → likely the runner's own bib
    vert = 1.2 if by1 > fy1 else 0.6

    return horiz * vert


def attribute_group(
    entries: List[str],
    image_index: Dict[str, Path],
    embeddings_dir: Path,
    yolo_model: Any,
    ocr: Any,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run bib OCR for all images in a face group and attribute a bib number
    using spatially-weighted voting.

    Returns a summary dict with best_guess, confidence, needs_review, etc.
    """
    from .infer import detections_to_dict, run_ocr_on_crop_paths, run_yolo_detect, save_crops
    from .production import infer_to_production_json

    yolo_conf = float(params.get("yolo_conf", 0.86))
    yolo_iou = float(params.get("yolo_iou", 0.45))
    imgsz = int(params.get("imgsz", 1280))
    device = params.get("device")
    yolo_classes = params.get("yolo_classes")
    ocr_conf = float(params.get("ocr_conf", 0.95))
    pad_frac = float(params.get("pad", 0.01))
    delete_crops = bool(params.get("delete_crops", False))
    allowed_ids = params.get("allowed_ids")
    ocr_char_set = params.get("ocr_char_set", "numeric")
    min_box_area = float(params.get("min_box_area", 10000.0))
    sigma = float(params.get("spatial_sigma", 1.5))
    flag_threshold = float(params.get("flag_threshold", 0.5))
    ambiguity_margin = float(params.get("ambiguity_margin", 0.15))
    crops_base_dir: Optional[Path] = params.get("crops_base_dir")

    # Accumulate weighted votes: text -> total_weight
    vote_totals: Dict[str, float] = {}
    num_images_with_bibs = 0

    for entry_id in entries:
        # Load face metadata
        try:
            meta = load_face_meta(embeddings_dir, entry_id)
        except Exception:
            continue

        orig_filename = meta.get("original_filename")
        face_bbox = meta.get("bbox")
        if not orig_filename or not face_bbox or len(face_bbox) != 4:
            continue

        img_path = image_index.get(orig_filename)
        if img_path is None or not img_path.exists():
            continue

        # Run YOLO + OCR
        try:
            dets = run_yolo_detect(
                model=yolo_model,
                img_path=img_path,
                conf=yolo_conf,
                iou=yolo_iou,
                imgsz=imgsz,
                device=device,
                classes=yolo_classes,
            )

            if not dets:
                continue

            if crops_base_dir is not None:
                effective_crops_dir = crops_base_dir / entry_id
            else:
                effective_crops_dir = Path("/tmp/raceocr_facegroups") / entry_id

            crop_meta = save_crops(
                img_path=img_path,
                detections=dets,
                crops_dir=effective_crops_dir,
                pad_frac=pad_frac,
            )

            ocr_candidates = run_ocr_on_crop_paths(
                ocr=ocr,
                crop_meta=crop_meta,
                ocr_conf=ocr_conf,
            )

            img_results = {
                "input_image_path": str(img_path),
                "detections": detections_to_dict(dets),
                "candidate_filter": {
                    "allowed_ids": allowed_ids,
                    "ocr_char_set": ocr_char_set,
                    "min_box_area": min_box_area,
                },
                "ocr": {"conf": ocr_conf},
                "ocr_candidates": ocr_candidates,
            }
            prod = infer_to_production_json(img_results)

            if delete_crops:
                for cm in crop_meta:
                    p = cm.get("crop_path")
                    if p:
                        try:
                            Path(p).unlink(missing_ok=True)
                        except Exception:
                            pass

            boxes = prod.get("boxes") or []
            valid_boxes = [b for b in boxes if b.get("ocr_result")]
            if not valid_boxes:
                continue

            num_images_with_bibs += 1

            for box in valid_boxes:
                text = box["ocr_result"]
                box_conf = float(box.get("box_confidence") or 0.0)
                ocr_conf_val = float(box.get("ocr_confidence") or 0.0)
                bib_xyxy = box.get("xyxy") or []

                if len(bib_xyxy) != 4:
                    continue

                affinity = spatial_affinity(face_bbox, bib_xyxy, sigma=sigma)
                weight = box_conf * ocr_conf_val * affinity

                vote_totals[text] = vote_totals.get(text, 0.0) + weight

        except Exception:
            continue

    # Aggregate
    if not vote_totals:
        return {
            "best_guess": None,
            "confidence": 0.0,
            "needs_review": True,
            "vote_breakdown": {},
            "num_images": len(entries),
            "num_images_with_bibs": num_images_with_bibs,
            "face_entries": entries,
        }

    sorted_candidates = sorted(vote_totals.items(), key=lambda x: x[1], reverse=True)
    total_weight = sum(w for _, w in sorted_candidates)

    best_text, best_weight = sorted_candidates[0]
    confidence = best_weight / total_weight if total_weight > 0 else 0.0

    # Ambiguity flag: top-2 candidates within ambiguity_margin of each other
    ambiguous = False
    if len(sorted_candidates) >= 2:
        second_weight = sorted_candidates[1][1]
        if best_weight > 0 and second_weight / best_weight > (1.0 - ambiguity_margin):
            ambiguous = True

    needs_review = confidence < flag_threshold or ambiguous

    return {
        "best_guess": best_text,
        "confidence": round(confidence, 4),
        "needs_review": needs_review,
        "vote_breakdown": {k: round(v, 4) for k, v in sorted_candidates},
        "num_images": len(entries),
        "num_images_with_bibs": num_images_with_bibs,
        "face_entries": entries,
    }


def run_face_groups(
    groups_path: Path,
    embeddings_dir: Path,
    images_dir: Path,
    out_path: Path,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Main entry point: load face groups, run OCR per image, attribute bib numbers,
    write results JSON to out_path.
    """
    from .config import default_cache_dir, yolo_cache_path
    from .infer import init_paddle_ocr, load_yolo
    from .util import write_json

    groups = load_groups(groups_path)
    num_groups = len(groups)
    print(f"[face-groups] loaded {num_groups} groups from {groups_path}")

    image_index = build_image_index(images_dir)
    print(f"[face-groups] indexed {len(image_index)} images under {images_dir}")

    # Resolve YOLO weights
    yolo_weights = params.get("yolo_weights")
    weights_path = Path(yolo_weights) if yolo_weights else yolo_cache_path(default_cache_dir())
    if not weights_path.exists():
        raise SystemExit(
            f"YOLO weights not found at {weights_path}. Run `raceocr setup` or pass --yolo-weights."
        )

    yolo_model = load_yolo(weights_path)
    ocr = init_paddle_ocr(ocr_device=params.get("ocr_device", "cpu"))

    groups_out: Dict[str, Any] = {}
    num_needs_review = 0

    for i, (group_id, entries) in enumerate(groups.items(), start=1):
        print(f"[face-groups] {i}/{num_groups} {group_id} ({len(entries)} entries)...")
        result = attribute_group(
            entries=entries,
            image_index=image_index,
            embeddings_dir=embeddings_dir,
            yolo_model=yolo_model,
            ocr=ocr,
            params=params,
        )
        groups_out[group_id] = result
        if result.get("needs_review"):
            num_needs_review += 1

    meta_out = {
        "approach": "spatial_weighted",
        "spatial_sigma": float(params.get("spatial_sigma", 1.5)),
        "flag_threshold": float(params.get("flag_threshold", 0.5)),
        "ambiguity_margin": float(params.get("ambiguity_margin", 0.15)),
        "yolo_conf": float(params.get("yolo_conf", 0.86)),
        "ocr_conf": float(params.get("ocr_conf", 0.95)),
        "min_box_area": float(params.get("min_box_area", 10000.0)),
        "ocr_char_set": params.get("ocr_char_set", "numeric"),
        "num_groups_total": num_groups,
        "num_groups_needs_review": num_needs_review,
        "groups_json": str(groups_path),
        "embeddings_dir": str(embeddings_dir),
        "images_dir": str(images_dir),
    }

    output = {
        "groups": groups_out,
        "meta": meta_out,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, output)
    print(f"[face-groups] wrote results to {out_path}")
    print(f"[face-groups] {num_needs_review}/{num_groups} groups flagged for review")

    return output
