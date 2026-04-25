from __future__ import annotations

import json
import math
from datetime import datetime, timezone
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


def spatial_affinity(
    face_bbox: List[float],
    bib_xyxy: List[float],
    sigma: float = 1.0,
    max_bib_dist_factor: float = 1.0,
) -> float:
    """
    Compute spatial affinity between a face bounding box and a bib bounding box.

    Two hard plausibility gates are applied before scoring. A bib that fails
    either gate is fully discarded (returns 0.0):

      Gate 1 — vertical: bib bottom above face top (bib_y2 < face_y1) is
        physically impossible for the target runner's own bib.

      Gate 2 — horizontal: nearest bib edge more than
        max_bib_dist_factor × bib_width from the face center is too far
        away to plausibly belong to this runner.

    Bibs that pass both gates are scored with a gaussian horizontal
    alignment (tighter sigma = stronger preference for bib directly below
    the face) and a vertical multiplier (1.2 if below face top, 0.6 if
    overlapping or above).

    face_bbox:           [x1, y1, x2, y2]
    bib_xyxy:            [bx1, by1, bx2, by2]
    sigma:               gaussian sigma as a multiple of face_width (default 1.0)
    max_bib_dist_factor: hard horizontal cutoff in bib-widths (default 1.5)

    Returns a value in [0, ~1.2].
    """
    fx1, fy1, fx2, _ = face_bbox
    bx1, by1, bx2, by2 = bib_xyxy

    face_cx = (fx1 + fx2) / 2.0
    face_width = fx2 - fx1

    if face_width <= 0:
        return 0.0

    # Gate 1: bib entirely above the face — physically impossible for target runner
    if by2 < fy1:
        return 0.0

    # Gate 2: nearest bib edge too far horizontally — not plausibly this runner's bib
    bib_width = bx2 - bx1
    if bib_width > 0:
        edge_dist = max(0.0, bx1 - face_cx, face_cx - bx2)
        if edge_dist > max_bib_dist_factor * bib_width:
            return 0.0

    bib_cx = (bx1 + bx2) / 2.0

    # Gaussian horizontal alignment
    sigma_px = face_width * sigma
    horiz = math.exp(-0.5 * ((bib_cx - face_cx) / sigma_px) ** 2)

    # Vertical multiplier: bib starting below face top → likely the runner's own bib
    vert = 1.2 if by1 > fy1 else 0.6

    return horiz * vert


def attribute_group(
    entries: List[str],
    embeddings_dir: Path,
    image_boxes: Dict[str, List[Dict]],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attribute a bib number to a face group using spatially-weighted voting over
    pre-computed album boxes.

    image_boxes: {bare_filename -> boxes list from album production JSON}

    Returns a summary dict with best_guess, confidence, status, etc.
    """
    sigma = float(params.get("spatial_sigma", 1.0))
    max_bib_dist_factor = float(params.get("max_bib_dist_factor", 1.0))
    flag_threshold = float(params.get("flag_threshold", 0.5))
    ambiguity_margin = float(params.get("ambiguity_margin", 0.15))

    vote_totals: Dict[str, float] = {}
    num_images_with_bibs = 0
    num_plausible_bib_detections = 0
    source_images: List[str] = []
    seen_source_images: set = set()

    for entry_id in entries:
        try:
            meta = load_face_meta(embeddings_dir, entry_id)
        except Exception:
            continue

        orig_filename = meta.get("original_filename")
        face_bbox = meta.get("bbox")
        if not orig_filename or not face_bbox or len(face_bbox) != 4:
            continue

        # Track source images regardless of whether boxes are found
        if orig_filename not in seen_source_images:
            source_images.append(orig_filename)
            seen_source_images.add(orig_filename)

        boxes = image_boxes.get(orig_filename, [])
        # Album output is pre-filtered; only keep boxes with a non-empty ocr_result
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

            affinity = spatial_affinity(
                face_bbox, bib_xyxy,
                sigma=sigma,
                max_bib_dist_factor=max_bib_dist_factor,
            )
            if affinity > 0:
                num_plausible_bib_detections += 1
            weight = box_conf * ocr_conf_val * affinity

            vote_totals[text] = vote_totals.get(text, 0.0) + weight

    # Aggregate
    if not vote_totals or num_plausible_bib_detections < 2:
        return {
            "best_guess": None,
            "confidence": 0.0,
            "status": "insufficient_plausible_bibs",
            "vote_breakdown": {},
            "source_images": source_images,
            "num_images": len(entries),
            "num_images_with_bibs": num_images_with_bibs,
            "num_plausible_bib_detections": num_plausible_bib_detections,
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

    status = "multiple_plausible_bib_ids" if (confidence < flag_threshold or ambiguous) else "confident"

    return {
        "best_guess": best_text,
        "confidence": round(confidence, 4),
        "status": status,
        "vote_breakdown": {k: round(v, 4) for k, v in sorted_candidates},
        "source_images": source_images,
        "num_images": len(entries),
        "num_images_with_bibs": num_images_with_bibs,
        "num_plausible_bib_detections": num_plausible_bib_detections,
        "face_entries": entries,
    }


def run_face_groups(
    groups_path: Path,
    embeddings_dir: Path,
    album_results_path: Path,
    out_path: Path,
    params: Dict[str, Any],
    started_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Main entry point: load face groups and album OCR results, attribute bib numbers
    via spatial voting, write results JSON to out_path.
    """
    from .util import write_json

    groups = load_groups(groups_path)
    num_groups = len(groups)
    print(f"[face-groups] loaded {num_groups} groups from {groups_path}")

    with open(album_results_path, "r", encoding="utf-8") as f:
        album_data = json.load(f)

    image_boxes: Dict[str, List[Dict]] = {}
    for img_entry in album_data.get("images") or []:
        fname = Path(img_entry.get("orig_img") or "").name
        image_boxes[fname] = img_entry.get("boxes") or []

    print(f"[face-groups] loaded album results from {album_results_path} ({len(image_boxes)} images indexed)")

    groups_out: Dict[str, Any] = {}

    for i, (group_id, entries) in enumerate(groups.items(), start=1):
        print(f"[face-groups] {i}/{num_groups} {group_id} ({len(entries)} entries)...")
        result = attribute_group(
            entries=entries,
            embeddings_dir=embeddings_dir,
            image_boxes=image_boxes,
            params=params,
        )
        groups_out[group_id] = result

    # Cross-group duplicate detection: flag groups sharing the same best_guess
    from collections import Counter
    guess_counts = Counter(
        info["best_guess"] for info in groups_out.values()
        if info["best_guess"] is not None
    )
    duplicate_guesses = {g for g, count in guess_counts.items() if count > 1}
    for info in groups_out.values():
        if info["best_guess"] in duplicate_guesses:
            info["status"] = "cross_group_duplicate"

    num_confident = sum(1 for g in groups_out.values() if g["status"] == "confident")
    num_multiple = sum(1 for g in groups_out.values() if g["status"] == "multiple_plausible_bib_ids")
    num_duplicate = sum(1 for g in groups_out.values() if g["status"] == "cross_group_duplicate")
    num_insufficient = sum(1 for g in groups_out.values() if g["status"] == "insufficient_plausible_bibs")

    finished_at = datetime.now(timezone.utc)
    if started_at is None:
        started_at = finished_at
    duration_s = (finished_at - started_at).total_seconds()

    meta_out = {
        "approach": "spatial_weighted",
        "spatial_sigma": float(params.get("spatial_sigma", 1.0)),
        "max_bib_dist_factor": float(params.get("max_bib_dist_factor", 1.0)),
        "flag_threshold": float(params.get("flag_threshold", 0.5)),
        "ambiguity_margin": float(params.get("ambiguity_margin", 0.15)),
        "num_groups_total": num_groups,
        "num_confident": num_confident,
        "num_multiple_plausible_bib_ids": num_multiple,
        "num_cross_group_duplicate": num_duplicate,
        "num_insufficient_plausible_bibs": num_insufficient,
        "groups_json": str(groups_path),
        "embeddings_dir": str(embeddings_dir),
        "album_results": str(album_results_path),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round(duration_s, 3),
    }

    output = {
        "groups": groups_out,
        "meta": meta_out,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, output)
    print(f"[face-groups] wrote results to {out_path}")
    print(f"[face-groups] confident: {num_confident}, multiple_plausible_bib_ids: {num_multiple}, cross_group_duplicate: {num_duplicate}, insufficient_plausible_bibs: {num_insufficient}")

    return output
