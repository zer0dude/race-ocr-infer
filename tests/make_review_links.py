#!/usr/bin/env python3
"""
make_review_links.py

Given the output of `raceocr face-groups`, create a symlink-based folder
structure for human review. Groups sharing the same bib number are
collected under a common bib sub-folder, and a face-crop image is
produced per detection entry to make review faster.

  <out_dir>/
    confident/
      <bib_number>/                    <- groups that share the same confident bib
        <group_id>/
          <image1.jpg> -> /abs/path/to/original
          <entry_id>_face.jpg          <- cropped face from that detection
    review/
      <bib_number>/
        <group_id>/                    <- needs-review groups with a guess
          ...
    no_bib/
      <group_id>/                      <- groups where no bib was ever detected
        ...
    noise/
      <entry_id>_face.jpg              <- flat; one face crop per noise detection
      <image.jpg> -> ...

Usage:
    python tests/make_review_links.py \\
        --result  data/face-group-test/runs/face_groups_result.json \\
        --groups  data/face-group-test/runs/groups_refined/refined_groups.json \\
        --embeddings-dir data/face-group-test/runs/embeddings \\
        --images-dir     data/face-group-test/test \\
        --out            data/face-group-test/review_links
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def build_image_index(images_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for p in images_dir.rglob("*"):
        if p.suffix.lower() in exts:
            index[p.name] = p
    return index


def load_face_meta(embeddings_dir: Path, entry_id: str) -> Optional[Dict[str, Any]]:
    """Return {original_filename, bbox} dict for this entry, or None on failure."""
    meta_path = embeddings_dir / f"{entry_id}_meta.json"
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("original_filename") and data.get("bbox"):
            return data
        return None
    except Exception:
        return None


def crop_face(img_path: Path, bbox: List[float], out_path: Path, pad_frac: float = 0.3) -> bool:
    """
    Crop the face region from img_path using bbox [x1, y1, x2, y2],
    with proportional padding on each side. Saves to out_path as JPEG.
    Returns True on success.
    """
    try:
        from PIL import Image
        img = Image.open(img_path)
        w, h = img.size
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bw = x2 - x1
        bh = y2 - y1
        pad_x = bw * pad_frac
        pad_y = bh * pad_frac
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w, int(x2 + pad_x))
        cy2 = min(h, int(y2 + pad_y))
        cropped = img.crop((cx1, cy1, cx2, cy2))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.convert("RGB").save(out_path, "JPEG", quality=90)
        return True
    except Exception:
        return False


def make_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target.resolve())


# ---------------------------------------------------------------------------
# Per-entry processing
# ---------------------------------------------------------------------------

def process_entry(
    entry_id: str,
    group_dir: Path,
    embeddings_dir: Path,
    image_index: Dict[str, Path],
    counts: Dict[str, int],
) -> None:
    """Symlink the original image and write a face crop for one entry."""
    meta = load_face_meta(embeddings_dir, entry_id)
    if not meta:
        counts["missing_meta"] += 1
        return

    orig_filename: str = meta["original_filename"]
    bbox: List[float] = meta["bbox"]

    img_path = image_index.get(orig_filename)
    if img_path is None:
        counts["missing_img"] += 1
        return

    # Symlink to original image (deduplicate if two entries share the same image)
    link = group_dir / img_path.name
    if link.exists() or link.is_symlink():
        stem, suffix = img_path.stem, img_path.suffix
        link = group_dir / f"{stem}__{entry_id}{suffix}"
    make_symlink(img_path, link)

    # Face crop
    face_out = group_dir / f"{entry_id}_face.jpg"
    if crop_face(img_path, bbox, face_out):
        counts["crops"] += 1
    else:
        counts["crop_failures"] += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Create symlink review folders from face-groups output.")
    parser.add_argument("--result", required=True, help="Path to face_groups_result.json")
    parser.add_argument("--groups", required=True, help="Path to refined_groups.json (for noise entries)")
    parser.add_argument("--embeddings-dir", required=True, help="Embeddings folder with *_meta.json files")
    parser.add_argument("--images-dir", required=True, help="Folder containing original images (searched recursively)")
    parser.add_argument("--out", required=True, help="Output folder for symlink tree")
    parser.add_argument(
        "--face-pad",
        type=float,
        default=0.3,
        help="Padding around face bbox as fraction of bbox size for crops (default: 0.3)",
    )
    args = parser.parse_args()

    result_path = Path(args.result)
    groups_path = Path(args.groups)
    embeddings_dir = Path(args.embeddings_dir)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out)
    face_pad = float(args.face_pad)

    print(f"[make_review_links] loading results from {result_path}")
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    with open(groups_path, "r", encoding="utf-8") as f:
        raw_groups = json.load(f)
    noise_entries: List[str] = raw_groups.get("noise") or []

    print(f"[make_review_links] indexing images under {images_dir}")
    image_index = build_image_index(images_dir)
    print(f"[make_review_links] indexed {len(image_index)} images")

    groups: Dict = result.get("groups") or {}

    counts: Dict[str, int] = {
        "confident": 0, "multiple_plausible_bib_ids": 0,
        "cross_group_duplicate": 0, "insufficient_plausible_bibs": 0,
        "noise": 0, "crops": 0, "crop_failures": 0,
        "missing_meta": 0, "missing_img": 0,
    }

    # ---- confident + review + no_bib ----------------------------------------
    for group_id, info in groups.items():
        status: str = info.get("status", "insufficient_plausible_bibs")
        best_guess: Optional[str] = info.get("best_guess")
        entries: List[str] = info.get("face_entries") or []

        if status == "insufficient_plausible_bibs":
            group_dir = out_dir / "no_bib" / group_id
            counts["insufficient_plausible_bibs"] += 1
        elif status == "confident":
            group_dir = out_dir / "confident" / best_guess / group_id
            counts["confident"] += 1
        else:
            # "multiple_plausible_bib_ids" or "cross_group_duplicate"
            group_dir = out_dir / "review" / status / best_guess / group_id
            counts[status] = counts.get(status, 0) + 1

        for entry_id in entries:
            process_entry(entry_id, group_dir, embeddings_dir, image_index, counts)

    # ---- noise ---------------------------------------------------------------
    noise_dir = out_dir / "noise"
    for entry_id in noise_entries:
        meta = load_face_meta(embeddings_dir, entry_id)
        if not meta:
            counts["missing_meta"] += 1
            continue
        img_path = image_index.get(meta["original_filename"])
        if img_path is None:
            counts["missing_img"] += 1
            continue

        link = noise_dir / img_path.name
        if link.exists() or link.is_symlink():
            stem, suffix = img_path.stem, img_path.suffix
            link = noise_dir / f"{stem}__{entry_id}{suffix}"
        make_symlink(img_path, link)

        face_out = noise_dir / f"{entry_id}_face.jpg"
        if crop_face(img_path, meta["bbox"], face_out, pad_frac=face_pad):
            counts["crops"] += 1
        else:
            counts["crop_failures"] += 1
        counts["noise"] += 1

    print()
    print("[make_review_links] done.")
    print(f"  confident                : {counts['confident']:>4}  -> {out_dir}/confident/<bib>/")
    print(f"  multiple plausible bibs  : {counts['multiple_plausible_bib_ids']:>4}  -> {out_dir}/review/multiple_plausible_bib_ids/<bib>/")
    print(f"  cross-group duplicates   : {counts['cross_group_duplicate']:>4}  -> {out_dir}/review/cross_group_duplicate/<bib>/")
    print(f"  insufficient plausible   : {counts['insufficient_plausible_bibs']:>4}  -> {out_dir}/no_bib/")
    print(f"  noise entries            : {counts['noise']:>4}  -> {out_dir}/noise/")
    print(f"  face crops made  : {counts['crops']:>4}")
    if counts["crop_failures"]:
        print(f"  crop failures    : {counts['crop_failures']:>4}")
    if counts["missing_meta"] or counts["missing_img"]:
        print(f"  missing meta     : {counts['missing_meta']:>4}  (entries with no _meta.json)")
        print(f"  missing images   : {counts['missing_img']:>4}  (meta found but image not in index)")


if __name__ == "__main__":
    main()
