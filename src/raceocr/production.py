from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


def utc_ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def ensure_json_suffix(name: str) -> str:
    name = name.strip()
    if not name.lower().endswith(".json"):
        name += ".json"
    return name


def make_production_out_path(
    *,
    mode: str,
    input_label: str,
    runs_dir: Path,
    output_name: Optional[str] = None,
    include_ts_default: bool = True,
) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)

    if output_name:
        fn = ensure_json_suffix(output_name)
    else:
        if include_ts_default:
            fn = f"{mode}_{input_label}_{utc_ts_compact()}.json"
        else:
            fn = f"{mode}_{input_label}.json"

    return runs_dir / fn


def group_ocr_candidates_by_det(ocr_candidates: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by_det: Dict[int, List[Dict[str, Any]]] = {}
    for r in ocr_candidates or []:
        det_index = r.get("det_index")
        if det_index is None:
            continue

        try:
            det_index = int(det_index)
        except Exception:
            continue

        text = str(r.get("text", "") or "").strip()
        conf = float(r.get("conf", 0.0) or 0.0)

        by_det.setdefault(det_index, []).append(
            {
                "text": text,
                "conf": conf,
            }
        )

    return by_det


def text_allowed_by_char_set(text: str, ocr_char_set: str) -> bool:
    if not text:
        return False

    if ocr_char_set == "numeric":
        return text.isdigit()

    if ocr_char_set == "alnum":
        return text.isalnum()

    if ocr_char_set == "any":
        return True

    raise ValueError(f"Unsupported ocr_char_set: {ocr_char_set}")


def filter_candidate_list(
    candidates: List[Dict[str, Any]],
    allowed_ids: Optional[List[str]],
    ocr_char_set: str,
) -> List[Dict[str, Any]]:
    allowed_set = set(allowed_ids) if allowed_ids else None

    out: List[Dict[str, Any]] = []
    for c in candidates or []:
        text = str(c.get("text", "") or "").strip()
        conf = float(c.get("conf", 0.0) or 0.0)

        if not text:
            continue

        if allowed_set is not None and text not in allowed_set:
            continue

        if not text_allowed_by_char_set(text, ocr_char_set):
            continue

        out.append(
            {
                "text": text,
                "conf": conf,
            }
        )

    out.sort(key=lambda x: x["conf"], reverse=True)
    return out


def xyxy_box_area(xyxy: Any) -> float:
    """
    Compute bounding box area from [x1, y1, x2, y2].
    Returns 0.0 on malformed input.
    """
    try:
        if xyxy is None or len(xyxy) != 4:
            return 0.0

        x1, y1, x2, y2 = [float(v) for v in xyxy]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        return w * h
    except Exception:
        return 0.0


def box_passes_area_filter(xyxy: Any, min_box_area: float) -> bool:
    return xyxy_box_area(xyxy) >= float(min_box_area)


def infer_to_production_json(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert internal infer results to a compact production JSON.

    Filtering, candidate sorting, and best-candidate selection are all handled here.
    """
    orig_img = results.get("input_image_path") or results.get("orig_img") or ""
    detections = results.get("detections") or []
    ocr_candidates = results.get("ocr_candidates") or []

    candidate_filter = results.get("candidate_filter") or {}
    allowed_ids = candidate_filter.get("allowed_ids")
    ocr_char_set = candidate_filter.get("ocr_char_set") or "numeric"
    min_box_area = float(candidate_filter.get("min_box_area", 10000.0) or 10000.0)

    by_det_raw = group_ocr_candidates_by_det(ocr_candidates)

    boxes_out: List[Dict[str, Any]] = []
    for det_idx, det in enumerate(detections):
        xyxy = det.get("xyxy")
        box_conf = float(det.get("conf", 0.0))
        cls_name = det.get("cls_name", "")

        if not box_passes_area_filter(xyxy, min_box_area):
            continue

        raw_candidates = by_det_raw.get(det_idx, [])
        cand_list = filter_candidate_list(
            raw_candidates,
            allowed_ids=allowed_ids,
            ocr_char_set=ocr_char_set,
        )
        best = cand_list[0] if cand_list else {"text": "", "conf": 0.0}

        boxes_out.append(
            {
                "xyxy": xyxy,
                "box_confidence": box_conf,
                "box_class": cls_name,
                "ocr_result": best.get("text", "") or "",
                "ocr_confidence": float(best.get("conf", 0.0) or 0.0),
                "ocr_method": "paddleocr",
                "ocr_candidates": cand_list,
            }
        )

    meta = {
        "yolo_weights": (results.get("yolo") or {}).get("weights"),
        "yolo_conf": (results.get("yolo") or {}).get("conf"),
        "ocr_conf_thresh": (results.get("ocr") or {}).get("conf"),
        "allowed_ids": allowed_ids,
        "ocr_char_set": ocr_char_set,
        "min_box_area": min_box_area,
    }

    return {
        "orig_img": orig_img,
        "boxes": boxes_out,
        "meta": meta,
    }


def album_to_production_json(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Album mode is a simple batch infer over all images in a folder.
    """
    orig_album = results.get("input_folder_path") or results.get("orig_album") or ""
    per_image_internal = results.get("per_image_results") or []
    candidate_filter = results.get("candidate_filter") or {}

    images_out: List[Dict[str, Any]] = []
    for img_res in per_image_internal:
        prod = infer_to_production_json(img_res)
        prod.pop("meta", None)
        images_out.append(prod)

    meta_out = {
        "num_images_total": int(results.get("num_images_total") or 0),
        "num_images_processed": int(results.get("num_images_processed") or 0),
        "num_images_failed": int(results.get("num_images_failed") or 0),
        "failed_images": results.get("failed_images") or [],
        "yolo_weights": (results.get("yolo") or {}).get("weights"),
        "yolo_conf": (results.get("yolo") or {}).get("conf"),
        "yolo_iou": (results.get("yolo") or {}).get("iou"),
        "imgsz": (results.get("yolo") or {}).get("imgsz"),
        "device": (results.get("yolo") or {}).get("device"),
        "ocr_conf_thresh": (results.get("ocr") or {}).get("conf"),
        "allowed_ids": candidate_filter.get("allowed_ids"),
        "ocr_char_set": candidate_filter.get("ocr_char_set") or "numeric",
        "min_box_area": float(candidate_filter.get("min_box_area", 10000.0) or 10000.0),
    }

    return {
        "orig_album": orig_album,
        "images": images_out,
        "meta": meta_out,
    }


def write_production_json(obj: Dict[str, Any], out_path: Path) -> None:
    import json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")