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
        by_det.setdefault(det_index, []).append({"text": r.get("text", ""), "conf": float(r.get("conf", 0.0))})

    for k in list(by_det.keys()):
        by_det[k].sort(key=lambda x: x["conf"], reverse=True)
    return by_det


def infer_to_production_json(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert internal infer results to a compact production JSON that matches the client's pipeline style.
    Expects results dict from cli.py infer run (contains detections + ocr_candidates + filter_words_used + yolo/ocr meta).
    """
    orig_img = results.get("input_image_path") or results.get("orig_img") or ""
    detections = results.get("detections") or []
    ocr_candidates = results.get("ocr_candidates") or []

    by_det = group_ocr_candidates_by_det(ocr_candidates)

    boxes_out: List[Dict[str, Any]] = []
    for det_idx, det in enumerate(detections):
        xyxy = det.get("xyxy")
        box_conf = float(det.get("conf", 0.0))
        cls_name = det.get("cls_name", "")

        cand_list = by_det.get(det_idx, [])
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
        "filter_words_used": results.get("filter_words_used") or [],
    }

    return {
        "orig_img": orig_img,
        "boxes": boxes_out,
        "meta": meta,
    }


def album_to_production_json(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert internal album results to a compact production JSON.
    """
    orig_album = results.get("input_folder_path") or results.get("orig_album") or ""
    num_images = int(results.get("num_images") or 0)

    ranked = results.get("ranked_guesses") or []
    ranked_out = []
    for r in ranked:
        text = r.get("canonical", "")
        count = int(r.get("count_images") or 0)
        total = int(r.get("num_images") or num_images or 0)
        ratio = float(r.get("ratio") or 0.0)
        ranked_out.append({"text": text, "count": count, "total": total, "ratio": ratio})

    meta = {
        "ocr_conf_thresh": (results.get("ocr") or {}).get("conf"),
        "filter_words_used": results.get("filter_words_used") or [],
    }

    return {
        "orig_album": orig_album,
        "num_images": num_images,
        "best_guess": results.get("best_guess"),
        "best_guess_ratio": float(results.get("best_guess_ratio") or 0.0),
        "album_conf_thresh": float(results.get("album_conf_thresh") or 0.75),
        "num_guesses_above_thresh": int(results.get("num_guesses_above_thresh") or 0),
        "needs_manual_check": bool(results.get("needs_manual_check")),
        "ranked_guesses": ranked_out,
        "meta": meta,
    }


def write_production_json(obj: Dict[str, Any], out_path: Path) -> None:
    import json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")