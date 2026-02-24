from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

from PIL import Image, ImageDraw, ImageFont

# Ultralytics is a runtime dependency we add in Step 4.
from ultralytics import YOLO


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    xyxy: Tuple[float, float, float, float]


def load_yolo(weights: Path) -> YOLO:
    return YOLO(str(weights))


def run_yolo_detect(
    model: YOLO,
    img_path: Path,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 1280,
    device: Optional[str] = None,
) -> List[Detection]:
    """
    Runs YOLO on a single image, returns normalized detections list.
    """
    # device can be "cpu", "0", "cuda:0" etc; ultralytics accepts str or int
    pred = model.predict(
        source=str(img_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    if not pred:
        return []

    r0 = pred[0]
    names = r0.names  # dict: class_id -> class_name
    boxes = r0.boxes
    if boxes is None or boxes.xyxy is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy()

    out: List[Detection] = []
    for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
        cls_id = int(k)
        cls_name = names.get(cls_id, str(cls_id))
        out.append(
            Detection(
                cls_id=cls_id,
                cls_name=cls_name,
                conf=float(c),
                xyxy=(float(x1), float(y1), float(x2), float(y2)),
            )
        )
    # sort high->low confidence
    out.sort(key=lambda d: d.conf, reverse=True)
    return out


def render_detections(
    img_path: Path,
    detections: List[Detection],
    out_path: Path,
) -> None:
    """
    Saves a visualization image with bounding boxes + labels.
    """
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)

    # Optional default font; PIL may fallback if missing.
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for d in detections:
        x1, y1, x2, y2 = d.xyxy
        # use green border for rectangle
        draw.rectangle([x1, y1, x2, y2], width=3, outline="green")
        label = f"{d.cls_name} {d.conf:.2f}"
        # compute text box size in a Pillow-version-safe way
        if font:
            try:
                # Pillow >= 8: textbbox exists; Pillow 10 removed textsize
                l, t, r, b = draw.textbbox((0, 0), label, font=font)
                tw, th = (r - l), (b - t)
            except Exception:
                # fallback: font bbox
                try:
                    l, t, r, b = font.getbbox(label)
                    tw, th = (r - l), (b - t)
                except Exception:
                    tw, th = (len(label) * 6, 10)
        else:
            tw, th = (len(label) * 6, 10)

        pad = 2
        # background rectangle (clamp y so we don't go negative)
        # use red for rectangle outline and white for text
        y_top = max(0, y1 - th - 2 * pad)
        draw.rectangle([x1, y_top, x1 + tw + 2 * pad, y_top + th + 2 * pad], width=2, outline="red")
        draw.text((x1 + pad, y_top + pad), label, font=font, fill="white")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def detections_to_dict(dets: List[Detection]) -> List[Dict[str, Any]]:
    return [
        {
            "cls_id": d.cls_id,
            "cls_name": d.cls_name,
            "conf": d.conf,
            "xyxy": list(d.xyxy),
        }
        for d in dets
    ]

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def padded_xyxy(
    xyxy: Tuple[float, float, float, float],
    pad_frac: float,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    px = bw * pad_frac
    py = bh * pad_frac

    nx1 = int(clamp(x1 - px, 0, img_w - 1))
    ny1 = int(clamp(y1 - py, 0, img_h - 1))
    nx2 = int(clamp(x2 + px, 0, img_w - 1))
    ny2 = int(clamp(y2 + py, 0, img_h - 1))

    # Ensure valid box
    if nx2 <= nx1:
        nx2 = min(img_w - 1, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(img_h - 1, ny1 + 1)

    return nx1, ny1, nx2, ny2


def save_crops(
    img_path: Path,
    detections: List[Detection],
    crops_dir: Path,
    pad_frac: float,
) -> List[Dict[str, Any]]:
    """
    Saves crops for each detection and returns crop metadata list.
    """
    im = Image.open(img_path).convert("RGB")
    w, h = im.size

    crops_dir.mkdir(parents=True, exist_ok=True)

    crop_meta: List[Dict[str, Any]] = []
    for i, d in enumerate(detections):
        cx1, cy1, cx2, cy2 = padded_xyxy(d.xyxy, pad_frac, w, h)
        crop = im.crop((cx1, cy1, cx2, cy2))

        # filename includes rank index, class, conf
        fn = f"{img_path.stem}_det{i:03d}_c{d.cls_id}_{d.cls_name}_conf{d.conf:.3f}.jpg"
        out_path = crops_dir / fn
        crop.save(out_path)

        crop_meta.append(
            {
                "det_index": i,
                "cls_id": d.cls_id,
                "cls_name": d.cls_name,
                "conf": d.conf,
                "xyxy": list(d.xyxy),
                "xyxy_padded": [cx1, cy1, cx2, cy2],
                "crop_path": str(out_path),
            }
        )

    return crop_meta

def init_paddle_ocr(device: str = "cpu"):
    """
    Initialize PaddleOCR in a 'pretrained-only' config with orientation/unwarp/textline disabled.
    """
    import os
    # Skip network connectivity checks to model hosters (saves time, avoids noise)
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    from paddleocr import PaddleOCR

    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",
        device=device,
    )


def load_filter_words(filter_words: str, filter_words_file: str | None) -> List[str]:
    words: List[str] = []
    if filter_words:
        words.extend([w.strip() for w in filter_words.split(",") if w.strip()])

    if filter_words_file:
        p = Path(filter_words_file)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s and not s.startswith("#"):
                    words.append(s)

    # normalize to lowercase for matching
    out = []
    seen = set()
    for w in words:
        wl = w.lower()
        if wl not in seen:
            out.append(wl)
            seen.add(wl)
    return out


def is_filtered(text: str, filter_words_lc: List[str]) -> bool:
    """
    Simple rule (lean v0):
    - case-insensitive substring match against any filter word
    """
    tl = text.lower()
    return any(fw in tl for fw in filter_words_lc)


def run_ocr_on_crop_paths(
    ocr,
    crop_meta: List[Dict[str, Any]],
    ocr_conf: float,
    filter_words_lc: List[str],
) -> List[Dict[str, Any]]:
    """
    Runs OCR on each crop image path listed in crop_meta.
    Returns global ranked list of OCR candidates with provenance.
    Also injects per-crop OCR results into crop_meta entries under key "ocr".

    Supports both:
      - Classic PaddleOCR output: [[[box, (text, conf)], ...]] or [[box, (text, conf)], ...]
      - PaddleOCR v3 / PaddleX pipeline dict output:
          [{'rec_texts': [...], 'rec_scores': [...], 'rec_polys': [...], ...}]
    """
    candidates: List[Dict[str, Any]] = []

    for cm in crop_meta:
        crop_path = cm.get("crop_path")
        if not crop_path:
            cm["ocr"] = []
            continue

        # PaddleOCR API variants exist; handle the common ones
        try:
            ocr_out = ocr.ocr(crop_path)
        except TypeError:
            # some versions expect cls=False
            ocr_out = ocr.ocr(crop_path, cls=False)

        per_crop: List[Dict[str, Any]] = []

        # --- Case 1: PaddleOCR v3 / PaddleX pipeline dict output ---
        # Example:
        #   [{'rec_texts': [...], 'rec_scores': [...], 'rec_polys': [...], ...}]
        if isinstance(ocr_out, list) and len(ocr_out) == 1 and isinstance(ocr_out[0], dict):
            d0 = ocr_out[0]
            texts = d0.get("rec_texts") or []
            scores = d0.get("rec_scores") or []
            polys = d0.get("rec_polys") or [None] * len(texts)

            for text, conf, poly in zip(texts, scores, polys):
                if text is None:
                    continue

                text_str = str(text).strip()
                conf_f = float(conf)

                if conf_f < float(ocr_conf):
                    continue
                if not text_str:
                    continue
                if is_filtered(text_str, filter_words_lc):
                    continue

                rec = {
                    "text": text_str,
                    "conf": conf_f,
                    "crop_path": crop_path,
                    "det_index": cm.get("det_index"),
                    "cls_id": cm.get("cls_id"),
                    "cls_name": cm.get("cls_name"),
                    "line_box": poly,
                }
                per_crop.append(rec)
                candidates.append(rec)

        # --- Case 2: Classic PaddleOCR list output ---
        else:
            # Normalize to list-of-lines
            lines = []
            if ocr_out is None:
                lines = []
            elif isinstance(ocr_out, list):
                # Common formats:
                # 1) [ [ [box, (text, conf)], ... ] ]
                # 2) [ [box, (text, conf)], ... ]
                if len(ocr_out) == 0:
                    lines = []
                elif isinstance(ocr_out[0], list) and len(ocr_out) == 1 and (
                    len(ocr_out[0]) == 0 or isinstance(ocr_out[0][0], (list, tuple))
                ):
                    lines = ocr_out[0]
                else:
                    lines = ocr_out
            else:
                lines = []

            for item in lines:
                # Expect: [box, (text, conf)] or (box, (text, conf))
                try:
                    box = item[0]
                    text, conf = item[1]
                except Exception:
                    continue

                if text is None:
                    continue

                text_str = str(text).strip()
                conf_f = float(conf)

                if conf_f < float(ocr_conf):
                    continue
                if not text_str:
                    continue
                if is_filtered(text_str, filter_words_lc):
                    continue

                rec = {
                    "text": text_str,
                    "conf": conf_f,
                    "crop_path": crop_path,
                    "det_index": cm.get("det_index"),
                    "cls_id": cm.get("cls_id"),
                    "cls_name": cm.get("cls_name"),
                    "line_box": box,
                }
                per_crop.append(rec)
                candidates.append(rec)

        # rank within crop
        per_crop.sort(key=lambda r: r["conf"], reverse=True)
        cm["ocr"] = per_crop

    # global rank
    candidates.sort(key=lambda r: r["conf"], reverse=True)
    return candidates