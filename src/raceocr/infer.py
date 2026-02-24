from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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