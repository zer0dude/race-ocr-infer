from __future__ import annotations

import os
from pathlib import Path


DEFAULT_YOLO_URL = (
    "https://raw.githubusercontent.com/zer0dude/race-ocr/main/"
    "production_weights/yolo11s_bib-headband-racetag/weights/best.pt"
)


def default_cache_dir() -> Path:
    # Linux: ~/.cache/raceocr
    # macOS: ~/.cache/raceocr
    base = Path(os.path.expanduser("~/.cache"))
    return base / "raceocr"


def yolo_cache_path(cache_dir: Path) -> Path:
    return cache_dir / "yolo" / "best.pt"