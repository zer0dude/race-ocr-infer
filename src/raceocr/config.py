from __future__ import annotations

import os
from pathlib import Path


# IMPORTANT:
# GitHub "blob" URLs are HTML pages. For downloads, use the "raw" URL.
DEFAULT_YOLO_URL = (
    "https://raw.githubusercontent.com/zer0dude/race-ocr/main/"
    "production_weights/yolo11s_bib-headband-racetag/weights/best.pt"
)


def default_cache_dir() -> Path:
    # Linux: ~/.cache/raceocr
    # macOS: ~/.cache/raceocr (fine for our use; keep simple)
    base = Path(os.path.expanduser("~/.cache"))
    return base / "raceocr"


def yolo_cache_path(cache_dir: Path) -> Path:
    return cache_dir / "yolo" / "best.pt"