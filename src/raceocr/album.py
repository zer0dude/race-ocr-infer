from __future__ import annotations

from pathlib import Path
from typing import List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    """
    Return sorted image files in a folder (non-recursive).
    """
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    paths.sort()
    return paths


def list_images_recursive(folder: Path) -> List[Path]:
    """
    Return sorted image files in a folder tree (recursive).
    """
    paths = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    paths.sort()
    return paths