from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

from .util import ensure_dir


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> Tuple[Path, int]:
    """
    Download URL to dest (atomic replace).
    Returns (dest, bytes_written).
    """
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)

    # Atomic-ish replace
    tmp.replace(dest)
    size = dest.stat().st_size
    return dest, size


def ensure_yolo_weights(
    *,
    url: str,
    dest: Path,
    sha256: Optional[str] = None,
    force: bool = False,
) -> Path:
    """
    Ensure YOLO weights exist at dest; download if missing or force=True.
    Optionally verify sha256.
    """
    if dest.exists() and not force:
        if sha256:
            got = sha256_file(dest)
            if got.lower() != sha256.lower():
                raise RuntimeError(
                    f"SHA256 mismatch for existing file: {dest}\n"
                    f"expected={sha256}\n"
                    f"got     ={got}\n"
                    "Use --force to re-download."
                )
        return dest

    downloaded, size = download_file(url, dest)

    if sha256:
        got = sha256_file(downloaded)
        if got.lower() != sha256.lower():
            raise RuntimeError(
                f"SHA256 mismatch after download: {downloaded}\n"
                f"expected={sha256}\n"
                f"got     ={got}\n"
            )

    return downloaded

def warm_paddleocr_cpu() -> None:
    """
    Minimal PaddleOCR warm routine to ensure models are downloaded and the pipeline initializes.
    Runs on CPU to avoid CUDA/Torch conflicts.
    """
    import os

    # Skip Paddle model source connectivity checks (faster/less noisy)
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    # Optional stability: disable mkldnn if needed in some environments (safe to leave enabled too)
    # os.environ.setdefault("FLAGS_use_mkldnn", "0")

    from paddleocr import PaddleOCR
    from PIL import Image
    import tempfile

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",
        device="cpu",
    )

    # Run once on a tiny dummy image to force full pipeline init
    img = Image.new("RGB", (16, 16), color=(255, 255, 255))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        img.save(tmp.name)
        try:
            _ = ocr.ocr(tmp.name)
        except TypeError:
            _ = ocr.ocr(tmp.name, cls=False)