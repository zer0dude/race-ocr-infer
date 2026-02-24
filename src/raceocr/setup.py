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