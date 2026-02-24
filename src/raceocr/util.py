from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_timestamp_compact() -> str:
    # Example: 20260224_153012Z
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def safe_stem(name: str, max_len: int = 80) -> str:
    """
    Make a filesystem-safe stem from a path stem or user-provided label.
    Keeps alnum, dash, underscore. Converts others to underscore.
    """
    out = []
    for ch in name:
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    if not s:
        s = "input"
    return s[:max_len]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_env_info() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
    }