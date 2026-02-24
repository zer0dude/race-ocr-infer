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
    """
    JSON writer that safely handles numpy arrays/scalars and other non-JSON types.
    """
    def to_jsonable(x: Any) -> Any:
        # Handle numpy types if present (without hard dependency)
        try:
            import numpy as np  # type: ignore
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (np.generic,)):
                return x.item()
        except Exception:
            pass

        # Common python containers
        if isinstance(x, dict):
            return {str(k): to_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [to_jsonable(v) for v in x]
        if isinstance(x, set):
            return [to_jsonable(v) for v in sorted(x)]

        # Path-like
        try:
            from pathlib import Path as _Path
            if isinstance(x, _Path):
                return str(x)
        except Exception:
            pass

        # Fallback: primitives or string
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        return str(x)

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)
        


def get_env_info() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
    }