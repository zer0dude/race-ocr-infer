from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .util import ensure_dir, safe_stem, utc_timestamp_compact, write_json


def make_run_dir(mode: str, input_label: str, out_dir: str | Path = "artifacts") -> Path:
    """
    Creates a deterministic run folder:
      <out_dir>/run_<timestamp>_<mode>_<input_label>/
    """
    out_dir = Path(out_dir)
    ts = utc_timestamp_compact()
    label = safe_stem(input_label)
    run_dir = out_dir / f"run_{ts}_{mode}_{label}"
    ensure_dir(run_dir)
    return run_dir


def write_params(run_dir: Path, params: Dict[str, Any]) -> Path:
    path = run_dir / "params.json"
    write_json(path, params)
    return path


def write_results(run_dir: Path, results: Dict[str, Any]) -> Path:
    path = run_dir / "results.json"
    write_json(path, results)
    return path