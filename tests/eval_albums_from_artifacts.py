#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

def norm(s: Any) -> str:
    return str(s).strip().upper() if s is not None else ""


def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory (default: artifacts)")
    ap.add_argument("--out-dir", default="tests/runs", help="Where to write summaries (default: tests/runs)")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all album results
    results_files = sorted(artifacts_dir.glob("run_*_album_*/results.json"))
    if not results_files:
        print(f"No album results.json found under: {artifacts_dir}")
        return 2

    records: List[Dict[str, Any]] = []

    for rp in results_files:
        d = load_json(rp)
        # folder name is the suffix of the run folder: run_<ts>_album_<folder>
        run_folder = rp.parent.name
        # expected id is everything after "_album_"
        try:
            expected = run_folder.split("_album_", 1)[1]
        except Exception:
            expected = ""

        best_guess = norm(d.get("best_guess"))
        expected_norm = norm(expected)

        num_images = int(d.get("num_images") or 0)
        is_confident = bool(d.get("is_confident"))
        needs_manual_check = bool(d.get("needs_manual_check"))
        best_ratio = float(d.get("best_guess_ratio") or 0.0)

        correct = (best_guess == expected_norm)
        correct_if_auto = correct and is_confident  # “taking manual check into account”

        records.append(
            {
                "run_folder": run_folder,
                "results_json": str(rp),
                "expected": expected_norm,
                "best_guess": best_guess,
                "best_guess_ratio": best_ratio,
                "num_images": num_images,
                "is_confident": is_confident,
                "needs_manual_check": needs_manual_check,
                "correct": correct,
                "correct_if_auto": correct_if_auto,
            }
        )

    total = len(records)
    correct_any = sum(r["correct"] for r in records)
    overall_acc = correct_any / total if total else 0.0

    auto_records = [r for r in records if r["is_confident"]]
    auto_total = len(auto_records)
    auto_correct = sum(r["correct"] for r in auto_records)
    auto_acc = auto_correct / auto_total if auto_total else 0.0
    auto_coverage = auto_total / total if total else 0.0

    manual_records = [r for r in records if r["needs_manual_check"]]
    manual_total = len(manual_records)

    # >=4 image albums: how many are correct and not-needs-manual-check?
    ge4 = [r for r in records if int(r["num_images"]) >= 4]
    ge4_total = len(ge4)
    ge4_correct_auto = sum(1 for r in ge4 if r["correct"] and (not r["needs_manual_check"]))
    ge4_ratio_correct_auto = (ge4_correct_auto / ge4_total) if ge4_total else 0.0

    manual_by_size = {1: 0, 2: 0, 3: 0, ">=4": 0}
    for r in manual_records:
        n = r["num_images"]
        if n <= 1:
            manual_by_size[1] += 1
        elif n == 2:
            manual_by_size[2] += 1
        elif n == 3:
            manual_by_size[3] += 1
        else:
            manual_by_size[">=4"] += 1

    manual_check_albums = sorted(
        {
            r["expected"]  # expected is the folder id/name
            for r in manual_records
            if r.get("expected")
        }
    )

    summary = {
        "counts": {
            "albums_total": total,
            "albums_auto_accepted": auto_total,
            "albums_manual_check": manual_total,
        },
        "accuracy": {
            # 1) disregarding manual check
            "best_guess_correct": correct_any,
            "best_guess_accuracy": overall_acc,
            # 2) taking manual check into account (only auto-accepted)
            "auto_accept_correct": auto_correct,
            "auto_accept_accuracy": auto_acc,
            "auto_accept_coverage": auto_coverage,
            "ge4_images_auto_accept_correct": {
                "albums_ge4": ge4_total,
                "correct_and_not_manual_check_ge4": ge4_correct_auto,
                "ratio": ge4_ratio_correct_auto,
            },
        },
        "manual_check_breakdown": {
            # 3) how many manual checks are because of album size
            "by_num_images": manual_by_size,
            "manual_check_albums": manual_check_albums,
        },
        "records": records,
    }

    # print concise summary
    print("=== EVAL SUMMARY ===")
    print(f"Albums total: {total}")
    print(f"Overall best_guess correct: {correct_any}/{total} ({overall_acc:.3f})")
    print(f"Auto-accepted coverage: {auto_total}/{total} ({auto_coverage:.3f})")
    print(f"Auto-accepted correct: {auto_correct}/{auto_total} ({auto_acc:.3f})" if auto_total else "Auto-accepted correct: n/a")
    print(f"≥4 images albums: {ge4_total}")
    print(f"≥4 images correct AND not manual-check: {ge4_correct_auto}/{ge4_total} ({ge4_ratio_correct_auto:.3f})" if ge4_total else "≥4 images metric: n/a")
    print(f"Manual check: {manual_total}")
    print(f"Manual check by #images: {manual_by_size}")
    print(f"Manual-check albums ({len(manual_check_albums)}): {manual_check_albums}")

    # save
    run_id = datetime.utcnow().strftime("eval_%Y%m%d_%H%M%SZ")
    out_path = out_dir / f"{run_id}_album_eval_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary JSON: {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())