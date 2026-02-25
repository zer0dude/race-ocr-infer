from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    paths.sort()
    return paths


def fold_text(s: str) -> str:
    # Safe normalization only: strip + uppercase
    return (s or "").strip().upper()


def edit_distance_leq_1(a: str, b: str) -> bool:
    """
    True if Levenshtein edit distance(a,b) <= 1.
    Optimized for our constraint; no external deps.
    """
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False

    # Ensure a is the shorter
    if la > lb:
        a, b = b, a
        la, lb = lb, la

    # Now lb is either la or la+1
    i = j = 0
    edits = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if la == lb:
            # substitution
            i += 1
            j += 1
        else:
            # insertion in longer string
            j += 1

    # trailing char in longer string
    if j < lb or i < la:
        edits += 1
    return edits <= 1


def substring_support(a: str, b: str) -> bool:
    """
    True if one string supports the other via substring/partial match
    with guardrails to avoid over-merging.

    Rules:
    - both strings length >= 3
    - short contained in long
    - length difference <= 3
    - short/long length ratio >= 0.6
    """
    a = a.strip()
    b = b.strip()
    if len(a) < 3 or len(b) < 3:
        return False
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if short not in long:
        return False
    if (len(long) - len(short)) > 3:
        return False
    if (len(short) / max(1, len(long))) < 0.6:
        return False
    return True


def similar(a: str, b: str) -> bool:
    """
    Conservative similarity:
    - exact match always
    - edit distance <=1 only if min len >=3
    - substring support with guardrails
    """
    if a == b:
        return True
    if min(len(a), len(b)) >= 3 and edit_distance_leq_1(a, b):
        return True
    if substring_support(a, b):
        return True
    return False


@dataclass
class Cluster:
    # canonical label we present
    canonical: str
    # variant -> set(images) where observed
    variant_images: Dict[str, set] = field(default_factory=dict)
    # image -> best_conf contributed to this cluster (one vote per image)
    image_best_conf: Dict[str, float] = field(default_factory=dict)
    # image -> chosen variant (best conf for that image in this cluster)
    image_best_variant: Dict[str, str] = field(default_factory=dict)

    def members(self) -> List[str]:
        return list(self.variant_images.keys())

    def add_observation(self, image_id: str, variant: str, conf: float) -> None:
        self.variant_images.setdefault(variant, set()).add(image_id)

        # one vote per image per cluster; keep best confidence for that image
        prev = self.image_best_conf.get(image_id)
        if prev is None or conf > prev:
            self.image_best_conf[image_id] = conf
            self.image_best_variant[image_id] = variant

    def count_images(self) -> int:
        return len(self.image_best_conf)

    def avg_conf(self) -> float:
        if not self.image_best_conf:
            return 0.0
        return sum(self.image_best_conf.values()) / len(self.image_best_conf)

    def choose_canonical(self) -> None:
        """
        Choose canonical from variants by:
        1) highest image support (variant appears in most images)
        2) then highest average conf among images where that variant was best (tie-break)
        3) then longest length
        """
        best = self.canonical
        best_key = (-1, -1.0, -1)  # support, avg_conf, length

        # compute per-variant support
        for v, imgs in self.variant_images.items():
            support = len(imgs)

            # avg conf when this variant was the per-image best for this cluster
            confs = [c for img, c in self.image_best_conf.items() if self.image_best_variant.get(img) == v]
            avgc = sum(confs) / len(confs) if confs else 0.0

            key = (support, avgc, len(v))
            if key > best_key:
                best_key = key
                best = v

        self.canonical = best

    def to_ranked_dict(self, num_images: int) -> Dict[str, Any]:
        self.choose_canonical()
        # variants as counts-of-images (readable for debugging)
        variants = {v: len(imgs) for v, imgs in self.variant_images.items()}
        # sort variants by support desc
        variants_sorted = dict(sorted(variants.items(), key=lambda kv: kv[1], reverse=True))

        c = self.count_images()
        ratio = (c / num_images) if num_images > 0 else 0.0

        return {
            "canonical": self.canonical,
            "count_images": c,
            "num_images": num_images,
            "ratio": ratio,
            "avg_conf": self.avg_conf(),
            "variants": variants_sorted,
        }


def assign_to_cluster(clusters: List[Cluster], text: str) -> Optional[int]:
    """
    Return cluster index that text belongs to, else None.
    Checks similarity against canonical and existing members.
    """
    for idx, cl in enumerate(clusters):
        if similar(text, cl.canonical):
            return idx
        for m in cl.members():
            if similar(text, m):
                return idx
    return None


def aggregate_album(
    per_image_candidates: Dict[str, List[Dict[str, Any]]],
    num_images: int,
    album_conf_thresh: float = 0.75,
) -> Dict[str, Any]:
    """
    per_image_candidates: image_id -> list of {text, conf, ...}
    Returns ranked guesses + confidence boolean with uniqueness rule.
    """
    clusters: List[Cluster] = []

    # Build clusters from all observations
    for image_id, cands in per_image_candidates.items():
        for r in cands:
            raw = r.get("text")
            if not raw:
                continue
            text = fold_text(str(raw))
            if not text:
                continue
            conf = float(r.get("conf", 0.0))

            idx = assign_to_cluster(clusters, text)
            if idx is None:
                cl = Cluster(canonical=text)
                cl.add_observation(image_id=image_id, variant=text, conf=conf)
                clusters.append(cl)
            else:
                clusters[idx].add_observation(image_id=image_id, variant=text, conf=conf)

    ranked = [cl.to_ranked_dict(num_images) for cl in clusters]
    ranked.sort(key=lambda d: (d["count_images"], d["avg_conf"], len(d["canonical"])), reverse=True)

    best = ranked[0] if ranked else None
    num_above = sum(1 for d in ranked if d["ratio"] >= album_conf_thresh)

    is_confident = bool(best) and best["ratio"] >= album_conf_thresh and num_above == 1
    needs_manual_check = not is_confident

    return {
        "album_conf_thresh": album_conf_thresh,
        "best_guess": best["canonical"] if best else None,
        "best_guess_ratio": best["ratio"] if best else 0.0,
        "num_guesses_above_thresh": num_above,
        "is_confident": is_confident,
        "needs_manual_check": needs_manual_check,
        "ranked_guesses": ranked,
    }