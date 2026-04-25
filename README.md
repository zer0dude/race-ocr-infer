# raceocr

Developed for **zosportu.sk** by **Brian Zelun Jin** (GitHub: **zer0dude**).

`race-ocr-infer` is a CLI tool that reads athlete identifiers from sports event photos using:
- **YOLO** object detection (race bibs / headbands / bike tags)
- **PaddleOCR** text recognition

The tool is designed to run on a machine with GPU and CUDA, produce a **compact production JSON** for automation, and generate **debug artifacts** for traceability.

It is part of a larger pipeline. The `face-groups` command bridges the output of an upstream **face clustering step** (which groups event photos by the faces that appear in them) with the bib OCR pipeline, attributing a bib number to each face group using spatially-weighted voting.

---

## Requirements

- Python 3.10+
- **CUDA 12.8** — required for GPU inference with the PyTorch version used by Ultralytics YOLO. CPU-only mode works without CUDA (omit `--device` to let Ultralytics auto-detect, or pass `--device cpu` explicitly).
- See `pyproject.toml` for Python package dependencies.

---

## What you get

For each run, `raceocr` produces:

1) **Production JSON** (stable contract for downstream systems)  
   Written to `./runs/` by default.

2) **Artifacts** (debug + traceability)  
   Written to `./artifacts/` by default. Includes run metadata, optional YOLO visualizations, and intermediate files.

---

## Quick start

### 1) Clone and create a virtual environment

```bash
git clone https://github.com/zer0dude/race-ocr-infer
cd race-ocr-infer

python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### 2) Install `raceocr`

```bash
pip install -e .
```

> **GPU note:** CUDA 12.8 is required for YOLO GPU inference. If you intend to run with `--device 0`, ensure CUDA 12.8 is installed on your system before installing the package.

### 3) Download weights and warm caches

```bash
raceocr setup
```

This:
- downloads YOLO weights into `~/.cache/raceocr/yolo/best.pt`
- warms PaddleOCR models on CPU by default so the first run is fast and predictable

To skip OCR warming:

```bash
raceocr setup --no-warm-ocr
```

---

## Usage

### `infer` — single image

By default, `infer`:
- runs YOLO only on class `race_bibs`
- keeps OCR lines above the configured OCR confidence threshold
- applies OCR candidate filtering in **production conversion**, not during OCR extraction
- allows only **numeric** OCR results by default
- removes YOLO boxes smaller than **10000 px²** during production conversion

Minimal example:

```bash
raceocr infer --img path/to/image.jpg
```

Use a whitelist of valid IDs for the image:

```bash
raceocr infer --img path/to/image.jpg --allowed-ids 243,248,251
```

Allow all YOLO classes instead of only race bibs:

```bash
raceocr infer --img path/to/image.jpg --yolo-classes all
```

Allow alphanumeric OCR results:

```bash
raceocr infer --img path/to/image.jpg --ocr-char-set alnum
```

Override the minimum box area filter:

```bash
raceocr infer --img path/to/image.jpg --min-box-area 15000
```

Create YOLO visualization and delete crops after OCR:

```bash
raceocr infer --img path/to/image.jpg --create-vis --delete-crops
```

### `album` — batch inference over a folder of images

`album` is **batch inferencing mode**: it runs the same pipeline as `infer` over all images in a folder, then produces a **single stitched production JSON** containing the per-image results.

Minimal example (top-level images only):

```bash
raceocr album --dir path/to/album_folder
```

Run over an image tree where photos are organized in subdirectories (e.g. by bib number or photographer):

```bash
raceocr album --dir path/to/image_tree --recursive --output-name album_all.json
```

When `--recursive` is used, `orig_img` paths in the production JSON include the full path through subdirectories. `face-groups` strips to the bare filename when building its lookup index, so subdirectory layout is transparent.

Use a whitelist of valid IDs for the album:

```bash
raceocr album --dir path/to/album_folder --allowed-ids 400
```

Allow all YOLO classes:

```bash
raceocr album --dir path/to/album_folder --yolo-classes all
```

Allow arbitrary OCR strings:

```bash
raceocr album --dir path/to/album_folder --ocr-char-set any
```

Override the minimum box area filter:

```bash
raceocr album --dir path/to/album_folder --min-box-area 15000
```

Create YOLO visualization per image and delete crops after OCR:

```bash
raceocr album --dir path/to/album_folder --create-vis --delete-crops
```

### `face-groups` — attribute bib numbers to face groups

`face-groups` is designed for integration with an upstream **face clustering** step. It is a **pure spatial-voting consumer**: it reads pre-computed bib boxes from an album production JSON and attributes a bib number to each face group using spatially-weighted voting. All YOLO and OCR parameters are configured at the `album` step.

**Two-step workflow:**

**Step 1** — run `album --recursive` over the full image tree (YOLO + OCR, GPU-intensive):

```bash
raceocr album \
  --dir          path/to/image_tree \
  --recursive \
  --device       0 \
  --delete-crops \
  --runs-dir     path/to/output \
  --output-name  album_allimages.json
```

**Step 2** — run `face-groups`, consuming the album output (pure CPU, fast):

```bash
raceocr face-groups \
  --groups         path/to/refined_groups.json \
  --embeddings-dir path/to/embeddings \
  --album-results  path/to/output/album_allimages.json \
  --out            path/to/output/face_groups_result.json
```

**How attribution works:**  
For every bib detected in every image of a group, a vote weight is computed as:

```
weight = yolo_confidence × ocr_confidence × spatial_affinity
```

`spatial_affinity` first applies two hard **plausibility gates** — bibs that fail either are fully discarded (weight = 0):

1. **Vertical gate:** if the bib's bottom edge is above the face's top edge, discard. A person's own bib is always below their head; a bib above the face top belongs to a different person in the background.
2. **Horizontal gate:** if the nearest edge of the bib is more than `max_bib_dist_factor × bib_width` (default 1.0×) from the face center, discard. Uses bib width as the reference unit so the threshold scales naturally with camera distance.

Bibs that pass both gates are then scored:
- **Horizontal alignment**: gaussian falloff from the face center (sigma = face_width × 1.0 by default) — bib directly below the face scores highest
- **Vertical position**: ×1.2 multiplier if bib top is below the face top, ×0.6 if overlapping

If no bib in any image passes the plausibility gates, the group returns `best_guess: null` — this correctly handles bystanders (who have no bib) and athletes whose bib was never in a plausible position.

Votes accumulate per OCR string across all images. The string with the highest total weight is the `best_guess`. Each group is assigned a `status` string (see output schema below) based on confidence, ambiguity, and cross-group duplicate detection.

#### Input: `refined_groups.json`

Produced by the upstream face clustering step. Groups images by face identity.

```json
{
  "groups": {
    "group_0": ["FAJ_3204_0", "FAJ_3211_1"],
    "group_1": ["DSC_9084_2"]
  },
  "noise": ["FAJ_1000_0"]
}
```

Each entry `"<image_id>_<face_index>"` corresponds to a `<entry>_meta.json` file in the embeddings folder that records `original_filename` and `bbox` (face bounding box in the original image).

Noise entries are skipped — they are faces the clustering model could not confidently assign to any group.

#### Output schema: `face-groups`

```json
{
  "groups": {
    "group_79": {
      "best_guess": "423",
      "confidence": 0.9572,
      "status": "confident",
      "source_images": ["FAJ_3204.jpg", "FLS01658.jpg", "FAJ_3211.jpg"],
      "vote_breakdown": { "423": 11.58, "367": 0.52 },
      "num_images": 14,
      "num_images_with_bibs": 13,
      "num_plausible_bib_detections": 22,
      "face_entries": ["FAJ_3204_0", "FLS01658_0", "..."]
    },
    "group_122": {
      "best_guess": "389",
      "confidence": 0.5019,
      "status": "multiple_plausible_bib_ids",
      "source_images": ["LUM-167.jpg", "LUPT2759.jpg"],
      "vote_breakdown": { "389": 1.08, "89": 1.07 },
      "num_images": 2,
      "num_images_with_bibs": 2,
      "num_plausible_bib_detections": 4,
      "face_entries": ["LUM-167_3", "LUPT2759_1"]
    },
    "group_X": {
      "best_guess": "423",
      "confidence": 0.91,
      "status": "cross_group_duplicate",
      "source_images": ["FAJ_3300.jpg"],
      "vote_breakdown": { "423": 8.14 },
      "num_images": 5,
      "num_images_with_bibs": 4,
      "num_plausible_bib_detections": 7,
      "face_entries": ["FAJ_3300_1", "..."]
    },
    "group_33": {
      "best_guess": null,
      "confidence": 0.0,
      "status": "insufficient_plausible_bibs",
      "source_images": ["DSC_9084.jpg"],
      "vote_breakdown": {},
      "num_images": 3,
      "num_images_with_bibs": 0,
      "num_plausible_bib_detections": 0,
      "face_entries": ["DSC_9084_0", "..."]
    }
  },
  "meta": {
    "approach": "spatial_weighted",
    "spatial_sigma": 1.0,
    "max_bib_dist_factor": 1.0,
    "flag_threshold": 0.5,
    "ambiguity_margin": 0.15,
    "num_groups_total": 157,
    "num_confident": 65,
    "num_multiple_plausible_bib_ids": 30,
    "num_cross_group_duplicate": 14,
    "num_insufficient_plausible_bibs": 48,
    "groups_json": "path/to/refined_groups.json",
    "embeddings_dir": "path/to/embeddings",
    "album_results": "path/to/album_allimages.json",
    "started_at": "2026-04-25T10:00:00+00:00",
    "finished_at": "2026-04-25T10:00:02+00:00",
    "duration_seconds": 2.1
  }
}
```

**Status values:**

| status | meaning | `best_guess` |
|---|---|---|
| `confident` | clear winner, not shared with another group | non-null |
| `multiple_plausible_bib_ids` | ambiguous — top candidates too close in weight | non-null |
| `cross_group_duplicate` | this bib was also attributed to another group | non-null |
| `insufficient_plausible_bibs` | fewer than 2 plausible bib detections in all images | null |

Notes:
- `source_images` is an ordered, deduplicated list of original image filenames that contributed detections for this group — useful for production processing without re-joining entry IDs to filenames.
- `num_plausible_bib_detections` counts bib detections that passed both hard plausibility gates (affinity > 0). Groups with fewer than 2 receive `insufficient_plausible_bibs`.
- `vote_breakdown` lists all candidate strings that received any votes, sorted by total weight descending.
- Cross-group duplicate detection runs as a post-processing pass: if two or more groups share the same `best_guess`, all are overwritten to `cross_group_duplicate` regardless of prior status.

#### Review symlink tool

`tests/make_review_links.py` takes the `face-groups` output JSON and creates a symlink folder tree for visual review. Groups are routed into buckets by `status`, with the bib number as a sub-folder to quickly spot conflicts:

```
review_links/
  confident/
    <bib>/
      <group_id>/            symlinks to original images + face crops
  review/
    multiple_plausible_bib_ids/
      <bib>/
        <group_id>/          ambiguous groups — top candidates too close
    cross_group_duplicate/
      <bib>/
        <group_id>/          groups whose best_guess is shared with another group
  no_bib/
    <group_id>/              insufficient_plausible_bibs groups
  noise/                     flat folder of noise-entry face crops and image symlinks
```

```bash
python tests/make_review_links.py \
  --result       path/to/face_groups_result.json \
  --groups       path/to/refined_groups.json \
  --embeddings-dir path/to/embeddings \
  --images-dir   path/to/original/images \
  --out          path/to/review_links
```

---

## Production JSON vs Artifacts

### Production JSON (default: `./runs/`)

This is the **automation interface** intended to be downloaded by a client system, then used for downstream logic such as sorting images and pairing faces with numbers.

You can control location and filename with:
- `--runs-dir` (default: `runs`)
- `--output-name` (optional; defaults to `infer_<stem>_<timestamp>.json` / `album_<folder>_<timestamp>.json`)

#### Production JSON schema: `infer`

Example shape:

```json
{
  "orig_img": "data/400/DSC01752.jpg",
  "boxes": [
    {
      "xyxy": [1878.78, 2252.19, 2043.35, 2408.32],
      "box_confidence": 0.8723,
      "box_class": "race_bibs",
      "ocr_result": "243",
      "ocr_confidence": 0.8919,
      "ocr_method": "paddleocr",
      "ocr_candidates": [
        { "text": "243", "conf": 0.8919 }
      ]
    }
  ],
  "meta": {
    "yolo_weights": "/home/ubuntu/.cache/raceocr/yolo/best.pt",
    "yolo_conf": 0.86,
    "ocr_conf_thresh": 0.95,
    "allowed_ids": ["243", "248", "251"],
    "ocr_char_set": "numeric",
    "min_box_area": 10000.0
  }
}
```

Notes:
- `boxes[].xyxy` are YOLO detection coordinates in **original image space**.
- `ocr_result` is the **best OCR candidate per box** after production-stage filtering.
- `ocr_candidates` is the per-box candidate list that survived filtering and is ranked by confidence.
- Production filtering includes:
  - optional ID whitelist filtering via `--allowed-ids`
  - OCR character-set filtering via `--ocr-char-set`
  - minimum YOLO box area filtering via `--min-box-area`
- Boxes smaller than the configured minimum area are removed entirely from production output.
- If no OCR candidate survives filtering for a remaining box, `ocr_result` is an empty string and `ocr_confidence` is `0.0`.

#### Production JSON schema: `album`

`album` stitches together the per-image `infer` outputs with per-image `meta` removed, and adds a single album-level `meta`.

Example shape:

```json
{
  "orig_album": "data/400",
  "images": [
    {
      "orig_img": "data/400/DSC01752.jpg",
      "boxes": [
        {
          "xyxy": [1878.78, 2252.19, 2043.35, 2408.32],
          "box_confidence": 0.8723,
          "box_class": "race_bibs",
          "ocr_result": "243",
          "ocr_confidence": 0.8919,
          "ocr_method": "paddleocr",
          "ocr_candidates": [
            { "text": "243", "conf": 0.8919 }
          ]
        }
      ]
    },
    {
      "orig_img": "data/400/DSC01753.jpg",
      "boxes": []
    }
  ],
  "meta": {
    "num_images_total": 7,
    "num_images_processed": 7,
    "num_images_failed": 0,
    "failed_images": [],
    "yolo_weights": "/home/ubuntu/.cache/raceocr/yolo/best.pt",
    "yolo_conf": 0.86,
    "yolo_iou": 0.45,
    "imgsz": 1280,
    "device": null,
    "ocr_conf_thresh": 0.95,
    "allowed_ids": ["400"],
    "ocr_char_set": "numeric",
    "min_box_area": 10000.0
  }
}
```

Interpretation:
- `images[]` contains stable per-image results (`orig_img` + `boxes`).
- Album-level `meta` summarizes counts and run configuration.
- If any image fails, it is counted in `num_images_failed` and listed in `failed_images` with an error string.
- The same production filtering rules apply in album mode as in infer mode.

---

## Full CLI reference

### `raceocr setup`

- `--cache-dir PATH`  
  Override cache directory (default: `~/.cache/raceocr`)

- `--yolo-url URL`  
  Override YOLO weights URL

- `--yolo-sha256 HEX`  
  Optional integrity check for YOLO weights

- `--force`  
  Re-download YOLO weights even if present

- `--warm-ocr / --no-warm-ocr`  
  Warm PaddleOCR models (default: on)

### `raceocr infer`

Required:
- `--img PATH` input image

Main options:
- `--ocr-conf FLOAT` (default: `0.95`)
- `--allowed-ids "a,b,c"` optional whitelist of valid OCR outputs
- `--ocr-char-set {numeric,alnum,any}` (default: `numeric`)
- `--min-box-area FLOAT` minimum YOLO bounding box area in px² kept in production JSON (default: `10000`)
- `--ocr-device {cpu,gpu}` (default: `cpu`)
- `--yolo-weights PATH` (default: cached weights from `raceocr setup`)
- `--yolo-conf FLOAT` (default: `0.86`)
- `--yolo-iou FLOAT` (default: `0.45`)
- `--yolo-classes {race_bibs,... | 0,... | all}` (default: `race_bibs`)
- `--imgsz INT` (default: `1280`)
- `--device STR` for YOLO device such as `"cpu"`, `"0"`, or `"cuda:0"` (default: Ultralytics auto)
- `--pad FLOAT` crop padding fraction (default: `0.01`)

Artifacts / production output:
- `--out-dir PATH` artifacts directory (default: `./artifacts`)
- `--create-vis` write visualization image (default: off)
- `--delete-crops` delete crops after OCR (default: off)
- `--runs-dir PATH` production JSON directory (default: `./runs`)
- `--output-name NAME.json` set production JSON filename (optional)

### `raceocr album`

Required:
- `--dir PATH` album folder

Main options:
- `--recursive` search the folder recursively for images (default: off, top-level only)
- `--ocr-conf FLOAT` (default: `0.95`)
- `--allowed-ids "a,b,c"` optional whitelist of valid OCR outputs
- `--ocr-char-set {numeric,alnum,any}` (default: `numeric`)
- `--min-box-area FLOAT` minimum YOLO bounding box area in px² kept in production JSON (default: `10000`)
- `--ocr-device {cpu,gpu}` (default: `cpu`)
- `--yolo-weights PATH` (default: cached weights)
- `--yolo-conf FLOAT` (default: `0.86`)
- `--yolo-iou FLOAT` (default: `0.45`)
- `--yolo-classes {race_bibs,... | 0,... | all}` (default: `race_bibs`)
- `--imgsz INT` (default: `1280`)
- `--device STR` for YOLO device
- `--pad FLOAT` crop padding fraction (default: `0.01`)

Artifacts / production output:
- `--out-dir PATH` artifacts directory (default: `./artifacts`)
- `--create-vis` write visualization per image (default: off)
- `--delete-crops` delete crops after OCR to save disk (default: off)
- `--runs-dir PATH` production JSON directory (default: `./runs`)
- `--output-name NAME.json` set production JSON filename (optional)

### `raceocr face-groups`

Required:
- `--groups PATH` refined_groups.json from the upstream face clustering step
- `--embeddings-dir PATH` folder containing `<entry>_meta.json` files
- `--album-results PATH` album production JSON produced by `raceocr album --recursive`
- `--out PATH` output JSON path

Attribution tuning:
- `--spatial-sigma FLOAT` gaussian sigma for horizontal alignment as a multiple of face width (default: `1.0`); applies only to bibs that pass the hard plausibility gates
- `--max-bib-dist FLOAT` hard horizontal cutoff in bib-widths from face center (default: `1.0`); bibs whose nearest edge exceeds this are discarded entirely
- `--flag-threshold FLOAT` confidence below which a group receives `status=multiple_plausible_bib_ids` (default: `0.5`)
- `--ambiguity-margin FLOAT` if runner-up weight ÷ top weight exceeds `1 - margin`, group receives `status=multiple_plausible_bib_ids` (default: `0.15`)

Artifacts:
- `--out-dir PATH` artifacts directory (default: `./artifacts`)

---

## Project structure and responsibilities

The tool is intentionally split by responsibility so that OCR extraction, orchestration, and production formatting remain clearly separated.

- `cli.py` is the command-line entrypoint. It parses arguments, orchestrates the run, writes debug artifacts, and calls production conversion at the end.
- `infer.py` contains the single-image pipeline building blocks: YOLO loading and inference, rendering, crop creation, PaddleOCR initialization, and raw OCR candidate extraction.
- `album.py` is intentionally small and focused. It provides album-level helpers such as listing input images for folder-based batch processing (flat or recursive).
- `production.py` owns **production-facing post-processing**. This is where OCR candidates are grouped, filtered, sorted, and converted into the final stable JSON contract. Logic such as allowed ID whitelisting, OCR character-set filtering, minimum box area filtering, and choosing `ocr_result` belongs here.
- `face_groups.py` implements the face-group bib attribution pipeline. It loads face group data from the upstream clustering step, reads pre-computed bib boxes from an album production JSON (produced by `raceocr album --recursive`), and aggregates spatially-weighted votes into a per-group best guess with confidence and review flags. All YOLO and OCR computation is delegated to the `album` step.
- `setup.py` defines package installation behavior, while the project's runtime setup helpers are used to download YOLO weights and warm OCR caches through the `raceocr setup` command.

This factoring is deliberate: `infer.py` extracts evidence, `production.py` decides what counts as a valid production answer, `face_groups.py` handles cross-image attribution, and `cli.py` ties the system together.

---

## Troubleshooting

### CUDA 12.8 requirement for GPU inference

YOLO uses PyTorch via Ultralytics, which requires **CUDA 12.8** for GPU inference. Running on a machine with a different CUDA version and passing `--device 0` will produce CUDA-related errors.

To check your CUDA version:

```bash
nvcc --version
# or
nvidia-smi
```

If your CUDA version does not match, either install CUDA 12.8 or run with `--device cpu` (slower, but fully compatible on any machine).

### YOLO GPU + PaddleOCR GPU conflicts

This tool defaults to:
- **YOLO on GPU** (PyTorch CUDA)
- **PaddleOCR on CPU**

Reason: running YOLO (PyTorch) and PaddleOCR (PaddlePaddle) both on GPU inside a single environment can trigger CUDA or NCCL version conflicts.
CPU OCR is stable and keeps the project in one environment.

If you later need PaddleOCR GPU inference for speed:
- try `--ocr-device gpu`
- if you run into CUDA stack conflicts, consider running PaddleOCR in a **separate environment** or container from YOLO

### Where are weights and models stored?

- YOLO weights: `~/.cache/raceocr/yolo/best.pt`
- PaddleOCR models: cached under the PaddleX or PaddleOCR directories in your home folder, depending on version

If you want to re-download PaddleOCR models, remove the corresponding cache folders and run `raceocr setup` again.

---

## Licensing and related repositories

- PaddleOCR license: https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE  
- Ultralytics / YOLO repository (licenses and usage): https://github.com/ultralytics/ultralytics  
- Training and finetuning work for this specific use case: https://github.com/zer0dude/race-ocr  

In accordance with the model licenses, all derivative work for this project is open and public in these repositories.
