# raceocr

Developed for **zosportu.sk** by **Brian Zelun Jin** (GitHub: **zer0dude**).

`race-ocr-infer` is a CLI tool that reads athlete identifiers from sports event photos using:
- **YOLO** object detection (race bibs / headbands / bike tags)
- **PaddleOCR** text recognition

The tool is designed to run on a machine with GPU and CUDA, produce a **compact production JSON** for automation, and generate **debug artifacts** for traceability.

It is part of a larger pipeline. The `face-groups` command bridges the output of an upstream **face clustering step** (which groups event photos by the faces that appear in them) with the bib OCR pipeline, attributing a bib number to each face group using spatially-weighted voting.

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

`album` is **batch inferencing mode**: it runs the same pipeline as `infer` over all images in a folder (non-recursive), then produces a **single stitched production JSON** containing the per-image results.

Minimal example:

```bash
raceocr album --dir path/to/album_folder
```

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

`face-groups` is designed for integration with an upstream **face clustering** step. It takes as input a set of face groups (images grouped by the face that appears in them, with bounding boxes per detected face) and runs bib OCR on each image. For each face group it produces a **best-guess bib number** with a confidence score, using **spatially-weighted voting** to distinguish the target runner's own bib from those of companion runners.

**How attribution works:**  
For every bib detected in every image of a group, a vote weight is computed as:

```
weight = yolo_confidence × ocr_confidence × spatial_affinity
```

`spatial_affinity` rewards bibs that are:
- **horizontally aligned** with the target face (gaussian falloff — bib directly below the face scores highest, a companion's offset bib scores lower)
- **vertically below** the face (×1.2 bonus if bib top is below the face top, ×0.6 penalty if above)

Votes accumulate per OCR string across all images. The string with the highest total weight is the `best_guess`. Groups are flagged `needs_review: true` when confidence is below 0.5 or when the top two candidates are within 15% of each other's weight (both thresholds are configurable).

Minimal example:

```bash
raceocr face-groups \
  --groups  path/to/refined_groups.json \
  --embeddings-dir path/to/embeddings \
  --images-dir path/to/original/images \
  --out path/to/output.json
```

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
      "needs_review": false,
      "vote_breakdown": { "423": 11.58, "367": 0.52 },
      "num_images": 14,
      "num_images_with_bibs": 13,
      "face_entries": ["FAJ_3204_0", "FLS01658_0", "..."]
    },
    "group_122": {
      "best_guess": "389",
      "confidence": 0.5019,
      "needs_review": true,
      "vote_breakdown": { "389": 1.08, "89": 1.07 },
      "num_images": 2,
      "num_images_with_bibs": 2,
      "face_entries": ["LUM-167_3", "LUPT2759_1"]
    },
    "group_33": {
      "best_guess": null,
      "confidence": 0.0,
      "needs_review": true,
      "vote_breakdown": {},
      "num_images": 3,
      "num_images_with_bibs": 0,
      "face_entries": ["DSC_9084_0", "..."]
    }
  },
  "meta": {
    "approach": "spatial_weighted",
    "spatial_sigma": 1.5,
    "flag_threshold": 0.5,
    "ambiguity_margin": 0.15,
    "yolo_conf": 0.86,
    "ocr_conf": 0.95,
    "min_box_area": 10000.0,
    "ocr_char_set": "numeric",
    "num_groups_total": 157,
    "num_groups_needs_review": 49
  }
}
```

Notes:
- `best_guess` is `null` when no bib was detected in any image of the group.
- `needs_review: true` is always set when `best_guess` is `null`.
- `vote_breakdown` lists all candidate strings that received any votes, sorted by total weight descending.
- Groups with `needs_review: false` are confident attributions safe for automation.
- Groups with `needs_review: true` and a non-null `best_guess` still have a most-likely answer but warrant human review.

#### Review symlink tool

`scripts/make_review_links.py` takes the `face-groups` output JSON and creates a symlink folder tree for visual review:

```
review_links/
  confident/                 one folder per confident group
    group_79__bib_423/       folder name encodes the best guess
      FAJ_3204.jpg -> ...    symlinks to original images
      FLS01658.jpg -> ...
  review/                    groups flagged needs_review=true with a guess
    group_122__bib_389/
  no_bib/                    groups where no bib was ever detected
    group_33/
  noise/                     flat folder of all noise-entry original images
    FAJ_1000.jpg -> ...
```

```bash
python scripts/make_review_links.py \
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
- `--images-dir PATH` folder containing original images (searched recursively)
- `--out PATH` output JSON path

Attribution tuning:
- `--spatial-sigma FLOAT` gaussian sigma for horizontal alignment as a multiple of face width (default: `1.5`); lower = tighter discrimination against companion runners
- `--flag-threshold FLOAT` confidence below which `needs_review` is set (default: `0.5`)
- `--ambiguity-margin FLOAT` if runner-up weight ÷ top weight exceeds `1 - margin`, flag as ambiguous (default: `0.15`)

Main OCR / YOLO options (same defaults as `infer` / `album`):
- `--ocr-conf FLOAT` (default: `0.95`)
- `--ocr-char-set {numeric,alnum,any}` (default: `numeric`)
- `--min-box-area FLOAT` (default: `10000`)
- `--allowed-ids "a,b,c"` optional whitelist
- `--ocr-device {cpu,gpu}` (default: `cpu`)
- `--yolo-conf FLOAT` (default: `0.86`)
- `--yolo-iou FLOAT` (default: `0.45`)
- `--yolo-classes ...` (default: `race_bibs`)
- `--imgsz INT` (default: `1280`)
- `--device STR` YOLO device (default: Ultralytics auto)
- `--yolo-weights PATH` (default: cached weights)
- `--pad FLOAT` (default: `0.01`)
- `--delete-crops` delete crops after OCR (default: off)

Artifacts:
- `--out-dir PATH` artifacts directory (default: `./artifacts`)

---

## Project structure and responsibilities

The tool is intentionally split by responsibility so that OCR extraction, orchestration, and production formatting remain clearly separated.

- `cli.py` is the command-line entrypoint. It parses arguments, orchestrates the run, writes debug artifacts, and calls production conversion at the end.
- `infer.py` contains the single-image pipeline building blocks: YOLO loading and inference, rendering, crop creation, PaddleOCR initialization, and raw OCR candidate extraction.
- `album.py` is intentionally small and focused. It provides album-level helpers such as listing input images for folder-based batch processing.
- `production.py` owns **production-facing post-processing**. This is where OCR candidates are grouped, filtered, sorted, and converted into the final stable JSON contract. Logic such as allowed ID whitelisting, OCR character-set filtering, minimum box area filtering, and choosing `ocr_result` belongs here.
- `face_groups.py` implements the face-group bib attribution pipeline. It loads face group data from the upstream clustering step, builds a spatial affinity model, runs OCR per image in each group, and aggregates spatially-weighted votes into a per-group best guess with confidence and review flags.
- `setup.py` defines package installation behavior, while the project’s runtime setup helpers are used to download YOLO weights and warm OCR caches through the `raceocr setup` command.

This factoring is deliberate: `infer.py` extracts evidence, `production.py` decides what counts as a valid production answer, `face_groups.py` handles cross-image attribution, and `cli.py` ties the system together.

---

## Troubleshooting

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
