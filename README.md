# raceocr

Developed for **zosportu.sk** by **Brian Zelun Jin** (GitHub: **zer0dude**).

`race-ocr-infer` is a CLI tool that reads athlete identifiers from sports event photos using:
- **YOLO** object detection (race bibs / headbands / bike tags)
- **PaddleOCR** text recognition

The tool is designed to run on a machine with GPU and CUDA, produce a **compact production JSON** for automation, and generate **debug artifacts** for traceability.

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

Create YOLO visualization per image and delete crops after OCR:

```bash
raceocr album --dir path/to/album_folder --create-vis --delete-crops
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
    "yolo_conf": 0.25,
    "ocr_conf_thresh": 0.75,
    "allowed_ids": ["243", "248", "251"],
    "ocr_char_set": "numeric"
  }
}
```

Notes:
- `boxes[].xyxy` are YOLO detection coordinates in **original image space**.
- `ocr_result` is the **best OCR candidate per box** after production-stage filtering.
- `ocr_candidates` is the per-box candidate list that survived filtering and is ranked by confidence.
- If no OCR candidate survives filtering, `ocr_result` is an empty string and `ocr_confidence` is `0.0`.

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
    "yolo_conf": 0.25,
    "yolo_iou": 0.45,
    "imgsz": 1280,
    "device": null,
    "ocr_conf_thresh": 0.75,
    "allowed_ids": ["400"],
    "ocr_char_set": "numeric"
  }
}
```

Interpretation:
- `images[]` contains stable per-image results (`orig_img` + `boxes`).
- Album-level `meta` summarizes counts and run configuration.
- If any image fails, it is counted in `num_images_failed` and listed in `failed_images` with an error string.

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
- `--ocr-conf FLOAT` (default: `0.75`)
- `--allowed-ids "a,b,c"` optional whitelist of valid OCR outputs
- `--ocr-char-set {numeric,alnum,any}` (default: `numeric`)
- `--ocr-device {cpu,gpu}` (default: `cpu`)
- `--yolo-weights PATH` (default: cached weights from `raceocr setup`)
- `--yolo-conf FLOAT` (default: `0.25`)
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
- `--ocr-conf FLOAT` (default: `0.75`)
- `--allowed-ids "a,b,c"` optional whitelist of valid OCR outputs
- `--ocr-char-set {numeric,alnum,any}` (default: `numeric`)
- `--ocr-device {cpu,gpu}` (default: `cpu`)
- `--yolo-weights PATH` (default: cached weights)
- `--yolo-conf FLOAT` (default: `0.25`)
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

---

## Project structure and responsibilities

The tool is intentionally split by responsibility so that OCR extraction, orchestration, and production formatting remain clearly separated.

- `cli.py` is the command-line entrypoint. It parses arguments, orchestrates the run, writes debug artifacts, and calls production conversion at the end.
- `infer.py` contains the single-image pipeline building blocks: YOLO loading and inference, rendering, crop creation, PaddleOCR initialization, and raw OCR candidate extraction.
- `album.py` is intentionally small and focused. It provides album-level helpers such as listing input images for folder-based batch processing.
- `production.py` owns **production-facing post-processing**. This is where OCR candidates are grouped, filtered, sorted, and converted into the final stable JSON contract. Logic such as allowed ID whitelisting, OCR character-set filtering, and choosing `ocr_result` belongs here.
- `setup.py` defines package installation behavior, while the project’s runtime setup helpers are used to download YOLO weights and warm OCR caches through the `raceocr setup` command.

This factoring is deliberate: `infer.py` extracts evidence, `production.py` decides what counts as a valid production answer, and `cli.py` ties the system together.

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
