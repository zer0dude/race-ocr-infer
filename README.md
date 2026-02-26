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
   Written to `./artifacts/` by default. Includes run metadata, (optional) YOLO visualizations, and intermediate files.

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
- warms PaddleOCR models on CPU (default) so the first run is fast and predictable

To skip OCR warming:

```bash
raceocr setup --no-warm-ocr
```

---

## Usage

### `infer` — single image

```bash
raceocr infer --img path/to/image.jpg --filter-words "adidas,nike,zona:1"
```

Optional:
- create YOLO visualization
- delete crops after OCR to save disk

```bash
raceocr infer --img path/to/image.jpg --create-vis --delete-crops --filter-words "adidas,nike,zona:1"
```

### `album` — folder of images for one athlete

```bash
raceocr album --dir path/to/album_folder --filter-words "adidas,nike,zona:1"
```

Optional:
- create YOLO visualization per image
- delete crops after OCR to save disk

```bash
raceocr album --dir path/to/album_folder --create-vis --delete-crops --filter-words "adidas,nike,zona:1"
```

---

## Production JSON vs Artifacts

### Production JSON (default: `./runs/`)

This is the **automation interface** intended to be downloaded by a client system,
then used for downstream logic like sorting images and pairing faces with numbers.

You can control location + filename:

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
        {"text": "243", "conf": 0.8919}
      ]
    }
  ],
  "meta": {
    "yolo_weights": "/home/ubuntu/.cache/raceocr/yolo/best.pt",
    "yolo_conf": 0.25,
    "ocr_conf_thresh": 0.75,
    "filter_words_used": ["adidas", "nike", "zona:1"]
  }
}
```

Notes:
- `boxes[].xyxy` are the YOLO detection coordinates in **original image space** (important for face-to-number pairing).
- `ocr_result` is the **best OCR candidate per box** after filtering and confidence threshold.
- `ocr_candidates` is a short per-box list, ranked by confidence.

#### Production JSON schema: `album`

Example shape:

```json
{
  "orig_album": "data/400",
  "num_images": 7,
  "best_guess": "400",
  "best_guess_ratio": 0.8571,
  "album_conf_thresh": 0.75,
  "num_guesses_above_thresh": 1,
  "needs_manual_check": false,
  "ranked_guesses": [
    {"text": "400", "count": 6, "total": 7, "ratio": 0.8571},
    {"text": "413", "count": 3, "total": 7, "ratio": 0.4286}
  ],
  "meta": {
    "ocr_conf_thresh": 0.75,
    "filter_words_used": ["adidas", "nike", "zona:1"]
  }
}
```

Interpretation:
- `needs_manual_check = false` means a unique best guess exceeded the configured vote threshold.
- `needs_manual_check = true` means that no conclusive best guess could be found. Thus, the system advises a manual check. 
- `ranked_guesses` helps discover persistent non-ID tokens (e.g., sponsors) to add to `--filter-words`.

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
- `--filter-words "a,b,c"` (comma-separated)
- `--filter-words-file FILE` (one word per line)
- `--yolo-weights PATH` (default: cached weights from `raceocr setup`)
- `--yolo-conf FLOAT` (default: `0.25`)
- `--yolo-iou FLOAT` (default: `0.45`)
- `--imgsz INT` (default: `1280`)
- `--device STR` (YOLO device: `"cpu"`, `"0"`, `"cuda:0"`, etc.; default: Ultralytics auto)
- `--pad FLOAT` crop padding fraction (default: `0.01`)
- `--ocr-device {cpu,gpu}` (default: `cpu`)

Artifacts / production output:
- `--out-dir PATH` artifacts directory (default: `./artifacts`)
- `--create-vis` write visualization image (default: off)
- `--delete-crops` delete crops after OCR (default: off)
- `--runs-dir PATH` production JSON directory (default: `./runs`)
- `--output-name NAME.json` set production JSON filename (optional)

### `raceocr album`

Required:
- `--dir PATH` album folder (one athlete)

Main options:
- `--album-conf-thresh FLOAT` vote threshold (default: `0.75`)
- `--ocr-conf FLOAT` (default: `0.75`)
- `--filter-words "a,b,c"`
- `--filter-words-file FILE`
- `--yolo-weights PATH` (default: cached weights)
- `--yolo-conf FLOAT` (default: `0.25`)
- `--yolo-iou FLOAT` (default: `0.45`)
- `--imgsz INT` (default: `1280`)
- `--device STR` (YOLO device)
- `--pad FLOAT` crop padding fraction (default: `0.01`)

Artifacts / production output:
- `--out-dir PATH` artifacts directory (default: `./artifacts`)
- `--create-vis` write visualization per image (default: off)
- `--delete-crops` delete crops after OCR (default: off)
- `--runs-dir PATH` production JSON directory (default: `./runs`)
- `--output-name NAME.json` set production JSON filename (optional)

---

## Troubleshooting

### YOLO GPU + PaddleOCR GPU conflicts

This tool defaults to:
- **YOLO on GPU** (PyTorch CUDA)
- **PaddleOCR on CPU**

Reason: running YOLO (PyTorch) and PaddleOCR (PaddlePaddle) both on GPU inside a single environment can trigger CUDA/NCCL version conflicts.
CPU OCR is stable and keeps the project in one environment.

If you later need PaddleOCR GPU inference for speed:
- consider running PaddleOCR in a **separate environment** (or container) from YOLO, unless the underlying CUDA stack incompatibilities are resolved.

### Where are weights and models stored?

- YOLO weights: `~/.cache/raceocr/yolo/best.pt`
- PaddleOCR models: cached under the PaddleX / PaddleOCR directories in your home folder (varies by version).

If you want to re-download PaddleOCR models, remove the corresponding cache folders and run `raceocr setup` again.

---

## Licensing and related repositories

- PaddleOCR license: https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE  
- Ultralytics / YOLO repository (licenses + usage): https://github.com/ultralytics/ultralytics  
- Training and finetuning work for this specific use case: https://github.com/zer0dude/race-ocr  

In accordance with the model licenses, all derivative work for this project is open and public in these repositories.

---