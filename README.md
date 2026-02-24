Perfect — let’s go **lean**: smallest structure that still enforces good habits (single env, two commands, artifacts, reproducible setup). We can always “split into modules” later when something starts to hurt.

## Minimal repo skeleton (v0)

```text
race-ocr-infer/
  README.md
  pyproject.toml
  .gitignore

  src/
    raceocr/
      __init__.py
      __main__.py          # python -m raceocr ...
      cli.py               # CLI entry + subcommands (setup/infer/album)
      config.py            # defaults + locating cache dirs + manifest constants
      setup.py             # download weights + warm PaddleOCR
      infer.py             # single-image pipeline
      album.py             # folder pipeline + aggregation
      io.py                # input discovery + JSON writing + artifact dirs
      util.py              # logging/timing helpers, hashing, small utils
```

That’s it. No `docs/`, no `tests/` folder yet. We’ll add them when we need them.

### What this skeleton still guarantees

* **One package** `raceocr`
* **One CLI** with 3 commands:

  * `raceocr setup`
  * `raceocr infer`
  * `raceocr album`
* **One artifacts layout** created by `io.py`
* Internal files are “modules”, but still minimal count.

---

## Minimal CLI spec (v0)

### `raceocr setup`

Purpose: make a fresh machine ready.
Flags (v0):

* `--cache-dir PATH` (default: `~/.cache/raceocr`)
* `--yolo-url URL` (default: your GitHub weights URL)
* `--yolo-sha256 HEX` (optional for now; add soon)
* `--device {cpu,gpu}` (used for Paddle warm start)

Behavior:

* download YOLO weights into cache
* initialize PaddleOCR once to trigger its internal downloads
* print where everything ended up

### `raceocr infer`

Flags (v0):

* `--img PATH`
* `--out-dir PATH` (default: `./artifacts`)
* `--filter-words "foo,bar,baz"` and/or `--filter-words-file`
* `--ocr-conf FLOAT` (default 0.75)
* `--yolo-weights PATH` (optional; default from cache)
* `--save-crops` `--save-vis`
* `--verbose` `--debug`

Output: `results.json` in run folder, includes:

* image path
* artifact dir
* ranked OCR candidates with confidences
* per-detection provenance (so debugging is possible)

### `raceocr album`

Flags (v0):

* `--dir PATH`
* same `--filter-words`, `--ocr-conf`, `--out-dir`, `--save-*`, `--verbose`
* aggregation options later (fuzzy grouping, weighted score), but not in v0 unless needed

Output: `results.json` containing:

* folder path
* artifact dir
* `ranked_counts`: list of `{text, count, num_images, ratio}`

---

## Step-by-step build plan from “repo exists” → working tool

You’ve already completed Step 0 (repo created, Cursor SSH, venv). So we start at Step 1.

### Step 1 — Installable package + CLI stub (no model logic)

Goal: `raceocr --help` works.

Deliverables:

* `pyproject.toml` with entrypoint `raceocr = raceocr.cli:app` (or argparse equivalent)
* `src/raceocr/__main__.py` so `python -m raceocr` works
* `cli.py` defines commands: `setup`, `infer`, `album`
* each command prints resolved args and creates a run folder

✅ Success criteria:

* `pip install -e .`
* `raceocr setup --help`
* `raceocr infer --help`
* `raceocr album --help`

### Step 2 — Artifacts + JSON writer (still no models)

Goal: every run creates a deterministic folder and writes `params.json` + empty `results.json`.

Deliverables:

* `io.py`:

  * `make_run_dir(mode, input_stem, out_dir)`
  * `write_json(path, obj)`
* `util.py` basic logger and timer context manager

✅ Success criteria:

* `raceocr infer --img some.jpg` creates `artifacts/run_.../params.json` and a placeholder `results.json`

### Step 3 — `setup` downloads YOLO weights (Paddle warm start optional here)

Goal: `raceocr setup` downloads weights to cache and prints the path.

Deliverables:

* `setup.py`:

  * `download(url, dest)`
  * store weights as `~/.cache/raceocr/yolo/best.pt` (or similar)
* In `config.py`, centralize default cache path and default yolo URL.

✅ Success criteria:

* fresh machine: `raceocr setup` ends with “YOLO weights at: …/best.pt”

*(Paddle warm start can be Step 5 if Paddle installs are annoying — we can keep setup lean.)*

### Step 4 — YOLO detect + crop (OCR still stubbed)

Goal: `infer` can produce crops and detection metadata.

Deliverables:

* `infer.py`:

  * load YOLO weights
  * run detection on image
  * crop boxes (+ optional padding)
  * save crops if `--save-crops`
  * write intermediate JSON with detections + crop paths

✅ Success criteria:

* run on an image and see crop files + JSON with box coords

### Step 5 — PaddleOCR on crops + ranked candidates

Goal: full single-image pipeline works end-to-end.

Deliverables:

* `infer.py` adds:

  * initialize PaddleOCR once
  * OCR each crop
  * collect `{text, conf, crop_path, detection_id}`
  * apply `--ocr-conf`
* Apply `filter_words`

✅ Success criteria:

* `raceocr infer --img ... --filter-words "ADIDAS,Nike"` outputs ranked OCR list

### Step 6 — Album mode = loop infer over folder + aggregate counts

Goal: folder pipeline produces `3245: 13/19` style output.

Deliverables:

* `album.py`:

  * list image files in dir
  * reuse YOLO model + OCR instance across images
  * call a shared internal function (or import `infer.run_infer(image, …)` returning structured results)
  * aggregate counts (post-filter, post-threshold)
  * write album results JSON

✅ Success criteria:

* `raceocr album --dir athlete_folder` outputs ranked frequency table

### Step 7 — Make it pleasant + stable (only after it works)

Add only what hurts:

* `--filter-words-file`
* confidence-weighted aggregation
* fuzzy grouping for partials / edit distance
* `--save-vis` overlay images
* checksums in `setup`
* minimal tests for `filter_words` and `aggregate_counts`

---

## What I need from you to start Step 1 cleanly (no extra questions)

Just pick the CLI framework preference:

* **Option A (leanest):** `argparse` (zero deps)
* **Option B (nice UX):** `typer` (small dep, great help texts)

If you don’t want to decide, I’ll default to **argparse** for minimalism and we can swap later.

Next message: tell me **argparse or typer**, and I’ll tell you exactly which files to create first and what each should contain (still without pasting full code if you want to stay “skeleton-first”).
