# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A FastAPI web app that runs the same Bangladesh NID card image through several OCR / vision-LLM pipelines side-by-side and parses structured fields (name in English/Bengali, father, mother, DOB, NID number) from each. The goal is comparative evaluation, not production OCR.

The currently-active pipelines (rendered as columns in the UI) are:

1. **Surya** — bbox-based OCR + `surya_parser.parse_surya_nid_fields`.
2. **Ollama (vision)** — sends the preprocessed image to a vision LLM (default `gemma4:latest`) and asks for structured JSON directly.
3. **Surya → Ollama (text)** — feeds Surya's raw OCR text to the same LLM as a text-only extraction step.

Tesseract, EasyOCR, the EasyOCR→Ollama variant, and the LightOnOCR→Ollama variant have all been **removed from the request flow** but their engine functions (`run_tesseract`, `run_easyocr`, `run_ollama_ocr`) are intentionally kept in `ocr_engines.py` for future use.

## Running

```bash
pip install -r requirements.txt   # also needs system tesseract with Bengali: `apt install tesseract-ocr tesseract-ocr-ben` (only if Tesseract is re-enabled)
python app.py                     # serves on http://0.0.0.0:8000 with reload
```

Ollama is reached via `OLLAMA_HOST` (default `http://localhost:11434`); models are configured via `OLLAMA_MODEL` (default `gemma4:latest`) and `OLLAMA_OCR_MODEL` (default `maternion/LightOnOCR-2`, currently unused in the request flow).

There are no tests, no linter config, and no build step.

## Architecture

Request flow in `app.py`: upload → save to `uploads/` under a UUID filename → `preprocess_image()` returns `(preprocessed_array, preprocessed_path)` and writes a `*_preprocessed.png` alongside → the active engines run on the preprocessed image (Surya, then Ollama vision, then Surya→Ollama text) → results rendered in `templates/index.html`. The template is engine-agnostic: it iterates `{% for r in results %}`, so adding/removing a column is a pure `app.py` change. Per-engine debug artifacts are written to `uploads/`: `<stem>_surya_lines.json` for the bbox dump and `<stem>_ollama_response.json` / `<stem>_surya_ollama_response.json` for raw LLM responses.

Modules:

- **`preprocessing.py`** — normalizes every input to a **canonical 1600×1008 canvas** (ID-1 aspect). Pipeline: detect the card quadrilateral (Canny + `approxPolyDP`, require area ≥30% of image, 4-point convex) → perspective-warp it to fill the canvas (crop + deskew + resize in one step; fallback is a plain stretch-resize if no quad is detected) → grayscale → gentle Gaussian denoise → **background flatten (MORPH_CLOSE 31×31 + divide)** to suppress holographic seal/watermarks → histogram normalize → unsharp mask. **No binarization, no thresholding.** The flatten step works because a large morphological close fills small dark holes (letter strokes), producing a background estimate that text doesn't pollute — dividing by that estimate wipes the watermark without touching letters. Don't swap in Otsu/adaptive thresholding; the output is intentionally grayscale. The canonical canvas size is a **contract**: downstream bbox AOIs and the card-type classifier both assume 1600×1008.

- **`ocr_engines.py`** — one function per engine returning `(raw_text, elapsed_seconds, lines)` where `lines` is a list of `{text, bbox: [x1,y1,x2,y2], confidence}` in canvas pixels (or `[]` for the LLM engines, which return JSON instead of bboxes).
  - **Active**:
    - `run_surya`: `RecognitionPredictor` + `DetectionPredictor`, `sort_lines=True`, `math_mode=False`. Emits `<br>`, `<sup>`, `<sub>` tags that the parser cleans.
    - `run_ollama(image_path)`: vision call to `OLLAMA_MODEL` with `format="json"`, `think=False`, `temperature=0`, `seed=42`, `num_ctx=2048`, `num_predict=512`. System prompt enforces the six-field schema and bans translation/transliteration (preserve source script).
    - `run_ollama_parse_text(raw_text)`: text-only call to `OLLAMA_MODEL` for structured extraction from another engine's OCR text. `format="json"`, `think=False`, `temperature=0`, `seed=0`, `num_ctx=1024`, `num_predict=256`. Short-circuits to `"{}"` when input is empty or looks like an error string.
  - **Preserved but not invoked**: `run_tesseract` (pytesseract `image_to_data`, `lang="ben+eng"`, `--psm 6`, custom `tessdata-dir`), `run_easyocr` (`readtext(detail=1, paragraph=False)`), `run_ollama_ocr` (vision LLM as plain-text transcriber, used as stage 1 of an Ollama→Ollama pipeline).
  - Lazy module-level singletons (`_easyocr_reader`, `_surya_recognition_predictor`, `_surya_detection_predictor`, `_ollama_client`) — model/client initialization is slow; preserve this pattern.
  - Every engine is wrapped in a broad `try/except` that returns the error string as "text" so one engine failing never breaks the comparison view. Concatenated raw text passes through `_clean_ocr_text()` which strips common noise chars for display.
  - `_log_ollama()` prints one stdout line per Ollama call with model, wall time, Ollama-reported `total/load/prompt_eval/eval` durations, prompt & generated token counts, tok/s, and output char count.

- **`bbox_parser.py`** — uniform parser used by Tesseract and EasyOCR (when those engines are wired in). Classifies each card as **Type A (labeled)** or **Type B (sparse smartcard)** by Canny edge density in a fixed chip ROI on the preprocessed canvas (Type B has a physical chip; Type A doesn't — classification is image-based, **not** text-based, because OCR quality is too uneven to drive it). Per card type, each field has a fixed AOI (fraction of canvas). Field selection uses **center-point-in-AOI** containment; same-row AOI hits are merged left-to-right. Regex is used only (a) to strip label prefixes from selected values and (b) scoped to the DOB and NID AOIs to pull out `DD Month YYYY` dates and 10/13/17-digit NIDs from whatever text landed in those boxes. AOIs for Type A and Type B are tuned independently; don't assume they're symmetric. Currently imported but unused in the active flow.

- **`surya_parser.py`** — Surya-specific structured parser; the active Surya pipeline uses this, not `bbox_parser`. **`parser.py`** — legacy regex parser, kept for reference.

- **`_coerce_nid_json(s)`** in `app.py` — shared helper that wraps `json.loads`, whitelists the six expected keys (`name_en, name_bn, father, mother, dob, nid_number`), coerces values to strings, and returns an empty-string dict on parse failure. Both Ollama columns route through this.

## Conventions worth knowing

- The Ollama system prompts (`_OLLAMA_SYSTEM_PROMPT`, `_OLLAMA_TEXT_SYSTEM_PROMPT`) explicitly ban translation/transliteration/romanization and require preserving Bengali script in Bengali fields. Don't soften this — the LLM has been observed romanizing father/mother names without it.
- `parse_nid_fields_by_bbox(lines, image, engine=...)` **requires** the preprocessed array as its second argument — it drives the card-type classifier. Don't call it without one.
- All bbox coordinates in line dumps and AOIs are on the 1600×1008 canonical canvas; never work in original-image pixels downstream of preprocessing.
- Adding/removing a comparison column is an `app.py`-only change: add the engine call, the optional debug dump, the `_coerce_nid_json` line, and the `results` entry. The template needs no edit.
- Preprocessed images are written next to originals in `uploads/` with a `_preprocessed.png` suffix — both are served via the `/uploads` static mount so the template can show before/after.
- The `uploads/` directory is gitignored and auto-created at startup.
