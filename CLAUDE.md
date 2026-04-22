# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A FastAPI web app that runs the same Bangladesh NID card image through three OCR engines (Tesseract, EasyOCR, Surya) side-by-side and parses structured fields (name in English/Bengali, father, mother, DOB, NID number) from each result. The goal is comparative evaluation, not production OCR.

## Running

```bash
pip install -r requirements.txt   # also needs system tesseract with Bengali: `apt install tesseract-ocr tesseract-ocr-ben`
python app.py                     # serves on http://0.0.0.0:8000 with reload
```

There are no tests, no linter config, and no build step.

## Architecture

Request flow in `app.py`: upload → save to `uploads/` under a UUID filename → `preprocess_image()` returns `(preprocessed_array, preprocessed_path)` and writes a `*_preprocessed.png` alongside → all three OCR engines run on the preprocessed image → each engine's `{text, bbox, confidence}` lines (plus the preprocessed array) are fed through `parse_nid_fields_by_bbox()` → results rendered in `templates/index.html`. Per-engine raw line dumps are written to `uploads/<stem>_<engine>_lines.json` for debugging.

Modules:

- **`preprocessing.py`** — normalizes every input to a **canonical 1600×1008 canvas** (ID-1 aspect). Pipeline: detect the card quadrilateral (Canny + `approxPolyDP`, require area ≥30% of image, 4-point convex) → perspective-warp it to fill the canvas (crop + deskew + resize in one step; fallback is a plain stretch-resize if no quad is detected) → grayscale → gentle Gaussian denoise → **background flatten (MORPH_CLOSE 31×31 + divide)** to suppress holographic seal/watermarks → histogram normalize → unsharp mask. **No binarization, no thresholding.** The flatten step works because a large morphological close fills small dark holes (letter strokes), producing a background estimate that text doesn't pollute — dividing by that estimate wipes the watermark without touching letters. Don't swap in Otsu/adaptive thresholding; the output is intentionally grayscale. The canonical canvas size is a **contract**: downstream bbox AOIs and the card-type classifier both assume 1600×1008.

- **`ocr_engines.py`** — one function per engine returning `(raw_text, elapsed_seconds, lines)` where `lines` is a list of `{text, bbox: [x1,y1,x2,y2], confidence}` in canvas pixels.
  - Tesseract: `pytesseract.image_to_data` grouped by `(block_num, par_num, line_num)` into line bboxes. `lang="ben+eng"`, `--psm 6`, custom `tessdata-dir` pointing at `trained_data/`.
  - EasyOCR: `readtext(detail=1, paragraph=False)`; 4-point polygon → axis-aligned bbox.
  - Surya: `RecognitionPredictor` + `DetectionPredictor`, `sort_lines=True`, `math_mode=False`. Emits `<br>`, `<sup>`, `<sub>` tags that the parser cleans.
  - EasyOCR and Surya models are lazy-loaded into module-level singletons (`_easyocr_reader`, `_surya_recognition_predictor`, `_surya_detection_predictor`) because model initialization is slow; preserve this pattern.
  - Every engine is wrapped in a broad `try/except` that returns the error string as "text" so one engine failing never breaks the comparison view. Concatenated raw text passes through `_clean_ocr_text()` which strips common noise chars for display.

- **`bbox_parser.py`** — the single, uniform parser consumed by all three engines. Classifies each card as **Type A (labeled)** or **Type B (sparse smartcard)** by Canny edge density in a fixed chip ROI on the preprocessed canvas (Type B has a physical chip; Type A doesn't — classification is image-based, **not** text-based, because OCR quality is too uneven to drive it). Per card type, each field has a fixed AOI (fraction of canvas). Field selection uses **center-point-in-AOI** containment; same-row AOI hits are merged left-to-right. Regex is used only (a) to strip label prefixes from selected values and (b) scoped to the DOB and NID AOIs to pull out `DD Month YYYY` dates and 10/13/17-digit NIDs from whatever text landed in those boxes. When a field fails, first check its AOI against the observed bbox center in `uploads/<stem>_<engine>_lines.json` — the AOI is the most common thing that needs nudging. AOIs for Type A and Type B are tuned independently; don't assume they're symmetric.

- **`surya_parser.py`**, **`parser.py`** — legacy regex-based parsers, no longer wired into `app.py`. Kept for reference; safe to delete.

## Conventions worth knowing

- `parse_nid_fields_by_bbox(lines, image, engine=...)` **requires** the preprocessed array as its second argument — it drives the card-type classifier. Don't call it without one.
- All bbox coordinates in line dumps and AOIs are on the 1600×1008 canonical canvas; never work in original-image pixels downstream of preprocessing.
- Tesseract is invoked with `lang="ben+eng"` and `--psm 6` (single uniform block). Both matter for NID layout.
- Preprocessed images are written next to originals in `uploads/` with a `_preprocessed.png` suffix — both are served via the `/uploads` static mount so the template can show before/after.
- The `uploads/` directory is gitignored and auto-created at startup.
