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

Request flow in `app.py`: upload → save to `uploads/` under a UUID filename → `preprocess_image()` writes a `*_preprocessed.png` alongside → all three OCR engines run on the preprocessed image → each raw text is fed through `parse_nid_fields()` → results rendered in `templates/index.html`.

Three orthogonal modules, each deliberately standalone:

- **`preprocessing.py`** — grayscale pipeline: upscale to ~1000px min side, grayscale, gentle Gaussian denoise, **background flatten (MORPH_CLOSE 31×31 + divide)** to suppress holographic seal/watermarks, histogram normalize, unsharp mask, Hough-based deskew. **No binarization, no thresholding.** The flatten step works because a large morphological close fills the small dark holes (letter strokes), producing a background estimate that text doesn't pollute — dividing by that estimate wipes the watermark without touching letters. Don't swap in Otsu/adaptive thresholding; the output is intentionally grayscale.

- **`ocr_engines.py`** — one function per engine returning `(text, elapsed_seconds)`. EasyOCR and Surya models are lazy-loaded into module-level singletons (`_easyocr_reader`, `_surya_recognition_predictor`, `_surya_detection_predictor`) because model initialization is slow; preserve this pattern. Surya is invoked with task `"ocr_with_boxes"` and is script-agnostic (no language list — the model auto-detects Bengali + Latin). Every engine is wrapped in a broad `try/except` that returns the error string as "text" so one engine failing never breaks the comparison view. All outputs pass through `_clean_ocr_text()` which strips common noise chars before parsing.

- **`parser.py`** — regex-based field extraction tuned specifically against **OCR misreads** of Bangladesh NID labels. The regexes intentionally match known garbled variants:
  - `NID` ↔ `ND`, `NO` ↔ `N0`
  - `Name` ↔ `Nare` / `Narne` / `Nam`
  - `Date of Birth` ↔ `Dare oi Birtn` / `Date: of Birth`
  - `নাম` ↔ `থম` / `নায়`, `পিতা` ↔ `পতা`, `মাতা` ↔ `মত` / `সতা`

  When adding a new field or fixing a miss, add the observed garbled form to the existing alternation rather than writing a separate branch. Each extractor has labeled-regex → line-adjacency → structural-fallback tiers; keep that ordering.

## Conventions worth knowing

- Tesseract is invoked with `lang="ben+eng"` and `--psm 6` (single uniform block). Both matter for NID layout.
- Preprocessed images are written next to originals in `uploads/` with a `_preprocessed.png` suffix — both are served via the `/uploads` static mount so the template can show before/after.
- The `uploads/` directory is gitignored and auto-created at startup.
