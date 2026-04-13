import re
import time
import traceback

import cv2
import numpy as np

# Lazy-loaded singletons
_easyocr_reader = None
_doctr_predictor = None


def _clean_ocr_text(text: str) -> str:
    """Remove OCR noise characters and clean up raw text."""
    # Remove common OCR noise symbols
    text = re.sub(r'[`~|{}\[\]\\@#$^&*_<>]+', '', text)
    # Remove standalone single non-alphanumeric characters on a line
    text = re.sub(r'^\s*[^\w\s\u0980-\u09FF]\s*$', '', text, flags=re.MULTILINE)
    # Remove lines that are only dashes, dots, or equals
    text = re.sub(r'^\s*[\-=\.]+\s*$', '', text, flags=re.MULTILINE)
    # Collapse multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def run_tesseract(image_path: str) -> tuple[str, float]:
    """Run Tesseract OCR. Returns (raw_text, elapsed_seconds)."""
    try:
        import pytesseract
        from PIL import Image

        start = time.time()
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang="ben+eng", config="--psm 6")
        elapsed = time.time() - start
        return _clean_ocr_text(text), elapsed
    except Exception as e:
        return f"[Tesseract error: {e}]\n{traceback.format_exc()}", 0.0


def run_easyocr(image_path: str) -> tuple[str, float]:
    """Run EasyOCR. Returns (raw_text, elapsed_seconds)."""
    try:
        import easyocr

        global _easyocr_reader
        if _easyocr_reader is None:
            _easyocr_reader = easyocr.Reader(["bn", "en"], gpu=False)

        start = time.time()
        results = _easyocr_reader.readtext(image_path, detail=0, paragraph=True)
        elapsed = time.time() - start
        text = "\n".join(results) if results else ""
        return _clean_ocr_text(text), elapsed
    except Exception as e:
        return f"[EasyOCR error: {e}]\n{traceback.format_exc()}", 0.0


def run_doctr(image_path: str) -> tuple[str, float]:
    """Run docTR OCR. Returns (raw_text, elapsed_seconds)."""
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor

        global _doctr_predictor
        if _doctr_predictor is None:
            _doctr_predictor = ocr_predictor(pretrained=True)

        start = time.time()
        doc = DocumentFile.from_images(image_path)
        result = _doctr_predictor(doc)
        elapsed = time.time() - start

        # Extract text from result
        text = result.render()
        return _clean_ocr_text(text), elapsed
    except Exception as e:
        return f"[docTR error: {e}]\n{traceback.format_exc()}", 0.0
