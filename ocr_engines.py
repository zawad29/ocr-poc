from transformers.models import falcon
import json
import re
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

_TESSDATA_DIR = Path(__file__).parent / "trained_data"

# Lazy-loaded singletons
_easyocr_reader = None
_surya_recognition_predictor = None
_surya_detection_predictor = None


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

        config = (
            "--oem 1 "          # LSTM only (best accuracy)
            "--psm 6 "          # Assume a uniform block of text
                                # Try --psm 4 if card has a single column
                                # Try --psm 11 for sparse text (address fields)
            "-c tessedit_char_blacklist=|}{[]<>~^"  # Remove garbage chars
            "-c preserve_interword_spaces=1"         # Keep spacing
            "-c textord_heavy_nr=1"                  # Reduce noise region marking
            f' --tessdata-dir "{_TESSDATA_DIR}"'     # Use project-local custom models
        )

        start = time.time()
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang="ben+eng", config=config)
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


def run_surya(image_path: str) -> tuple[str, float, list[dict]]:
    """Run Surya OCR. Returns (raw_text, elapsed_seconds, text_lines)."""
    try:
        from PIL import Image
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor

        global _surya_recognition_predictor, _surya_detection_predictor
        if _surya_recognition_predictor is None:
            _surya_recognition_predictor = RecognitionPredictor(FoundationPredictor())
            _surya_detection_predictor = DetectionPredictor()

        image = Image.open(image_path)
        start = time.time()
        predictions = _surya_recognition_predictor(
            images=[image],
            det_predictor=_surya_detection_predictor,
            math_mode=False,
            sort_lines=True
        )
        elapsed = time.time() - start

        text = "\n".join(line.text for line in predictions[0].text_lines)

        lines = [
            {
                "text": line.text,
                "confidence": getattr(line, "confidence", None),
                "bbox": getattr(line, "bbox", None),
                "polygon": getattr(line, "polygon", None),
            }
            for line in predictions[0].text_lines
        ]

        input_path = Path(image_path)
        output_json_path = input_path.with_name(f"{input_path.stem}_output.json")
        payload = {
            "image": input_path.name,
            "elapsed_seconds": elapsed,
            "text_lines": lines,
        }
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

        return _clean_ocr_text(text), elapsed, lines
    except Exception as e:
        print(e)
        return f"[Surya error: {e}]\n{traceback.format_exc()}", 0.0, []
