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


def run_tesseract(image_path: str) -> tuple[str, float, list[dict]]:
    """Run Tesseract OCR. Returns (raw_text, elapsed_seconds, text_lines).

    `text_lines` is a list of `{text, bbox, confidence}` where bbox is
    `[x1, y1, x2, y2]` in image pixels. Words are grouped into lines by
    Tesseract's `(block_num, par_num, line_num)`."""
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
        data = pytesseract.image_to_data(
            img, lang="ben+eng", config=config, output_type=pytesseract.Output.DICT
        )
        elapsed = time.time() - start

        groups: dict[tuple, list[int]] = {}
        for i, txt in enumerate(data["text"]):
            if not txt or not txt.strip():
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            groups.setdefault(key, []).append(i)

        lines: list[dict] = []
        for key, idxs in groups.items():
            idxs.sort(key=lambda i: data["left"][i])
            words = [data["text"][i].strip() for i in idxs if data["text"][i].strip()]
            if not words:
                continue
            x1 = min(data["left"][i] for i in idxs)
            y1 = min(data["top"][i] for i in idxs)
            x2 = max(data["left"][i] + data["width"][i] for i in idxs)
            y2 = max(data["top"][i] + data["height"][i] for i in idxs)
            confs = [float(data["conf"][i]) for i in idxs if float(data["conf"][i]) >= 0]
            mean_conf = (sum(confs) / len(confs) / 100.0) if confs else None
            lines.append({
                "text": " ".join(words),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": mean_conf,
            })

        lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
        text = "\n".join(l["text"] for l in lines)
        return _clean_ocr_text(text), elapsed, lines
    except Exception as e:
        return f"[Tesseract error: {e}]\n{traceback.format_exc()}", 0.0, []


def run_easyocr(image_path: str) -> tuple[str, float, list[dict]]:
    """Run EasyOCR. Returns (raw_text, elapsed_seconds, text_lines).

    `text_lines` carries `{text, bbox, confidence}`; bboxes are axis-aligned
    rectangles derived from EasyOCR's 4-point polygon output."""
    try:
        import easyocr

        global _easyocr_reader
        if _easyocr_reader is None:
            _easyocr_reader = easyocr.Reader(["bn", "en"], gpu=False)

        start = time.time()
        results = _easyocr_reader.readtext(image_path, detail=1, paragraph=False)
        elapsed = time.time() - start

        lines: list[dict] = []
        for polygon, txt, conf in results:
            if not txt or not txt.strip():
                continue
            xs = [float(p[0]) for p in polygon]
            ys = [float(p[1]) for p in polygon]
            lines.append({
                "text": txt.strip(),
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "confidence": float(conf) if conf is not None else None,
            })

        lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
        text = "\n".join(l["text"] for l in lines)
        return _clean_ocr_text(text), elapsed, lines
    except Exception as e:
        return f"[EasyOCR error: {e}]\n{traceback.format_exc()}", 0.0, []


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
            }
            for line in predictions[0].text_lines
        ]

        return _clean_ocr_text(text), elapsed, lines
    except Exception as e:
        print(e)
        return f"[Surya error: {e}]\n{traceback.format_exc()}", 0.0, []
