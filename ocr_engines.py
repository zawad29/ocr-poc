from transformers.models import falcon
import json
import os
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
_ollama_client = None
_paddleocr_vl_pipeline = None

# Layout-element labels emitted by PaddleOCR-VL that carry text we want to
# treat as OCR lines. Non-text labels (table, image, chart, formula, figure,
# seal, …) are dropped.
_PADDLE_TEXT_LABELS = {
    "text", "ocr", "paragraph_title", "title", "doc_title",
    "header", "footer", "abstract", "content", "reference",
}

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:latest")

_OLLAMA_TEXT_SYSTEM_PROMPT = (
    "You receive raw OCR output from a Bangladesh National ID (NID) card. "
    "The OCR text may have fields out of order, noise, or missing characters. "
    "Return ONLY a single JSON object with exactly these string keys: "
    "name_en, name_bn, father, mother, dob, nid_number. "
    "CRITICAL: Copy each field value EXACTLY as it appears in the OCR text, character for character. "
    "Do NOT translate, transliterate, romanize, clean up spelling, or guess missing characters. "
    "If a value appears in Bengali script (অ-৿) in the OCR text, the output MUST remain in Bengali script. "
    "If it appears in Latin letters, keep it in Latin letters. "
    "name_en is the English/Latin name. "
    "name_bn is the Bengali name (Bengali Unicode only). "
    "father and mother are the parent names as written in the OCR text — preserve the exact script. "
    "dob is the date of birth in 'DD Month YYYY' form exactly as written. "
    "nid_number is the digits of the ID/NID number, no spaces. "
    "Use an empty string for any field that is not present in the OCR text. "
    "Do not include any other keys, commentary, markdown, or code fences."
)

_OLLAMA_SYSTEM_PROMPT = (
    "You extract fields from a Bangladesh National ID (NID) card image. "
    "Return ONLY a single JSON object with exactly these string keys: "
    "name_en, name_bn, father, mother, dob, nid_number. "
    "CRITICAL: Copy every value EXACTLY as printed on the card, character for character. "
    "Do NOT translate, transliterate, romanize, or convert between scripts. "
    "If a value is printed in Bengali script (\u0985-\u09FF), the output MUST remain in Bengali script. "
    "If it is printed in Latin letters, keep it in Latin letters. "
    "name_en is the English/Latin name as printed. "
    "name_bn is the Bengali name as printed (Bengali Unicode characters only). "
    "father and mother are the parent names as printed \u2014 preserve the exact script of the source. "
    "dob is the date of birth in 'DD Month YYYY' form exactly as written. "
    "nid_number is the digits of the ID/NID number, no spaces. "
    "Use an empty string for any field that is not clearly visible. "
    "Do not include any other keys, commentary, markdown, or code fences."
)


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


def run_paddleocr_vl(image_path: str) -> tuple[str, float, list[dict]]:
    """Run PaddleOCR-VL on a preprocessed canvas image. Returns
    (raw_text, elapsed_seconds, text_lines) shaped identically to run_surya:
    each line is `{text, bbox: [x1,y1,x2,y2], confidence}` in canvas pixels.

    PaddleOCR-VL doesn't expose per-block confidence (see PaddleOCR#16899),
    so confidence is always None — same convention Surya uses for missing
    fields. Only text-bearing layout blocks are kept; tables/images/figures
    are filtered out via `_PADDLE_TEXT_LABELS`.

    Memory tuning: the 0.9B VLM + paddle's CUDA workspace peak at ~5.7 GB,
    which doesn't fit native on a 6 GB card. `FLAGS_use_cuda_managed_memory`
    enables CUDA unified memory so paddle can spill to host RAM when VRAM
    is exhausted — adds modest overhead (≈8–10 s/image on the v1.5 pipeline
    on a 6 GB RTX 4050) but is the only way to run in-process here. Flags
    must be set before any paddle import; we set them here instead of
    module-load time so other engines aren't affected if paddleocr is
    never invoked."""
    os.environ.setdefault("FLAGS_use_cuda_managed_memory", "true")
    os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
    os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", "0.95")
    os.environ.setdefault("FLAGS_reallocate_gpu_memory_in_mb", "1")
    try:
        from paddleocr import PaddleOCRVL

        global _paddleocr_vl_pipeline
        if _paddleocr_vl_pipeline is None:
            # pipeline_version="v1.5" (the default) — v1 misreads Bengali as
            # Devanagari/Telugu; v1.5's multilingual coverage handles Bangla
            # correctly. Slightly larger but still fits via managed memory.
            _paddleocr_vl_pipeline = PaddleOCRVL(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_chart_recognition=False,
                use_seal_recognition=False,
            )

        start = time.time()
        output = _paddleocr_vl_pipeline.predict(image_path)
        elapsed = time.time() - start

        lines: list[dict] = []
        for res in output:
            payload = getattr(res, "json", None)
            if isinstance(payload, dict):
                data = payload.get("res", payload)
            else:
                data = {}
            for block in (data.get("parsing_res_list") or []):
                label = (block.get("block_label") or "").lower()
                if label and label not in _PADDLE_TEXT_LABELS:
                    continue
                text = (block.get("block_content") or "").strip()
                bbox = block.get("block_bbox")
                if bbox is not None and not isinstance(bbox, (list, tuple)):
                    bbox = list(bbox)  # numpy array → list
                if not text or not bbox or len(bbox) != 4:
                    continue
                lines.append({
                    "text": text,
                    "bbox": [float(c) for c in bbox],
                    "confidence": None,
                })

        lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
        text = "\n".join(l["text"] for l in lines)
        return _clean_ocr_text(text), elapsed, lines
    except Exception as e:
        return f"[PaddleOCR-VL error: {e}]\n{traceback.format_exc()}", 0.0, []


def run_ollama(image_path: str) -> tuple[str, float, list[dict]]:
    """Send the preprocessed image to an Ollama vision LLM and ask for structured JSON.

    Returns (raw_response_text, elapsed_seconds, []). The third slot is an empty
    list because this engine has no per-line bboxes; the caller parses the JSON
    directly instead of going through bbox_parser."""
    try:
        import ollama

        global _ollama_client
        if _ollama_client is None:
            _ollama_client = ollama.Client(host=OLLAMA_HOST)

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        start = time.time()
        response = _ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _OLLAMA_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Extract the NID fields from this image and return the JSON object.",
                    "images": [image_bytes],
                },
            ],
            format="json",
            think=False,
            stream=False,
            options={
                "temperature": 0,
                "seed": 42,
                "num_ctx": 2048,
                "num_predict": 512,
            },
        )
        elapsed = time.time() - start

        content = response["message"]["content"] if isinstance(response, dict) else response.message.content
        return content or "", elapsed, []
    except Exception as e:
        return f"[Ollama error: {e}]\n{traceback.format_exc()}", 0.0, []



def run_ollama_parse_text(raw_text: str) -> tuple[str, float, list[dict]]:
    """Feed OCR text (from another engine) to the Ollama LLM for structured JSON extraction.

    Text-only path — no image is sent. Returns (raw_response_text, elapsed_seconds, [])."""
    if not raw_text or raw_text.startswith("[") and "error:" in raw_text[:30]:
        return "{}", 0.0, []
    try:
        import ollama

        global _ollama_client
        if _ollama_client is None:
            _ollama_client = ollama.Client(host=OLLAMA_HOST)

        start = time.time()
        response = _ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _OLLAMA_TEXT_SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            format="json",
            think=False,
            stream=False,
            options={
                "temperature": 0,
                "seed": 0,
                "num_ctx": 1024,
                "num_predict": 256,
            },
        )
        elapsed = time.time() - start

        content = response["message"]["content"] if isinstance(response, dict) else response.message.content
        return content or "", elapsed, []
    except Exception as e:
        return f"[Ollama error: {e}]\n{traceback.format_exc()}", 0.0, []
