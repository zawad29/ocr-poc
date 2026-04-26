import json
import os
import uuid

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from preprocessing import preprocess_image
from ocr_engines import (
    run_surya, run_ollama, run_ollama_parse_text,
    OLLAMA_MODEL,
)
from bbox_parser import parse_nid_fields_by_bbox
from surya_parser import parse_surya_nid_fields
from preprocessing_lab import router as preprocessing_lab_router

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")
app.include_router(preprocessing_lab_router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/", response_class=HTMLResponse)
async def process(request: Request, image: UploadFile = File(...)):
    # Save uploaded file
    ext = os.path.splitext(image.filename or "upload.png")[1] or ".png"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    original_path = os.path.join(UPLOAD_DIR, unique_name)

    contents = await image.read()
    with open(original_path, "wb") as f:
        f.write(contents)

    # Preprocess
    try:
        preprocessed_array, preprocessed_path = preprocess_image(original_path)
        preprocessed_filename = os.path.basename(preprocessed_path)
    except Exception as e:
        preprocessed_array = None
        preprocessed_filename = None

    # Run OCR engines on preprocessed image
    ocr_path = preprocessed_path if preprocessed_filename else original_path
    surya_text, surya_time, surya_lines = run_surya(ocr_path)
    ollama_text, ollama_time, _ = run_ollama(ocr_path)
    surya_llm_text, surya_llm_time, _ = run_ollama_parse_text(surya_text)


    # Debug: dump raw bbox output per engine next to the preprocessed image
    stem, _ = os.path.splitext(os.path.basename(ocr_path))
    for engine_name, engine_lines in (
        ("surya", surya_lines),
    ):
        debug_path = os.path.join(UPLOAD_DIR, f"{stem}_{engine_name}_lines.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(
                {"engine": engine_name, "image": os.path.basename(ocr_path),
                 "text_lines": engine_lines},
                f, ensure_ascii=False, indent=2, default=str,
            )

    # Dump raw Ollama response for debugging
    ollama_debug_path = os.path.join(UPLOAD_DIR, f"{stem}_ollama_response.json")
    with open(ollama_debug_path, "w", encoding="utf-8") as f:
        json.dump(
            {"engine": "ollama", "model": OLLAMA_MODEL, "image": os.path.basename(ocr_path),
             "raw_text": ollama_text},
            f, ensure_ascii=False, indent=2,
        )

    # Dump Surya→Ollama raw response for debugging
    surya_ollama_debug_path = os.path.join(UPLOAD_DIR, f"{stem}_surya_ollama_response.json")
    with open(surya_ollama_debug_path, "w", encoding="utf-8") as f:
        json.dump(
            {"engine": "surya_ollama", "model": OLLAMA_MODEL, "image": os.path.basename(ocr_path),
             "surya_ocr_text": surya_text, "llm_raw_text": surya_llm_text},
            f, ensure_ascii=False, indent=2,
        )

    # Coerce Ollama JSON responses into the six-field schema
    _ollama_keys = ("name_en", "name_bn", "father", "mother", "dob", "nid_number")

    def _coerce_nid_json(s: str) -> dict:
        try:
            obj = json.loads(s)
            if not isinstance(obj, dict):
                obj = {}
        except (json.JSONDecodeError, TypeError):
            obj = {}
        return {k: str(obj.get(k, "") or "") for k in _ollama_keys}

    ollama_parsed = _coerce_nid_json(ollama_text)
    surya_llm_parsed = _coerce_nid_json(surya_llm_text)

    # Parse results
    results = [
        {
            "engine": "Surya",
            "raw_text": surya_text,
            "time": f"{surya_time:.2f}s",
            "parsed": parse_surya_nid_fields(surya_lines),
        },
        {
            "engine": f"Ollama ({OLLAMA_MODEL})",
            "raw_text": ollama_text,
            "time": f"{ollama_time:.2f}s",
            "parsed": ollama_parsed,
        },
        {
            "engine": f"Surya → Ollama ({OLLAMA_MODEL})",
            "raw_text": surya_llm_text,
            "time": f"{surya_time + surya_llm_time:.2f}s",
            "parsed": surya_llm_parsed,
        },
    ]

    return templates.TemplateResponse(request, "index.html", {
        "original_image": f"/uploads/{unique_name}",
        "preprocessed_image": f"/uploads/{preprocessed_filename}" if preprocessed_filename else None,
        "results": results,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
