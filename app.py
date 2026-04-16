import os
import uuid

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from preprocessing import preprocess_image
from ocr_engines import run_tesseract, run_easyocr, run_doctr
from parser import parse_nid_fields
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
        _, preprocessed_path = preprocess_image(original_path)
        preprocessed_filename = os.path.basename(preprocessed_path)
    except Exception as e:
        preprocessed_filename = None

    # Run OCR engines on preprocessed image
    ocr_path = preprocessed_path if preprocessed_filename else original_path
    tesseract_text, tesseract_time = run_tesseract(ocr_path)
    easyocr_text, easyocr_time = run_easyocr(ocr_path)
    doctr_text, doctr_time = run_doctr(ocr_path)

    # Parse results
    results = [
        {
            "engine": "Tesseract",
            "raw_text": tesseract_text,
            "time": f"{tesseract_time:.2f}s",
            "parsed": parse_nid_fields(tesseract_text),
        },
        {
            "engine": "EasyOCR",
            "raw_text": easyocr_text,
            "time": f"{easyocr_time:.2f}s",
            "parsed": parse_nid_fields(easyocr_text),
        },
        {
            "engine": "docTR",
            "raw_text": doctr_text,
            "time": f"{doctr_time:.2f}s",
            "parsed": parse_nid_fields(doctr_text),
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
