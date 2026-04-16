"""Disposable preprocessing comparison lab.

Provides /preprocessing-lab for side-by-side tuning of two pipelines:
- Option 1: a tunable copy of the current preprocessing.preprocess_image pipeline.
- Option 2: CLAHE + adaptive Gaussian threshold + optional morphological line removal.

This file is intentionally self-contained. To remove the lab, delete this file,
delete templates/preprocessing_lab.html, and drop the two include_router lines
from app.py.
"""

import os
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ocr_engines import run_tesseract, run_easyocr

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

templates = Jinja2Templates(directory="templates")
router = APIRouter()


# ---------- Option 1: tunable copy of current pipeline ----------

def preprocess_v1(
    image_path: str,
    target_min_side: int = 1000,
    max_long_side: int = 3000,
    denoise_kernel: int = 3,
    bg_enable: bool = True,
    bg_kernel: int = 31,
    bg_method: str = "divide",
    unsharp_sigma: float = 3.0,
    unsharp_weight: float = 1.5,
    deskew_angle_thresh: float = 0.3,
) -> tuple[np.ndarray, str]:
    """Current pipeline + background flattening to suppress watermarks.

    Pipeline: upscale → grayscale → denoise → background flatten → normalize
              → unsharp mask → deskew. Output stays grayscale (no binarization).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Step 1: scale
    h, w = img.shape[:2]
    min_side = min(h, w)
    if min_side < target_min_side:
        scale = target_min_side / min_side
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif max(h, w) > max_long_side:
        scale = max_long_side / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Step 2: grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: gentle Gaussian denoise (kernel forced odd, min 1)
    k = max(1, denoise_kernel | 1)
    denoised = cv2.GaussianBlur(gray, (k, k), 0)

    # Step 4: background flatten (suppresses watermarks/seals while keeping text dark)
    if bg_enable:
        bk = max(3, int(bg_kernel) | 1)
        bg_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (bk, bk))
        background = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, bg_struct)
        if bg_method == "subtract":
            diff = cv2.absdiff(denoised, background)
            flattened = cv2.bitwise_not(diff)
        else:  # divide
            src_f = denoised.astype(np.float32)
            bg_f = background.astype(np.float32) + 1.0
            flattened = np.clip((src_f / bg_f) * 255.0, 0, 255).astype(np.uint8)
    else:
        flattened = denoised

    # Step 5: normalize brightness/contrast
    normalized = cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX)

    # Step 6: unsharp mask
    blurred = cv2.GaussianBlur(normalized, (0, 0), unsharp_sigma)
    neg_weight = 1.0 - unsharp_weight
    sharpened = cv2.addWeighted(normalized, unsharp_weight, blurred, neg_weight, 0)

    # Step 7: deskew
    result = _deskew(sharpened, angle_thresh=deskew_angle_thresh)

    base, _ = os.path.splitext(os.path.basename(image_path))
    out_path = os.path.join(os.path.dirname(image_path), f"{base}_v1.png")
    cv2.imwrite(out_path, result)
    return result, out_path


def _deskew(image: np.ndarray, angle_thresh: float = 0.3) -> np.ndarray:
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=10)
    if lines is None or len(lines) < 5:
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 15:
            angles.append(angle)
    if not angles:
        return image

    median_angle = float(np.median(angles))
    if abs(median_angle) < angle_thresh:
        return image

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ---------- Option 2: CLAHE + adaptive threshold + morph line removal ----------

def preprocess_v2(
    image_path: str,
    target_min_side: int = 1000,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
    thresh_block: int = 31,
    thresh_c: int = 10,
    morph_enable: bool = True,
    morph_h_len: int = 40,
    morph_v_len: int = 40,
) -> tuple[np.ndarray, str]:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Step 0: upscale (same semantics as v1 for fair comparison)
    h, w = img.shape[:2]
    min_side = min(h, w)
    if min_side < target_min_side:
        scale = target_min_side / min_side
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Step 1: grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: CLAHE
    tile = max(1, clahe_tile)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(tile, tile))
    equalized = clahe.apply(gray)

    # Step 3: adaptive Gaussian threshold (blockSize must be odd and >= 3)
    block = max(3, thresh_block | 1)
    binarized = cv2.adaptiveThreshold(
        equalized, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        int(thresh_c),
    )

    result = binarized

    # Step 4: morphological line removal
    # Text in `binarized` is black on white; invert so lines are white for MORPH_OPEN to isolate.
    if morph_enable and (morph_h_len > 0 or morph_v_len > 0):
        inverted = cv2.bitwise_not(binarized)
        line_mask = np.zeros_like(inverted)

        if morph_h_len > 0:
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(morph_h_len), 1))
            h_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, h_kernel, iterations=1)
            line_mask = cv2.bitwise_or(line_mask, h_lines)

        if morph_v_len > 0:
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(morph_v_len)))
            v_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, v_kernel, iterations=1)
            line_mask = cv2.bitwise_or(line_mask, v_lines)

        # Erase lines from inverted image (text stays white), then invert back to black-on-white.
        cleaned_inverted = cv2.bitwise_and(inverted, cv2.bitwise_not(line_mask))
        result = cv2.bitwise_not(cleaned_inverted)

    base, _ = os.path.splitext(os.path.basename(image_path))
    out_path = os.path.join(os.path.dirname(image_path), f"{base}_v2.png")
    cv2.imwrite(out_path, result)
    return result, out_path


# ---------- Option 3: background-flattened + adaptive threshold + closing ----------

def preprocess_v3(
    image_path: str,
    target_min_side: int = 1000,
    bg_kernel: int = 31,
    bg_method: str = "divide",
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
    thresh_block: int = 31,
    thresh_c: int = 10,
    close_kernel: int = 2,
    morph_enable: bool = False,
    morph_h_len: int = 40,
    morph_v_len: int = 40,
) -> tuple[np.ndarray, str]:
    """
    Option 2 failed on this image because:
      (a) the watermark has local-contrast edges that adaptive threshold keeps,
      (b) text came out broken because no step rejoins thin strokes.
    This variant adds background flattening BEFORE threshold and stroke closing AFTER.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Upscale
    h, w = img.shape[:2]
    min_side = min(h, w)
    if min_side < target_min_side:
        scale = target_min_side / min_side
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Background flattening.
    # MORPH_CLOSE with a large kernel fills dark holes (text), leaving an estimate of
    # the light background + watermark. Divide original by that estimate to push the
    # watermark toward white while keeping dark text dark.
    k = max(3, int(bg_kernel) | 1)
    bg_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, bg_struct)

    if bg_method == "subtract":
        diff = cv2.absdiff(gray, background)
        flattened = cv2.bitwise_not(diff)  # re-invert so text stays dark
    else:  # "divide" — classic shading-correction
        gray_f = gray.astype(np.float32)
        bg_f = background.astype(np.float32) + 1.0
        flattened = np.clip((gray_f / bg_f) * 255.0, 0, 255).astype(np.uint8)

    # CLAHE on flattened image for final contrast boost
    tile = max(1, int(clahe_tile))
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(tile, tile))
    equalized = clahe.apply(flattened)

    # Adaptive Gaussian threshold
    block = max(3, int(thresh_block) | 1)
    binarized = cv2.adaptiveThreshold(
        equalized, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        int(thresh_c),
    )

    # Stroke closing: rejoin broken letter parts.
    # Close on the inverted image (text=white) with a small kernel, then invert back.
    if close_kernel and close_kernel > 0:
        ck = cv2.getStructuringElement(cv2.MORPH_RECT, (int(close_kernel), int(close_kernel)))
        inverted = cv2.bitwise_not(binarized)
        closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, ck)
        binarized = cv2.bitwise_not(closed)

    result = binarized

    # Optional line removal (same logic as v2)
    if morph_enable and (morph_h_len > 0 or morph_v_len > 0):
        inverted = cv2.bitwise_not(binarized)
        line_mask = np.zeros_like(inverted)

        if morph_h_len > 0:
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(morph_h_len), 1))
            h_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, h_kernel, iterations=1)
            line_mask = cv2.bitwise_or(line_mask, h_lines)

        if morph_v_len > 0:
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(morph_v_len)))
            v_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, v_kernel, iterations=1)
            line_mask = cv2.bitwise_or(line_mask, v_lines)

        cleaned_inverted = cv2.bitwise_and(inverted, cv2.bitwise_not(line_mask))
        result = cv2.bitwise_not(cleaned_inverted)

    base, _ = os.path.splitext(os.path.basename(image_path))
    out_path = os.path.join(os.path.dirname(image_path), f"{base}_v3.png")
    cv2.imwrite(out_path, result)
    return result, out_path


# ---------- Defaults / form helpers ----------

DEFAULTS = {
    "v1_target_min_side": 1000,
    "v1_max_long_side": 3000,
    "v1_denoise_kernel": 3,
    "v1_bg_enable": True,
    "v1_bg_kernel": 31,
    "v1_bg_method": "divide",
    "v1_unsharp_sigma": 3.0,
    "v1_unsharp_weight": 1.5,
    "v1_deskew_angle_thresh": 0.3,
    "v2_target_min_side": 1000,
    "clahe_clip": 2.0,
    "clahe_tile": 8,
    "thresh_block": 31,
    "thresh_c": 10,
    "morph_enable": True,
    "morph_h_len": 40,
    "morph_v_len": 40,
    "v3_target_min_side": 1000,
    "v3_bg_kernel": 31,
    "v3_bg_method": "divide",
    "v3_clahe_clip": 2.0,
    "v3_clahe_tile": 8,
    "v3_thresh_block": 31,
    "v3_thresh_c": 10,
    "v3_close_kernel": 2,
    "v3_morph_enable": False,
    "v3_morph_h_len": 40,
    "v3_morph_v_len": 40,
}


# ---------- Routes ----------

@router.get("/preprocessing-lab", response_class=HTMLResponse)
async def lab_index(request: Request):
    return templates.TemplateResponse(
        request,
        "preprocessing_lab.html",
        {"params": DEFAULTS},
    )


@router.post("/preprocessing-lab", response_class=HTMLResponse)
async def lab_process(
    request: Request,
    image: UploadFile = File(...),
    v1_target_min_side: int = Form(DEFAULTS["v1_target_min_side"]),
    v1_max_long_side: int = Form(DEFAULTS["v1_max_long_side"]),
    v1_denoise_kernel: int = Form(DEFAULTS["v1_denoise_kernel"]),
    v1_bg_enable: bool = Form(False),
    v1_bg_kernel: int = Form(DEFAULTS["v1_bg_kernel"]),
    v1_bg_method: str = Form(DEFAULTS["v1_bg_method"]),
    v1_unsharp_sigma: float = Form(DEFAULTS["v1_unsharp_sigma"]),
    v1_unsharp_weight: float = Form(DEFAULTS["v1_unsharp_weight"]),
    v1_deskew_angle_thresh: float = Form(DEFAULTS["v1_deskew_angle_thresh"]),
    v2_target_min_side: int = Form(DEFAULTS["v2_target_min_side"]),
    clahe_clip: float = Form(DEFAULTS["clahe_clip"]),
    clahe_tile: int = Form(DEFAULTS["clahe_tile"]),
    thresh_block: int = Form(DEFAULTS["thresh_block"]),
    thresh_c: int = Form(DEFAULTS["thresh_c"]),
    morph_enable: bool = Form(False),  # unchecked checkbox sends nothing
    morph_h_len: int = Form(DEFAULTS["morph_h_len"]),
    morph_v_len: int = Form(DEFAULTS["morph_v_len"]),
    v3_target_min_side: int = Form(DEFAULTS["v3_target_min_side"]),
    v3_bg_kernel: int = Form(DEFAULTS["v3_bg_kernel"]),
    v3_bg_method: str = Form(DEFAULTS["v3_bg_method"]),
    v3_clahe_clip: float = Form(DEFAULTS["v3_clahe_clip"]),
    v3_clahe_tile: int = Form(DEFAULTS["v3_clahe_tile"]),
    v3_thresh_block: int = Form(DEFAULTS["v3_thresh_block"]),
    v3_thresh_c: int = Form(DEFAULTS["v3_thresh_c"]),
    v3_close_kernel: int = Form(DEFAULTS["v3_close_kernel"]),
    v3_morph_enable: bool = Form(False),
    v3_morph_h_len: int = Form(DEFAULTS["v3_morph_h_len"]),
    v3_morph_v_len: int = Form(DEFAULTS["v3_morph_v_len"]),
):
    params = {
        "v1_target_min_side": v1_target_min_side,
        "v1_max_long_side": v1_max_long_side,
        "v1_denoise_kernel": v1_denoise_kernel,
        "v1_bg_enable": v1_bg_enable,
        "v1_bg_kernel": v1_bg_kernel,
        "v1_bg_method": v1_bg_method,
        "v1_unsharp_sigma": v1_unsharp_sigma,
        "v1_unsharp_weight": v1_unsharp_weight,
        "v1_deskew_angle_thresh": v1_deskew_angle_thresh,
        "v2_target_min_side": v2_target_min_side,
        "clahe_clip": clahe_clip,
        "clahe_tile": clahe_tile,
        "thresh_block": thresh_block,
        "thresh_c": thresh_c,
        "morph_enable": morph_enable,
        "morph_h_len": morph_h_len,
        "morph_v_len": morph_v_len,
        "v3_target_min_side": v3_target_min_side,
        "v3_bg_kernel": v3_bg_kernel,
        "v3_bg_method": v3_bg_method,
        "v3_clahe_clip": v3_clahe_clip,
        "v3_clahe_tile": v3_clahe_tile,
        "v3_thresh_block": v3_thresh_block,
        "v3_thresh_c": v3_thresh_c,
        "v3_close_kernel": v3_close_kernel,
        "v3_morph_enable": v3_morph_enable,
        "v3_morph_h_len": v3_morph_h_len,
        "v3_morph_v_len": v3_morph_v_len,
    }

    ext = os.path.splitext(image.filename or "upload.png")[1] or ".png"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    original_path = os.path.join(UPLOAD_DIR, unique_name)

    contents = await image.read()
    with open(original_path, "wb") as f:
        f.write(contents)

    v1_error = None
    v1_filename = None
    v1_text = ""
    v1_time = 0.0
    try:
        _, v1_path = preprocess_v1(
            original_path,
            target_min_side=v1_target_min_side,
            max_long_side=v1_max_long_side,
            denoise_kernel=v1_denoise_kernel,
            bg_enable=v1_bg_enable,
            bg_kernel=v1_bg_kernel,
            bg_method=v1_bg_method,
            unsharp_sigma=v1_unsharp_sigma,
            unsharp_weight=v1_unsharp_weight,
            deskew_angle_thresh=v1_deskew_angle_thresh,
        )
        v1_filename = os.path.basename(v1_path)
        v1_text, v1_time = run_easyocr(v1_path)
    except Exception as e:
        v1_error = str(e)

    v2_error = None
    v2_filename = None
    v2_text = ""
    v2_time = 0.0
    try:
        _, v2_path = preprocess_v2(
            original_path,
            target_min_side=v2_target_min_side,
            clahe_clip=clahe_clip,
            clahe_tile=clahe_tile,
            thresh_block=thresh_block,
            thresh_c=thresh_c,
            morph_enable=morph_enable,
            morph_h_len=morph_h_len,
            morph_v_len=morph_v_len,
        )
        v2_filename = os.path.basename(v2_path)
        v2_text, v2_time = run_easyocr(v2_path)
    except Exception as e:
        v2_error = str(e)

    v3_error = None
    v3_filename = None
    v3_text = ""
    v3_time = 0.0
    try:
        _, v3_path = preprocess_v3(
            original_path,
            target_min_side=v3_target_min_side,
            bg_kernel=v3_bg_kernel,
            bg_method=v3_bg_method,
            clahe_clip=v3_clahe_clip,
            clahe_tile=v3_clahe_tile,
            thresh_block=v3_thresh_block,
            thresh_c=v3_thresh_c,
            close_kernel=v3_close_kernel,
            morph_enable=v3_morph_enable,
            morph_h_len=v3_morph_h_len,
            morph_v_len=v3_morph_v_len,
        )
        v3_filename = os.path.basename(v3_path)
        v3_text, v3_time = run_easyocr(v3_path)
    except Exception as e:
        v3_error = str(e)

    return templates.TemplateResponse(
        request,
        "preprocessing_lab.html",
        {
            "params": params,
            "original_image": f"/uploads/{unique_name}",
            "v1_image": f"/uploads/{v1_filename}" if v1_filename else None,
            "v2_image": f"/uploads/{v2_filename}" if v2_filename else None,
            "v3_image": f"/uploads/{v3_filename}" if v3_filename else None,
            "v1_text": v1_text,
            "v1_time": f"{v1_time:.2f}s",
            "v1_error": v1_error,
            "v2_text": v2_text,
            "v2_time": f"{v2_time:.2f}s",
            "v2_error": v2_error,
            "v3_text": v3_text,
            "v3_time": f"{v3_time:.2f}s",
            "v3_error": v3_error,
        },
    )
