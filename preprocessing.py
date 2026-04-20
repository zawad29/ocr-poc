import cv2
import numpy as np
import os


CANONICAL_W = 1600
CANONICAL_H = 1008  # ID-1 aspect ratio (85.60 x 53.98 mm => 1.587)


def preprocess_image(image_path: str) -> tuple[np.ndarray, str]:
    """
    OCR preprocessing for ID card images with security patterns.
    Returns (preprocessed_array, saved_path).

    Pipeline:
    1. Normalize to canonical canvas (CANONICAL_W x CANONICAL_H):
       - Detect the card quadrilateral and perspective-warp it to fill the
         canvas (crop + deskew + resize in one step).
       - If no card border is detectable, stretch-resize to the canonical size.
    2. Grayscale conversion.
    3. Gentle Gaussian denoise.
    4. Background flatten (MORPH_CLOSE 31x31 + divide) to suppress
       holographic/seal watermarks while leaving letters dark.
    5. Histogram normalize.
    6. Unsharp mask sharpening.

    The output canvas size is a contract for downstream bbox-based parsers:
    every preprocessed image is exactly CANONICAL_W x CANONICAL_H so spatial
    thresholds stay consistent across cards. Output is grayscale — no
    binarization or thresholding.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Step 1: normalize to canonical canvas
    quad = _detect_card_quad(img)
    if quad is not None:
        img = _warp_to_canonical(img, quad)
    else:
        img = cv2.resize(
            img, (CANONICAL_W, CANONICAL_H), interpolation=cv2.INTER_CUBIC
        )

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Gentle Gaussian denoise
    # Small kernel to smooth sensor noise without blurring text strokes
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 4: Background flatten
    # MORPH_CLOSE with a 31x31 kernel fills small dark holes (letter strokes),
    # yielding an estimate of just the light background + watermarks. Dividing
    # the denoised image by that estimate pushes the watermark toward white
    # while text stays dark.
    bg_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    background = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, bg_struct)
    src_f = denoised.astype(np.float32)
    bg_f = background.astype(np.float32) + 1.0
    flattened = np.clip((src_f / bg_f) * 255.0, 0, 255).astype(np.uint8)

    # Step 5: Normalize brightness and contrast
    # Stretches histogram to full 0-255 range for consistent OCR input
    normalized = cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX)

    # Step 6: Unsharp mask — sharpens text edges
    blurred = cv2.GaussianBlur(normalized, (0, 0), 3)
    result = cv2.addWeighted(normalized, 1.5, blurred, -0.5, 0)

    # Save preprocessed image
    base, _ = os.path.splitext(os.path.basename(image_path))
    preprocessed_path = os.path.join(
        os.path.dirname(image_path), f"{base}_preprocessed.png"
    )
    cv2.imwrite(preprocessed_path, result)

    return result, preprocessed_path


def _detect_card_quad(img: np.ndarray) -> np.ndarray | None:
    """Find the 4-corner card quadrilateral. Returns points ordered
    TL, TR, BR, BL as float32, or None if no suitable quad is detected."""
    h, w = img.shape[:2]
    img_area = float(h * w)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.30 * img_area:
            break  # remaining contours are smaller; stop
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        return _order_points(approx.reshape(4, 2).astype(np.float32))
    return None


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL using sum/diff of coordinates:
    TL has smallest x+y, BR largest; TR has smallest y-x, BL largest."""
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _warp_to_canonical(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-warp the detected quad to fill the canonical canvas."""
    dst = np.array(
        [[0, 0], [CANONICAL_W, 0], [CANONICAL_W, CANONICAL_H], [0, CANONICAL_H]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(
        img, M, (CANONICAL_W, CANONICAL_H),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
