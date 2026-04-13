import cv2
import numpy as np
import os


def preprocess_image(image_path: str) -> tuple[np.ndarray, str]:
    """
    OCR preprocessing for ID card images with security patterns.
    Returns (preprocessed_array, saved_path).

    Pipeline:
    1. Upscale to ~300 DPI equivalent resolution
    2. Grayscale conversion
    3. Gaussian denoise (gentle, preserves text)
    4. Normalize brightness/contrast
    5. Unsharp mask sharpening
    6. Deskew via Hough line detection

    No binarization or aggressive filtering — modern OCR engines perform
    best on clean grayscale, especially for ID cards with holographic
    or patterned security backgrounds.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Step 1: Scale to optimal size for OCR
    # Target: shortest side ~1000px (roughly 300 DPI for ID card size)
    h, w = img.shape[:2]
    min_side = min(h, w)
    if min_side < 1000:
        scale = 1000 / min_side
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif max(h, w) > 3000:
        scale = 3000 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Gentle Gaussian denoise
    # Small kernel to smooth sensor noise without blurring text strokes
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 4: Normalize brightness and contrast
    # Stretches histogram to full 0-255 range for consistent OCR input
    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)

    # Step 5: Unsharp mask — sharpens text edges
    blurred = cv2.GaussianBlur(normalized, (0, 0), 3)
    sharpened = cv2.addWeighted(normalized, 1.5, blurred, -0.5, 0)

    # Step 6: Deskew
    result = _deskew(sharpened)

    # Save preprocessed image
    base, _ = os.path.splitext(os.path.basename(image_path))
    preprocessed_path = os.path.join(
        os.path.dirname(image_path), f"{base}_preprocessed.png"
    )
    cv2.imwrite(preprocessed_path, result)

    return result, preprocessed_path


def _deskew(image: np.ndarray) -> np.ndarray:
    """Detect and correct skew angle using Hough Line Transform."""
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

    median_angle = np.median(angles)

    if abs(median_angle) < 0.3:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated
