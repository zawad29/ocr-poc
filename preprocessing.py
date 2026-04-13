import cv2
import numpy as np
import os


def preprocess_image(image_path: str) -> tuple[np.ndarray, str]:
    """Preprocess NID card image for OCR. Returns (preprocessed_array, saved_path)."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Resize if too large
    h, w = img.shape[:2]
    max_dim = 2000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=15, C=8
    )

    # Deskew
    thresh = _deskew(thresh)

    # Save preprocessed image
    base, ext = os.path.splitext(os.path.basename(image_path))
    preprocessed_path = os.path.join(
        os.path.dirname(image_path), f"{base}_preprocessed.png"
    )
    cv2.imwrite(preprocessed_path, thresh)

    return thresh, preprocessed_path


def _deskew(image: np.ndarray) -> np.ndarray:
    """Detect and correct skew angle."""
    coords = np.column_stack(np.where(image > 0))
    if len(coords) < 100:
        return image

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # Normalize angle
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    # Only correct if skew is meaningful but not extreme
    if abs(angle) < 0.5 or abs(angle) > 15:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated
