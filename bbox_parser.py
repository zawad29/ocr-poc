"""Uniform bbox-based NID parser.

Consumes `[{text, bbox, confidence}]` from any OCR engine whose input was the
canonical 1600x1008 canvas produced by `preprocessing.preprocess_image`, and
picks field values by checking which lines' bbox centers fall inside fixed
Areas of Interest (AOIs).

Two card layouts:
- Type A (labeled): every field has a label on its row (নাম:, Name:, পিতা:,
  মাতা:, Date of Birth:, ID NO:). Example: input/Image_1.png.
- Type B (sparse smartcard): most fields unlabeled, values sit in fixed
  positions down the right side of the card. A gold/silver smartcard chip
  occupies the right-middle region. Example: input/Image_2.png.

Classification is **image-based, not text-based**: the smartcard chip on Type
B produces a dense edge-pattern in a known ROI on the canonical canvas. OCR
text is too unreliable across engines to drive this decision.

Field *selection* uses only bbox geometry (center-point-in-AOI). Regex is
used only to clean the selected text (strip label prefixes) and to pull DOB
(`DD Month YYYY`) and NID (10/13/17 digits) patterns out of their AOI text.
"""

import re

import cv2
import numpy as np

CANVAS_W = 1600
CANVAS_H = 1008

# Card-type classifier: fraction of Canny edges found inside the chip ROI of
# the canonical canvas. Type B (smartcard) has contact-pattern edges here;
# Type A has only background/watermark. Measured: Type A ≈ 0.02, Type B ≈ 0.06.
CHIP_ROI_FRAC = (0.72, 0.30, 0.98, 0.70)
CHIP_EDGE_THRESHOLD = 0.035

_FIELDS = ("name_bn", "name_en", "father", "mother", "dob", "nid_number")
_LABEL_PREFIX_TOKENS = (
    "নাম", "পিতা", "মাতা",
    "Name", "Date of Birth", "DOB", "ID NO", "NID No", "NID Number", "ID No",
)

# AOIs: (x1, y1, x2, y2) as fractions of the 1600x1008 canvas.
# Each AOI covers the *center region* of the value line. Adjacent AOIs are
# vertically non-overlapping so a line's center unambiguously maps to one field.
AOI = {
    "A": {
        "name_bn":    (0.35, 0.33, 0.85, 0.44),
        "name_en":    (0.35, 0.44, 0.95, 0.55),
        "father":     (0.35, 0.56, 0.75, 0.66),
        "mother":     (0.35, 0.66, 0.80, 0.77),
        "dob":        (0.25, 0.77, 0.80, 0.87),
        "nid_number": (0.25, 0.87, 0.80, 0.95),
    },
    "B": {
        "name_bn":    (0.28, 0.28, 0.78, 0.40),
        "name_en":    (0.28, 0.42, 0.78, 0.52),
        "father":     (0.28, 0.53, 0.70, 0.63),
        "mother":     (0.28, 0.64, 0.70, 0.76),
        "dob":        (0.28, 0.77, 0.75, 0.85),
        "nid_number": (0.28, 0.85, 0.75, 0.93),
    },
}

HTML_TAG_RE = re.compile(r"<[^>]+>")
LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:" + "|".join(re.escape(t) for t in _LABEL_PREFIX_TOKENS) + r")\s*[:.\-]?\s*",
    re.IGNORECASE,
)
DOB_PREFIX_RE = re.compile(
    r"^\s*(?:Date\s*of\s*Birth|DOB)\s*[:.\-]?\s*", re.IGNORECASE
)

# DOB format: "10 July, 1999" (day + month name + optional comma + 4-digit year).
# Case-insensitive, comma optional, month may be short (Oct) or full (October).
DOB_RE = re.compile(
    r"\b(\d{1,2})\s+([A-Za-z]{3,9}),?\s+(\d{4})\b"
)

# NID candidates: contiguous runs of digits and spaces. After stripping spaces,
# only lengths 10, 13, or 17 are valid NID numbers.
NID_CANDIDATE_RE = re.compile(r"\d(?:[\d\s]*\d)?")
_VALID_NID_LENGTHS = (10, 13, 17)


def parse_nid_fields_by_bbox(
    lines: list[dict], image: np.ndarray, engine: str | None = None
) -> dict:
    """Parse NID fields from bbox-carrying OCR lines on the canonical canvas.

    `image` must be the preprocessed canonical-canvas array — it drives the
    card-type classifier."""
    fields = {k: None for k in _FIELDS}
    card_type = _classify_image(image)
    print(f"[bbox_parser] {engine or 'unknown'}: card type = {card_type}")
    if not lines:
        return fields

    normalized = _normalize(lines)

    # Name/father/mother: pick a single line by center-point containment.
    for field in ("name_bn", "name_en", "father", "mother"):
        picked = _select(normalized, AOI[card_type][field])
        if picked:
            fields[field] = _clean_field(field, picked["text"])

    # DOB and NID: regex, but scoped to their AOI. Gather all lines whose
    # center falls in the AOI, concatenate, then pattern-match.
    dob_text = _collect_in_aoi(normalized, AOI[card_type]["dob"])
    fields["dob"] = _extract_dob(dob_text)

    nid_text = _collect_in_aoi(normalized, AOI[card_type]["nid_number"])
    fields["nid_number"] = _extract_nid(nid_text)

    return fields


def _collect_in_aoi(lines: list[dict], aoi_frac: tuple) -> str:
    """Concatenate (sorted top-to-bottom, left-to-right) the text of every line
    whose bbox center sits inside the AOI."""
    x1 = aoi_frac[0] * CANVAS_W
    y1 = aoi_frac[1] * CANVAS_H
    x2 = aoi_frac[2] * CANVAS_W
    y2 = aoi_frac[3] * CANVAS_H
    hits = []
    for line in lines:
        bx1, by1, bx2, by2 = line["bbox"]
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            hits.append(line)
    hits.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
    return " ".join(l["text"] for l in hits)


def _extract_dob(text: str) -> str | None:
    m = DOB_RE.search(text)
    if not m:
        return None
    day, month, year = m.group(1), m.group(2), m.group(3)
    return f"{int(day)} {month.capitalize()} {year}"


def _extract_nid(text: str) -> str | None:
    """Return the first digit run whose length (spaces stripped) is 10, 13, or 17."""
    for match in NID_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"\s+", "", match.group(0))
        if len(digits) in _VALID_NID_LENGTHS:
            return digits
    return None


def _normalize(lines: list[dict]) -> list[dict]:
    """Strip HTML tags, drop entries with missing/invalid bbox."""
    out = []
    for line in lines:
        raw = (line.get("text") or "").strip()
        bbox = line.get("bbox")
        if not raw or not bbox or len(bbox) != 4:
            continue
        text = HTML_TAG_RE.sub("", raw).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        out.append({
            "text": text,
            "bbox": [float(c) for c in bbox],
            "confidence": line.get("confidence"),
        })
    return out


def _classify_image(image: np.ndarray) -> str:
    """Classify card type from the preprocessed canonical canvas.

    Type B (smartcard) has a chip in the right-middle region whose contact
    pattern produces dense Canny edges; Type A does not. Threshold is set
    halfway between measured densities of known samples (A ≈ 0.02, B ≈ 0.06)."""
    if image is None or image.size == 0:
        return "A"
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    x1 = int(CHIP_ROI_FRAC[0] * w)
    y1 = int(CHIP_ROI_FRAC[1] * h)
    x2 = int(CHIP_ROI_FRAC[2] * w)
    y2 = int(CHIP_ROI_FRAC[3] * h)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return "A"
    edges = cv2.Canny(roi, 80, 200)
    density = float(edges.mean()) / 255.0
    return "B" if density >= CHIP_EDGE_THRESHOLD else "A"


def _select(lines: list[dict], aoi_frac: tuple) -> dict | None:
    """Pick the line whose bbox center is inside the AOI; merge same-row
    siblings left-to-right when they both center inside."""
    x1 = aoi_frac[0] * CANVAS_W
    y1 = aoi_frac[1] * CANVAS_H
    x2 = aoi_frac[2] * CANVAS_W
    y2 = aoi_frac[3] * CANVAS_H
    aoi_cx = (x1 + x2) / 2
    aoi_cy = (y1 + y2) / 2

    hits = []
    for line in lines:
        bx1, by1, bx2, by2 = line["bbox"]
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            hits.append(line)

    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]

    # Group same-row hits (y-overlap >= 0.5 * min height) and concatenate them
    # left-to-right. Between-row groups: pick the one closest to AOI center.
    rows = []
    for line in sorted(hits, key=lambda l: (l["bbox"][1] + l["bbox"][3]) / 2):
        placed = False
        for row in rows:
            ly1, ly2 = line["bbox"][1], line["bbox"][3]
            lh = max(ly2 - ly1, 1)
            ry1 = min(r["bbox"][1] for r in row)
            ry2 = max(r["bbox"][3] for r in row)
            rh = max(ry2 - ry1, 1)
            overlap = min(ly2, ry2) - max(ly1, ry1)
            if overlap >= 0.5 * min(lh, rh):
                row.append(line)
                placed = True
                break
        if not placed:
            rows.append([line])

    def row_center_dist(row):
        cx = sum((r["bbox"][0] + r["bbox"][2]) / 2 for r in row) / len(row)
        cy = sum((r["bbox"][1] + r["bbox"][3]) / 2 for r in row) / len(row)
        return (cx - aoi_cx) ** 2 + (cy - aoi_cy) ** 2

    best_row = min(rows, key=row_center_dist)
    best_row.sort(key=lambda l: l["bbox"][0])
    merged_text = " ".join(r["text"] for r in best_row)
    bx1 = min(r["bbox"][0] for r in best_row)
    by1 = min(r["bbox"][1] for r in best_row)
    bx2 = max(r["bbox"][2] for r in best_row)
    by2 = max(r["bbox"][3] for r in best_row)
    return {"text": merged_text, "bbox": [bx1, by1, bx2, by2]}


def _clean_field(field: str, text: str) -> str | None:
    t = HTML_TAG_RE.sub("", text or "").strip()
    if field == "dob":
        t = DOB_PREFIX_RE.sub("", t)
    elif field == "nid_number":
        t = re.sub(r"\D", "", t)
        return t or None
    else:
        t = LABEL_PREFIX_RE.sub("", t)
    t = re.sub(r"^[\-:\.=\s]+", "", t)
    t = re.sub(r"[\-:\.=\s]+$", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or None
