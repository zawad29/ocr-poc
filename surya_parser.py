"""Surya-specific NID parser.

Consumes Surya's structured output (line text + bbox) and uses geometry to
locate label-value pairs. Handles two NID layouts observed in `input/`:

- **Labeled card** (e.g. Image_1.png) — every field has an explicit label line
  (`নাম:`, `Name:`, `পিতা:`, `মাতা:`, `Date of Birth:`, `ID NO:`). Labels and
  values sit on the same row with the value to the right of the label, OR the
  value inline after the colon.

- **Unlabeled/sparse card** (e.g. Image_2.png) — most fields have no label on
  the card. Labels that do exist (`Name`, `NID No.`) may sit directly above
  or to the left of their value. The Bengali name / father / mother appear in
  fixed top-to-bottom reading order.

Quirks the parser handles:
- Surya sometimes returns HTML tags in text (`<b>...</b>`, `<br>` joining
  multiple physical lines into one output line).
- A stray date stamp near the card logo must not be confused with DOB.
- Label and value bboxes can be siblings (same row) OR stacked (value below).
"""

import re

from parser import parse_nid_fields

HEADER_TOKENS = (
    "গণপ্রজাতন্ত্রী",
    "বাংলাদেশ",
    "সরকার",
    "Government",
    "People's Republic",
    "Republic of Bangladesh",
    "National ID Card",
    "জাতীয় পরিচয়",
)

LABEL_VARIANTS = {
    "name_bn": ("নাম",),
    "name_en": ("Name",),
    "father": ("পিতা",),
    "mother": ("মাতা",),
    "dob": ("Date of Birth", "DOB"),
    "nid_number": ("ID NO", "NID No", "NID Number", "ID No"),
}

_ALL_LABELS = tuple({v for variants in LABEL_VARIANTS.values() for v in variants})

HTML_TAG_RE = re.compile(r"<[^>]+>")
BR_SPLIT_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
# Surya emits superscript/subscript OCR fragments inside <sup>...</sup> or
# <sub>...</sub>. These are artifacts (small stray marks, sub-line misreads),
# not field content — drop them with their contents before further parsing.
SUP_SUB_RE = re.compile(r"<\s*(sup|sub)\s*>.*?<\s*/\s*\1\s*>", re.IGNORECASE | re.DOTALL)
DATE_RE = re.compile(r"\b\d{1,2}\s+\w{3,9}\s+\d{4}\b")
NID_DIGITS_RE = re.compile(r"\d{3}[\s:]*\d{3}[\s:]*\d{4,}")
UPPERCASE_NAME_RE = re.compile(r"^[A-Z][A-Z\s\.\-']{4,}$")
LABEL_TRAIL_RE = re.compile(r"[:\.\-\s]+$")


def parse_surya_nid_fields(lines: list[dict]) -> dict:
    """Parse structured NID fields from Surya's text_lines output."""
    fields = {
        "name_bn": None,
        "name_en": None,
        "father": None,
        "mother": None,
        "dob": None,
        "nid_number": None,
    }
    if not lines:
        return fields

    normalized = _normalize(lines)
    body = _filter_header(normalized)

    for key, value in _extract_labeled(body).items():
        if value:
            fields[key] = value

    for key, value in _extract_positional(body, fields).items():
        if value and not fields[key]:
            fields[key] = value

    missing = [k for k, v in fields.items() if not v]
    if missing:
        text = "\n".join(line["text"] for line in normalized)
        fallback = parse_nid_fields(text)
        for key in missing:
            if fallback.get(key):
                fields[key] = fallback[key]

    return fields


def _normalize(lines: list[dict]) -> list[dict]:
    """Strip HTML tags, keep each bbox as one entry. `<br>` within a bbox is
    replaced with a space so sup/sub fragments stay attached to their line."""
    out = []
    for line in lines:
        raw = (line.get("text") or "").strip()
        bbox = line.get("bbox")
        if not raw or not bbox or len(bbox) != 4:
            continue
        stripped = SUP_SUB_RE.sub(" ", raw)
        joined = BR_SPLIT_RE.sub(" ", stripped)
        text = HTML_TAG_RE.sub("", joined).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        out.append({"text": text, "bbox": [float(c) for c in bbox]})
    out.sort(key=lambda l: l["bbox"][1])
    return out


def _filter_header(lines: list[dict]) -> list[dict]:
    return [l for l in lines if not any(tok in l["text"] for tok in HEADER_TOKENS)]


def _extract_labeled(lines: list[dict]) -> dict:
    """For each field, find the first line whose text starts with a known label
    and either read the value inline after the colon or locate the value line
    geometrically (same row → right, else directly below with x-overlap)."""
    result = {}
    for field, variants in LABEL_VARIANTS.items():
        for line in lines:
            matched = _match_label(line["text"], variants)
            if matched is None:
                continue
            value = None
            inline = _value_after_label(line["text"], matched)
            if inline and not _is_pure_label_line(inline):
                value = _finalize(field, inline)
            if not value:
                sibling = _find_sibling_value(line, lines)
                if sibling:
                    value = _finalize(field, sibling["text"])
            if value:
                result[field] = value
                break
    return result


def _match_label(text: str, variants: tuple) -> str | None:
    for variant in variants:
        if re.match(rf"^\s*{re.escape(variant)}\s*[:\.\-]?", text, re.IGNORECASE):
            return variant
    return None


def _value_after_label(text: str, label: str) -> str:
    return re.sub(
        rf"^\s*{re.escape(label)}\s*[:\.\-]?\s*", "", text, count=1, flags=re.IGNORECASE
    ).strip()


def _is_pure_label_line(text: str) -> bool:
    stripped = LABEL_TRAIL_RE.sub("", text.strip())
    return any(stripped.lower() == label.lower() for label in _ALL_LABELS)


def _find_sibling_value(label_line: dict, lines: list[dict]) -> dict | None:
    """Prefer a same-row line to the right of the label. If none, use the line
    directly below with horizontal overlap."""
    lx1, ly1, lx2, ly2 = label_line["bbox"]
    label_h = max(ly2 - ly1, 1)
    label_cx = (lx1 + lx2) / 2

    best_right, best_right_dx = None, None
    for cand in lines:
        if cand is label_line or _is_pure_label_line(cand["text"]):
            continue
        cx1, cy1, cx2, cy2 = cand["bbox"]
        if min(ly2, cy2) - max(ly1, cy1) < 0.5 * label_h:
            continue
        cand_cx = (cx1 + cx2) / 2
        if cand_cx <= label_cx:
            continue
        dx = cand_cx - label_cx
        if best_right_dx is None or dx < best_right_dx:
            best_right, best_right_dx = cand, dx
    if best_right:
        return best_right

    label_cy = (ly1 + ly2) / 2
    best_below, best_below_gap = None, None
    for cand in lines:
        if cand is label_line or _is_pure_label_line(cand["text"]):
            continue
        cx1, cy1, cx2, cy2 = cand["bbox"]
        cand_cy = (cy1 + cy2) / 2
        if cand_cy <= label_cy:
            continue
        gap = cy1 - ly2
        if gap > 1.5 * label_h:
            continue
        if min(lx2, cx2) - max(lx1, cx1) <= 0:
            continue
        if best_below_gap is None or gap < best_below_gap:
            best_below, best_below_gap = cand, gap
    return best_below


def _extract_positional(lines: list[dict], already: dict) -> dict:
    """Fill fields still missing from labeled extraction using reading-order
    rules. Pure-label lines are excluded. Bengali-script non-label lines are
    assigned in order to name_bn → father → mother."""
    result = {}
    non_label = [l for l in lines if not _is_pure_label_line(l["text"])]

    if not already.get("dob"):
        dob = _find_first(
            non_label,
            lambda l: ("Date of Birth" in l["text"]) or re.search(r"\bDOB\b", l["text"], re.I),
        ) or _find_first(non_label, lambda l: bool(DATE_RE.search(l["text"])))
        if dob:
            result["dob"] = _finalize("dob", dob["text"])

    if not already.get("nid_number"):
        nid = _find_first(
            non_label,
            lambda l: re.search(r"(?:NID\s*No|ID\s*NO)", l["text"], re.I)
            and any(c.isdigit() for c in l["text"]),
        ) or _find_first(non_label, lambda l: bool(NID_DIGITS_RE.search(l["text"])))
        if nid:
            result["nid_number"] = _finalize("nid_number", nid["text"])

    if not already.get("name_en"):
        name_en = _find_first(
            non_label,
            lambda l: not any(c.isdigit() for c in l["text"])
            and bool(UPPERCASE_NAME_RE.match(l["text"])),
        )
        if name_en:
            result["name_en"] = _finalize("name_en", name_en["text"])

    bengali_lines = [l for l in non_label if _dominant_script(l["text"]) == "bengali"]
    bengali_groups = _group_by_vertical_gap(bengali_lines)
    g_iter = iter(bengali_groups)
    for slot in ("name_bn", "father", "mother"):
        try:
            group = next(g_iter)
        except StopIteration:
            break
        if already.get(slot):
            continue
        text = " ".join(l["text"] for l in sorted(group, key=lambda l: l["bbox"][1]))
        result[slot] = _finalize(slot, text)

    return result


def _group_by_vertical_gap(lines: list[dict], factor: float = 0.8) -> list[list[dict]]:
    """Cluster vertically-adjacent lines into field groups. A new group starts
    when the gap to the previous line exceeds `factor * median_line_height`.
    This keeps multi-fragment fields (e.g. sup/sub mother name) together while
    preserving the gap between distinct fields like father and mother."""
    if not lines:
        return []
    ordered = sorted(lines, key=lambda l: l["bbox"][1])
    heights = sorted(max(l["bbox"][3] - l["bbox"][1], 1) for l in ordered)
    median_h = heights[len(heights) // 2]
    threshold = max(median_h * factor, 10)

    groups = [[ordered[0]]]
    for prev, curr in zip(ordered, ordered[1:]):
        gap = curr["bbox"][1] - prev["bbox"][3]
        if gap > threshold:
            groups.append([curr])
        else:
            groups[-1].append(curr)
    return groups


def _find_first(lines, pred):
    for line in lines:
        if pred(line):
            return line
    return None


def _dominant_script(text: str) -> str:
    bengali = len(re.findall(r"[\u0980-\u09FF]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if bengali == 0 and latin == 0:
        return "other"
    return "bengali" if bengali >= latin else "latin"


def _finalize(field: str, value: str) -> str | None:
    value = HTML_TAG_RE.sub("", value or "")
    value = _clean(value)
    if field == "dob":
        value = re.sub(
            r"^(?:Date\s*of\s*Birth|DOB)\s*[:\.\-]?\s*", "", value, flags=re.IGNORECASE
        ).strip()
    elif field == "nid_number":
        value = re.sub(
            r"^(?:ID\s*NO|NID\s*(?:No|Number)?)\.?\s*[:\.\-]?\s*", "", value, flags=re.IGNORECASE
        )
        value = re.sub(r"\D", "", value)
    return value or None


def _clean(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^[\-:\.=\s]+", "", s)
    s = re.sub(r"[\-:\.=\s]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()
