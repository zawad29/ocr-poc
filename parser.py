import re


def parse_nid_fields(raw_text: str) -> dict:
    """Parse structured fields from raw OCR text of a Bangladesh NID card."""
    fields = {
        "name_bn": None,
        "name_en": None,
        "father": None,
        "mother": None,
        "dob": None,
        "nid_number": None,
    }

    if not raw_text:
        return fields

    fields["nid_number"] = _extract_nid_number(raw_text)
    fields["dob"] = _extract_dob(raw_text)
    fields["name_en"] = _extract_name_en(raw_text)
    fields["name_bn"] = _extract_name_bn(raw_text)
    fields["father"] = _extract_father(raw_text)
    fields["mother"] = _extract_mother(raw_text)

    return fields


def _clean(s: str) -> str:
    """Strip OCR artifacts from extracted text."""
    s = s.strip()
    # Remove leading punctuation artifacts
    s = re.sub(r'^[\-:\.=\s]+', '', s)
    # Remove trailing punctuation artifacts
    s = re.sub(r'[\-:\.=\s]+$', '', s)
    # Collapse multiple spaces
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def _extract_nid_number(text: str) -> str | None:
    # Try labeled patterns first (handle OCR misreads: ND/NID, N0/NO)
    m = re.search(
        r'(?:N?ID\s*(?:No|Number)?|ID\s*N[O0]|ND\s*No)\s*[:\.]?\s*(\d[\d\s:]{8,19})',
        text, re.IGNORECASE
    )
    if m:
        return re.sub(r'[\s:]+', '', m.group(1))

    # Try digit sequences with spaces or colons (e.g., "595 537 5075" or "595:537 5075")
    m = re.search(r'(\d{3}[\s:]+\d{3}[\s:]+\d{4,})', text)
    if m:
        return re.sub(r'[\s:]+', '', m.group(1))

    # Fallback: standalone long digit sequences
    for pattern in [r'\b(\d{17})\b', r'\b(\d{13})\b', r'\b(\d{10})\b']:
        m = re.search(pattern, text)
        if m:
            return m.group(1)

    return None


def _extract_dob(text: str) -> str | None:
    # Look for Date of Birth label
    m = re.search(
        r'(?:Date\s*of\s*Birth|DOB|জন্ম\s*তারিখ)\s*[:\.\-]?\s*(.+?)(?:\n|ID|NID|$)',
        text, re.IGNORECASE
    )
    if m:
        dob = _clean(m.group(1))
        # Remove quotes and extra punctuation from OCR
        dob = re.sub(r"['\"]", " ", dob)
        dob = re.sub(r'\s+', ' ', dob).strip()
        if dob:
            return dob

    # Fallback: common date patterns
    m = re.search(r'(\d{1,2}\s+\w{3,9}\s+\d{4})', text)
    if m:
        return m.group(1).strip()

    return None


def _extract_name_en(text: str) -> str | None:
    # Look for "Name" label followed by English text on same line
    m = re.search(
        r'Name\s*[:\.]?\s*([A-Za-z][A-Za-z\s\'\.]+)',
        text, re.IGNORECASE
    )
    if m:
        name = _clean(m.group(1))
        name = name.replace("'", " ").strip()
        name = re.sub(r'\s+', ' ', name)
        if len(name) > 3:
            return name

    # Look for "Name" on one line and the English name on the next line
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if re.search(r'\bNam[e]?\b', line, re.IGNORECASE) and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            # Check if next line is mostly uppercase Latin
            if re.match(r'^[A-Z][A-Z\s\.]{5,}', next_line):
                return _clean(next_line)

    # Fallback: find lines with mostly uppercase Latin letters (typical for NID names)
    for line in lines:
        line = line.strip()
        if re.match(r'^[A-Z][A-Z\s\.]{5,}$', line):
            return line

    return None


def _extract_name_bn(text: str) -> str | None:
    # Look for নাম label followed by Bengali text
    m = re.search(
        r'নাম\s*[:\.]?\s*([\u0980-\u09FF][\u0980-\u09FF\s\.]+)',
        text
    )
    if m:
        name = _clean(m.group(1))
        # Stop at next label (পিতা, মাতা, Name)
        name = re.split(r'(?:পিতা|মাতা|Name|নাম)', name)[0]
        return _clean(name) if _clean(name) else None

    # Fallback: first line with mostly Bengali characters after header lines
    lines = text.split('\n')
    header_keywords = ['গণপ্রজাতন্ত্রী', 'জাতীয়', 'Government', 'National', 'Republic']
    past_header = False
    for line in lines:
        line = line.strip()
        if any(kw in line for kw in header_keywords):
            past_header = True
            continue
        if past_header:
            bengali_chars = len(re.findall(r'[\u0980-\u09FF]', line))
            total_chars = len(re.sub(r'\s', '', line))
            if bengali_chars > 5 and total_chars > 0 and bengali_chars / total_chars > 0.6:
                # Skip if this looks like father/mother line
                if 'পিতা' not in line and 'মাতা' not in line:
                    return _clean(line)

    return None


def _extract_father(text: str) -> str | None:
    # Bengali label পিতা
    m = re.search(
        r'পিতা\s*[:\.]?\s*([\u0980-\u09FF][\u0980-\u09FF\s\.]+)',
        text
    )
    if m:
        name = _clean(m.group(1))
        # Stop before next label (মাতা or OCR misread মত)
        name = re.split(r'(?:মাতা|মত(?:[:\s]|$)|Name|Date)', name)[0]
        return _clean(name) if _clean(name) else None

    # English label
    m = re.search(
        r'Father\s*[:\.]?\s*([A-Za-z][\w\s\.]+)',
        text, re.IGNORECASE
    )
    if m:
        return _clean(m.group(1))

    return None


def _extract_mother(text: str) -> str | None:
    # Bengali label মাতা (also handle OCR misread as মত or মাত)
    m = re.search(
        r'(?:মাতা|মাত|মত)\s*[:\.]?\s*([\u0980-\u09FF][\u0980-\u09FF\s\.]+)',
        text
    )
    if m:
        name = _clean(m.group(1))
        # Stop before next label
        name = re.split(r'(?:Date|জন্ম|Birth|ID|NID)', name)[0]
        return _clean(name) if _clean(name) else None

    # English label
    m = re.search(
        r'Mother\s*[:\.]?\s*([A-Za-z][\w\s\.]+)',
        text, re.IGNORECASE
    )
    if m:
        return _clean(m.group(1))

    return None
