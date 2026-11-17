"""Vietnamese text-to-number converter and normalizer for spoken voice numbers."""
from __future__ import annotations

import re

VI_DIGITS = {
    "không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4,
    "năm": 5, "lăm": 5, "nhăm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9,
}

_UNIT_SCALES = {
    "nghìn": 1_000,
    "ngàn": 1_000,
    "k": 1_000,
    "triệu": 1_000_000,
    "tỷ": 1_000_000_000,
}


def _parse_under_thousand(tokens: list[str]) -> int | None:
    """Parse Vietnamese integer < 1000 from word tokens (e.g. 'hai trăm năm mươi', 'mười lăm')."""
    if not tokens:
        return None
    total = 0
    i = 0
    matched = False
    n = len(tokens)

    # Hundreds place: X trăm ...
    if i + 1 < n and tokens[i + 1] == "trăm" and tokens[i] in VI_DIGITS:
        total += VI_DIGITS[tokens[i]] * 100
        i += 2
        matched = True
        if i < n and tokens[i] in ("linh", "lẻ"):
            i += 1
            if i < n and tokens[i] in VI_DIGITS:
                total += VI_DIGITS[tokens[i]]
                return total
            return total

    # Tens place
    if i < n:
        if tokens[i] == "mười":
            total += 10
            i += 1
            matched = True
            if i < n and tokens[i] in VI_DIGITS:
                total += VI_DIGITS[tokens[i]]
                i += 1
        elif i + 1 < n and tokens[i + 1] == "mươi" and tokens[i] in VI_DIGITS:
            total += VI_DIGITS[tokens[i]] * 10
            i += 2
            matched = True
            if i < n and tokens[i] in VI_DIGITS:
                total += VI_DIGITS[tokens[i]]
                i += 1
        elif tokens[i] in VI_DIGITS:
            # Handle shortened tens like "hai lăm" (25), "ba mốt" (31) if followed by digit
            if i + 1 < n and tokens[i + 1] in VI_DIGITS and VI_DIGITS[tokens[i]] >= 2:
                total += VI_DIGITS[tokens[i]] * 10 + VI_DIGITS[tokens[i + 1]]
                i += 2
            else:
                total += VI_DIGITS[tokens[i]]
                i += 1
            matched = True

    return total if matched else None


def parse_vietnamese_number(text: str) -> int | None:
    """Parse Vietnamese digits or compound number words into a non-negative integer."""
    if not text:
        return None
    t = text.strip().lower()

    # 1. Handle numeric literals with optional thousand separators / scale suffixes
    m_scaled = re.search(
        r"\b(\d+(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\s*(nghìn|ngàn|triệu|tỷ|k)?\b",
        t,
    )
    if m_scaled:
        raw_num = m_scaled.group(1)
        scale_word = m_scaled.group(2)
        if re.match(r"^\d{1,3}(?:[.,]\d{3})+$", raw_num):
            val: float = float(raw_num.replace(".", "").replace(",", ""))
        else:
            val = float(raw_num.replace(",", "."))
        if scale_word in _UNIT_SCALES:
            val *= _UNIT_SCALES[scale_word]
        return int(round(val))

    # 2. Compound Vietnamese word numbers (with recursive scale decomposition)
    tokens = re.findall(r"[a-zà-ỹ]+", t)
    if not tokens:
        return None

    return _parse_compound_words(tokens)


def _parse_compound_words(tokens: list[str]) -> int | None:
    """Recursively parse compound Vietnamese number words using descending scale delimiters."""
    if not tokens:
        return None

    for scale_word in ("tỷ", "triệu", "nghìn", "ngàn"):
        if scale_word in tokens:
            idx = tokens.index(scale_word)
            scale = _UNIT_SCALES[scale_word]
            prefix_val = _parse_compound_words(tokens[:idx])
            if prefix_val is None:
                continue
            tail_tokens = tokens[idx + 1 :]
            rem_val = _parse_compound_words(tail_tokens) or 0
            # Handle shorthand like "một triệu hai" -> 1_200_000
            if rem_val < 10 and len(tail_tokens) == 1 and scale >= 1000:
                rem_val *= scale // 10
            return prefix_val * scale + rem_val

    return _parse_under_thousand(tokens)


_SPOKEN_NUM_TOKENS = set(VI_DIGITS.keys()) | {
    "trăm",
    "linh",
    "lẻ",
    "mười",
    "mươi",
    "tỷ",
    "triệu",
    "nghìn",
    "ngàn",
}


def extract_vietnamese_amount(text: str) -> int | None:
    """Extract the primary numerical amount or quantity from a full Vietnamese utterance.

    Prioritizes scaled numeric literals (e.g. '1.5 triệu', '250.000') and contiguous
    spoken compound number spans (e.g. 'hai triệu năm trăm nghìn') while ignoring clock-time
    literals like '8 giờ' or '8 rưỡi'.
    """
    if not text:
        return None
    t = text.strip().lower()

    # 1. Explicit scaled numeric amounts first (e.g. 1.5 triệu, 500k, 250.000)
    for m in re.finditer(r"\b(\d+(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\s*(nghìn|ngàn|triệu|tỷ|k)?\b", t):
        after = t[m.end() : m.end() + 12].lstrip()
        # Skip time/date digits if not carrying a monetary scale suffix
        if not m.group(2) and re.match(r"^(giờ|h\b|phút|rưỡi|tháng|năm|/|:)", after):
            continue
        if m.group(2) or "." in m.group(1) or "," in m.group(1) or len(m.group(1)) >= 4:
            return parse_vietnamese_number(m.group(0))

    # 2. Extract longest contiguous spoken Vietnamese number-word spans
    tokens = re.findall(r"[a-zà-ỹ]+", t)
    spans: list[list[str]] = []
    cur: list[str] = []
    for tok in tokens:
        if tok in _SPOKEN_NUM_TOKENS:
            cur.append(tok)
        elif cur:
            spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)

    best_val: int | None = None
    for sp in sorted(spans, key=len, reverse=True):
        val = _parse_compound_words(sp)
        if val is not None:
            best_val = val
            break
    if best_val is not None:
        return best_val

    # 3. Fallback to non-time standalone digits
    for m in re.finditer(r"\b(\d+)\b", t):
        after = t[m.end() : m.end() + 12].lstrip()
        if not re.match(r"^(giờ|h\b|phút|rưỡi|tháng|năm|/|:)", after):
            return int(m.group(1))
    return None


_NON_MONEY_CLASSIFIERS = (
    r"^(giờ|h\b|phút|giây|rưỡi|tháng|năm|ngày|hôm|lần|bước|/|:|"
    r"tô|bát|dĩa|đĩa|phần|suất|hộp|gói|chai|lon|ly|cốc|tách|"
    r"cái|chiếc|quả|trái|cuốn|quyển|vé|ghế|chỗ|người|khách|cổ phiếu|cổ)"
)


def extract_monetary_amount(text: str) -> int | None:
    """Extract only genuine VND monetary amounts from a Vietnamese utterance.

    Ignores clock times ('8 giờ rưỡi') and item quantity counts ('1 tô phở',
    '2 hộp yến mạch') when no monetary scale or currency unit is present.
    """
    if not text:
        return None
    t = text.strip().lower()

    # 1. Explicit numeric money with unit/scale suffix or grouped thousands (e.g. 250.000, 1.5 triệu, 500k)
    for m in re.finditer(
        r"\b(\d+(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\s*(nghìn|ngàn|triệu|tỷ|k|đồng|vnđ|vnd|đ)?\b",
        t,
    ):
        after = t[m.end() : m.end() + 16].lstrip()
        if not m.group(2) and re.match(_NON_MONEY_CLASSIFIERS, after):
            continue
        if m.group(2) or "." in m.group(1) or "," in m.group(1) or len(m.group(1)) >= 4:
            val_str = m.group(1)
            unit = m.group(2) or ""
            if unit in _UNIT_SCALES:
                return parse_vietnamese_number(f"{val_str} {unit}")
            clean_digits = val_str.replace(".", "").replace(",", "")
            if clean_digits.isdigit():
                return int(clean_digits)

    # 2. Compound Vietnamese spoken number spans with monetary scales (nghìn/triệu/tỷ/ngàn)
    tokens = re.findall(r"[a-zà-ỹ]+", t)
    spans: list[list[str]] = []
    cur: list[str] = []
    for tok in tokens:
        if tok in _SPOKEN_NUM_TOKENS:
            cur.append(tok)
        elif cur:
            spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)

    for sp in sorted(spans, key=len, reverse=True):
        if not any(scale in sp for scale in ("nghìn", "ngàn", "triệu", "tỷ")):
            continue
        val = _parse_compound_words(sp)
        if val is not None:
            return val
    return None
