"""Vietnamese Relative DateTime Engine for ZAC 2026 LifeMate.
Resolves relative colloquial Vietnamese temporal references (hôm nay, ngày mai, ngày mốt,
thứ X tuần sau, X giờ chiều/sáng, v.v.) anchored to `current_time` from sample context.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

WEEKDAY_MAP = {
    "thứ hai": 0, "thứ 2": 0, "t2": 0,
    "thứ ba": 1, "thứ 3": 1, "t3": 1,
    "thứ tư": 2, "thứ 4": 2, "t4": 2,
    "thứ năm": 3, "thứ 5": 3, "t5": 3,
    "thứ sáu": 4, "thứ 6": 4, "t6": 4,
    "thứ bảy": 5, "thứ 7": 5, "t7": 5,
    "chủ nhật": 6, "cn": 6,
}

_TOMORROW_RE = re.compile(r"(?:ngày\s+mai|sáng\s+mai|chiều\s+mai|tối\s+mai|\bmai\b)")
_DAY_AFTER_RE = re.compile(r"(?:ngày\s+mốt|hôm\s+kia|ngày\s+kia)")
_TODAY_RE = re.compile(r"(?:hôm\s+nay|chiều\s+nay|sáng\s+nay|tối\s+nay)")

_SPOKEN_HOURS = {
    "mười hai": 12,
    "mười một": 11,
    "mười": 10,
    "chín": 9,
    "tám": 8,
    "bảy": 7,
    "sáu": 6,
    "năm": 5,
    "bốn": 4,
    "tư": 4,
    "ba": 3,
    "hai": 2,
    "một": 1,
}
_SPOKEN_HOUR_ALT = "|".join(re.escape(k) for k in _SPOKEN_HOURS)
_DIGIT_TIME_RE = re.compile(r"(\d{1,2})\s*(?:(h|giờ|:)\s*(\d{1,2}|rưỡi)?|rưỡi\b)")
_SPOKEN_TIME_RE = re.compile(rf"\b({_SPOKEN_HOUR_ALT})\s+giờ(?:\s+(rưỡi|\d{{1,2}}))?\b")


class VietnameseRelativeDateTimeEngine:
    """Deterministic Vietnamese relative date/time resolution."""

    @staticmethod
    def resolve_date_string(query_text: str, current_time_iso: str) -> str | None:
        """Extract and resolve relative date from text. Returns YYYY-MM-DD or None."""
        if not current_time_iso:
            return None
        try:
            base_dt = datetime.fromisoformat(current_time_iso)
        except Exception:
            return None

        t = query_text.lower()
        target_date = base_dt.date()
        matched = False

        if _TOMORROW_RE.search(t):
            target_date += timedelta(days=1)
            matched = True
        elif _DAY_AFTER_RE.search(t):
            target_date += timedelta(days=2)
            matched = True
        elif "hôm qua" in t:
            target_date -= timedelta(days=1)
            matched = True
        elif _TODAY_RE.search(t):
            matched = True
        else:
            for wk_str, wk_idx in WEEKDAY_MAP.items():
                pattern = rf"\b{wk_str}\b"
                if re.search(pattern, t):
                    current_wk = base_dt.weekday()
                    delta_days = (wk_idx - current_wk) % 7
                    if "tuần sau" in t or "tuần tới" in t:
                        delta_days += 7
                    elif delta_days == 0 and "tuần sau" not in t:
                        delta_days = 7
                    target_date += timedelta(days=delta_days)
                    matched = True
                    break

        if matched:
            return target_date.strftime("%Y-%m-%d")
        return None

    @classmethod
    def resolve_datetime_iso(cls, query_text: str, current_time_iso: str, tz_offset: str = "+07:00") -> str | None:
        """Resolve full ISO datetime with hours and minutes anchored to current_time."""
        if not current_time_iso:
            return None
        try:
            base_dt = datetime.fromisoformat(current_time_iso)
        except Exception:
            return None

        date_str = cls.resolve_date_string(query_text, current_time_iso)
        if not date_str:
            return None

        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        target_hour = 9  # Default morning 09:00 if not specified
        target_minute = 0

        t = query_text.lower()
        h: int | None = None
        m = 0

        digit_m = _DIGIT_TIME_RE.search(t)
        if digit_m:
            h = int(digit_m.group(1))
            min_tok = digit_m.group(3)
            if min_tok == "rưỡi" or (digit_m.group(2) is None and "rưỡi" in digit_m.group(0)):
                m = 30
            elif min_tok:
                m = int(min_tok)
        else:
            spoken_m = _SPOKEN_TIME_RE.search(t)
            if spoken_m:
                h = _SPOKEN_HOURS.get(spoken_m.group(1))
                min_tok = spoken_m.group(2)
                if min_tok == "rưỡi":
                    m = 30
                elif min_tok:
                    m = int(min_tok)

        if h is not None and 0 <= h <= 23 and 0 <= m <= 59:
            if any(w in t for w in ("chiều", "tối", "pm")) and h < 12:
                h += 12
            elif "sáng" in t and h == 12:
                h = 0
            target_hour = h
            target_minute = m

        dt = datetime(target_date.year, target_date.month, target_date.day, target_hour, target_minute, 0)
        return dt.strftime("%Y-%m-%dT%H:%M:%S") + tz_offset
