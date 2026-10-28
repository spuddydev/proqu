"""Parsing and formatting of sizes and durations.

Every value needs an explicit unit. A bare number is refused, because guessing
between bytes and gigabytes, or seconds and minutes, misallocates silently.
"""

import math
import re

_SIZE_UNITS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
# at most 12 digits, so no value can overflow on its way to an integer
_NUMBER = r"\d{1,12}(?:\.\d{1,12})?"
_SIZE_RE = re.compile(rf"^({_NUMBER})\s*(?:([KMGT])(?:I?B)?|B)$", re.IGNORECASE | re.ASCII)

_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_DURATION_PART = re.compile(rf"({_NUMBER})([smhd])", re.ASCII)
# whole components, optionally separated by spaces, as in "1h 30m"
_DURATION_RE = re.compile(rf"^\s*(?:{_NUMBER}[smhd]\s*)+$", re.ASCII)


def parse_size(text: str) -> int:
    """Parse a size such as 512M, 6G, 1.5T or 500B into bytes."""
    match = _SIZE_RE.match(text.strip())
    if not match:
        raise ValueError(
            f"cannot read size {text!r}: give a number and a unit, for example 512M or 6G"
        )
    number, unit = match.groups()
    size = int(float(number) * _SIZE_UNITS[(unit or "").upper()])
    if size <= 0:
        raise ValueError(f"size {text!r} must be more than zero")
    return size


def parse_duration(text: str) -> int:
    """Parse a duration such as 90s, 20m, 2h or 1h30m into whole seconds."""
    if not _DURATION_RE.match(text):
        lower = text.lower()
        if lower != text and _DURATION_RE.match(lower):
            raise ValueError(
                f"cannot read duration {text!r}: units are lowercase, try {lower.strip()}"
            )
        raise ValueError(
            f"cannot read duration {text!r}: give numbers with units, for example 90s, 20m or 1h30m"
        )
    parts = _DURATION_PART.findall(text)
    # half seconds round up, so 0.5s is one second rather than nothing
    seconds = math.floor(sum(float(n) * _DURATION_UNITS[u] for n, u in parts) + 0.5)
    if seconds <= 0:
        raise ValueError(f"duration {text!r} must be more than zero")
    return seconds


def format_size(size: int) -> str:
    """Format bytes with the largest unit that keeps the number at least 1.

    Rounds down, so a shown limit is always accepted back as a request.
    """
    for unit in ("T", "G", "M", "K"):
        if size >= _SIZE_UNITS[unit]:
            tenths = size * 10 // _SIZE_UNITS[unit]
            return f"{tenths // 10}.{tenths % 10}{unit}"
    return f"{size}B"


def format_duration(seconds: float) -> str:
    """Format seconds as a compact duration such as 1h30m, 1m30s or 45s."""
    seconds = max(0, round(seconds))
    if seconds == 0:
        return "0s"
    out = ""
    for unit, length in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        count, seconds = divmod(seconds, length)
        if count:
            out += f"{count}{unit}"
    return out
