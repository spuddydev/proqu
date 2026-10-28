"""Parsing and formatting of sizes and durations.

Every value needs an explicit unit. A bare number is refused, because guessing
between bytes and gigabytes, or seconds and minutes, misallocates silently.
"""

import re

_SIZE_UNITS = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
_SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([KMGT])(?:I?B)?$", re.IGNORECASE | re.ASCII)

_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)([smhd])", re.ASCII)


def parse_size(text: str) -> int:
    """Parse a size such as 512M, 6G or 1.5T into bytes."""
    match = _SIZE_RE.match(text.strip())
    if not match:
        raise ValueError(
            f"cannot read size {text!r}: give a number and a unit, for example 512M or 6G"
        )
    number, unit = match.groups()
    size = int(float(number) * _SIZE_UNITS[unit.upper()])
    if size <= 0:
        raise ValueError(f"size {text!r} must be more than zero")
    return size


def parse_duration(text: str) -> int:
    """Parse a duration such as 90s, 20m, 2h or 1h30m into whole seconds."""
    compact = text.strip().replace(" ", "")
    parts = _DURATION_PART.findall(compact)
    if not parts or "".join(n + u for n, u in parts) != compact:
        lower = compact.lower()
        if lower != compact and "".join(n + u for n, u in _DURATION_PART.findall(lower)) == lower:
            raise ValueError(f"cannot read duration {text!r}: units are lowercase, try {lower}")
        raise ValueError(
            f"cannot read duration {text!r}: give numbers with units, for example 90s, 20m or 1h30m"
        )
    seconds = round(sum(float(n) * _DURATION_UNITS[u] for n, u in parts))
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
    """Format seconds as a compact duration such as 1h30m or 45s."""
    seconds = max(0, round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    out = ""
    for unit, length in (("d", 86400), ("h", 3600), ("m", 60)):
        count, seconds = divmod(seconds, length)
        if count:
            out += f"{count}{unit}"
    return out or "0m"
