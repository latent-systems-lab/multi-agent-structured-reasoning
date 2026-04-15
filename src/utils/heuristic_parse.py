import re
from typing import Dict

_WEIGHT_RE = re.compile(r"([A-Za-z]{1,5})\s*[:=]\s*([-+]?\d*\.?\d+)\s*(%?)")

def parse_weights_from_text(text: str) -> Dict[str, float]:
    """Extract weight assignments from free-form text.

    Recognizes simple patterns like ``SYM: value`` or ``SYM: 10%``.
    Percentages are converted to decimals.
    """
    if not isinstance(text, str):
        return {}

    weights: Dict[str, float] = {}
    for sym, val, pct in _WEIGHT_RE.findall(text):
        try:
            num = float(val)
            if pct:
                num /= 100.0
            weights[sym] = num
        except ValueError:
            continue
    return weights


_CONF_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*%\s*confidence", re.IGNORECASE),
    re.compile(r"confidence[^\d]*(\d+(?:\.\d+)?)\s*%", re.IGNORECASE),
    re.compile(r"confidence[^\d]*(\d+(?:\.\d+)?)", re.IGNORECASE),
]

def parse_confidence_from_text(text: str) -> float | None:
    """Extract a confidence score from text.

    Supports formats like ``75% confidence`` or ``confidence: 0.75``.
    Returns a value between 0 and 1 if found, else ``None``.
    """
    if not isinstance(text, str):
        return None

    for pattern in _CONF_PATTERNS:
        m = pattern.search(text)
        if m:
            val = float(m.group(1))
            if "%" in m.group(0) or val > 1:
                val /= 100.0
            return val
    return None
