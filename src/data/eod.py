"""End-of-day market data accessors."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING, cast

import requests  # type: ignore[import-untyped]

from utils.timegate import timegate
from .warehouse import write_artifact

if TYPE_CHECKING:  # pragma: no cover
    from core.schemas import ISODate, Symbol


CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
THROTTLE_PER_SEC = 4
_LAST_CALL = 0.0


def set_throttle(per_sec: int) -> None:
    """Set the maximum number of API calls per second."""

    global THROTTLE_PER_SEC
    THROTTLE_PER_SEC = per_sec


def _throttle() -> None:
    """Simple rate limiter respecting ``THROTTLE_PER_SEC``."""

    global _LAST_CALL
    elapsed = time.time() - _LAST_CALL
    wait = max(0.0, 1.0 / THROTTLE_PER_SEC - elapsed)
    if wait:
        time.sleep(wait)
    _LAST_CALL = time.time()


def _load_json(path: Path) -> Any | None:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


@timegate(as_of_field="end_date")  # type: ignore[misc]
def get_prices_window(
    symbols: List[Symbol], end_date: ISODate, lookback_days: int
) -> Dict[Symbol, List[Tuple[ISODate, float]]]:
    """Return a rolling window of closing prices for each symbol.

    Data is first loaded from the local cache. If missing, the EOD Historical
    Data API is queried and the result cached to ``data_cache``.
    """

    api_token = os.getenv("EOD_API_KEY")
    start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(
        days=lookback_days - 1
    )
    start_date = start_dt.date().isoformat()
    out: Dict[Symbol, List[Tuple[ISODate, float]]] = {}

    for sym in symbols:
        cache_file = CACHE_DIR / f"prices_{sym}_{start_date}_{end_date}.json"
        cached = _load_json(cache_file)
        if cached is not None:
            cached_list = cast(List[List[Any]], cached)
            out[sym] = [(d, float(c)) for d, c in cached_list]
            continue
        if api_token is None:
            raise EnvironmentError("EOD_API_KEY not set and cache miss for prices")
        _throttle()
        url = (
            f"https://eodhd.com/api/eod/{sym}?from={start_date}&to={end_date}"
            f"&api_token={api_token}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("historical", [])
        series: List[Tuple[ISODate, float]] = []
        for row in data:
            date = row.get("date")
            if date and date <= end_date:
                series.append((date, float(row.get("close", 0.0))))
        series.sort(key=lambda x: x[0])
        out[sym] = series
        write_artifact(str(cache_file), [(d, p) for d, p in series])
    return out


@timegate()  # type: ignore[misc]
def get_fundamentals(
    symbols: List[Symbol], as_of: ISODate
) -> Dict[Symbol, Dict[str, float]]:
    """Return fundamental data as of ``as_of``.

    Numeric fields returned by the EOD Historical Data ``fundamentals`` endpoint
    are retained and cached locally.
    """

    api_token = os.getenv("EOD_API_KEY")
    out: Dict[Symbol, Dict[str, float]] = {}
    for sym in symbols:
        cache_file = CACHE_DIR / f"fundamentals_{sym}_{as_of}.json"
        cached = _load_json(cache_file)
        if cached is not None:
            cached_dict = cast(Dict[str, Any], cached)
            out[sym] = {k: float(v) for k, v in cached_dict.items()}
            continue
        if api_token is None:
            raise EnvironmentError(
                "EOD_API_KEY not set and cache miss for fundamentals"
            )
        _throttle()
        url = f"https://eodhd.com/api/fundamentals/{sym}?api_token={api_token}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        records = resp.json()
        selected: Dict[str, float] = {}
        for row in records:
            date = row.get("date")
            if date and date <= as_of:
                for k, v in row.items():
                    if isinstance(v, (int, float)):
                        selected[k] = float(v)
                break
        out[sym] = selected
        write_artifact(str(cache_file), selected)
    return out


@timegate()  # type: ignore[misc]
def get_headlines(
    symbols: List[Symbol], as_of: ISODate, limit_per_asset: int = 5
) -> Dict[Symbol, List[str]]:
    """Return news headlines up to ``as_of``.

    Headlines are sourced from the EOD Historical Data ``news`` endpoint and
    cached to ``data_cache``.
    """

    api_token = os.getenv("EOD_API_KEY")
    out: Dict[Symbol, List[str]] = {}
    for sym in symbols:
        cache_file = CACHE_DIR / f"headlines_{sym}_{as_of}_{limit_per_asset}.json"
        cached = _load_json(cache_file)
        if cached is not None:
            out[sym] = list(cast(List[str], cached))
            continue
        if api_token is None:
            raise EnvironmentError("EOD_API_KEY not set and cache miss for headlines")
        _throttle()
        url = (
            "https://eodhd.com/api/news"
            f"?symbols={sym}&limit={limit_per_asset}&to={as_of}&api_token={api_token}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        items = resp.json()
        headlines: List[str] = []
        for item in items:
            date = (item.get("date") or item.get("publishedDate", ""))[:10]
            if date <= as_of:
                title = item.get("title") or item.get("headline")
                if title:
                    headlines.append(str(title))
        out[sym] = headlines[:limit_per_asset]
        write_artifact(str(cache_file), headlines[:limit_per_asset])
    return out
