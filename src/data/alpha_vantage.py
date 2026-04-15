"""Alpha Vantage market data accessors."""

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
from utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from core.schemas import ISODate, Symbol
import yfinance as yf


CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
THROTTLE_PER_SEC = 4
_LAST_CALL = 0.0

logger = get_logger(__name__)


def set_throttle(per_sec: int) -> None:
    """Set the maximum number of API calls per second."""

    global THROTTLE_PER_SEC
    THROTTLE_PER_SEC = per_sec


def _is_stock(symbol: str) -> bool:
    """Heuristic: treat plain tickers as stocks; skip API for others.

    Guidance: stocks don't include characters like '=' or '^' in this project.
    We use this to avoid calling Alpha Vantage fundamentals/news for non-stocks
    such as futures (e.g., 'CL=F') or indices (e.g., '^GSPC').
    """
    return ("=" not in symbol) and ("^" not in symbol)


def _throttle() -> None:
    """Simple rate limiter respecting ``THROTTLE_PER_SEC``."""
    global _LAST_CALL
    if THROTTLE_PER_SEC <= 0:
        return
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


def _write_and_return(path: Path, payload: Dict[str, float]) -> Dict[str, float]:
    """Persist ``payload`` to ``path`` and return it.

    This small helper mirrors the existing caching pattern in this module and
    keeps the call sites concise.
    """

    write_artifact(str(path), payload)
    return payload

@timegate(as_of_field="end_date")  # type: ignore[misc]
def get_prices_window(
    symbols: List[Symbol], end_date: ISODate, lookback_days: int
) -> Dict[Symbol, List[Tuple[ISODate, float]]]:
    """Return a rolling window of closing prices for each symbol.

    * Equity and ETF tickers are sourced from Yahoo Finance via :mod:`yfinance`.
    * FX pairs, commodities and Treasury futures use Alpha Vantage endpoints as
      described in the manuscript (``FX_DAILY`` or ``TIME_SERIES_DAILY``).
    """

    start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=lookback_days - 1)
    start_date = start_dt.date().isoformat()
    yf_end = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).date().isoformat()
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")

    out: Dict[Symbol, List[Tuple[ISODate, float]]] = {}
    for sym in symbols:
        cache_file = CACHE_DIR / f"prices_{sym}_{start_date}_{end_date}.json"
        cached = _load_json(cache_file)
        if cached is not None:
            out[sym] = [(d, float(c)) for d, c in cast(List[List[Any]], cached)]
            if out[sym]:
                continue

        if _is_stock(sym):
            try:
                hist = yf.Ticker(sym).history(start=start_date, end=yf_end)
            except Exception as exc:  # pragma: no cover - network failures
                logger.error("Failed to fetch price history", symbol=sym, error=str(exc))
                out[sym] = []
                continue
            series: List[Tuple[ISODate, float]] = [
                (dt.strftime("%Y-%m-%d"), float(row["Close"]))
                for dt, row in hist.iterrows()
            ]
        else:
            if api_key is None:
                raise EnvironmentError(
                    "ALPHAVANTAGE_API_KEY required for non-equity instruments"
                )
            _throttle()
            if sym.endswith("=X") and len(sym) >= 6:
                from_curr, to_curr = sym[:3], sym[3:6]
                url = (
                    "https://www.alphavantage.co/query"
                    f"?function=FX_DAILY&from_symbol={from_curr}&to_symbol={to_curr}&apikey={api_key}"
                )
                key = "Time Series FX (Daily)"
                close_field = "4. close"
            else:
                url = (
                    "https://www.alphavantage.co/query"
                    f"?function=TIME_SERIES_DAILY_ADJUSTED&symbol={sym}&apikey={api_key}"
                )
                key = "Time Series (Daily)"
                close_field = "4. close"
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                payload = resp.json().get(key, {})
            except Exception as exc:  # pragma: no cover - network failures
                logger.error("Failed to fetch Alpha Vantage history", symbol=sym, error=str(exc))
                out[sym] = []
                continue
            series = []
            for date, row in payload.items():
                if start_date <= date <= end_date:
                    try:
                        price = float(row[close_field])
                    except (TypeError, ValueError, KeyError):
                        continue
                    series.append((date, price))
            series.sort(key=lambda x: x[0])

        out[sym] = series
        write_artifact(str(cache_file), [(d, p) for d, p in series])

    return out


@timegate()  # type: ignore[misc]
def get_fundamentals(
    symbols: List[Symbol], as_of: ISODate
) -> Dict[Symbol, Dict[str, float]]:
    """Return fundamental data as of ``as_of``.

    Numeric fields returned by the Alpha Vantage ``OVERVIEW`` endpoint are
    retained and cached locally.
    """

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    out: Dict[Symbol, Dict[str, float]] = {}
    for sym in symbols:
        # Skip non-stocks (per heuristic) to avoid API errors for FX/commodities/indices
        if not _is_stock(sym):
            out[sym] = {}
            cache_file = CACHE_DIR / f"fundamentals_{sym}_{as_of}.json"
            write_artifact(str(cache_file), {})
            continue
        cache_file = CACHE_DIR / f"fundamentals_{sym}_{as_of}.json"
        cached = _load_json(cache_file)
        if cached is not None:
            cached_dict = cast(Dict[str, Any], cached)
            # If prior cache is empty for a stock, treat as miss and refetch
            if len(cached_dict) > 0:
                out[sym] = {k: float(v) for k, v in cached_dict.items()}
                continue
        if api_key is None:
            raise EnvironmentError(
                "ALPHAVANTAGE_API_KEY not set and cache miss for fundamentals"
            )
        _throttle()
        url = (
            "https://www.alphavantage.co/query"
            f"?function=OVERVIEW&symbol={sym}&apikey={api_key}"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            records = resp.json()
        except Exception as exc:  # pragma: no cover - network failures
            logger.error(
                "Failed to fetch fundamentals", symbol=sym, error=str(exc)
            )
            out[sym] = {}
            continue
        selected: Dict[str, float] = {}
        for k, v in records.items():
            try:
                selected[k] = float(v)
            except (TypeError, ValueError):
                continue
        out[sym] = selected
        write_artifact(str(cache_file), selected)
    return out


@timegate()  # type: ignore[misc]
def get_insights(
    symbols: List[Symbol], as_of: ISODate
) -> Dict[Symbol, Dict[str, float]]:
    """Return contextual market data for non-equity symbols.

    * For stock tickers (heuristically determined), this simply proxies to
      :func:`get_fundamentals`.
    * For FX pairs formatted like ``EURUSD=X`` the latest exchange rate is
      returned using the ``CURRENCY_EXCHANGE_RATE`` endpoint.
    * For select commodity and bond futures a representative economic
      indicator is queried (e.g., ``CL=F`` maps to the West Texas Intermediate
      crude oil price via ``function=WTI`` and ``ZN=F`` maps to the
      ``TREASURY_YIELD`` endpoint for the 10 year maturity).

    Results are cached under ``data_cache`` similarly to fundamentals.
    """

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    out: Dict[Symbol, Dict[str, float]] = {}
    for sym in symbols:
        # Delegate to stock fundamentals when applicable
        if _is_stock(sym):
            out[sym] = get_fundamentals([sym], as_of)[sym]
            continue

        cache_file = CACHE_DIR / f"insights_{sym}_{as_of}.json"
        cached = _load_json(cache_file)
        if cached is not None:
            cached_dict = cast(Dict[str, Any], cached)
            out[sym] = {k: float(v) for k, v in cached_dict.items()}
            continue

        if api_key is None:
            raise EnvironmentError(
                "ALPHAVANTAGE_API_KEY not set and cache miss for insights"
            )

        data: Dict[str, float] = {}
        if sym.endswith("=X") and len(sym) >= 6:
            # Currency pair, e.g., EURUSD=X
            from_curr, to_curr = sym[:3], sym[3:6]
            _throttle()
            url = (
                "https://www.alphavantage.co/query"
                f"?function=CURRENCY_EXCHANGE_RATE&from_currency={from_curr}"
                f"&to_currency={to_curr}&apikey={api_key}"
            )
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                records = resp.json().get("Realtime Currency Exchange Rate", {})
                rate = float(records.get("5. Exchange Rate", 0.0))
                data = {"exchange_rate": rate}
            except Exception as exc:  # pragma: no cover - network failures
                logger.error("Failed to fetch FX rate", symbol=sym, error=str(exc))
        elif sym == "CL=F":
            # Crude oil future -> WTI price
            _throttle()
            url = (
                "https://www.alphavantage.co/query?function=WTI"
                f"&interval=monthly&apikey={api_key}"
            )
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                value = float(resp.json()["data"][0]["value"])
                data = {"price": value}
            except Exception as exc:  # pragma: no cover - network failures
                logger.error("Failed to fetch WTI price", symbol=sym, error=str(exc))
        elif sym == "ZN=F":
            # 10Y Treasury note future -> treasury yield
            _throttle()
            url = (
                "https://www.alphavantage.co/query?function=TREASURY_YIELD"
                f"&interval=monthly&maturity=10year&apikey={api_key}"
            )
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                value = float(resp.json()["data"][0]["value"])
                data = {"yield": value}
            except Exception as exc:  # pragma: no cover - network failures
                logger.error(
                    "Failed to fetch treasury yield", symbol=sym, error=str(exc)
                )
        # Other assets remain with empty dict
        out[sym] = _write_and_return(cache_file, data)

    return out


@timegate()  # type: ignore[misc]
def get_headlines(
    symbols: List[Symbol], as_of: ISODate, limit_per_asset: int = 5
) -> Dict[Symbol, List[str]]:
    """Return recent news headlines for the supplied symbols."""

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if api_key is None:
        raise EnvironmentError("ALPHAVANTAGE_API_KEY required for headlines")

    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")
    start = (as_of_dt - timedelta(days=7)).strftime("%Y%m%dT0000")
    out: Dict[Symbol, List[str]] = {}

    for sym in symbols:
        cache_file = CACHE_DIR / f"headlines_{sym}_{as_of}_{limit_per_asset}.json"
        cached = _load_json(cache_file)
        if cached is not None:
            out[sym] = list(cast(List[str], cached))
            continue

        _throttle()
        url = (
            "https://www.alphavantage.co/query"
            f"?function=NEWS_SENTIMENT&tickers={sym}&time_from={start}&sort=LATEST&apikey={api_key}"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            feed = resp.json().get("feed", [])
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to fetch headlines", symbol=sym, error=str(exc))
            out[sym] = []
            continue

        headlines: List[str] = []
        for item in feed:
            published = item.get("time_published", "")[:8]
            if published and published <= as_of_dt.strftime("%Y%m%d"):
                title = item.get("title")
                if title:
                    headlines.append(str(title))
            if len(headlines) >= limit_per_asset:
                break

        out[sym] = headlines
        write_artifact(str(cache_file), headlines)

    return out
