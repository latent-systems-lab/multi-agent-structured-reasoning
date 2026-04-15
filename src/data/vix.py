"""VIX index loader from local CSV.

Reads a VIX history CSV with columns: DATE,OPEN,HIGH,LOW,CLOSE where DATE is
in MM/DD/YYYY format. Returns the latest CLOSE on or before the requested date.

CSV path can be overridden via the VIX_CSV_PATH environment variable. By
default the loader looks for "VIX_HISTORY.csv" in the current working
directory.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.logging import get_logger

logger = get_logger(__name__)


def _vix_csv_path() -> Path:
    path = os.getenv("VIX_CSV_PATH", "data_cache/VIX_History.csv")
    return Path(path)


def get_vix_close(as_of: str) -> Optional[float]:
    """Return VIX CLOSE for the latest date <= ``as_of``.

    Parameters
    ----------
    as_of:
        ISO date string YYYY-MM-DD.

    Returns
    -------
    Optional[float]
        CLOSE price if a record is found, otherwise None.
    """

    csv_path = _vix_csv_path()
    if not csv_path.exists():  # pragma: no cover - depends on environment
        logger.warning("VIX CSV not found", path=str(csv_path))
        return None

    try:
        cutoff = datetime.strptime(as_of, "%Y-%m-%d").date()
    except Exception:  # pragma: no cover - defensive
        return None

    latest_date = None
    latest_close: Optional[float] = None

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_str = row.get("DATE") or row.get("Date") or ""
            try:
                dt = datetime.strptime(date_str, "%m/%d/%Y").date()
            except Exception:
                continue
            if dt <= cutoff and (latest_date is None or dt > latest_date):
                try:
                    latest_close = float(row.get("CLOSE") or row.get("Close") or row.get("close") or 0.0)
                    latest_date = dt
                except (TypeError, ValueError):
                    continue

    return latest_close

