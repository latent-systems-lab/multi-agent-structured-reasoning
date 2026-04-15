"""Simple data warehouse helpers."""

from __future__ import annotations

import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, cast, Dict


def load_calendar(start: str, end: str) -> list[str]:
    """Return a list of trading dates between ``start`` and ``end``.

    The calendar is generated as simple business days (Mon–Fri) and cached to
    ``data_cache`` to avoid recomputation.
    """

    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"calendar_{start}_{end}.json"
    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as f:
            return cast(List[str], json.load(f))

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    days: list[str] = []
    cur = start_dt
    while cur <= end_dt:
        if cur.weekday() < 5:  # Monday=0 .. Friday=4
            days.append(cur.date().isoformat())
        cur += timedelta(days=1)

    write_artifact(str(cache_file), days)
    return days


def write_artifact(path: str, obj: Any) -> None:
    """Persist an object to disk (e.g., JSON or CSV)."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()

    if suffix == ".json":
        with p.open("w", encoding="utf-8") as f:
            json.dump(obj, f)
    elif suffix == ".csv":
        with p.open("w", newline="", encoding="utf-8") as f:
            if isinstance(obj, list):
                if obj and isinstance(obj[0], dict):
                    fieldnames = list(cast(List[dict[str, Any]], obj)[0].keys())
                    dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
                    dict_writer.writeheader()
                    dict_writer.writerows(cast(List[dict[str, Any]], obj))
                else:
                    csv.writer(f).writerows(cast(List[List[Any]], obj))
            elif isinstance(obj, dict):
                dict_writer = csv.DictWriter(f, fieldnames=["key", "value"])
                dict_writer.writeheader()
                for k, v in obj.items():
                    dict_writer.writerow({"key": k, "value": v})
            else:
                f.write(str(obj))
    else:
        raise ValueError(f"Unsupported artifact format: {suffix}")


def append_positions_rows(path: str, rows: List[Dict[str, float | str]]) -> None:
    """Append position records to a CSV, creating header on first write."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # Stable ordering of columns
    fieldnames = ["date", "symbol", "price", "weight", "notional", "units"]
    need_header = not p.exists() or p.stat().st_size == 0
    with p.open("a", newline="", encoding="utf-8") as f:
        dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            dict_writer.writeheader()
        for r in rows:
            dict_writer.writerow({k: r.get(k, "") for k in fieldnames})
