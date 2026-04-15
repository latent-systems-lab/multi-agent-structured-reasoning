"""Reporting utilities for backtest results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:  # pragma: no cover
    from core.schemas import ChairDecision


def write_decision_csv(path: str, decisions: list[ChairDecision]) -> None:
    """Write a list of :class:`ChairDecision` objects to CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not decisions:
        # Write an empty file with just a header for consistency.
        with out.open("w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["date"])
        return

    # Collect union of all symbols to create stable, analysis-friendly columns.
    symbols = sorted({sym for d in decisions for sym in d.weights})
    fieldnames = [
        "date",
        "utility",
        "synthesis",
        "protocol_id",
        "rounds_taken",
        "sc_M",
        "token_in",
        "token_out",
        "latency_ms",
        "supporting",
        "dissenting",
        "data_refs",
    ] + symbols

    with out.open("w", newline="") as f:
        dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
        dict_writer.writeheader()
        for d in decisions:
            row: Dict[str, object] = {
                "date": d.date,
                "utility": d.utility,
                "synthesis": d.synthesis,
                "protocol_id": d.protocol_id,
                "rounds_taken": d.rounds_taken,
                "sc_M": d.sc_M,
                "token_in": d.token_in,
                "token_out": d.token_out,
                "latency_ms": d.latency_ms,
                "supporting": json.dumps({k: p.model_dump() for k, p in d.supporting.items()}),
                "dissenting": json.dumps(d.dissenting),
                "data_refs": json.dumps([ref.hash for ref in d.data_refs]),
            }
            for sym in symbols:
                row[sym] = d.weights.get(sym, 0.0)
            dict_writer.writerow(row)


def append_decision_row(path: str, decision: "ChairDecision", symbols: List[str]) -> None:
    """Append a single decision row to CSV using fixed symbol columns.

    The header is created if the file does not exist. ``symbols`` should be a
    stable list (e.g., the configured universe and optionally ``CASH``) so the
    CSV schema remains consistent across days.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date",
        "utility",
        "synthesis",
        "protocol_id",
        "rounds_taken",
        "sc_M",
        "token_in",
        "token_out",
        "latency_ms",
        "supporting",
        "dissenting",
        "data_refs",
    ] + symbols

    need_header = not out.exists() or out.stat().st_size == 0
    with out.open("a", newline="") as f:
        dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            dict_writer.writeheader()
        row: Dict[str, object] = {
            "date": decision.date,
            "utility": decision.utility,
            "synthesis": decision.synthesis,
            "protocol_id": decision.protocol_id,
            "rounds_taken": decision.rounds_taken,
            "sc_M": decision.sc_M,
            "token_in": decision.token_in,
            "token_out": decision.token_out,
            "latency_ms": decision.latency_ms,
            "supporting": json.dumps({k: p.model_dump() for k, p in decision.supporting.items()}),
            "dissenting": json.dumps(decision.dissenting),
            "data_refs": json.dumps([ref.hash for ref in decision.data_refs]),
        }
        for sym in symbols:
            row[sym] = decision.weights.get(sym, 0.0)
        dict_writer.writerow(row)


def write_decision_json(path: str, decisions: list[ChairDecision]) -> None:
    """Write a list of :class:`ChairDecision` objects to JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for d in decisions:
        rec = d.model_dump()
        rec["data_refs"] = [r["hash"] for r in rec.get("data_refs", [])]
        data.append(rec)
    with out.open("w") as f:
        json.dump(data, f)


def write_ops_json(path: str, tokens_latency: list[Dict[str, int | str]]) -> None:
    """Write token and latency metrics to JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        json.dump(tokens_latency, f)


def write_token_usage_json(path: str, token_usage: list[Dict[str, object]]) -> None:
    """Write daily token usage metadata to JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        json.dump(token_usage, f)


def write_returns_csv(path: str, returns: list[Dict[str, float]]) -> None:
    """Write daily returns to CSV."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", "return", "cumulative"]
    with out.open("w", newline="") as f:
        dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
        dict_writer.writeheader()
        for r in returns:
            dict_writer.writerow(r)


def append_returns_row(path: str, date: str, r: float, cumulative: float) -> None:
    """Append a single returns row to CSV (header on first write)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", "return", "cumulative"]
    need_header = not out.exists() or out.stat().st_size == 0
    with out.open("a", newline="") as f:
        dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            dict_writer.writeheader()
        dict_writer.writerow({"date": date, "return": r, "cumulative": cumulative})


def write_returns_json(path: str, returns: list[Dict[str, float]]) -> None:
    """Write daily returns to JSON."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(returns, f)


def write_metrics_csv(path: str, metrics: Dict[str, Dict[str, float]]) -> None:
    """Write performance metrics to CSV."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not metrics:
        with out.open("w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["protocol"])
        return

    fieldnames = ["protocol"] + sorted({k for m in metrics.values() for k in m})
    with out.open("w", newline="") as f:
        dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
        dict_writer.writeheader()
        for name, m in metrics.items():
            row: Dict[str, float | str] = {"protocol": name}
            row.update(m)
            dict_writer.writerow(row)


def write_metrics_json(path: str, metrics: Dict[str, Dict[str, float]]) -> None:
    """Write performance metrics to JSON."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(metrics, f)
