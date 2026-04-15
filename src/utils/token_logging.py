"""Token usage logging helpers.

This module provides utility functions to record token usage for individual
agents and aggregate daily totals.  Functions here are intentionally lightweight
and avoid additional dependencies so that they can be reused across scripts.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from langchain_core.messages import BaseMessage


def _merge_counts(dest: MutableMapping[str, int], src: Mapping[str, int]) -> None:
    """Add counts from ``src`` into ``dest`` in-place."""

    for key, value in src.items():
        try:
            dest[key] = dest.get(key, 0) + int(value)
        except Exception:
            # Ignore non-numeric values
            continue


def _extract_usage(message: BaseMessage) -> Mapping[str, object] | None:
    """Best-effort extraction of usage metadata from a message."""

    meta = getattr(message, "usage_metadata", None)
    if not meta:
        meta = getattr(message, "response_metadata", None)
        if isinstance(meta, Mapping):
            meta = meta.get("token_usage") or meta.get("usage") or meta
    if not meta:
        meta = getattr(message, "token_usage", None)
    if isinstance(meta, Mapping):
        return meta
    return None


def log_agent_usage(agent: str, messages: Sequence[BaseMessage]) -> None:
    """Aggregate token usage across messages and append a CSV row.

    The output directory and date are obtained from the ``TOKEN_LOG_OUTDIR`` and
    ``TOKEN_LOG_DATE`` environment variables.  If either is unset, the function
    returns without writing anything.
    """

    outdir = os.getenv("TOKEN_LOG_OUTDIR")
    day = os.getenv("TOKEN_LOG_DATE")
    if not outdir or not day:
        return

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    input_details: dict[str, int] = {}
    output_details: dict[str, int] = {}

    for message in messages:
        usage = _extract_usage(message)
        if not usage:
            continue
        in_t = int(
            usage.get("prompt_token_count")
            or usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or 0
        )
        out_t = int(
            usage.get("candidates_token_count")
            or usage.get("output_tokens")
            or usage.get("completion_tokens")
            or 0
        )
        total = int(usage.get("total_tokens") or (in_t + out_t))

        input_tokens += in_t
        output_tokens += out_t
        total_tokens += total

        _merge_counts(input_details, usage.get("input_token_details", {}))
        _merge_counts(output_details, usage.get("output_token_details", {}))

    csv_row = {
        "date": day,
        "agent": agent,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_token_details": json.dumps(input_details) if input_details else "",
        "output_token_details": json.dumps(output_details) if output_details else "",
    }

    out_path = Path(outdir) / "agent_token_usage.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "agent",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "input_token_details",
                "output_token_details",
            ],
        )
        if need_header:
            writer.writeheader()
        writer.writerow(csv_row)


def append_daily_usage(outdir: str, day: str) -> None:
    """Aggregate per-agent usage for ``day`` and append to daily CSV."""

    src = Path(outdir) / "agent_token_usage.csv"
    if not src.exists():
        return

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    input_details: dict[str, int] = {}
    output_details: dict[str, int] = {}

    with src.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("date") != day:
                continue
            try:
                input_tokens += int(row.get("input_tokens", 0) or 0)
                output_tokens += int(row.get("output_tokens", 0) or 0)
                total_tokens += int(row.get("total_tokens", 0) or 0)
            except Exception:
                pass
            if row.get("input_token_details"):
                try:
                    details = json.loads(row["input_token_details"])
                    if isinstance(details, Mapping):
                        _merge_counts(input_details, details)
                except Exception:
                    pass
            if row.get("output_token_details"):
                try:
                    details = json.loads(row["output_token_details"])
                    if isinstance(details, Mapping):
                        _merge_counts(output_details, details)
                except Exception:
                    pass

    dest = Path(outdir) / "daily_token_usage.csv"
    need_header = not dest.exists() or dest.stat().st_size == 0
    with dest.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "agent",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "input_token_details",
                "output_token_details",
            ],
        )
        if need_header:
            writer.writeheader()
        writer.writerow(
            {
                "date": day,
                "agent": "all",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "input_token_details": json.dumps(input_details) if input_details else "",
                "output_token_details": json.dumps(output_details) if output_details else "",
            }
        )
