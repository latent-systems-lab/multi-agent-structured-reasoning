"""Comparison reporting for multiple protocols."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .metrics import (
    bootstrap_test,
    cvar_95,
    hit_rate,
    max_drawdown,
    randomization_test,
    sharpe,
)
from .reporter import write_metrics_csv, write_metrics_json


def comparison_report(
    returns: Dict[str, List[float]], baseline: str, outdir: str
) -> None:
    """Generate comparison metrics and statistical tests.

    Parameters
    ----------
    returns:
        Mapping of protocol id to list of daily returns.
    baseline:
        Key of the baseline strategy within ``returns``.
    outdir:
        Directory where reports will be written.
    """

    Path(outdir).mkdir(parents=True, exist_ok=True)
    base = returns[baseline]
    results: Dict[str, Dict[str, float]] = {}
    for name, series in returns.items():
        stats = {
            "sharpe": sharpe(series),
            "max_drawdown": max_drawdown(series),
            "cvar_95": cvar_95(series),
            "hit_rate": hit_rate(series),
        }
        if name != baseline:
            stats["bootstrap_p"] = bootstrap_test(series, base)
            stats["randomization_p"] = randomization_test(series, base)
        results[name] = stats
    write_metrics_csv(str(Path(outdir) / "comparison.csv"), results)
    write_metrics_json(str(Path(outdir) / "comparison.json"), results)
