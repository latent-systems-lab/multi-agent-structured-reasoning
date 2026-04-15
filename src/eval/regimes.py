"""Regime-based walk-forward evaluation utilities."""

from __future__ import annotations

from typing import Dict, List, Sequence

from .metrics import cvar_95, hit_rate, max_drawdown, sharpe


def partition_returns(
    dates: Sequence[str], returns: Sequence[float], regimes: Sequence[Dict[str, str]]
) -> Dict[str, List[float]]:
    """Partition ``returns`` into regimes defined by date ranges."""
    out: Dict[str, List[float]] = {}
    for idx, reg in enumerate(regimes):
        name = reg.get("name", f"regime_{idx}")
        start = reg["start"]
        end = reg["end"]
        out[name] = [r for d, r in zip(dates, returns) if start <= d <= end]
    return out


def walk_forward_regimes(
    dates: Sequence[str], returns: Sequence[float], regimes: Sequence[Dict[str, str]]
) -> Dict[str, Dict[str, float]]:
    """Compute performance metrics for each regime.

    Parameters
    ----------
    dates:
        Sequence of dates corresponding to ``returns``.
    returns:
        Sequence of periodic returns.
    regimes:
        List of mappings with ``start``/``end`` keys defining regime windows and
        an optional ``name``.
    """
    partitions = partition_returns(dates, returns, regimes)
    results: Dict[str, Dict[str, float]] = {}
    for name, series in partitions.items():
        results[name] = {
            "sharpe": sharpe(series),
            "max_drawdown": max_drawdown(series),
            "cvar_95": cvar_95(series),
            "hit_rate": hit_rate(series),
        }
    return results

