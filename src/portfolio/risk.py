"""Risk-related calculations."""

from __future__ import annotations

from typing import Dict

import numpy as np


def cvar_95(returns: list[float]) -> float:
    """Compute the 95% Conditional Value-at-Risk."""

    if not returns:
        raise ValueError("returns cannot be empty")

    arr = np.sort(np.array(returns))
    var_level = 0.05
    cutoff = np.quantile(arr, var_level)
    tail = arr[arr <= cutoff]
    return float(tail.mean())


def stress_pnl(weights: Dict[str, float], scenarios: Dict[str, float]) -> float:
    """Return stressed PnL under given scenarios."""

    pnl = 0.0
    for symbol, w in weights.items():
        if symbol in scenarios:
            pnl += w * scenarios[symbol]
    return pnl
