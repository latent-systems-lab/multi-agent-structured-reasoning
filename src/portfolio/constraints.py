"""Portfolio constraint helpers."""

from __future__ import annotations

from typing import Dict
import os


def feasible_bounds(universe: list[str], optimize_cash: bool | None = None) -> Dict[str, float]:
    """Return bounds and caps for the given trading universe.

    When ``optimize_cash`` is enabled, the optimiser treats a synthetic CASH
    variable as part of the portfolio. Non-cash assets may take negative weights
    (shorts) within the per-name bounds while CASH remains long-only:
    - Non-cash assets are constrained to [-max_weight, max_weight]
    - CASH is constrained to [0, 1]
    - Sum of all weights equals 1
    """
    if optimize_cash is None:
        optimize_cash = os.getenv("OPTIMIZE_CASH", "0") not in {"0", "false", "False"}
    if not universe:
        raise ValueError("universe must contain at least one symbol")

    n = len(universe)
    equal_weight = 1.0 / n

    max_weight = min(0.30, 4 * equal_weight)
    min_weight = -max_weight

    if optimize_cash:
        return {
            "include_cash": 1.0,
            "min_weight": -max_weight,
            "max_weight": max_weight,
            "gross_cap": 1.2,
            "net_cap": 0.5,
        }
    else:
        gross_cap = 1.2  # sum(|w|)
        net_cap = 0.5  # |sum(w)|

        return {
            "min_weight": min_weight,
            "max_weight": max_weight,
            "gross_cap": gross_cap,
            "net_cap": net_cap,
        }
