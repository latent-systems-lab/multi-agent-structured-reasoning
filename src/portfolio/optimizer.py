"""Portfolio optimisation routines."""

from __future__ import annotations

from typing import Dict

import cvxpy as cp
import numpy as np

from core.schemas import ChairCandidate


def project_to_feasible(
    candidate: ChairCandidate, bounds: Dict[str, float]
) -> Dict[str, float]:
    """Project candidate weights into the feasible set."""
    symbols = list(candidate.weights.keys())
    w0 = np.array([candidate.weights[s] for s in symbols])

    w = cp.Variable(len(symbols))

    if "include_cash" in bounds and "CASH" in symbols:
        idx_cash = symbols.index("CASH")
        noncash_idx = [i for i, s in enumerate(symbols) if i != idx_cash]
        constraints = [
            cp.sum(w) == 1.0,
            w[idx_cash] >= 0.0,
            w[idx_cash] <= 1.0,
        ]
        if noncash_idx:
            constraints += [
                w[noncash_idx] >= bounds["min_weight"],
                w[noncash_idx] <= bounds["max_weight"],
            ]
    else:
        constraints = [
            cp.sum(w) <= bounds["net_cap"],
            cp.sum(w) >= -bounds["net_cap"],
            cp.norm1(w) <= bounds["gross_cap"],
            w <= bounds["max_weight"],
            w >= bounds["min_weight"],
        ]

    objective = cp.Minimize(cp.sum_squares(w - w0))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    if w.value is None:
        raise RuntimeError("Projection optimisation failed")

    projected = {s: float(val) for s, val in zip(symbols, w.value)}
    return projected
