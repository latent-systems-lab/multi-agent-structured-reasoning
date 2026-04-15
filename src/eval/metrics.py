"""Performance metric utilities."""

from __future__ import annotations

import math  # Needed for square root calculations
import random
import statistics
from typing import Dict, List

from utils.logging import get_logger

logger = get_logger(__name__)


def sharpe(returns: List[float]) -> float:
    """Return the annualised Sharpe ratio.

    The function assumes that ``returns`` are periodic returns expressed as
    decimals (e.g. ``0.01`` for 1%).  If the standard deviation of returns is
    zero or ``returns`` is empty the ratio is defined to be ``0``.  A trading
    year of 252 periods is assumed for annualisation which keeps the output
    friendly for downstream analysis in pandas/Polars.
    """

    if not returns:
        return 0.0

    mean = statistics.mean(returns)
    # ``statistics.stdev`` requires at least two data points.  When only a
    # single observation is available the Sharpe ratio is undefined so we
    # return ``0``.
    std = statistics.stdev(returns) if len(returns) > 1 else 0.0
    if std == 0.0:
        return 0.0

    return mean / std * math.sqrt(252)


def max_drawdown(returns: List[float]) -> float:
    """Return the maximum drawdown.

    ``returns`` should be a sequence of periodic returns expressed as decimals.
    The drawdown is computed from the running cumulative return and the result
    is reported as a positive number representing the magnitude of the largest
    peak-to-trough decline.
    """

    if not returns:
        return 0.0

    cumulative = 1.0
    peak = 1.0
    max_dd = 0.0

    for r in returns:
        cumulative *= 1.0 + r
        peak = max(peak, cumulative)
        drawdown = cumulative / peak - 1.0
        if drawdown < max_dd:
            max_dd = drawdown

    return abs(max_dd)


def hit_rate(returns: List[float]) -> float:
    """Return the proportion of positive-return periods."""

    if not returns:
        return 0.0

    hits = sum(1 for r in returns if r > 0)
    return hits / len(returns)


def cvar_95(returns: List[float]) -> float:
    """Return the 95% Conditional Value-at-Risk (CVaR)."""

    if not returns:
        return 0.0

    sorted_returns = sorted(returns)
    cutoff = max(1, int(len(sorted_returns) * 0.05))
    tail = sorted_returns[:cutoff]
    return sum(tail) / len(tail)


def bootstrap_test(
    a: List[float], b: List[float], n_iter: int = 10000, seed: int | None = None, block: int = 5
) -> float:
    """Return moving block bootstrap p-value for difference in means."""

    if not a or not b:
        return 1.0

    rng = random.Random(seed)
    obs = statistics.mean(a) - statistics.mean(b)

    def _block_sample(series: List[float]) -> List[float]:
        if len(series) <= block:
            return [rng.choice(series) for _ in series]
        draws: List[float] = []
        while len(draws) < len(series):
            start = rng.randint(0, len(series) - block)
            draws.extend(series[start : start + block])
        return draws[: len(series)]

    count = 0
    for _ in range(n_iter):
        sample_a = _block_sample(a)
        sample_b = _block_sample(b)
        diff = statistics.mean(sample_a) - statistics.mean(sample_b)
        if obs >= 0:
            if diff >= obs:
                count += 1
        else:
            if diff <= obs:
                count += 1
    return count / n_iter


def randomization_test(
    a: List[float], b: List[float], n_iter: int = 1000, seed: int | None = None
) -> float:
    """Return randomization test p-value for difference in means."""

    if not a or not b:
        return 1.0

    rng = random.Random(seed)
    obs = statistics.mean(a) - statistics.mean(b)
    combined = a + b
    n_a = len(a)
    count = 0
    for _ in range(n_iter):
        shuffled = combined[:]
        rng.shuffle(shuffled)
        sample_a = shuffled[:n_a]
        sample_b = shuffled[n_a:]
        diff = statistics.mean(sample_a) - statistics.mean(sample_b)
        if obs >= 0:
            if diff >= obs:
                count += 1
        else:
            if diff <= obs:
                count += 1
    return count / n_iter


def simulate_portfolio(
    weights: Dict[str, Dict[str, float]],
    prices: Dict[str, Dict[str, float]],
) -> List[float]:
    """Return daily returns for a series of portfolio weights.

    The function assumes ``weights`` and ``prices`` are keyed by ISO date
    strings.  ``weights[date]`` contains the portfolio allocation to apply over
    the period from ``date`` to the next trading date.  ``prices`` holds closing
    prices for each symbol on each date.  Returns are computed by taking the
    weighted price relatives between consecutive dates.  Dates missing either
    weights or price data are skipped silently.
    """

    dates = sorted(prices.keys())
    returns: List[float] = []
    cumulative = 1.0
    for prev, curr in zip(dates[:-1], dates[1:]):
        w_prev = weights.get(prev)
        p_prev = prices.get(prev)
        p_curr = prices.get(curr)
        if w_prev is None or p_prev is None or p_curr is None:
            continue
        day_ret = 0.0
        for sym, w in w_prev.items():
            if sym in p_prev and sym in p_curr and p_prev[sym] != 0:
                day_ret += w * (p_curr[sym] / p_prev[sym] - 1.0)
        returns.append(day_ret)
        cumulative *= 1.0 + day_ret
        logger.debug("cumulative return", date=curr, cumulative=cumulative)
    return returns
