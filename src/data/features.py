from __future__ import annotations

from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from core.schemas import ISODate, Symbol


def _sma(arr: np.ndarray) -> float:
    return float(np.mean(arr))


def _rsi(values: np.ndarray, period: int = 14) -> float:
    if values.size < period + 1:
        return 50.0
    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_features(
    prices_window: Dict[Symbol, List[Tuple[ISODate, float]]],
) -> Dict[Symbol, Dict[str, float]]:
    """Derive technical indicators from historical prices.

    Extends the previous implementation with additional signals required by the
    scenario and risk agents: multi-horizon momentum, volatility estimates,
    relative strength and a liquidity proxy.  A simple equal-weight market
    benchmark is used to estimate per-asset beta coefficients.
    """

    features: Dict[Symbol, Dict[str, float]] = {}
    returns_map: Dict[Symbol, np.ndarray] = {}

    for sym, series in prices_window.items():
        series_sorted = sorted(series, key=lambda x: x[0])
        closes = np.array([p for _, p in series_sorted], dtype=float)
        feats: Dict[str, float] = {}

        if closes.size >= 2:
            feats["return_1d"] = float(closes[-1] / closes[-2] - 1.0)
        if closes.size >= 5:
            feats["sma_5"] = _sma(closes[-5:])
            feats["momentum_5"] = float(closes[-1] / closes[-5] - 1.0)
        if closes.size >= 10:
            feats["sma_10"] = _sma(closes[-10:])
        if closes.size >= 20:
            feats["sma_20"] = _sma(closes[-20:])
            feats["momentum_20"] = float(closes[-1] / closes[-20] - 1.0)
        if closes.size >= 50:
            feats["sma_50"] = _sma(closes[-50:])
        if closes.size >= 21:
            rets = np.diff(closes[-21:]) / closes[-21:-1]
            feats["volatility_20"] = float(np.std(rets, ddof=1))
        if closes.size >= 61:
            rets_long = np.diff(closes[-61:]) / closes[-61:-1]
            feats["volatility_60"] = float(np.std(rets_long, ddof=1))

        feats["rsi_14"] = float(_rsi(closes, 14))

        if closes.size >= 2:
            returns = np.diff(closes) / closes[:-1]
            returns_map[sym] = returns
            tail = returns[-20:] if returns.size >= 20 else returns
            if tail.size > 0:
                feats["liquidity_proxy"] = float(1.0 / (np.mean(np.abs(tail)) + 1e-6))
        else:
            returns_map[sym] = np.array([], dtype=float)

        features[sym] = feats

    lengths = [len(arr) for arr in returns_map.values() if len(arr) > 0]
    if lengths:
        min_len = min(lengths)
        if min_len >= 2:
            aligned = np.stack(
                [arr[-min_len:] for arr in returns_map.values() if len(arr) >= min_len]
            )
            bench = np.mean(aligned, axis=0)
            var_bench = float(np.var(bench, ddof=1)) if len(bench) > 1 else 0.0
            if var_bench > 0.0:
                for sym, arr in returns_map.items():
                    if len(arr) >= min_len:
                        sym_slice = arr[-min_len:]
                        cov = (
                            float(np.cov(sym_slice, bench, ddof=1)[0, 1])
                            if len(bench) > 1
                            else 0.0
                        )
                        features.setdefault(sym, {})["beta"] = (
                            float(cov / var_bench) if var_bench else 0.0
                        )

    return features
