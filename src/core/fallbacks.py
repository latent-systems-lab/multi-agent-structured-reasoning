"""Deterministic fallbacks for specialist agents."""

from __future__ import annotations

from typing import Dict, List, Tuple, Any

from core.schemas import RiskAssessment, AnalystProposal
from portfolio.constraints import feasible_bounds
from portfolio.risk import cvar_95, stress_pnl


def _portfolio_returns(weights: Dict[str, float], prices_window: Dict[str, List[Tuple[str, float]]]) -> List[float]:
    returns_by_sym: Dict[str, List[float]] = {}
    for sym, series in prices_window.items():
        if len(series) < 2:
            continue
        prices = [p for _, p in series]
        returns_by_sym[sym] = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
    if not returns_by_sym:
        return []
    n = min(len(r) for r in returns_by_sym.values())
    out: List[float] = []
    for i in range(n):
        out.append(sum(weights.get(sym, 0.0) * returns_by_sym[sym][i] for sym in returns_by_sym))
    return out


def _max_drawdown(returns: List[float]) -> float:
    value = peak = 1.0
    mdd = 0.0
    for r in returns:
        value *= 1 + r
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > mdd:
            mdd = drawdown
    return mdd


def fallback_risk(
    weights: Dict[str, float],
    prices_window: Dict[str, List[Tuple[str, float]]],
    universe: List[str],
    stress_scenarios: Dict[str, float] | None = None,
) -> RiskAssessment:
    """Compute simple portfolio risk metrics without LLM assistance."""
    gross = sum(abs(w) for w in weights.values())
    net = sum(weights.values())
    port_returns = _portfolio_returns(weights, prices_window)
    cvar = cvar_95(port_returns) if port_returns else 0.0
    mdd = _max_drawdown(port_returns) if port_returns else 0.0
    bounds = feasible_bounds(universe)
    violations: List[str] = []
    if gross > bounds["gross_cap"]:
        violations.append(f"Gross {gross:.2f} exceeds {bounds['gross_cap']:.2f}")
    if abs(net) > bounds["net_cap"]:
        violations.append(f"Net {net:.2f} exceeds {bounds['net_cap']:.2f}")
    scenarios = stress_scenarios or {"market_down_5": -0.05}
    for name, move in scenarios.items():
        shock = {sym: move for sym in weights}
        pnl = stress_pnl(weights, shock)
        if pnl < 0:
            violations.append(f"{name} stress loss {pnl:.2%}")
    return RiskAssessment(
        gross=gross,
        net=net,
        cvar_95=cvar,
        max_drawdown_est=mdd,
        violations=violations,
    )


def momentum(prices: List[Tuple[str, float]]) -> float:
    if len(prices) < 2:
        return 0.0
    start, end = prices[0][1], prices[-1][1]
    if start == 0:
        return 0.0
    return (end - start) / start


def normalize_weights(raw: Dict[str, float], universe: List[str]) -> Dict[str, float]:
    """Return ``raw`` weights scaled to sum to one.

    Negative inputs are preserved so callers can represent short positions.
    If the sum of weights is zero, the raw mapping is returned unchanged.
    """

    weights = {sym: float(raw.get(sym, 0.0)) for sym in universe}
    total = sum(weights.values())
    if total != 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def normalize_long_only(raw: Dict[str, float], universe: List[str]) -> Dict[str, float]:
    weights = {sym: max(0.0, float(raw.get(sym, 0.0))) for sym in universe}
    total = sum(weights.values())
    if total > 0:
        return {k: v / total for k, v in weights.items()}
    if universe:
        return {sym: 1.0 / len(universe) for sym in universe}
    return {}


def fallback_technical(
    prices_window: Dict[str, List[Tuple[str, float]]],
    universe: List[str],
    optimize_cash: bool = False,
) -> AnalystProposal:
    scores = {sym: momentum(prices_window.get(sym, [])) for sym in universe}
    weights = normalize_weights(scores, universe)
    if abs(sum(weights.values())) < 1e-12 and any(v > 0.0 for v in scores.values()):
        weights = normalize_long_only(scores, universe)
    if all(v == 0.0 for v in weights.values()) and universe:
        weights = {s: 1.0 / len(universe) for s in universe}
    if optimize_cash:
        risky_sum = sum(weights.values())
        cash = max(0.0, 1.0 - risky_sum)
        if cash > 0:
            weights = {**weights, "CASH": cash}
    return AnalystProposal(
        weights=weights,
        claim="Momentum-aligned allocation",
        evidence=[f"{s}: {scores.get(s, 0.0):+.3f}" for s in universe],
        risk_flags=[],
        rationale="Allocate proportionally to momentum (allow shorts); normalize to 1.",
        confidence=0.6,
        assumptions=[],
        data_refs=[],
    )


def _risk_signal_from_vix(vix: float) -> float:
    return max(-1.0, min(1.0, (20.0 - float(vix)) / 20.0))


def fallback_sentiment(
    market_context: Dict[str, float],
    universe: List[str],
    optimize_cash: bool = False,
) -> AnalystProposal:
    vix_val = float(market_context.get("vix", 20.0))
    signal = _risk_signal_from_vix(vix_val)
    weights = normalize_weights({sym: signal for sym in universe}, universe)
    if optimize_cash:
        risky_sum = sum(weights.values())
        cash = max(0.0, 1.0 - risky_sum)
        if cash > 0:
            weights = {**weights, "CASH": cash}
    return AnalystProposal(
        weights=weights,
        claim="Allocate equally with minor VIX-aware adjustments",
        evidence=["VIX risk-on/off signal"],
        risk_flags=[],
        rationale=f"VIX={vix_val:.2f} implies signal {signal:.2f}; maintain diversification",
        confidence=0.5,
        assumptions=[],
        data_refs=[],
    )


def fallback_scenario(
    market_context: Dict[str, float],
    features: Dict[str, Dict[str, float]],
    universe: List[str],
    optimize_cash: bool = False,
) -> AnalystProposal:
    macro_score = sum(market_context.values()) / len(market_context) if market_context else 0.0
    scores = {
        sym: features.get(sym, {}).get("beta", 0.0) * macro_score for sym in universe
    }
    weights = normalize_weights(scores, universe)
    if optimize_cash:
        risky_sum = sum(weights.values())
        cash = max(0.0, 1.0 - risky_sum)
        if cash > 0:
            weights = {**weights, "CASH": cash}
    return AnalystProposal(
        weights=weights,
        claim="Macro-aligned allocation",
        evidence=["Beta sensitivity to macro score"],
        risk_flags=[],
        rationale="Allocate by macro exposure",
        confidence=0.5,
        assumptions=[],
        data_refs=[],
    )
