import pytest

from core.schemas import ChairCandidate
from portfolio.constraints import feasible_bounds
from portfolio.optimizer import project_to_feasible
from portfolio.risk import cvar_95, stress_pnl


def test_feasible_bounds():
    universe = ["AAPL", "MSFT", "GOOG", "AMZN=F"]
    bounds = feasible_bounds(universe)
    assert bounds["min_weight"] == pytest.approx(-0.3)
    assert bounds["max_weight"] == pytest.approx(0.3)
    assert bounds["gross_cap"] == pytest.approx(1.2)
    assert bounds["net_cap"] == pytest.approx(0.5)


def test_project_to_feasible():
    universe = ["A", "B", "C"]
    bounds = feasible_bounds(universe)
    candidate = ChairCandidate(
        weights={"A": 0.5, "B": 0.5, "C": -0.5},
        utility=0.0,
        synthesis="",
        used_protocol="",
        supporting={},
    )
    projected = project_to_feasible(candidate, bounds)
    for w in projected.values():
        assert bounds["min_weight"] - 1e-5 <= w <= bounds["max_weight"] + 1e-5
    assert sum(abs(w) for w in projected.values()) <= bounds["gross_cap"] + 1e-6
    assert abs(sum(projected.values())) <= bounds["net_cap"] + 1e-6


def test_project_to_feasible_with_cash():
    universe = ["A", "B"]
    bounds = feasible_bounds(universe, optimize_cash=True)
    candidate = ChairCandidate(
        weights={"A": 0.2, "B": -0.3, "CASH": 1.1},
        utility=0.0,
        synthesis="",
        used_protocol="",
        supporting={},
    )
    projected = project_to_feasible(candidate, bounds)
    assert sum(projected.values()) == pytest.approx(1.0)
    assert 0.0 - 1e-6 <= projected["CASH"] <= 1.0 + 1e-6
    for sym in ["A", "B"]:
        assert (
            bounds["min_weight"] - 1e-6 <= projected[sym] <= bounds["max_weight"] + 1e-6
        )


def test_cvar_95():
    returns = [-0.1, -0.2, 0.0, 0.3]
    assert cvar_95(returns) == pytest.approx(-0.2)


def test_stress_pnl():
    weights = {"A": 0.5, "B": -0.3}
    scenarios = {"A": -0.02, "B": -0.05, "C": 0.1}
    expected = 0.5 * -0.02 + (-0.3) * -0.05
    assert stress_pnl(weights, scenarios) == pytest.approx(expected)
