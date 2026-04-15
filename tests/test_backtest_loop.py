import csv
import json
import pytest
from pathlib import Path

from core.runtime import run_backtest_loop
from core.schemas import ChairDecision, ChairCandidate


class DummyGraph:
    pass


def test_run_backtest_loop_writes_artifacts_and_metrics(tmp_path, monkeypatch):
    universe = ["AAPL"]
    calendar = ["2024-01-01"]

    # Stub external data fetchers
    monkeypatch.setattr("core.runtime.load_calendar", lambda start, end: calendar)
    monkeypatch.setattr(
        "core.runtime.get_prices_window",
        lambda u, end_date, lookback_days: {
            sym: [("2023-12-31", 100.0), (end_date, 100.0)] for sym in u
        },
    )
    monkeypatch.setattr(
        "core.runtime.compute_features",
        lambda pw: {sym: {"feat": 0.0} for sym in pw},
    )
    monkeypatch.setattr(
        "core.runtime.get_fundamentals",
        lambda u, as_of: {sym: {"pe": 1.0} for sym in u},
    )
    monkeypatch.setattr(
        "core.runtime.get_headlines",
        lambda u, as_of: {sym: ["headline"] for sym in u},
    )

    # Stub decision and optimisation helpers
    def fake_run_decision_day(graph, inputs, flags=None):
        return ChairDecision(
            date=inputs.date,
            weights={sym: 1.0 for sym in inputs.universe},
            utility=1.0,
            synthesis="syn",
            protocol_id="proto",
            rounds_taken=1,
            sc_M=1,
            token_in=0,
            token_out=0,
            latency_ms=0,
            data_refs=[],
        )

    monkeypatch.setattr("core.runtime.run_decision_day", fake_run_decision_day)
    # Avoid importing the real optimiser (requires cvxpy); instead provide a
    # lightweight stub module
    import types, sys

    optimizer_mod = types.ModuleType("portfolio.optimizer")
    optimizer_mod.project_to_feasible = lambda candidate, bounds: candidate.weights
    portfolio_pkg = types.ModuleType("portfolio")
    portfolio_pkg.optimizer = optimizer_mod
    monkeypatch.setitem(sys.modules, "portfolio", portfolio_pkg)
    monkeypatch.setitem(sys.modules, "portfolio.optimizer", optimizer_mod)

    # Capture artifacts and metrics
    artifacts = []
    monkeypatch.setattr(
        "core.runtime.write_artifact", lambda path, obj: artifacts.append((path, obj))
    )

    returns_calls = {}
    monkeypatch.setattr(
        "core.runtime.append_returns_row",
        lambda path, date, ret, cum: returns_calls.setdefault("csv", (path, ret)),
    )
    monkeypatch.setattr(
        "core.runtime.write_returns_json",
        lambda path, returns: returns_calls.setdefault("json", (path, returns)),
    )

    metrics_calls = {}
    monkeypatch.setattr(
        "core.runtime.write_metrics_csv",
        lambda path, metrics: metrics_calls.setdefault(
            Path(path).name, (path, metrics)
        ),
    )
    monkeypatch.setattr(
        "core.runtime.write_metrics_json",
        lambda path, metrics: metrics_calls.setdefault(
            Path(path).name, (path, metrics)
        ),
    )

    costs = {"tc_bps": 0.0, "slippage_bps": 0.0, "borrow_bps_annual": 0.0}
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "x")

    run_backtest_loop(
        graph=DummyGraph(),
        universe=universe,
        start="2024-01-01",
        end="2024-01-01",
        costs=costs,
        market_api={"key_env": "ALPHAVANTAGE_API_KEY"},
        seeds=None,
        outdir=str(tmp_path),
        regimes=[{"name": "full", "start": "2024-01-01", "end": "2024-01-01"}],
    )

    assert artifacts
    decision_path = Path(artifacts[0][0])
    assert decision_path.parent == tmp_path / "decisions"
    assert decision_path.name == "decision_2024-01-01.json"

    expected_metrics = {
        "strategy": {
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "cvar_95": 0.0,
            "hit_rate": 0.0,
        }
    }
    assert metrics_calls["metrics.csv"][1] == expected_metrics
    assert metrics_calls["metrics.json"][1] == expected_metrics
    assert "full" in metrics_calls["regimes.csv"][1]
    assert "regimes.json" in metrics_calls

    for key, (path, data) in returns_calls.items():
        assert Path(path).parent == tmp_path
    for path, data in metrics_calls.values():
        assert Path(path).parent == tmp_path

    decision_csv = tmp_path / "decisions.csv"
    ops_json = tmp_path / "ops.json"
    token_usage_json = tmp_path / "token_usage.json"
    assert decision_csv.exists()
    assert ops_json.exists()
    assert token_usage_json.exists()
    with decision_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["date"] == "2024-01-01"
    assert float(rows[0]["AAPL"]) == pytest.approx(1.0)
    assert rows[0]["supporting"] == "{}"
    assert rows[0]["dissenting"] == "[]"
    assert rows[0]["data_refs"] == "[]"
    with ops_json.open() as f:
        data = json.load(f)
    assert data == [
        {"date": "2024-01-01", "token_in": 0, "token_out": 0, "latency_ms": 0}
    ]
    with token_usage_json.open() as f:
        usage = json.load(f)
    assert usage == [
        {
            "date": "2024-01-01",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    ]


def test_run_backtest_loop_sets_throttle(tmp_path, monkeypatch):
    monkeypatch.setattr("core.runtime.load_calendar", lambda start, end: [])
    monkeypatch.setattr("core.runtime.write_ops_json", lambda *a, **k: None)
    monkeypatch.setattr("core.runtime.append_returns_row", lambda *a, **k: None)
    monkeypatch.setattr("core.runtime.write_returns_json", lambda *a, **k: None)
    monkeypatch.setattr("core.runtime.write_metrics_csv", lambda *a, **k: None)
    monkeypatch.setattr("core.runtime.write_metrics_json", lambda *a, **k: None)

    import types, sys

    optimizer_mod = types.ModuleType("portfolio.optimizer")

    def _check_candidate(candidate, bounds):
        assert sum(candidate.weights.values()) == pytest.approx(1.0)
        return candidate.weights

    optimizer_mod.project_to_feasible = _check_candidate
    portfolio_pkg = types.ModuleType("portfolio")
    portfolio_pkg.optimizer = optimizer_mod
    monkeypatch.setitem(sys.modules, "portfolio", portfolio_pkg)
    monkeypatch.setitem(sys.modules, "portfolio.optimizer", optimizer_mod)

    called = {}
    monkeypatch.setattr(
        "core.runtime.set_throttle",
        lambda per_sec: called.setdefault("per_sec", per_sec),
    )

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "x")

    run_backtest_loop(
        graph=DummyGraph(),
        universe=["A"],
        start="2024-01-01",
        end="2024-01-01",
        costs={},
        market_api={"throttle_per_sec": 9, "key_env": "ALPHAVANTAGE_API_KEY"},
        seeds=None,
        outdir=str(tmp_path),
    )

    assert called["per_sec"] == 9


def test_candidate_with_shorts_sums_to_one_before_projection():
    candidate = ChairDecision(
        date="2024-01-01",
        weights={"AAA": 0.2, "BBB": -0.3},
        utility=1.0,
        synthesis="syn",
        protocol_id="proto",
        rounds_taken=1,
        sc_M=1,
        token_in=0,
        token_out=0,
        latency_ms=0,
        data_refs=[],
    )
    cand = ChairCandidate(
        weights=candidate.weights.copy(),
        utility=candidate.utility,
        synthesis=candidate.synthesis,
        used_protocol=candidate.protocol_id,
        supporting={},
    )
    cash_seed = max(0.0, 1.0 - sum(cand.weights.values()))
    cand.weights["CASH"] = cash_seed
    assert sum(cand.weights.values()) == pytest.approx(1.0)

    equity = 1_000_000.0
    long_notional = 0.2 * equity
    short_notional = -0.3 * equity
    cash_notional = equity - long_notional + short_notional
    assert cash_notional == pytest.approx(500_000.0)
