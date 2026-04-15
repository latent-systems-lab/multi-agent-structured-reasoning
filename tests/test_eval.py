import csv
import json
from pathlib import Path

import pytest

from eval.metrics import cvar_95, bootstrap_test, randomization_test
from eval.reporter import write_metrics_csv, write_metrics_json
from eval.comparison import comparison_report
from eval.metrics import simulate_portfolio
from eval.regimes import walk_forward_regimes


def test_metrics_and_reporter(tmp_path: Path) -> None:
    returns_a = [0.1, -0.2, 0.05, 0.0]
    returns_b = [0.0, -0.1, 0.02, 0.01]
    assert cvar_95([-0.1, -0.2, 0.0, 0.3]) == pytest.approx(-0.2)
    p_boot = bootstrap_test(returns_a, returns_b, n_iter=100, seed=0)
    p_rand = randomization_test(returns_a, returns_b, n_iter=100, seed=0)
    assert 0.0 <= p_boot <= 1.0
    assert 0.0 <= p_rand <= 1.0

    metrics = {"proto": {"sharpe": 1.0}}
    csv_path = tmp_path / "m.csv"
    json_path = tmp_path / "m.json"
    write_metrics_csv(str(csv_path), metrics)
    write_metrics_json(str(json_path), metrics)
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["protocol"] == "proto"
    with json_path.open() as f:
        data = json.load(f)
    assert "proto" in data


def test_comparison_report(tmp_path: Path) -> None:
    returns = {
        "baseline": [0.0, 0.01, -0.02, 0.03],
        "proto": [0.01, 0.02, -0.01, 0.04],
    }
    outdir = tmp_path / "out"
    comparison_report(returns, baseline="baseline", outdir=str(outdir))
    csv_file = outdir / "comparison.csv"
    json_file = outdir / "comparison.json"
    assert csv_file.exists()
    assert json_file.exists()
    with json_file.open() as f:
        data = json.load(f)
    assert "baseline" in data and "proto" in data
    assert "bootstrap_p" in data["proto"]


def test_walk_forward_regimes() -> None:
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    returns = [0.01, -0.01, 0.02, -0.02]
    regimes = [
        {"name": "r1", "start": "2024-01-01", "end": "2024-01-02"},
        {"name": "r2", "start": "2024-01-03", "end": "2024-01-04"},
    ]
    metrics = walk_forward_regimes(dates, returns, regimes)
    assert set(metrics.keys()) == {"r1", "r2"}
    assert metrics["r1"]["hit_rate"] == pytest.approx(0.5)


def test_simulate_portfolio() -> None:
    weights = {
        "2024-01-01": {"A": 0.5, "B": 0.5},
        "2024-01-02": {"A": 0.5, "B": 0.5},
    }
    prices = {
        "2024-01-01": {"A": 100.0, "B": 200.0},
        "2024-01-02": {"A": 110.0, "B": 190.0},
        "2024-01-03": {"A": 121.0, "B": 180.0},
    }
    returns = simulate_portfolio(weights, prices)
    assert returns == pytest.approx([0.025, 0.023684], rel=1e-3)
