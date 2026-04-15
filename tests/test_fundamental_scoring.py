import pytest

from agents.fundamental import _score_fundamentals
from data import alpha_vantage as av


def test_weighted_scoring():
    fundamentals = {
        "PERatio": 10.0,
        "ProfitMargin": 0.2,
        "ReturnOnEquityTTM": 0.15,
        "PEGRatio": 0.5,
    }
    score = _score_fundamentals(fundamentals)
    assert score == pytest.approx(0.675954301075269, abs=1e-6)


def test_missing_metrics():
    fundamentals = {"PERatio": 10.0}
    score = _score_fundamentals(fundamentals)
    assert score == pytest.approx(0.967741935483871, abs=1e-6)


def test_get_fundamentals_skips_non_equities(monkeypatch, tmp_path):
    monkeypatch.setattr(av, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(av, "_throttle", lambda: None)
    monkeypatch.setattr(av, "write_artifact", lambda p, o: None)
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "KEY")

    calls = []

    class DummyResp:
        def json(self):
            return {"PERatio": "1.0"}

        def raise_for_status(self):
            pass

    def fake_get(url, timeout):
        calls.append(url)
        return DummyResp()

    monkeypatch.setattr(av.requests, "get", fake_get)

    out = av.get_fundamentals(["AAPL", "EURUSD=X"], as_of="2024-01-01")

    assert out["EURUSD=X"] == {}
    assert out["AAPL"]["PERatio"] == 1.0
    assert len(calls) == 1

