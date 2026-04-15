import pytest
from data import alpha_vantage as av

symbols = [
    "AAPL",
    "TSLA",
    "NVDA",
    "PG",
    "JPM",
    "SPY",
    "EURUSD=X",
    "USDJPY=X",
    "^XAU",
    "CL=F",
    "ZN=F",
]

class DummyResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def _setup(monkeypatch, tmp_path):
    monkeypatch.setattr(av, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(av, "_throttle", lambda: None)
    monkeypatch.setattr(av, "write_artifact", lambda p, o: None)


def test_get_fundamentals_uses_api_key(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "KEY")
    called = {}

    def fake_get(url, timeout):
        called["url"] = url
        return DummyResp({"PERatio": "1.0"})

    monkeypatch.setattr(av.requests, "get", fake_get)
    out = av.get_fundamentals(symbols, as_of="2024-01-01")
    assert called["url"].startswith("https://www.alphavantage.co/query")
    assert "function=OVERVIEW" in called["url"]
    assert "apikey=KEY" in called["url"]
    assert all(symbol in out for symbol in symbols)


def test_get_insights_currency(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "KEY")
    called = {}

    def fake_get(url, timeout):
        called["url"] = url
        payload = {
            "Realtime Currency Exchange Rate": {"5. Exchange Rate": "1.0"}
        }
        return DummyResp(payload)

    monkeypatch.setattr(av.requests, "get", fake_get)
    out = av.get_insights(["EURUSD=X"], as_of="2024-01-01")
    assert "function=CURRENCY_EXCHANGE_RATE" in called["url"]
    assert "from_currency=EUR" in called["url"]
    assert "to_currency=USD" in called["url"]
    assert "apikey=KEY" in called["url"]
    assert out["EURUSD=X"]["exchange_rate"] == 1.0


def test_get_insights_commodity(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "KEY")
    called = {}

    def fake_get(url, timeout):
        called["url"] = url
        return DummyResp({"data": [{"date": "2024-01-01", "value": "10"}]})

    monkeypatch.setattr(av.requests, "get", fake_get)
    out = av.get_insights(["CL=F"], as_of="2024-01-01")
    assert "function=WTI" in called["url"]
    assert "apikey=KEY" in called["url"]
    assert out["CL=F"]["price"] == 10.0


def test_get_insights_bond(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "KEY")
    called = {}

    def fake_get(url, timeout):
        called["url"] = url
        return DummyResp({"data": [{"date": "2024-01-01", "value": "5"}]})

    monkeypatch.setattr(av.requests, "get", fake_get)
    out = av.get_insights(["ZN=F"], as_of="2024-01-01")
    assert "function=TREASURY_YIELD" in called["url"]
    assert "maturity=10year" in called["url"]
    assert "apikey=KEY" in called["url"]
    assert out["ZN=F"]["yield"] == 5.0


@pytest.mark.parametrize(
    "func,args,kwargs",
    [
        (av.get_fundamentals, (symbols,), {"as_of": "2024-01-01"}),
        (av.get_insights, (symbols,), {"as_of": "2024-01-01"}),
    ],
)
def test_alpha_vantage_functions_require_key(monkeypatch, tmp_path, func, args, kwargs):
    monkeypatch.setattr(av, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    with pytest.raises(EnvironmentError):
        func(*args, **kwargs)
