import os
import pytest

from data import alpha_vantage as av

SYMBOLS = ["AAPL", "TSLA", "NVDA", "PG", "JPM", "SPY", "EURUSD=X", "USDJPY=X", "^XAU", "CL=F", "ZN=F"]
DAY = "2020-01-04"


@pytest.mark.skipif(
    not os.getenv("ALPHAVANTAGE_API_KEY"),
    reason="Missing ALPHAVANTAGE_API_KEY",
)
def test_alpha_vantage_get_prices_window_live():
    out = av.get_prices_window(SYMBOLS, end_date=DAY, lookback_days=1)
    for symbol in SYMBOLS:
        assert symbol in out and len(out[symbol]) == 1
        date, price = out[symbol][0]
        assert date == DAY
        assert price > 0


@pytest.mark.skipif(
    not os.getenv("ALPHAVANTAGE_API_KEY"),
    reason="Missing ALPHAVANTAGE_API_KEY",
)
def test_alpha_vantage_get_fundamentals_live():
    out = av.get_fundamentals(SYMBOLS, as_of=DAY)
    for symbol in SYMBOLS:
        assert symbol in out
        assert out[symbol] is not None


@pytest.mark.skipif(
    not os.getenv("ALPHAVANTAGE_API_KEY"),
    reason="Missing ALPHAVANTAGE_API_KEY",
)
def test_alpha_vantage_get_headlines_live():
    out = av.get_headlines(SYMBOLS, as_of=DAY)
    for symbol in SYMBOLS:
        assert symbol in out
