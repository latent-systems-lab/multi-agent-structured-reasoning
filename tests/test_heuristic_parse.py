import pytest

from utils.heuristic_parse import parse_weights_from_text, parse_confidence_from_text


def test_parse_weights_basic():
    text = "AAPL: 0.4, MSFT: -0.2, GOOGL: 0.8"
    assert parse_weights_from_text(text) == {
        "AAPL": 0.4,
        "MSFT": -0.2,
        "GOOGL": 0.8,
    }


def test_parse_weights_percent_and_invalid():
    text = "AAPL: 50%, MSFT: -25%, BAD: x, GOOGL: 25%"
    out = parse_weights_from_text(text)
    assert out == {"AAPL": 0.5, "MSFT": -0.25, "GOOGL": 0.25}


def test_parse_weights_compact():
    """Handles weights without commas or spaces."""

    text = "AAPL:0.2 TSLA:-0.1"
    assert parse_weights_from_text(text) == {"AAPL": 0.2, "TSLA": -0.1}


def test_parse_confidence_patterns():
    assert parse_confidence_from_text("Model shows 75% confidence in result") == pytest.approx(0.75)
    assert parse_confidence_from_text("Confidence: 0.6") == pytest.approx(0.6)
    assert parse_confidence_from_text("confidence 80%") == pytest.approx(0.8)


def test_parse_confidence_missing():
    assert parse_confidence_from_text("No explicit confidence given") is None


def test_parse_confidence_decimal():
    """Handles decimal confidence without percent sign."""

    assert parse_confidence_from_text("confidence 0.7") == pytest.approx(0.7)
