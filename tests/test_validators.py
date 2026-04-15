import pytest

from core.schemas import (
    AnalystProposal,
    ChairCandidate,
    ChairDecision,
    DataRef,
    PeerPrediction,
)
from core.validators import (
    validate_candidate,
    validate_decision,
    validate_proposal,
)


def _data_ref(symbol=None):
    return DataRef(source="src", symbol=symbol, as_of="2024-01-01", hash="h")


def test_validate_proposal_ok():
    p = AnalystProposal(
        weights={"A": 0.1, "B": -0.1},
        claim="c",
        evidence=["e"],
        risk_flags=["f"],
        rationale="r",
        confidence=0.5,
        assumptions=[],
        data_refs=[_data_ref("A"), _data_ref("B")],
    )
    assert validate_proposal(p) == p


def test_validate_proposal_range():
    p = AnalystProposal(
        weights={"A": 1.5},
        claim="c",
        evidence=[],
        risk_flags=[],
        rationale="r",
        confidence=0.5,
        assumptions=[],
        data_refs=[],
    )
    with pytest.raises(ValueError):
        validate_proposal(p)


def test_validate_proposal_dataref_symbol():
    p = AnalystProposal(
        weights={"A": 0.1},
        claim="c",
        evidence=[],
        risk_flags=[],
        rationale="r",
        confidence=0.5,
        assumptions=[],
        data_refs=[_data_ref("B")],
    )
    with pytest.raises(ValueError):
        validate_proposal(p)


def test_validate_candidate_universe():
    universe = ["A", "B"]
    c = ChairCandidate(
        weights={"A": 0.1, "B": -0.1},
        utility=0.0,
        synthesis="",
        used_protocol="",
        supporting={},
    )
    assert validate_candidate(c, universe) == c

    c_missing = ChairCandidate(
        weights={"A": 0.1},
        utility=0.0,
        synthesis="",
        used_protocol="",
        supporting={},
    )
    with pytest.raises(ValueError):
        validate_candidate(c_missing, universe)

    c_extra = ChairCandidate(
        weights={"A": 0.1, "B": -0.1, "C": 0.0},
        utility=0.0,
        synthesis="",
        used_protocol="",
        supporting={},
    )
    with pytest.raises(ValueError):
        validate_candidate(c_extra, universe)


def test_candidate_accepts_weights_abs_alias():
    c = ChairCandidate(
        weights_abs={"A": 0.1, "B": -0.1},
        utility=0.0,
        synthesis="",
        used_protocol="",
        supporting={},
    )
    assert c.weights == {"A": 0.1, "B": -0.1}


def _decision(weights):
    return ChairDecision(
        date="2024-01-01",
        weights=weights,
        utility=0.0,
        synthesis="",
        protocol_id="p",
        rounds_taken=1,
        sc_M=1,
        token_in=0,
        token_out=0,
        latency_ms=0,
        data_refs=[],
    )


def test_validate_decision():
    universe = ["A", "B"]
    d = _decision({"A": 0.1, "B": -0.1})
    assert validate_decision(d, universe, gross_cap=1.0, pos_cap=0.2) == d

    d_cap = _decision({"A": 0.3, "B": -0.1})
    validate_decision(d_cap, universe, gross_cap=1.0, pos_cap=0.2)
    assert d_cap.weights["A"] == pytest.approx(0.2)

    d_gross = _decision({"A": 0.6, "B": 0.6})
    assert validate_decision(d_gross, universe, gross_cap=1.0, pos_cap=1.0) == d_gross

    d_net = _decision({"A": 0.3, "B": 0.1})
    assert validate_decision(d_net, universe, gross_cap=1.0, pos_cap=0.5) == d_net

    d_missing = _decision({"A": 0.1})
    with pytest.raises(ValueError):
        validate_decision(d_missing, universe, gross_cap=1.0, pos_cap=0.5)

    d_extra = _decision({"A": 0.1, "B": -0.1, "C": 0.0})
    with pytest.raises(ValueError):
        validate_decision(d_extra, universe, gross_cap=1.0, pos_cap=0.5)


def test_peer_prediction_predicted_weights_json():
    pp = PeerPrediction(target="t", predicted_weights='{"AAPL":0.1}')
    assert pp.predicted_weights == {"AAPL": 0.1}


def test_peer_prediction_predicted_weights_comma():
    pp = PeerPrediction(target="t", predicted_weights="AAPL:0.1, TSLA:-0.05")
    assert pp.predicted_weights == {"AAPL": 0.1, "TSLA": -0.05}
