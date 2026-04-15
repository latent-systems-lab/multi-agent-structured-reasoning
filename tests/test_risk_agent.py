from core.schemas import GraphInputs, GraphState, ChairCandidate
from agents import risk
from core.protocols import ReasoningToggles


def test_default_stress_scenario():
    weights = {"A": 1.0}
    cand = ChairCandidate(weights=weights, utility=0.0, synthesis="", used_protocol="", supporting={})
    inputs = GraphInputs(
        date="2024-01-01",
        universe=["A"],
        prices_window={},
        features={},
        fundamentals={},
        headlines={},
        market_context={},
        prev_weights={"A": 0.0},
    )
    state = GraphState(inputs=inputs, chair_candidates=[cand])
    node = risk.create(ReasoningToggles())
    node(state)
    report = state.chair_candidates[-1].risk
    assert report is not None
    assert any("market_down_5" in v for v in report.violations)


def test_configured_stress_scenario():
    weights = {"A": 1.0}
    cand = ChairCandidate(weights=weights, utility=0.0, synthesis="", used_protocol="", supporting={})
    inputs = GraphInputs(
        date="2024-01-01",
        universe=["A"],
        prices_window={},
        features={},
        fundamentals={},
        headlines={},
        market_context={},
        prev_weights={"A": 0.0},
        stress_scenarios={"drop10": -0.1},
    )
    state = GraphState(inputs=inputs, chair_candidates=[cand])
    node = risk.create(ReasoningToggles())
    node(state)
    report = state.chair_candidates[-1].risk
    assert report is not None
    assert any("drop10" in v for v in report.violations)
