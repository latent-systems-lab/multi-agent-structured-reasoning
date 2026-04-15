import pytest

from core.schemas import GraphInputs, GraphState, AnalystProposal
from core.protocols import ReasoningToggles
from agents import fundamental, sentiment, scenario


def _base_inputs():
    return GraphInputs(
        date="2024-01-01",
        universe=["A", "B"],
        prices_window={"A": [("2024-01-01", 1.0)], "B": [("2024-01-01", 1.0)]},
        features={"A": {"beta": 1.0}, "B": {"beta": 0.5}},
        fundamentals={"A": {"PERatio": 10.0}, "B": {"PERatio": 20.0}},
        headlines={"A": ["good"], "B": ["bad"]},
        market_context={"vix": 20.0, "gdp": 1.0},
        prev_weights={"A": 0.0, "B": 0.0},
    )


def _assert_normalized(proposal: AnalystProposal):
    weights = proposal.weights
    assert sum(weights.values()) == pytest.approx(1.0)
    assert set(weights).issubset({"A", "B", "CASH"})
    for w in weights.values():
        assert 0.0 <= w <= 1.0


def test_fundamental_agent_invalid_llm_output(monkeypatch):
    class BadOutput:
        def model_dump(self):
            return {"weights": {"A": 1.0}}

    def fake_create(prompt, tools, schema, client=None, **kwargs):
        def _agent(*args, **kwargs):
            return BadOutput()

        return _agent

    monkeypatch.setattr(fundamental, "create_react_agent", fake_create)
    monkeypatch.setattr(fundamental, "get_gemini_llm", lambda: object())
    monkeypatch.setenv("OPTIMIZE_CASH", "1")

    state = GraphState(inputs=_base_inputs())
    node = fundamental.create(ReasoningToggles())
    new_state = node(state)
    proposal = new_state.proposals["fundamental"]
    assert isinstance(proposal, AnalystProposal)
    _assert_normalized(proposal)


def test_sentiment_agent_invalid_llm_output(monkeypatch):
    def fake_create(prompt, tools, schema, client=None, **kwargs):
        def _agent(*args, **kwargs):
            raise ValueError("malformed")

        return _agent

    monkeypatch.setattr(sentiment, "create_react_agent", fake_create)
    monkeypatch.setattr(sentiment, "get_gemini_llm", lambda: object())
    monkeypatch.setenv("OPTIMIZE_CASH", "1")

    state = GraphState(inputs=_base_inputs())
    node = sentiment.create(ReasoningToggles())
    new_state = node(state)
    proposal = new_state.proposals["sentiment"]
    assert isinstance(proposal, AnalystProposal)
    _assert_normalized(proposal)


def test_scenario_agent_invalid_llm_output(monkeypatch):
    def fake_create(prompt, tools, schema, client=None, **kwargs):
        def _agent(*args, **kwargs):
            raise ValueError("bad output")

        return _agent

    monkeypatch.setattr(scenario, "create_react_agent", fake_create)
    monkeypatch.setattr(scenario, "get_gemini_llm", lambda: object())
    monkeypatch.setenv("OPTIMIZE_CASH", "1")

    state = GraphState(inputs=_base_inputs())
    node = scenario.create(ReasoningToggles())
    new_state = node(state)
    proposal = new_state.proposals["scenario"]
    assert isinstance(proposal, AnalystProposal)
    _assert_normalized(proposal)
