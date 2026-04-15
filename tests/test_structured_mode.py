import pytest

from core.schemas import GraphInputs, GraphState, AnalystProposal
from core.protocols import ReasoningToggles


ANALYST_NAMES = ["fundamental", "technical", "sentiment", "scenario"]


def _base_inputs():
    return GraphInputs(
        date="2024-01-01",
        universe=["AAPL", "TSLA"],
        prices_window={
            "AAPL": [("2024-01-01", 1.0), ("2024-01-02", 1.1)],
            "TSLA": [("2024-01-01", 1.0), ("2024-01-02", 0.9)],
        },
        features={"AAPL": {"beta": 1.0}, "TSLA": {"beta": 0.5}},
        fundamentals={"AAPL": {"PERatio": 10.0}, "TSLA": {"PERatio": 20.0}},
        headlines={"AAPL": ["good"], "TSLA": ["bad"]},
        market_context={"vix": 20.0, "gdp": 1.0},
        prev_weights={"AAPL": 0.0, "TSLA": 0.0},
    )


@pytest.mark.parametrize("mode", ["weights_only", "off"])
@pytest.mark.parametrize("name", ANALYST_NAMES)
def test_structured_modes(name, mode, monkeypatch):
    module = __import__(f"agents.{name}", fromlist=["_dummy"])
    monkeypatch.setattr(module, "get_gemini_llm", lambda: object())

    def fake_create(prompt, tools, schema, **kwargs):
        if schema is None:
            output = "AAPL:0.2 TSLA:-0.1 confidence 0.7"
        else:
            output = schema(weights={"AAPL": 0.2, "TSLA": 0.1})
        return lambda *args, **kwargs: output

    monkeypatch.setattr(module, "create_react_agent", fake_create)

    state = GraphState(inputs=_base_inputs())
    toggles = ReasoningToggles(structured_mode=mode)
    node = module.create(toggles)
    new_state = node(state)
    proposal = new_state.proposals[name]
    assert isinstance(proposal, AnalystProposal)
    assert sum(proposal.weights.values()) == pytest.approx(1.0)
    assert set(proposal.weights).issubset({"AAPL", "TSLA", "CASH"})
    for w in proposal.weights.values():
        assert 0.0 <= w <= 1.0
    if mode == "off":
        assert proposal.confidence == pytest.approx(0.7)
    else:
        assert proposal.confidence == pytest.approx(0.5)
