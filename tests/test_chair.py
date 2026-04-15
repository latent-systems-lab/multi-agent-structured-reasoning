import numpy as np
import pytest

from agents import risk
from core.protocols import ReasoningToggles, SelfConsistency
from core.schemas import GraphInputs, GraphState, AnalystProposal
from agents.chair import run


def _inputs():
    return GraphInputs(
        date="2024-01-01",
        universe=["A", "B"],
        prices_window={},
        features={},
        fundamentals={},
        headlines={},
        market_context={},
        prev_weights={"A": 0.0, "B": 0.0},
    )


def _proposal(delta: float) -> AnalystProposal:
    return AnalystProposal(
        weights={"A": delta, "B": -delta},
        claim="test claim",
        evidence=[],
        risk_flags=[],
        rationale="",
        confidence=1.0,
        assumptions=[],
        data_refs=[],
    )


def test_median_aggregation():
    toggles = ReasoningToggles(
        self_consistency=SelfConsistency(enabled=True, M=3, aggregation="median_utility")
    )
    state = GraphState(inputs=_inputs())
    chair = run(toggles)
    weights = []
    utils = []
    for d in [0.1, 0.3, -0.2]:
        state.proposals = {"p": _proposal(d)}
        update = chair(state)
        state.chair_candidates.extend(update["chair_candidates"])
        weights.append(state.chair_candidates[-1].weights["A"])
        utils.append(state.chair_candidates[-1].utility)
    out = risk.create(toggles)(state)
    decision = out["decision"]
    median_idx = sorted(range(len(utils)), key=lambda i: utils[i])[len(utils) // 2]
    assert decision.weights["A"] == pytest.approx(weights[median_idx])


def test_mean_aggregation():
    toggles = ReasoningToggles(
        self_consistency=SelfConsistency(enabled=True, M=2, aggregation="mean_utility")
    )
    state = GraphState(inputs=_inputs())
    chair = run(toggles)
    for d in [0.2, -0.2]:
        state.proposals = {"p": _proposal(d)}
        update = chair(state)
        state.chair_candidates.extend(update["chair_candidates"])
    out = risk.create(toggles)(state)
    decision = out["decision"]
    assert decision.weights["A"] == pytest.approx(-0.2)
