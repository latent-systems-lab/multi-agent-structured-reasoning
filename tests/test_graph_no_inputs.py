from types import ModuleType
from typing import Callable
import sys

from core.graph import build_graph, _no_inputs
from core.protocols import ProtocolConfig, ProtocolId
from core.runtime import run_decision_day
from core.schemas import (
    AnalystProposal,
    ChairDecision,
    GraphInputs,
    GraphState,
)


def _stub_agent(name: str) -> ModuleType:
    def run(state: GraphState) -> GraphState:
        state.proposals[name] = AnalystProposal(
            weights={sym: 0.0 for sym in state.inputs.universe},
            claim="",
            evidence=[],
            risk_flags=[],
            rationale="",
            confidence=0.0,
            assumptions=[],
            data_refs=[],
        )
        return state

    m = ModuleType(f"agents.{name}")
    m.run = run  # type: ignore[attr-defined]
    return m


def _stub_chair() -> ModuleType:
    def run(_: object) -> Callable[[GraphState], GraphState]:
        def node(state: GraphState) -> GraphState:
            state.decision = ChairDecision(
                date=state.inputs.date,
                weights={sym: 0.0 for sym in state.inputs.universe},
                utility=0.0,
                synthesis="",
                protocol_id="test",
                rounds_taken=1,
                sc_M=1,
                token_in=0,
                token_out=0,
                latency_ms=0,
                data_refs=[],
            )
            return state

        return node

    m = ModuleType("agents.chair")
    m.run = run  # type: ignore[attr-defined]
    return m


def _inputs() -> GraphInputs:
    return GraphInputs(
        date="2024-01-01",
        universe=["A"],
        prices_window={},
        features={},
        fundamentals={},
        headlines={},
        market_context={},
        prev_weights={"A": 0.0},
    )


def test_run_decision_day_no_invalid_update(monkeypatch):
    monkeypatch.setitem(sys.modules, "agents.alpha", _stub_agent("alpha"))
    monkeypatch.setitem(sys.modules, "agents.beta", _stub_agent("beta"))
    monkeypatch.setitem(sys.modules, "agents.chair", _stub_chair())

    cfg = ProtocolConfig(
        id=ProtocolId.FLAT,
        roles=["alpha", "beta", "chair"],
        comm_pattern="parallel",
    )
    graph = build_graph(cfg)

    run_decision_day(graph, _inputs())


def test_no_inputs_preserves_model_lists():
    """Nodes returning lists of models should round-trip without dict conversion."""

    from core.schemas import ChairCandidate

    def node(state: GraphState) -> GraphState:
        state.chair_candidates = [
            ChairCandidate(
                weights={sym: 0.0 for sym in state.inputs.universe},
                utility=0.0,
                synthesis="",
                used_protocol="",
                supporting={},
            )
        ]
        return state

    wrapped = _no_inputs(node)
    out = wrapped(GraphState(inputs=_inputs()))
    assert isinstance(out["chair_candidates"][0], ChairCandidate)
