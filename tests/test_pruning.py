import sys
from types import ModuleType
from typing import Callable

from core.graph import build_graph
from core.protocols import ProtocolConfig, ProtocolId
from core.schemas import AnalystProposal, ChairDecision, GraphInputs, GraphState


def _stub_agent(name: str, prune: bool = False) -> ModuleType:
    def run(state: GraphState) -> GraphState:
        if getattr(state, "inputs", None) is not None and prune:
            state.inputs = None  # type: ignore[assignment]
        state.proposals[name] = AnalystProposal(
            weights={},
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
            weights = {sym: 0.0 for sym in getattr(state, "inputs", _inputs()).universe}
            state.decision = ChairDecision(
                date="2024-01-01",
                weights=weights,
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
        prices_window={"A": [("2024-01-01", 1.0)]},
        features={},
        fundamentals={"A": {"PERatio": 10.0}},
        headlines={},
        market_context={"vix": 20.0},
        prev_weights={"A": 0.0},
    )


def test_pruning_does_not_change_decision(monkeypatch):
    monkeypatch.setitem(sys.modules, "agents.alpha", _stub_agent("alpha", prune=False))
    monkeypatch.setitem(sys.modules, "agents.beta", _stub_agent("beta", prune=False))
    monkeypatch.setitem(sys.modules, "agents.chair", _stub_chair())

    cfg = ProtocolConfig(
        id=ProtocolId.FLAT,
        roles=["alpha", "beta", "chair"],
        comm_pattern="round_table",
    )
    graph = build_graph(cfg)
    baseline_state = graph.invoke({"inputs": _inputs()}, config={"configurable": {"thread_id": "t"}})
    baseline = baseline_state["decision"]

    monkeypatch.setitem(sys.modules, "agents.alpha", _stub_agent("alpha", prune=True))
    monkeypatch.setitem(sys.modules, "agents.beta", _stub_agent("beta", prune=True))
    monkeypatch.setitem(sys.modules, "agents.chair", _stub_chair())
    graph2 = build_graph(cfg)
    pruned_state = graph2.invoke({"inputs": _inputs()}, config={"configurable": {"thread_id": "t"}})
    pruned = pruned_state["decision"]

    assert pruned.weights == baseline.weights
