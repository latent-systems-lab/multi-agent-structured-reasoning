import pytest
from langgraph.errors import GraphInterrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

from core.runtime import run_decision_day
from core.schemas import ChairDecision, GraphInputs


def _build_graph():
    def chair_node(state):
        ans = interrupt("need input")
        decision = ChairDecision(
            date="2024-01-01",
            weights={"A": 0.0},
            utility=0.0,
            synthesis=ans or "",
            protocol_id="proto",
            rounds_taken=1,
            sc_M=1,
            token_in=0,
            token_out=0,
            latency_ms=0,
            data_refs=[],
        )
        return {"decision": decision}

    builder = StateGraph(dict)
    builder.add_node("chair", chair_node)
    builder.add_edge(START, "chair")
    builder.add_edge("chair", END)
    return builder.compile(checkpointer=MemorySaver())


def _inputs():
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


def test_run_decision_day_resume():
    g = _build_graph()
    with pytest.raises(GraphInterrupt):
        run_decision_day(g, _inputs(), thread_id="t1")
    decision = run_decision_day(g, _inputs(), Command(resume="ok"), thread_id="t1")
    assert decision.synthesis == "ok"
