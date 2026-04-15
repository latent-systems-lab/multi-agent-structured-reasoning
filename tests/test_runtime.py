from core.runtime import run_decision_day
from core.schemas import ChairDecision, GraphInputs


class _BaseGraph:
    def __init__(self, usage):
        self._usage = usage


class DummyGraph(_BaseGraph):
    async def ainvoke(self, state):
        decision = ChairDecision(
            date="2024-01-01",
            weights={"A": 0.0},
            utility=0.0,
            synthesis="",
            protocol_id="proto",
            rounds_taken=1,
            sc_M=1,
            token_in=0,
            token_out=0,
            latency_ms=0,
            data_refs=[],
        )
        return {"decision": decision, "usage": self._usage}


class AsyncGraph(_BaseGraph):
    async def ainvoke(self, state):
        decision = ChairDecision(
            date="2024-01-01",
            weights={"A": 0.0},
            utility=0.0,
            synthesis="",
            protocol_id="proto",
            rounds_taken=1,
            sc_M=1,
            token_in=0,
            token_out=0,
            latency_ms=0,
            data_refs=[],
        )
        return {"decision": decision, "usage": self._usage}


class MessagesGraph:
    async def ainvoke(self, state):
        decision = ChairDecision(
            date="2024-01-01",
            weights={"A": 0.0},
            utility=0.0,
            synthesis="",
            protocol_id="proto",
            rounds_taken=1,
            sc_M=1,
            token_in=0,
            token_out=0,
            latency_ms=0,
            data_refs=[],
        )
        msg1 = type("M", (), {"usage_metadata": {"input_tokens": 2, "output_tokens": 1}})()
        msg2 = type(
            "M",
            (),
            {"usage_metadata": {"prompt_token_count": 3, "candidates_token_count": 4}},
        )()
        return {"decision": decision, "messages": [msg1, msg2]}


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


def test_run_decision_day_records_tokens():
    g = DummyGraph({"prompt_token_count": 3, "candidates_token_count": 2})
    decision = run_decision_day(g, _inputs())
    assert decision.token_in == 3
    assert decision.token_out == 2


def test_run_decision_day_async_graph():
    g = AsyncGraph({"input_tokens": 4, "output_tokens": 1})
    decision = run_decision_day(g, _inputs())
    assert decision.token_in == 4
    assert decision.token_out == 1


def test_run_decision_day_sums_message_tokens():
    g = MessagesGraph()
    decision = run_decision_day(g, _inputs())
    assert decision.token_in == 5
    assert decision.token_out == 5
