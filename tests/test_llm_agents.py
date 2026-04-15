import pytest
import types
from typing import Any
from pydantic import BaseModel, ValidationError

from core.react_factory import create_react_agent, DEFAULT_MODEL, DEFAULT_TIMEOUT_S
from core import react_factory
from core import gemini_client
from core.schemas import (
    AnalystProposal,
    RiskAssessment,
    GraphInputs,
    GraphState,
    ChairCandidate,
)
from agents import fundamental, technical, sentiment, scenario, risk as risk_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from core.protocols import ReasoningToggles, SelfConsistency, Debate


@pytest.fixture
def price_fetcher():
    """Mock price fetching tool returning a constant price."""

    @tool
    def fetch_price(symbol: str) -> float:
        """Return mocked latest price for ``symbol``."""

        return 123.45

    return fetch_price


@pytest.fixture
def risk_calculator():
    """Mock risk calculator tool returning simple aggregates."""

    @tool
    def calc_risk(weights: dict[str, float]) -> RiskAssessment:
        """Return mocked risk metrics for ``weights``."""

        gross = float(sum(abs(w) for w in weights.values()))
        net = float(sum(weights.values()))
        return RiskAssessment(
            gross=gross,
            net=net,
            cvar_95=0.0,
            max_drawdown_est=0.0,
            violations=[],
        )

    return calc_risk


def test_create_react_agent_thinking_config(monkeypatch):
    captured: dict[str, Any] = {}

    class DummyLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def with_structured_output(self, schema):
            return self

    monkeypatch.setattr(react_factory, "ChatGoogleGenerativeAI", DummyLLM)
    monkeypatch.setattr(
        react_factory,
        "_create_react_agent",
        lambda model, tools, prompt, response_format: types.SimpleNamespace(
            invoke=lambda state, config=None: {"structured_response": response_format()}
        ),
    )

    class DummyModel(BaseModel):
        foo: str = "bar"

    create_react_agent(
        "prompt",
        [],
        DummyModel,
        structured_mode="strict",
        thinking_budget=-1,
        include_thoughts=True,
    )

    assert captured["thinking_budget"] == -1
    assert captured["include_thoughts"] is True
    assert captured["timeout"] == DEFAULT_TIMEOUT_S


def _patch_graph(monkeypatch):
    """Patch Gemini dependencies with deterministic stubs."""

    class DummyLLM:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema):
            self.schema = schema
            return self

    monkeypatch.setattr(react_factory, "ChatGoogleGenerativeAI", DummyLLM)
    monkeypatch.setattr(gemini_client, "get_gemini_llm", lambda: object())
    for mod in (fundamental, technical, sentiment, scenario, risk_agent):
        monkeypatch.setattr(mod, "get_gemini_llm", lambda: object())

    captured_prompts: list[str] = []

    def fake_create(model, tools, prompt, response_format, **kwargs):
        captured_prompts.append(getattr(prompt, "content", prompt))
        schema = response_format

        class _Graph:
            def invoke(self, state, config=None):
                if isinstance(schema, type) and issubclass(schema, AnalystProposal):
                    sym = state.get("symbols", ["A"])[0]
                    try:
                        tools[0].invoke({"symbol": sym})
                    except Exception:
                        tools[0].invoke({})
                    return {
                        "structured_response": schema(
                            weights={},
                            claim="claim",
                            evidence=[],
                            risk_flags=[],
                            rationale="rationale",
                            confidence=0.5,
                            assumptions=[],
                            data_refs=[],
                        )
                    }
                weights = state.get("weights", {})
                tools[0].invoke({"weights": weights})
                return {
                    "structured_response": schema(
                        gross=0.0,
                        net=0.0,
                        cvar_95=0.0,
                        max_drawdown_est=0.0,
                        violations=[],
                    )
                }

        return _Graph()

    monkeypatch.setattr(react_factory, "_create_react_agent", fake_create)
    return captured_prompts


def test_create_react_agent_price_tool(price_fetcher, monkeypatch):
    """create_react_agent wires tools and validates output schema."""

    class PriceModel(BaseModel):
        price: float

    def fake_create(model, tools, prompt, response_format, **kwargs):
        schema = response_format

        class _Graph:
            def invoke(self, state):
                symbol = state["messages"][-1].content
                price = tools[0].invoke({"symbol": symbol})
                return {"structured_response": schema(price=price)}

        return _Graph()

    monkeypatch.setattr(react_factory, "_create_react_agent", fake_create)
    class DummyLLM:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema):
            self.schema = schema
            return self

    monkeypatch.setattr(react_factory, "ChatGoogleGenerativeAI", DummyLLM)

    agent = create_react_agent(
        "prompt", [price_fetcher], PriceModel, structured_mode="strict", client=object()
    )
    result = agent({"messages": [HumanMessage(content="AAPL")]})

    assert isinstance(result, PriceModel)
    assert result.price == 123.45


def test_create_react_agent_risk_tool(risk_calculator, monkeypatch):
    """RiskAssessment schema is honoured and tool callable."""

    def fake_create(model, tools, prompt, response_format, **kwargs):
        class _Graph:
            def invoke(self, state):
                report = tools[0].invoke({"weights": state["weights"]})
                return {"structured_response": report}

        return _Graph()

    monkeypatch.setattr(react_factory, "_create_react_agent", fake_create)
    class DummyLLM:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema):
            self.schema = schema
            return self

    monkeypatch.setattr(react_factory, "ChatGoogleGenerativeAI", DummyLLM)

    agent = create_react_agent(
        "prompt", [risk_calculator], RiskAssessment, structured_mode="strict", client=object()
    )
    result = agent(
        {
            "messages": [HumanMessage(content="Assess risk")],
            "weights": {"A": 1.0},
        }
    )

    assert isinstance(result, RiskAssessment)
    assert result.gross == 1.0
    assert result.net == 1.0


def test_create_react_agent_reuses_client(monkeypatch):
    """Injected client is used without revalidating environment."""

    class DummyModel(BaseModel):
        pass

    captured = {}

    class DummyLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def validate_environment(self):  # pragma: no cover - exercised via subclass
            raise AssertionError("validate_environment should not run")

        def with_structured_output(self, schema):
            self.schema = schema
            return self

    monkeypatch.setattr(react_factory, "ChatGoogleGenerativeAI", DummyLLM)

    def fake_create(model, tools, prompt, response_format, **kwargs):
        schema = response_format

        class _Graph:
            def invoke(self, state):
                return {"structured_response": schema()}

        return _Graph()

    monkeypatch.setattr(react_factory, "_create_react_agent", fake_create)
    stub_client = object()
    create_react_agent(
        "prompt", [], DummyModel, structured_mode="strict", client=stub_client
    )

    assert captured["client"] is stub_client
    assert captured["model"] == DEFAULT_MODEL


def test_create_react_agent_defaults(monkeypatch):
    """Without client, default model and temperature are applied."""

    class DummyModel(BaseModel):
        pass

    captured = {}

    class DummyLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def with_structured_output(self, schema):
            self.schema = schema
            return self

    monkeypatch.setattr(react_factory, "ChatGoogleGenerativeAI", DummyLLM)

    def fake_create(model, tools, prompt, response_format, **kwargs):
        schema = response_format

        class _Graph:
            def invoke(self, state):
                return {"structured_response": schema()}

        return _Graph()

    monkeypatch.setattr(react_factory, "_create_react_agent", fake_create)

    create_react_agent("prompt", [], DummyModel, structured_mode="strict")

    assert captured["model"] == DEFAULT_MODEL
    assert captured["temperature"] == 0.0
    assert "client" not in captured


def test_create_react_agent_parsing_fallback(monkeypatch):
    """Fallback LLM parses result when structured_response is missing."""

    class PriceModel(BaseModel):
        price: int

    instantiations: list[dict[str, Any]] = []

    class DummyLLM:
        def __init__(self, **kwargs):
            instantiations.append(kwargs)

        def with_structured_output(self, schema):
            self.schema = schema
            return self

        def invoke(self, _prompt):
            return self.schema(price=42)

    monkeypatch.setattr(react_factory, "ChatGoogleGenerativeAI", DummyLLM)

    def fake_create(model, tools, prompt, response_format, **kwargs):
        class _Graph:
            def invoke(self, state, config=None):
                return {"structured_response": None}

        return _Graph()

    monkeypatch.setattr(react_factory, "_create_react_agent", fake_create)

    agent = create_react_agent("prompt", [], PriceModel, structured_mode="strict")
    result = agent({"messages": []})

    assert isinstance(result, PriceModel)
    assert result.price == 42
    assert len(instantiations) == 2


@pytest.mark.parametrize(
    "module,name",
    [
        (fundamental, "fundamental"),
        (technical, "technical"),
        (sentiment, "sentiment"),
        (scenario, "scenario"),
    ],
)
def test_specialist_agents_produce_proposals(module, name, monkeypatch):
    """Each specialist agent yields an AnalystProposal via create_react_agent."""

    prompts = _patch_graph(monkeypatch)

    inputs = GraphInputs(
        date="2024-01-01",
        universe=["A"],
        prices_window={"A": [("2024-01-01", 1.0), ("2024-01-02", 1.1)]},
        features={"A": {"beta": 1.0}},
        fundamentals={"A": {"pe": 10.0}},
        headlines={"A": ["Good news"]},
        market_context={"gdp": 1.0},
        prev_weights={"A": 0.0},
    )
    state = GraphState(inputs=inputs)

    toggles = ReasoningToggles(
        cot=True,
        self_consistency=SelfConsistency(enabled=True, M=3),
        debate=Debate(enabled=True),
        theory_of_mind=True,
    )

    node = module.create(toggles)
    new_state = node(state)

    assert isinstance(new_state.proposals[name], AnalystProposal)

    prompt = prompts[0]
    assert "step by step" in prompt
    assert "debate" in prompt
    assert "3" in prompt
    assert "beliefs and intentions" in prompt


def test_risk_agent_produces_assessment(monkeypatch):
    """Risk agent attaches a RiskAssessment using create_react_agent."""

    _patch_graph(monkeypatch)

    inputs = GraphInputs(
        date="2024-01-01",
        universe=["A"],
        prices_window={"A": [("2024-01-01", 1.0), ("2024-01-02", 1.1)]},
        features={"A": {"beta": 1.0}},
        fundamentals={"A": {"pe": 10.0}},
        headlines={"A": ["Good news"]},
        market_context={"gdp": 1.0},
        prev_weights={"A": 0.0},
    )
    state = GraphState(
        inputs=inputs,
        chair_candidates=[
            ChairCandidate(
                weights={"A": 1.0},
                utility=0.0,
                synthesis="",
                used_protocol="",
                supporting={},
            )
        ],
    )

    toggles = ReasoningToggles(
        cot=True,
        self_consistency=SelfConsistency(enabled=True, M=2),
        debate=Debate(enabled=True),
        theory_of_mind=True,
    )
    node = risk_agent.create(toggles)
    node(state)
    assert isinstance(state.chair_candidates[-1].risk, RiskAssessment)


def test_specialist_agents_reject_malformed_inputs(monkeypatch):
    """GraphInputs validation rejects malformed data before agent invocation."""

    _patch_graph(monkeypatch)

    with pytest.raises(ValidationError):
        state = GraphState(
            inputs={
                "date": 20240101,
                "universe": "A",  # should be list
                "prices_window": {"A": [("2024-01-01", 1.0)]},
                "features": {"A": {"beta": 1.0}},
                "fundamentals": {"A": {"pe": 10.0}},
                "headlines": {"A": ["Good news"]},
                "market_context": {"gdp": 1.0},
                "prev_weights": {"A": 0.0},
            }
        )
        node = fundamental.create(ReasoningToggles())
        node(state)


def test_technical_agent_recursion_limit_fallback(monkeypatch):
    """Technical agent falls back to momentum weights when LLM fails."""

    def fake_create(prompt, tools, schema, *, structured_mode="strict", client=None):
        def _agent(state, *, config=None):
            assert config and config.get("recursion_limit") == 50
            raise RuntimeError("recursion limit reached")

        return _agent

    monkeypatch.setattr(technical, "create_react_agent", fake_create)
    monkeypatch.setattr(technical, "get_gemini_llm", lambda: object())
    monkeypatch.setenv("OPTIMIZE_CASH", "0")

    inputs = GraphInputs(
        date="2024-01-01",
        universe=["A", "B"],
        prices_window={
            "A": [("2024-01-01", 1.0), ("2024-01-02", 1.2)],
            "B": [("2024-01-01", 1.0), ("2024-01-02", 0.8)],
        },
        features={},
        fundamentals={},
        headlines={},
        market_context={},
        prev_weights={"A": 0.0, "B": 0.0},
    )
    state = GraphState(inputs=inputs)

    node = technical.create(ReasoningToggles())
    out = node(state)

    weights = out.proposals["technical"].weights
    assert sum(weights.values()) == pytest.approx(1.0)
    assert set(weights) == {"A", "B"}
    assert weights["A"] > weights["B"]
