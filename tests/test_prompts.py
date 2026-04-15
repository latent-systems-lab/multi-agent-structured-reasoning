import json

from pydantic import BaseModel

from core.protocols import ReasoningToggles, SelfConsistency, Debate
from core.prompts import analyst_prompt, chair_prompt, critique_prompt


class _SchemaModel(BaseModel):
    field: str


def _schema():
    return _SchemaModel.model_json_schema()


def test_analyst_prompt_substitution():
    toggles = ReasoningToggles(
        cot=True,
        self_consistency=SelfConsistency(enabled=True, M=3),
        debate=Debate(enabled=True),
    )
    prompt = analyst_prompt("technical", toggles, _schema())
    assert "technical analyst" in prompt
    assert "step by step" in prompt
    assert "3" in prompt
    assert json.dumps(_schema(), indent=2) in prompt


def test_chair_prompt_substitution():
    toggles = ReasoningToggles(
        cot=True,
        self_consistency=SelfConsistency(enabled=True, M=2),
        debate=Debate(enabled=True),
    )
    prompt = chair_prompt(toggles, _schema())
    assert "chair" in prompt.lower()
    assert "chain-of-thought" in prompt
    assert "self-consistency" in prompt
    assert json.dumps(_schema(), indent=2) in prompt


def test_critique_prompt_substitution():
    toggles = ReasoningToggles(cot=True, debate=Debate(enabled=True))
    summary = "Weights changed by 5%"
    violations = ["risk limit exceeded"]
    prompt = critique_prompt("fundamental", summary, violations, toggles)
    assert "fundamental" in prompt
    assert summary in prompt
    assert violations[0] in prompt
    assert "step by step" in prompt


def test_theory_of_mind_instruction():
    toggles = ReasoningToggles(theory_of_mind=True)
    prompt = analyst_prompt("sentiment", toggles, _schema())
    assert "mind_reading" in prompt and "predict" in prompt


def test_peer_beliefs_in_prompt():
    toggles = ReasoningToggles()
    beliefs = {
        "technical": ["momentum is rising"],
        "fundamental": ["valuations are stretched"],
    }
    prompt = analyst_prompt("sentiment", toggles, _schema(), beliefs)
    assert "technical" in prompt and "momentum is rising" in prompt
    assert "fundamental" in prompt and "valuations are stretched" in prompt


def test_short_position_note(monkeypatch):
    monkeypatch.setenv("OPTIMIZE_CASH", "1")
    prompt = analyst_prompt("sentiment", ReasoningToggles(), _schema())
    assert "short positions" in prompt.lower()
    chair = chair_prompt(ReasoningToggles(), _schema())
    assert "short positions" in chair.lower()
