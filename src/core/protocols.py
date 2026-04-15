"""Protocol definitions and presets."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

class ProtocolId(str, Enum):
    ONE_SHOT = "one_shot"
    COT = "cot"
    TOM = "tom"
    SC = "sc"
    DEBATE = "debate"
    COT_SC = "cot_sc"
    TOM_SC = "tom_sc"
    DEBATE_SC = "debate_sc"
    COT_TOM = "cot_tom"
    DEBATE_TOM = "debate_tom"
    COT_DEBATE = "cot_debate"
    COT_DEBATE_TOM = "cot_debate_tom"
    COT_DEBATE_SC = "cot_debate_sc"
    COT_TOM_SC = "cot_tom_sc"
    DEBATE_TOM_SC = "debate_tom_sc"
    COT_DEBATE_TOM_SC = "cot_debate_tom_sc"
    JUDGE_ONLY = "judge_only"
    FLAT = "flat"
    PARALLEL = "parallel"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    VOTING = "voting"
    HYBRID = "hybrid"
    TWO_AGENT = "two_agent"



class SelfConsistency(BaseModel):
    enabled: bool = False
    M: int = 5
    aggregation: Literal["median_utility", "mean_utility"] = "median_utility"


class Debate(BaseModel):
    enabled: bool = False
    R_max: int = 2
    epsilon_l2: float = 0.01
    critique_style: Literal["targeted", "freeform"] = "targeted"


class ReasoningToggles(BaseModel):
    cot: bool = False
    temperature_chair: float = 0.2
    temperature_analyst: float = 0.4
    self_consistency: SelfConsistency = SelfConsistency()
    debate: Debate = Debate()
    theory_of_mind: bool = False
    episodic_top_k: int = 0
    experience_replay: bool = True
    structured_mode: Literal["strict", "weights_only", "off"] = Field(
        "strict",
        description=(
            "Schema enforcement: 'strict' validates full Pydantic schemas "
            "(legacy 'structured_json: true'); 'weights_only' only validates "
            "the 'weights' field; 'off' returns raw text (legacy 'structured_json: false')."
        ),
    )


class ProtocolConfig(BaseModel):
    id: ProtocolId
    roles: list[str] = Field(
        default_factory=lambda: [
            "fundamental",
            "technical",
            "sentiment",
            "risk",
            "scenario",
            "chair",
        ]
    )
    comm_pattern: Literal[
        "hierarchical",
        "round_table",
        "flat",
        "parallel",
        "iterative_refinement",
        "voting",
        "hybrid",
    ] = "hierarchical"
    toggles: ReasoningToggles = ReasoningToggles()
    stopping: Literal["none", "epsilon_or_rmax"] = "none"


def protocol_presets(pid: ProtocolId) -> ProtocolConfig:
    """Return a fully-populated ProtocolConfig for the given protocol id."""

    if pid == ProtocolId.ONE_SHOT:
        return ProtocolConfig(id=pid)
    if pid == ProtocolId.COT:
        return ProtocolConfig(id=pid, toggles=ReasoningToggles(cot=True))
    if pid == ProtocolId.COT_SC:
        return ProtocolConfig(
            id=pid,
            toggles=ReasoningToggles(
                cot=True, self_consistency=SelfConsistency(enabled=True)
            ),
        )
    if pid == ProtocolId.DEBATE:
        return ProtocolConfig(
            id=pid, toggles=ReasoningToggles(debate=Debate(enabled=True))
        )
    if pid == ProtocolId.COT_DEBATE:
        return ProtocolConfig(
            id=pid,
            toggles=ReasoningToggles(cot=True, debate=Debate(enabled=True)),
        )
    if pid == ProtocolId.TOM:
        return ProtocolConfig(
            id=pid,
            toggles=ReasoningToggles(theory_of_mind=True),
        )

    if pid == ProtocolId.SC:
        return ProtocolConfig(
            id=pid,
            toggles=ReasoningToggles(
                self_consistency=SelfConsistency(enabled=True),
                temperature_chair=0.7,
                temperature_analyst=0.4,
            ),
        )

    if pid == ProtocolId.TOM_SC:
        return ProtocolConfig(
            id=pid,
            toggles=ReasoningToggles(
                theory_of_mind=True,
                self_consistency=SelfConsistency(enabled=True),
                temperature_chair=0.7,
                temperature_analyst=0.4,
            ),
        )

    if pid == ProtocolId.DEBATE_SC:
        return ProtocolConfig(
            id=pid,
            toggles=ReasoningToggles(
                debate=Debate(enabled=True),
                self_consistency=SelfConsistency(enabled=True),
                temperature_chair=0.7,
                temperature_analyst=0.4,
            ),
            stopping="epsilon_or_rmax",
        )

    if pid == ProtocolId.COT_TOM:
        return ProtocolConfig(
            id=pid,
            toggles=ReasoningToggles(cot=True, theory_of_mind=True),
        )

    if pid == ProtocolId.DEBATE_TOM:
        return ProtocolConfig(
            id=pid,
            toggles=ReasoningToggles(
                debate=Debate(enabled=True),
                theory_of_mind=True,
            ),
            stopping="epsilon_or_rmax",
        )

    if pid == ProtocolId.COT_DEBATE_TOM:
        return ProtocolConfig(
            id=pid,
            toggles=ReasoningToggles(
                cot=True,
                debate=Debate(enabled=True),
                theory_of_mind=True,
            ),
            stopping="epsilon_or_rmax",
        )

    if pid == ProtocolId.COT_DEBATE_SC:
        return ProtocolConfig(
            id=pid,
            toggles=ReasoningToggles(
                cot=True,
                debate=Debate(enabled=True),
                self_consistency=SelfConsistency(enabled=True),
                temperature_chair=0.7,
                temperature_analyst=0.4,
            ),
            stopping="epsilon_or_rmax",
        )

    if pid == ProtocolId.COT_TOM_SC:
        return ProtocolConfig(
            id=pid,
            toggles=ReasoningToggles(
                cot=True,
                theory_of_mind=True,
                self_consistency=SelfConsistency(enabled=True),
                temperature_chair=0.7,
                temperature_analyst=0.4,
            ),
        )

    if pid == ProtocolId.DEBATE_TOM_SC:
        return ProtocolConfig(
            id=pid,
            toggles=ReasoningToggles(
                debate=Debate(enabled=True),
                theory_of_mind=True,
                self_consistency=SelfConsistency(enabled=True),
                temperature_chair=0.7,
                temperature_analyst=0.4,
            ),
            stopping="epsilon_or_rmax",
        )

    if pid == ProtocolId.COT_DEBATE_TOM_SC:
        return ProtocolConfig(
            id=pid,
            toggles=ReasoningToggles(
                cot=True,
                debate=Debate(enabled=True),
                theory_of_mind=True,
                self_consistency=SelfConsistency(enabled=True),
                temperature_chair=0.7,
                temperature_analyst=0.4,
            ),
            stopping="epsilon_or_rmax",
        )

    if pid == ProtocolId.JUDGE_ONLY:
        return ProtocolConfig(id=pid, roles=["chair"], comm_pattern="flat")
    if pid == ProtocolId.FLAT:
        return ProtocolConfig(id=pid, comm_pattern="flat")
    if pid == ProtocolId.PARALLEL:
        return ProtocolConfig(id=pid, comm_pattern="parallel")
    if pid == ProtocolId.ITERATIVE_REFINEMENT:
        return ProtocolConfig(id=pid, comm_pattern="iterative_refinement")
    if pid == ProtocolId.VOTING:
        return ProtocolConfig(id=pid, comm_pattern="voting")
    if pid == ProtocolId.HYBRID:
        return ProtocolConfig(id=pid, comm_pattern="hybrid")
    if pid == ProtocolId.TWO_AGENT:
        return ProtocolConfig(
            id=pid,
            roles=["fundamental", "chair"],
            comm_pattern="flat",
        )
    raise ValueError(f"Unsupported ProtocolId: {pid}")
