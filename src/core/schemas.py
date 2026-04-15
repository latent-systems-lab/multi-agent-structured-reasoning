"""Pydantic data models for agent I/O and graph state."""

from __future__ import annotations

from typing import Annotated, Dict, List, Optional, Tuple, Any, Union
import operator
import json

from langgraph.channels import BinaryOperatorAggregate

from pydantic import AliasChoices, BaseModel, Field, NonNegativeFloat, field_validator


def _coerce_weights_arg(values: Union[dict[str, float], str, Any]) -> dict[str, float]:
    """Accept dict or JSON string and return {str: float}.

    Any parsing error results in an empty dict rather than an exception.
    """

    def _convert_dict(d: dict[Any, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        try:
            for k, v in d.items():
                if v is not None:
                    out[str(k)] = float(v)
            return out
        except Exception:
            return {}

    if isinstance(values, dict):
        return _convert_dict(values)

    if isinstance(values, str):
        try:
            parsed = json.loads(values)
            if isinstance(parsed, dict):
                out = _convert_dict(parsed)
                if out:
                    return out
        except Exception:
            pass
        # also support "AAPL:0.1,TSLA:-0.05" style
        try:
            out: dict[str, float] = {}
            for part in values.split(","):
                if ":" in part:
                    k, v = part.split(":", 1)
                    out[k.strip()] = float(v.strip())
            if out:
                return out
        except Exception:
            pass
    return {}


Symbol = str
ISODate = str  # "YYYY-MM-DD"


class DataRef(BaseModel):
    source: str
    symbol: Optional[Symbol] = None
    as_of: ISODate
    hash: str  # content fingerprint for audit


class Experience(BaseModel):
    """Simple experience tuple for replay buffers."""

    obs: dict[str, Any] = Field(default_factory=dict)
    action: str = ""
    reward: float = 0.0
    next_obs: dict[str, Any] | None = None
    done: bool = False


class EpisodicSnippet(BaseModel):
    """Short snippet retrieved from episodic memory."""

    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class PeerPrediction(BaseModel):
    target: str  # analyst name you are predicting, e.g., "technical"
    predicted_weights: Dict[Symbol, float] = Field(default_factory=dict)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    summary: str = ""

    @field_validator("predicted_weights", mode="before")
    @classmethod
    def _coerce_predicted_weights(cls, v):
        return _coerce_weights_arg(v)


class MinimalProposal(BaseModel):
    weights: Dict[str, float]
    confidence: Optional[float] = None
    raw_text: Optional[str] = None

    @field_validator("weights", mode="before")
    @classmethod
    def _coerce_weights(cls, v: Any) -> dict[str, float]:
        """Coerce weight mappings via :func:`_coerce_weights_arg`.

        Any unparseable input yields an empty mapping rather than raising
        a validation error.
        """
        return _coerce_weights_arg(v)


class AnalystProposal(BaseModel):
    """Proposal from a specialist agent."""

    weights: dict[str, float] = Field(
        default_factory=dict,
        description="Absolute weights proposal over symbols (and possibly CASH) summing to 1.",
    )
    claim: str = Field(...)
    evidence: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    rationale: str = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    assumptions: list[str] = Field(default_factory=list)
    data_refs: list[DataRef] = Field(default_factory=list)
    mind_reading: list[PeerPrediction] = Field(
        default_factory=list,
        description="Optional predictions about other analysts' proposals (ToM).",
    )

    @field_validator("weights", mode="before")
    @classmethod
    def _coerce_weights(cls, v):
        """Coerce weights from JSON string or simple 'SYM: val, SYM: val' string."""
        if isinstance(v, str):
            # Try JSON first
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    return {str(k): float(val) for k, val in parsed.items()}
            except Exception:
                pass
            # Fallback: parse comma-separated "SYM: val" pairs
            items = [p.strip() for p in v.split(",") if p.strip()]
            out: dict[str, float] = {}
            for item in items:
                if ":" in item:
                    k, val = item.split(":", 1)
                    try:
                        out[k.strip()] = float(val.strip())
                    except Exception:
                        continue
            if out:
                return out
        return v


class RiskAssessment(BaseModel):
    """Portfolio risk evaluation of candidate weights."""

    gross: NonNegativeFloat
    net: float
    cvar_95: float
    max_drawdown_est: Optional[float] = None
    violations: List[str]


# Backwards compatibility alias
RiskReport = RiskAssessment


class ChairCandidate(BaseModel):
    """One candidate decision (e.g., one SC sample or one debate round)."""

    weights: Dict[Symbol, float] = Field(
        validation_alias=AliasChoices("weights", "weights_abs")
    )
    utility: float
    synthesis: str
    used_protocol: str
    supporting: Dict[str, AnalystProposal]
    risk: Optional[RiskAssessment] = None
    mind_loss: float = 0.0  # mean MSE across predictors (lower=better)
    base_expected_return: float = 0.0
    turnover: float = 0.0
    liquidity_penalty: float = 0.0
    consensus_score: float = 0.0
    scenario_alignment: float = 0.0


class ChairDecision(BaseModel):
    """Final decision for day t."""

    date: ISODate
    weights: Dict[Symbol, float] = Field(
        validation_alias=AliasChoices("weights", "weights_abs")
    )
    utility: float
    synthesis: str
    protocol_id: str
    rounds_taken: int
    sc_M: int
    token_in: int
    token_out: int
    latency_ms: int
    data_refs: List[DataRef]
    supporting: Dict[str, AnalystProposal] = Field(default_factory=dict)
    dissenting: List[str] = Field(default_factory=list)


class GraphInputs(BaseModel):
    """Inputs provided to the graph per day."""

    date: ISODate
    universe: List[Symbol]
    prices_window: Dict[Symbol, List[Tuple[ISODate, float]]]
    features: Dict[Symbol, Dict[str, float]]
    fundamentals: Dict[Symbol, Dict[str, float]]
    headlines: Dict[Symbol, List[str]]
    market_context: Dict[str, float]
    prev_weights: Dict[Symbol, float]
    stress_scenarios: Dict[str, float] = Field(default_factory=dict)


class GraphState(BaseModel):
    """LangGraph state type."""

    inputs: GraphInputs
    proposals: Annotated[
        Dict[str, AnalystProposal],
        BinaryOperatorAggregate(dict, operator.or_),
    ] = Field(default_factory=dict)
    chair_candidates: Annotated[
        List[ChairCandidate],
        BinaryOperatorAggregate(list, operator.add),
    ] = Field(default_factory=list)
    decision: Annotated[
        Optional[ChairDecision],
        BinaryOperatorAggregate(lambda: None, lambda _, new: new),
    ] = None
    beliefs: Annotated[
        Dict[str, List[str]],
        BinaryOperatorAggregate(dict, operator.or_),
    ] = Field(default_factory=dict)
    flags: Annotated[
        Dict[str, Any],
        BinaryOperatorAggregate(dict, operator.or_),
    ] = Field(
        default_factory=dict,
        description=(
            "Misc runtime flags. Reserved keys: "
            "'episodic_topk' (list[EpisodicSnippet]) and "
            "'replay_last' (Experience | None)."
        ),
    )
    chair_last_date: Optional[ISODate] = None
