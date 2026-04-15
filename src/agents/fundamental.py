"""Fundamental analyst agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Callable, List, cast
import os

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from core.schemas import AnalystProposal, MinimalProposal
from core.prompts import analyst_prompt
from core.protocols import ReasoningToggles
from core.react_factory import create_react_agent
from core.gemini_client import get_gemini_llm
from core.fallbacks import normalize_long_only, normalize_weights
from utils.logging import get_logger
from utils.heuristic_parse import parse_weights_from_text, parse_confidence_from_text

if TYPE_CHECKING:  # pragma: no cover
    from core.schemas import GraphState


logger = get_logger(__name__)


# Metric configuration: min/max are clipped ranges for normalisation, weight
# expresses relative importance and ``higher_is_better`` controls direction.
_METRIC_SPECS = {
    "PERatio": {"min": 2.0, "max": 250.0, "weight": 0.5, "higher_is_better": False},
    "ProfitMargin": {"min": 0.0, "max": 0.6, "weight": 0.25, "higher_is_better": True},
    "ReturnOnEquityTTM": {"min": 0.0, "max": 2.0, "weight": 0.15, "higher_is_better": True},
    "PEGRatio": {"min": 0.0, "max": 20.0, "weight": 0.1, "higher_is_better": False},
}


def _score_fundamentals(values: dict[str, float] | str) -> float:
    """Return a weighted score for fundamental metrics.

    Each supported metric is first normalised to ``[0, 1]`` using the
    configured min/max ranges. Metrics where a lower value is preferable
    (e.g. ``pe_ratio`` or ``debt_to_equity``) are inverted. The final score is
    a weighted sum of these normalised values.
    """
    

    total_weight = 0.0
    score = 0.0
    for name, spec in _METRIC_SPECS.items():
        if name not in values:
            continue
        raw = values[name]
        min_v = spec["min"]
        max_v = spec["max"]
        if max_v == min_v:
            norm = 0.0
        else:
            norm = (raw - min_v) / (max_v - min_v)
        # Clip to [0, 1]
        norm = max(0.0, min(1.0, norm))
        if not spec["higher_is_better"]:
            norm = 1.0 - norm
        weight = spec["weight"]
        score += norm * weight
        total_weight += weight
    return score / total_weight if total_weight else 0.0


def create(toggles: ReasoningToggles) -> Callable[["GraphState"], "GraphState"]:
    """Return a node function using ``toggles`` for reasoning instructions."""

    def node(state: "GraphState") -> "GraphState":
        inputs = getattr(state, "inputs", None)
        fundamentals = getattr(inputs, "fundamentals", None)
        if not fundamentals:
            return state

        @tool
        def fetch_fundamentals(symbol: str) -> Dict[str, float]:
            """Return fundamental features for ``symbol``."""

            fmap = cast(Dict[str, Dict[str, float]], fundamentals)
            fmap = fmap.get(symbol, cast(Dict[str, float], {}))
            return fmap

        mode = getattr(toggles, "structured_mode", "strict")
        if mode == "weights_only":
            schema = MinimalProposal
            schema_json = MinimalProposal.model_json_schema()
        elif mode == "off":
            schema = None
            schema_json = AnalystProposal.model_json_schema()
        else:
            schema = AnalystProposal
            schema_json = AnalystProposal.model_json_schema()

        prompt = analyst_prompt(
            "fundamental",
            toggles,
            schema_json,
            state.beliefs,
        )
        episodic = state.flags.get("episodic_topk", [])
        if episodic:
            newline = '\n'
            mem_lines = "\n".join(
                f"- {str(s).replace(newline, ' ').strip()}" for s in episodic
            )
            prompt = f"{prompt}\nEpisodic memory:\n{mem_lines}"
        replay = state.flags.get("replay_last")
        if replay:
            try:
                prompt = (
                    f"{prompt}\nRecent action recap: weights={replay.action}"
                    f" reward={float(getattr(replay, 'reward', 0.0)):.4f}"
                )
            except Exception:
                prompt = f"{prompt}\nRecent action recap available."
        enable_tools = os.getenv("ENABLE_TOOLS", "1") not in {"0", "false", "False"}
        tools = [fetch_fundamentals] if enable_tools else []

        # pull chair feedback (if any)
        feedback = getattr(state, "flags", {}).get("feedback")
        try:
            client = get_gemini_llm()
            agent_kwargs = {
                "structured_mode": mode,
                "client": client,
            }
            if toggles.cot:
                agent_kwargs.update({"thinking_budget": -1, "include_thoughts": True})
            agent = create_react_agent(prompt, tools, schema, **agent_kwargs)
            symbols = cast(List[str], inputs.universe)
            fmap = cast(Dict[str, Dict[str, float]], fundamentals)
            scores: Dict[str, float] = {
                sym: _score_fundamentals(fmap.get(sym, {})) - 0.5 for sym in symbols
            }
            symbols_str = ", ".join(symbols)
            
            # build the message list and INCLUDE feedback
            messages = []
            if feedback:
                messages.append(HumanMessage(content=f"Chair feedback (use this to refine your new proposal): {feedback}"))

            messages.append(HumanMessage(
                content=(
                    f"Analyse the universe fundamentals for symbols: {symbols_str} "
                    "and suggest weight adjustments. "
                    f"Precomputed scores: {scores}"
                )
            ))
            result = agent(
                {
                    "messages": messages
                },
                config={"recursion_limit": 25},
            )
            if mode == "off":
                text = str(result)
                raw_w = parse_weights_from_text(text)
                risky = {k: v for k, v in raw_w.items() if k != "CASH"}
                normalized_risky = normalize_long_only(risky, symbols)
                if os.getenv("OPTIMIZE_CASH", "0") not in {"0", "false", "False"}:
                    cash_weight = max(0.0, min(1.0, float(raw_w.get("CASH", 0.0))))
                    normalized = {k: v * (1.0 - cash_weight) for k, v in normalized_risky.items()}
                    normalized = {**normalized, "CASH": cash_weight}
                else:
                    normalized = normalized_risky
                confidence = parse_confidence_from_text(text) or 0.5
                proposal = AnalystProposal(
                    weights=normalized,
                    claim=text,
                    evidence=[],
                    risk_flags=[],
                    rationale=text,
                    confidence=confidence,
                    assumptions=[],
                    data_refs=[],
                )
            elif mode == "weights_only":
                proposal_obj = result
                raw = proposal_obj.weights
                risky = {k: v for k, v in raw.items() if k != "CASH"}
                normalized_risky = normalize_weights(risky, symbols)
                if os.getenv("OPTIMIZE_CASH", "0") not in {"0", "false", "False"}:
                    cash_weight = max(0.0, min(1.0, float(raw.get("CASH", 0.0))))
                    normalized = {k: v * (1.0 - cash_weight) for k, v in normalized_risky.items()}
                    normalized = {**normalized, "CASH": cash_weight}
                else:
                    normalized = normalized_risky
                proposal = AnalystProposal(
                    weights=normalized,
                    claim=proposal_obj.raw_text or "Weights-only proposal",
                    evidence=[],
                    risk_flags=[],
                    rationale=proposal_obj.raw_text or "Weights-only proposal",
                    confidence=proposal_obj.confidence or 0.5,
                    assumptions=[],
                    data_refs=[],
                )
            else:
                proposal = AnalystProposal.model_validate(result.model_dump())
                raw = proposal.weights
                risky = {k: v for k, v in raw.items() if k != "CASH"}
                normalized_risky = normalize_weights(risky, symbols)
                if os.getenv("OPTIMIZE_CASH", "0") not in {"0", "false", "False"}:
                    cash_weight = max(0.0, min(1.0, float(raw.get("CASH", 0.0))))
                    normalized = {k: v * (1.0 - cash_weight) for k, v in normalized_risky.items()}
                    normalized = {**normalized, "CASH": cash_weight}
                else:
                    normalized = normalized_risky
                proposal = proposal.model_copy(update={"weights": normalized})
            
        except Exception as exc:
            logger.exception("Fundamental agent failed; falling back to heuristic", exc_info=exc)
            fmap = cast(Dict[str, Dict[str, float]], fundamentals)
            universe = cast(List[str], inputs.universe)
            scores: Dict[str, float] = {
                sym: _score_fundamentals(fmap.get(sym, {})) - 0.5 for sym in universe
            }
            weights = normalize_weights(scores, universe)
            proposal = AnalystProposal(
                weights=weights,
                claim="Fundamental-aligned allocation",
                evidence=["Composite fundamental scores"],
                risk_flags=[],
                rationale="Allocate proportionally to fundamental scores (signed)",
                confidence=0.7,
                assumptions=[],
                data_refs=[],
            )

        state.proposals = {**state.proposals, "fundamental": proposal}
        state.beliefs = {**state.beliefs, "fundamental": proposal.assumptions}
        try:
            inputs.fundamentals = {}  # type: ignore[attr-defined]
        except Exception:
            pass
        return state

    return node


def run(state: "GraphState") -> "GraphState":
    """Backward compatibility wrapper using default ``ReasoningToggles``."""

    return create(ReasoningToggles())(state)
