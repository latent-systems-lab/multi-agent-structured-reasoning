"""Scenario analysis agent."""

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
from core.fallbacks import (
    fallback_scenario,
    normalize_long_only as _normalize_long_only,
    normalize_weights as _normalize_weights,
)
from utils.logging import get_logger
from utils.heuristic_parse import parse_weights_from_text, parse_confidence_from_text

if TYPE_CHECKING:  # pragma: no cover
    from core.schemas import GraphState


logger = get_logger(__name__)


def create(toggles: ReasoningToggles) -> Callable[["GraphState"], "GraphState"]:
    """Return a node function using ``toggles`` for reasoning instructions."""

    def node(state: "GraphState") -> "GraphState":
        if state.flags.get("scenario_done"):
            return state

        inputs = getattr(state, "inputs", None)
        if inputs is None or getattr(inputs, "market_context", None) is None:
            return state

        @tool  # type: ignore[misc]
        def macro_context() -> Dict[str, float]:
            """Return current macro context values."""

            return cast(Dict[str, float], inputs.market_context)

        @tool  # type: ignore[misc]
        def beta(symbol: str) -> float:
            """Return beta feature for ``symbol``."""

            features = cast(Dict[str, Dict[str, float]], inputs.features)
            return float(features.get(symbol, {}).get("beta", 0.0))

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
            "scenario",
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
        enable_tools = (
            os.getenv("ENABLE_TOOLS", "1").lower() not in {"0", "false"}
        )
        tools = [macro_context, beta] if enable_tools else []

        use_llm = os.getenv("ENABLE_LLM", "1").lower() not in {"0", "false"}
        recursion_limit = int(os.getenv("SCENARIO_RECURSION_LIMIT", "25"))
        optimize_cash = os.getenv("OPTIMIZE_CASH", "0") not in {"0", "false"}
        feedback = getattr(state, "flags", {}).get("feedback")

        if use_llm:
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
                symbols_str = ", ".join(symbols)
                
                # build the message list and INCLUDE feedback
                messages = []
                if feedback:
                    messages.append(HumanMessage(content=f"Chair feedback (use this to refine your new proposal): {feedback}"))
                messages.append(HumanMessage(
                    content=(
                        f"Use macro context and betas to suggest scenario adjustments. "
                        f"{symbols_str} and suggest weight adjustments."
                    )
                ))
                result = agent(
                    {
                        "messages": messages
                    },
                    config={"recursion_limit": recursion_limit},
                )
                universe = cast(List[str], state.inputs.universe)
                if mode == "off":
                    text = str(result)
                    raw_w = parse_weights_from_text(text)
                    weights = _normalize_long_only(raw_w, universe)
                    cash = float(raw_w.get("CASH", 0.0))
                    total = sum(weights.values()) + cash
                    if total > 0 and cash > 0:
                        weights = {k: v / total for k, v in weights.items()}
                        weights = {**weights, "CASH": cash / total}
                    confidence = parse_confidence_from_text(text) or 0.5
                    proposal = AnalystProposal(
                        weights=weights,
                        claim=text,
                        evidence=[],
                        risk_flags=[],
                        rationale=text,
                        confidence=confidence,
                        assumptions=[],
                        data_refs=[],
                    )
                elif mode == "weights_only":
                    mini = result
                    weights = _normalize_weights(mini.weights, universe)
                    cash = float(mini.weights.get("CASH", 0.0))
                    total = sum(weights.values()) + cash
                    if total > 0 and cash > 0:
                        weights = {k: v / total for k, v in weights.items()}
                        weights = {**weights, "CASH": cash / total}
                    proposal = AnalystProposal(
                        weights=weights,
                        claim=mini.raw_text or "Weights-only proposal",
                        evidence=[],
                        risk_flags=[],
                        rationale=mini.raw_text or "Weights-only proposal",
                        confidence=mini.confidence or 0.5,
                        assumptions=[],
                        data_refs=[],
                    )
                else:
                    proposal = AnalystProposal.model_validate(result)
                    weights = _normalize_weights(proposal.weights, universe)
                    cash = float(proposal.weights.get("CASH", 0.0))
                    total = sum(weights.values()) + cash
                    if total <= 0:
                        raise ValueError("No positive weights")
                    if cash > 0:
                        weights = {k: v / total for k, v in weights.items()}
                        weights = {**weights, "CASH": cash / total}
                    proposal = proposal.model_copy(update={"weights": weights})
                logger.info("scenario_path", path="llm")
            except Exception as exc:
                logger.exception(
                    "Scenario agent failed; falling back to heuristic", exc_info=exc
                )
                proposal = fallback_scenario(
                    cast(Dict[str, float], inputs.market_context),
                    cast(Dict[str, Dict[str, float]], inputs.features),
                    cast(List[str], inputs.universe),
                    optimize_cash,
                )
                logger.info("scenario_path", path="fallback")
        else:
            proposal = fallback_scenario(
                cast(Dict[str, float], inputs.market_context),
                cast(Dict[str, Dict[str, float]], inputs.features),
                cast(List[str], inputs.universe),
                optimize_cash,
            )
            logger.info("scenario_path", path="deterministic")

        state.proposals = {**state.proposals, "scenario": proposal}
        state.beliefs = {**state.beliefs, "scenario": proposal.assumptions}
        state.flags["scenario_done"] = True
        try:
            inputs.features = {}  # type: ignore[attr-defined]
            inputs.market_context = {}  # type: ignore[attr-defined]
        except Exception:
            pass
        return state

    return node


def run(state: "GraphState") -> "GraphState":
    """Backward compatibility wrapper using default ``ReasoningToggles``."""

    return create(ReasoningToggles())(state)
