"""Risk evaluation agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple, Callable, Union, cast, Any
import os

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from agents.chair import _utility as utility, _to_decision
from core.schemas import RiskAssessment, _coerce_weights_arg
from core.prompts import analyst_prompt
from core.protocols import ReasoningToggles
from core.react_factory import create_react_agent
from core.gemini_client import get_gemini_llm
from core.fallbacks import fallback_risk
from utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from core.schemas import GraphState


logger = get_logger(__name__)


def create(toggles: ReasoningToggles) -> Callable[["GraphState"], Dict[str, Any]]:
    """Return a node function using ``toggles`` for reasoning instructions."""

    def node(state: "GraphState") -> Dict[str, Any]:
        if state.flags.get("risk_done"):
            return {}
        if not state.chair_candidates:
            return {}

        inputs = getattr(state, "inputs", None)
        if inputs is None or getattr(inputs, "prices_window", None) is None:
            return {}

        cand = state.chair_candidates[-1]
        prices_window = cast(Dict[str, List[Tuple[str, float]]], inputs.prices_window)
        universe = cast(List[str], inputs.universe)
        stress_scenarios = cast(Dict[str, float] | None, inputs.stress_scenarios)

        @tool  # type: ignore[misc]
        def evaluate_risk(weights: Union[dict[str, float], str]) -> RiskAssessment:
            """Return risk assessment for ``weights`` using historical data."""
            w = _coerce_weights_arg(weights)
            return fallback_risk(w, prices_window, universe, stress_scenarios)

        prompt = analyst_prompt(
            "risk",
            toggles,
            RiskAssessment.model_json_schema(),
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
        tools = [evaluate_risk] if enable_tools else []

        mode = getattr(toggles, "structured_mode", "strict")
        schema = RiskAssessment if mode != "off" else None
        use_llm = (
            os.getenv("ENABLE_LLM", "1").lower() not in {"0", "false"}
            and schema is not None
        )
        recursion_limit = int(os.getenv("RISK_RECURSION_LIMIT", "25"))
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
                    
                # build the message list and INCLUDE feedback
                messages = []
                if feedback:
                    messages.append(HumanMessage(content=f"Chair feedback (use this to refine your new proposal): {feedback}"))
                messages.append(HumanMessage(
                    content=f"Assess portfolio risk for the given weights. {cand.weights}"
                ))

                result = agent(
                    {
                        "messages": messages
                    },
                    config={"recursion_limit": recursion_limit},
                )
                report = RiskAssessment.model_validate(result)
                logger.info("risk_path", path="llm")
            except Exception as exc:
                logger.exception(
                    "Risk agent failed; using deterministic fallback", exc_info=exc
                )
                report = RiskAssessment.model_validate(
                    fallback_risk(cand.weights, prices_window, universe, stress_scenarios)
                )
                logger.info("risk_path", path="fallback")
        else:
            report = RiskAssessment.model_validate(
                fallback_risk(cand.weights, prices_window, universe, stress_scenarios)
            )
            logger.info("risk_path", path="deterministic")

        cand.risk = report
        cand.utility = utility(cand)
        decision = _to_decision(
            cand,
            state,
            rounds=len(state.chair_candidates),
            toggles=toggles,
        )
        state.chair_candidates[-1] = cand
        state.flags["risk_done"] = True
        try:
            state.inputs = None  # type: ignore[assignment]
        except Exception:
            pass
        return {
            "decision": decision,
            "flags": state.flags,
        }

    return node


def run(state: "GraphState") -> Dict[str, Any]:
    """Backward compatibility wrapper using default ``ReasoningToggles``."""

    return create(ReasoningToggles())(state)
