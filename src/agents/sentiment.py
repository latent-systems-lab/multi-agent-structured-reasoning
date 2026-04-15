"""Sentiment analyst agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Callable, cast
import os

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from core.schemas import AnalystProposal, MinimalProposal
from core.prompts import analyst_prompt
from core.protocols import ReasoningToggles
from core.react_factory import create_react_agent
from core.gemini_client import get_gemini_llm
from core.fallbacks import (
    fallback_sentiment,
    _risk_signal_from_vix,
    normalize_long_only,
    normalize_weights,
)
from utils.logging import get_logger
from utils.heuristic_parse import parse_weights_from_text, parse_confidence_from_text

if TYPE_CHECKING:  # pragma: no cover
    from core.schemas import GraphState


logger = get_logger(__name__)


def create(toggles: ReasoningToggles) -> Callable[["GraphState"], "GraphState"]:
    """Return a node function using ``toggles`` for reasoning instructions."""

    def node(state: "GraphState") -> "GraphState":
        if state.flags.get("sentiment_done"):
            return state

        inputs = getattr(state, "inputs", None)
        if inputs is None or getattr(inputs, "market_context", None) is None:
            return state

        @tool  # type: ignore[misc]
        def fetch_vix() -> float:
            """Return the VIX index close for the current date from market_context."""
            return float(cast(Dict[str, float], inputs.market_context).get("vix", 20.0))

        @tool  # type: ignore[misc]
        def vix_signal(vix: float) -> float:
            """Compute risk appetite signal from VIX level (higher VIX = risk-off)."""
            return _risk_signal_from_vix(vix)

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
            "sentiment",
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
        tools = [fetch_vix, vix_signal] if enable_tools else []

        use_llm = os.getenv("ENABLE_LLM", "1").lower() not in {"0", "false"}
        recursion_limit = int(os.getenv("SENTIMENT_RECURSION_LIMIT", "25"))
        optimize_cash = os.getenv("OPTIMIZE_CASH", "0") not in {"0", "false"}
        feedback = getattr(state, "flags", {}).get("feedback")

        if use_llm:
            symbols = cast(List[str], inputs.universe)
            symbols_str = ", ".join(symbols)
            vix_val = float(cast(Dict[str, float], inputs.market_context).get("vix", 20.0))
            headlines_map = cast(Dict[str, List[str]], getattr(inputs, "headlines", {}))
            headlines_lines: List[str] = []
            for sym in symbols:
                raw_headlines = [h.strip() for h in headlines_map.get(sym, []) if h and h.strip()]
                if not raw_headlines:
                    continue
                truncated = "; ".join(raw_headlines[:3])
                if len(truncated) > 500:
                    truncated = f"{truncated[:497]}..."
                headlines_lines.append(f"{sym}: {truncated}")
            if headlines_lines:
                # keep prompt succinct by capping symbols with headlines
                max_assets = 6
                shown_lines = headlines_lines[:max_assets]
                if len(headlines_lines) > max_assets:
                    shown_lines.append("(additional headlines omitted for brevity)")
                headlines_text = "\n".join(f"- {line}" for line in shown_lines)
                headlines_prompt = (
                    "Recent headlines (consider their sentiment and implications):\n"
                    f"{headlines_text}"
                )
            else:
                headlines_prompt = "No recent headlines were provided."
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
                message_content = (
                    f"Use the VIX index (close={vix_val:.2f}) to assess market risk appetite and"
                    f" suggest weight adjustments across: {symbols_str}."
                    "\nIncorporate any relevant sentiment insights from the provided headlines."
                )
                if headlines_prompt:
                    message_content = f"{message_content}\n\n{headlines_prompt}"
                messages.append(HumanMessage(content=message_content))
                result = agent(
                    {
                        "messages": messages
                    },
                    config={"recursion_limit": recursion_limit},
                )
                if mode == "off":
                    text = str(result)
                    raw_w = parse_weights_from_text(text)
                    weights = normalize_long_only(raw_w, symbols)
                    if optimize_cash and "CASH" not in weights:
                        cash_resid = 1.0 - sum(weights.values())
                        if cash_resid > 0:
                            weights = {**weights, "CASH": cash_resid}
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
                    weights = normalize_weights(mini.weights, symbols)
                    if optimize_cash and "CASH" not in weights:
                        cash_resid = 1.0 - sum(weights.values())
                        if cash_resid > 0:
                            weights = {**weights, "CASH": cash_resid}
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
                    weights = normalize_weights(proposal.weights, symbols)
                    if optimize_cash and "CASH" not in weights:
                        cash_resid = 1.0 - sum(weights.values())
                        if cash_resid > 0:
                            weights = {**weights, "CASH": cash_resid}
                    proposal = proposal.model_copy(update={"weights": weights})
                logger.info("sentiment_path", path="llm")
            except Exception as exc:
                logger.exception(
                    "Sentiment agent failed; falling back to heuristic", exc_info=exc
                )
                proposal = fallback_sentiment(
                    cast(Dict[str, float], state.inputs.market_context),
                    symbols,
                    optimize_cash,
                )
                logger.info("sentiment_path", path="fallback")
        else:
            proposal = fallback_sentiment(
                cast(Dict[str, float], inputs.market_context),
                cast(List[str], inputs.universe),
                optimize_cash,
            )
            logger.info("sentiment_path", path="deterministic")

        state.proposals = {**state.proposals, "sentiment": proposal}
        state.beliefs = {**state.beliefs, "sentiment": proposal.assumptions}
        state.flags["sentiment_done"] = True
        try:
            inputs.headlines = {}  # type: ignore[attr-defined]
        except Exception:
            pass
        return state

    return node


def run(state: "GraphState") -> "GraphState":
    """Backward compatibility wrapper using default ``ReasoningToggles``."""

    return create(ReasoningToggles())(state)
