"""Technical analyst agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple, Callable, cast
import os

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from core.schemas import AnalystProposal, MinimalProposal
from core.prompts import analyst_prompt
from core.protocols import ReasoningToggles
from core.react_factory import create_react_agent
from core.gemini_client import get_gemini_llm
from core.fallbacks import (
    fallback_technical,
    momentum as _momentum,
    normalize_long_only as _normalize_long_only,
    normalize_weights as _normalize_weights,
)
from utils.logging import get_logger
from utils.heuristic_parse import parse_weights_from_text, parse_confidence_from_text

if TYPE_CHECKING:  # pragma: no cover
    from core.schemas import GraphState


logger = get_logger(__name__)


class PricePoint(BaseModel):
    date: str
    price: float


def create(toggles: ReasoningToggles) -> Callable[["GraphState"], "GraphState"]:
    """Return a node function using ``toggles`` for reasoning instructions."""

    def node(state: "GraphState") -> "GraphState":
        if state.flags.get("technical_done"):
            return state

        inputs = getattr(state, "inputs", None)
        if inputs is None or getattr(inputs, "prices_window", None) is None:
            return state

        prices_window = cast(Dict[str, List[Tuple[str, float]]], inputs.prices_window)
        universe = cast(List[str], inputs.universe)

        call_budget = int(os.getenv("TECHNICAL_CALL_BUDGET", "1"))
        recursion_limit = int(os.getenv("TECHNICAL_RECURSION_LIMIT", "50"))
        momentum_calls = {"momentum": 0, "fetch_prices": 0, "momentum_batch": 0}

        @tool  # type: ignore[misc]
        def momentum_batch(symbols: List[str]) -> Dict[str, float]:
            """Compute simple momentum for each symbol in one batch call.

            Args:
                symbols: list of tickers from the investment universe.

            Returns:
                Mapping {symbol -> momentum} using (last - first)/first over the window.
            """
            momentum_calls["momentum_batch"] += 1
            if momentum_calls["momentum_batch"] > call_budget:
                raise RuntimeError("technical momentum_batch call budget exceeded")

            out: Dict[str, float] = {}
            for sym in symbols:
                series = prices_window.get(sym, [])
                out[sym] = _momentum(series)
            logger.info(
                "technical_tool_call", tool="momentum_batch", symbols=len(symbols)
            )
            return out

        @tool  # type: ignore[misc]
        def fetch_prices(symbol: str) -> List[PricePoint]:
            """Return price window for ``symbol`` as list of {date, price}."""
            momentum_calls["fetch_prices"] += 1
            if momentum_calls["fetch_prices"] > call_budget:
                raise RuntimeError("technical fetch_prices call budget exceeded")
            series = prices_window.get(symbol, cast(List[Tuple[str, float]], []))
            logger.info("technical_tool_call", tool="fetch_prices", symbol=symbol)
            return [PricePoint(date=d, price=float(p)) for d, p in series]

        @tool  # type: ignore[misc]
        def momentum(prices: List[PricePoint]) -> float:
            """Compute simple momentum from a price series."""
            momentum_calls["momentum"] += 1
            if momentum_calls["momentum"] > call_budget:
                raise RuntimeError("technical momentum call budget exceeded")
            tuples: List[Tuple[str, float]] = [
                (pt.date, float(pt.price)) for pt in prices
            ]
            val = _momentum(tuples)
            logger.info(
                "technical_tool_call", tool="momentum", series=len(tuples), value=val
            )
            return val

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
            "technical",
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
            and call_budget > 0
        )
        # Put the batch tool first to make it the most attractive option
        tools = [momentum_batch, fetch_prices, momentum] if enable_tools else []

        # Build a strong, explicit task instruction to avoid endless loops
        symbols_str = ", ".join(universe)
        user_instruction = (
            "You are the TECHNICAL analyst. Use price trend momentum to propose portfolio "
            "weights for the universe. Follow this plan strictly:\n"
            "1) Prefer a single call to `momentum_batch` with the full universe "
            f"[{symbols_str}] to get all momenta at once. If unavailable, at most ONE "
            "call per symbol to `fetch_prices` and at most ONE `momentum` per symbol.\n"
            "2) Use raw momentum as signed scores (retain negatives). If all scores sum to 0, "
            "use equal weights across the universe.\n"
            "3) Return a valid `AnalystProposal` JSON NOW — do not iterate further.\n"
            "Requirements:\n"
            "- Weights must be only from the provided universe plus optional 'CASH'.\n"
            "- Normalize to sum EXACTLY 1.0 without clamping negatives (preserve explicit CASH if you include it).\n"
            "- Include a concise claim, bullet evidence from the computed momenta, "
            "any risk flags, a short rationale, and a confidence in [0,1]."
        )

        use_llm = os.getenv("ENABLE_LLM", "1").lower() not in {"0", "false"}
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
                # build the message list and INCLUDE feedback
                messages = []
                if feedback:
                    messages.append(HumanMessage(content=f"Chair feedback (use this to refine your new proposal): {feedback}"))
                messages.append(HumanMessage(content=user_instruction))
                result = agent(
                    {
                        "messages": messages
                    },
                    config={"recursion_limit": recursion_limit},
                )

                if mode == "off":
                    text = str(result)
                    raw_w = parse_weights_from_text(text)
                    weights = _normalize_long_only(raw_w, universe)
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
                    proposal = result
                    norm_w = _normalize_weights(proposal.weights, universe)
                    if norm_w != proposal.weights:
                        proposal = proposal.model_copy(update={"weights": norm_w})

                logger.info("technical_path", path="llm")

            except Exception as exc:
                logger.exception(
                    "Technical agent failed; falling back to heuristic", exc_info=exc
                )
                proposal = fallback_technical(prices_window, universe, optimize_cash)
                logger.info("technical_path", path="fallback")
        else:
            proposal = fallback_technical(prices_window, universe, optimize_cash)
            logger.info("technical_path", path="deterministic")

        state.proposals = {**state.proposals, "technical": proposal}
        state.beliefs = {**state.beliefs, "technical": proposal.assumptions}
        state.flags["technical_done"] = True
        return state

    return node


def run(state: "GraphState") -> "GraphState":
    """Backward compatibility wrapper using default ``ReasoningToggles``."""
    return create(ReasoningToggles())(state)
