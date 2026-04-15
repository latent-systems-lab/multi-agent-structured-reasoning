"""Tracing utilities for LangChain/LangGraph agents.

Provides a lightweight callback handler that logs key lifecycle events
through the project's structlog logger. This makes it easy to see what
the ReAct agent is doing (LLM calls, tool invocations, errors) without
changing agent code elsewhere.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

from .logging import get_logger


def _summarize(value: Any, limit: int = 300) -> str:
    """Return a short one-line summary of ``value`` for logs."""

    try:
        text = str(value)
    except Exception:
        text = repr(value)
    text = text.replace("\n", " ")
    return (text[: limit - 1] + "…") if len(text) > limit else text


class StructlogCallbackHandler(BaseCallbackHandler):
    """Minimal callback handler that logs LLM/tool events."""

    def __init__(self, name: str = __name__):
        self.logger = get_logger(name)

    # LLM events
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:  # type: ignore[override]
        if isinstance(serialized, dict):
            model = serialized.get("kwargs", {}).get("model") or serialized.get("name")
        else:
            model = None
        self.logger.info(
            "llm_start",
            model=model,
            prompts=len(prompts),
            preview=_summarize(prompts[0]) if prompts else None,
        )

    def on_llm_end(self, response, **kwargs: Any) -> None:  # type: ignore[override]
        try:
            generations = response.generations
            text = generations[0][0].text if generations and generations[0] else None
        except Exception:
            text = None
        self.logger.info("llm_end", preview=_summarize(text) if text else None)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:  # type: ignore[override]
        self.logger.error("llm_error", error=str(error))

    # Tool events
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:  # type: ignore[override]
        tool = serialized.get("name") if isinstance(serialized, dict) else None
        self.logger.info(
            "tool_start",
            tool=tool,
            input=_summarize(input_str),
        )

    def on_tool_end(self, output: str, **kwargs: Any) -> None:  # type: ignore[override]
        self.logger.info("tool_end", output=_summarize(output))

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:  # type: ignore[override]
        self.logger.error("tool_error", error=str(error))

    # Chain/graph events (best-effort)
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:  # type: ignore[override]
        chain_id = serialized.get("id") if isinstance(serialized, dict) else None
        self.logger.info("chain_start", chain=chain_id)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:  # type: ignore[override]
        keys = list(outputs.keys()) if isinstance(outputs, dict) else None
        self.logger.info("chain_end", keys=keys)

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:  # type: ignore[override]
        self.logger.error("chain_error", error=str(error))

