"""Wrapper for creating ReAct agents with optional structured output.

This module exposes :func:`create_react_agent` which constructs a LangGraph/
LangChain ReAct agent backed by Gemini. The returned callable can produce
structured outputs (Pydantic) or raw text depending on the ``structured_mode``
setting and includes bounded retries, request timeouts, and a robust
JSON-conversion fallback.

Env toggles
-----------
GEMINI_MODEL=models/gemini-2.5-flash-lite
GEMINI_MAX_RETRIES=1
GEMINI_TIMEOUT_S=20
LLM_MAX_RETRIES=1              # legacy/alt name, still honored
LLM_TEMPERATURE=0.0
GEMINI_API_KEY / GOOGLE_API_KEY
TRACE_REACT=1
"""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence, Type, List, Literal

from langgraph.prebuilt import create_react_agent as _create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from utils.logging import get_logger
from utils.tracing import StructlogCallbackHandler
from utils.helpers import _coerce_to_schema
from utils.token_logging import log_agent_usage

# -------- Defaults (overridable via env) ---------------------------------
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash-lite")
# Prefer GEMINI_MAX_RETRIES; fall back to LLM_MAX_RETRIES if provided.
DEFAULT_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", os.getenv("LLM_MAX_RETRIES", "1")))
DEFAULT_TIMEOUT_S = float(os.getenv("GEMINI_TIMEOUT_S", "20"))
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "20000"))
# ------------------------------------------------------------------------


def _build_llm_kwargs(
    model: str | None,
    client: Any | None,
    thinking_budget: int | None,
    include_thoughts: bool | None,
) -> dict[str, Any]:
    """Compose kwargs for ChatGoogleGenerativeAI with sane defaults."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    llm_kwargs: dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "max_retries": DEFAULT_MAX_RETRIES,
        "timeout": DEFAULT_TIMEOUT_S,
        "convert_system_message_to_human": True,
    }

    if LLM_MAX_OUTPUT_TOKENS:
        llm_kwargs["max_output_tokens"] = LLM_MAX_OUTPUT_TOKENS
    if api_key:
        llm_kwargs["api_key"] = api_key
    if thinking_budget is not None:
        llm_kwargs["thinking_budget"] = thinking_budget
    if include_thoughts is not None:
        llm_kwargs["include_thoughts"] = include_thoughts
    if client is not None:
        llm_kwargs["client"] = client

    return llm_kwargs


def create_react_agent(
    prompt: str,
    tools: Sequence[BaseTool],
    schema: Type[BaseModel] | None,
    *,
    structured_mode: Literal["strict", "weights_only", "off"] = "strict",
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
    thinking_budget: int | None = None,
    include_thoughts: bool | None = None,
    agent_name: str | None = None,
) -> Callable[..., Any]:
    """Return a callable ReAct agent optionally producing structured output.

    Parameters
    ----------
    prompt:
        System prompt that seeds the agent's reasoning.
    tools:
        Sequence of tools available to the agent.
    structured_mode:
        Controls structured output handling:
        ``"strict"`` enforces full JSON schemas,
        ``"weights_only"`` only validates weight formatting, and
        ``"off"`` returns raw model text without schema validation.
    model:
        Gemini model identifier (env GEMINI_MODEL overrides default).
    client:
        Optional pre-initialised Gemini client (reused when provided).
    thinking_budget:
        Gemini "thinking" token budget (-1 dynamic, 0 off, >0 explicit).
    include_thoughts:
        Whether to include returned thought summaries (if supported).
    agent_name:
        Identifier used for token usage logging.
    schema:
        Pydantic model defining the structured response.

    Returns
    -------
    Callable[[dict[str, Any]], Any]
        Callable that accepts a mapping with a ``"messages"`` key containing a
        list of :class:`~langchain_core.messages.BaseMessage` objects and returns
        either an instance of ``schema`` or raw text when ``structured_mode`` is
        ``"off"``.
    """
    if structured_mode != "off" and (
        not isinstance(schema, type) or not issubclass(schema, BaseModel)
    ):
        raise TypeError("schema must be a Pydantic BaseModel subclass")

    logger = get_logger(__name__)

    if os.getenv("AGENTS_LLM_DISABLED", "0").lower() in {"1", "true", "yes"}:
        def _disabled(_state: dict[str, Any], *, config: dict[str, Any] | None = None) -> BaseModel:
            raise RuntimeError("LLM disabled by env (AGENTS_LLM_DISABLED=1)")

        return _disabled

    llm_kwargs = _build_llm_kwargs(model, client, thinking_budget, include_thoughts)
    llm = ChatGoogleGenerativeAI(**llm_kwargs)

    system_msg = SystemMessage(content=prompt)
    if structured_mode != "off" and schema is not None:
        graph = _create_react_agent(
            model=llm, tools=tools, prompt=system_msg, response_format=schema
        )
    else:
        graph = _create_react_agent(model=llm, tools=tools, prompt=system_msg)

    def _run(
        state: dict[str, Any], *, config: dict[str, Any] | None = None
    ) -> Any:
        raw_messages = state.get("messages", [])
        messages: List[BaseMessage] = []
        for msg in raw_messages:
            if isinstance(msg, BaseMessage):
                messages.append(msg)
            else:
                messages.append(HumanMessage(content=str(msg)))
        state = {**state, "messages": messages}

        callbacks = []
        if os.getenv("TRACE_REACT", "0").lower() in {"1", "true", "yes"}:
            callbacks.append(StructlogCallbackHandler(name=__name__))
            logger.info(
                "react_trace_enabled",
                model=llm_kwargs.get("model"),
                tools=[getattr(t, "name", type(t).__name__) for t in tools],
            )

        try:
            cfg = dict(config or {})
            if callbacks:
                cfg.setdefault("callbacks", callbacks)
            if cfg:
                cfg.setdefault("recursion_limit", 25)
            result = graph.invoke(state, config=cfg) if cfg else graph.invoke(state)
        except Exception as e:
            logger.error("react_agent_invoke_error", error=str(e))
            raise

        try:
            msgs_for_logging = (
                result.get("messages", []) if isinstance(result, dict) else result
            )
            log_agent_usage(agent_name or "unknown", msgs_for_logging)
        except Exception as e:
            logger.warning("token_logging_failed", error=str(e))

        if structured_mode == "off" or schema is None:
            # Return the raw string from the final message when structured JSON
            # handling is disabled. ``graph.invoke`` may return either a list of
            # messages or a mapping containing them; in both cases extract the
            # last message's content.
            if isinstance(result, dict):
                msgs = result.get("messages")
                if msgs:
                    last = msgs[-1]
                    return getattr(last, "content", str(last))
            if isinstance(result, list) and result:
                last = result[-1]
                return getattr(last, "content", str(last))
            # For simple string or message objects
            return getattr(result, "content", str(result))

        try:
            return _coerce_to_schema(result, schema, logger)
        except Exception as exc:
            logger.warning(
                "structured_output_missing",
                error=str(exc),
                result_type=type(result).__name__,
            )
            try:
                parser = ChatGoogleGenerativeAI(**llm_kwargs).with_structured_output(schema)
                fallback_prompt = (
                    "Convert the following model output into a VALID JSON object that "
                    f"conforms to the schema {schema.__name__}. "
                    "If any field like `weights` appears as a string (e.g., 'SPX: 0.8, CASH: 0.2'), "
                    "convert it into a JSON object mapping symbols to numbers "
                    "(e.g., {\"SPX\": 0.8, \"CASH\": 0.2}).\n\n"
                    f"OUTPUT:\n{result}"
                )
                new_results = parser.invoke(
                    [("system", "You are a strict JSON converter."), ("human", fallback_prompt)]
                )
                return new_results
            except Exception as parse_err:
                logger.error("parsing_llm_failed", error=str(parse_err))
                raise

    return _run

