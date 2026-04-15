"""Graph construction utilities."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Dict

from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from core.protocols import ProtocolConfig
from core.schemas import GraphState
from core.state import StateSpec
from utils.logging import get_logger

logger = get_logger(__name__)
if TYPE_CHECKING:  # pragma: no cover
    from langgraph.graph import StateGraph
else:  # pragma: no cover
    StateGraph = object  # type: ignore


def _no_inputs(node_fn: Callable):
    """Ensure graph nodes never write to the immutable input channel."""

    def _wrapped(state):
        out = node_fn(state)
        if isinstance(out, GraphState):
            try:
                fields = type(out).model_fields
                data = {
                    name: getattr(out, name)
                    for name in fields
                    if name != "inputs"
                    and getattr(out, name) is not None
                    and getattr(out, name) != fields[name].get_default()
                }
            except Exception as e:  # pragma: no cover
                logger.exception("Failed to exclude 'inputs'")
                raise RuntimeError(f"Failed to exclude 'inputs' from {out}: {e}")
            for k in list(data.keys()):
                attr = getattr(out, k)
                if isinstance(attr, BaseModel):
                    data[k] = attr
                elif isinstance(attr, (list, tuple)) and any(
                    isinstance(a, BaseModel) for a in attr
                ):
                    data[k] = attr
            out = data
        elif isinstance(out, dict) and "inputs" in out:
            out = dict(out)
            out.pop("inputs", None)
        return out

    return _wrapped


def build_graph(cfg: ProtocolConfig) -> StateGraph:
    """Build the LangGraph execution graph for a protocol configuration."""
    from langgraph.graph import END, START, StateGraph as _StateGraph
    from core.schemas import GraphState as _GS

    builder: _StateGraph[GraphState] = _StateGraph(StateSpec)

    roles = list(cfg.roles)
    has_risk = "risk" in roles
    analyst_roles = [r for r in roles if r not in {"chair", "risk"}]
    proposers: Dict[str, Callable[[Any, Any], Any]] = {}

    for role in analyst_roles:
        mod = import_module(f"agents.{role}")

        if hasattr(mod, "propose"):
            proposers[role] = getattr(mod, "propose")
            continue

        if hasattr(mod, "create"):
            node_fn = mod.create(cfg.toggles)
        elif hasattr(mod, "run"):
            node_fn = mod.run
        else:
            raise RuntimeError(f"Agent {role} exposes neither propose/create/run")

        def _make_adapter(r: str, node: Callable):
            def _propose(inputs, ctx) -> Any:
                st = _GS(
                    inputs=inputs,
                    proposals={},
                    beliefs={},
                    flags={"feedback": getattr(ctx, "feedback", "")},
                    chair_candidates=[],
                    decision=None,
                )
                res = node(st)
                if isinstance(res, dict):
                    proposals = res.get("proposals", getattr(st, "proposals", {}))
                else:
                    proposals = getattr(res, "proposals", getattr(st, "proposals", {}))
                if r not in proposals:
                    raise RuntimeError(f"Agent {r} did not produce a proposal")
                return proposals[r]

            return _propose

        proposers[role] = _make_adapter(role, node_fn)

    chair_mod = import_module("agents.chair")
    if hasattr(chair_mod, "run"):
        try:
            chair_node = chair_mod.run(
                cfg.toggles, proposers, comm_pattern=cfg.comm_pattern
            )
        except TypeError:
            chair_node = chair_mod.run(cfg.toggles)
    elif hasattr(chair_mod, "create"):
        chair_node = chair_mod.create(cfg.toggles, proposers)
    else:
        raise RuntimeError("agents.chair must expose run(...) or create(...)")

    builder.add_node("chair", _no_inputs(chair_node))

    if has_risk:
        risk_mod = import_module("agents.risk")
        risk_node = (
            risk_mod.create(cfg.toggles) if hasattr(risk_mod, "create") else risk_mod.run
        )
        builder.add_node("risk", _no_inputs(risk_node))

    builder.add_edge(START, "chair")
    builder.add_edge("chair", "risk" if has_risk else END)
    if has_risk:
        builder.add_edge("risk", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
