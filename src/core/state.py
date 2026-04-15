"""Shared state type alias for LangGraph."""

from __future__ import annotations

from core.schemas import GraphState

# ``StateSpec`` is imported by ``langgraph`` builders to annotate the graph state.
StateSpec = GraphState
