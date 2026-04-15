import sys
from types import ModuleType

from core.graph import build_graph
from core.protocols import ProtocolConfig, ProtocolId


def _stub_chair() -> ModuleType:
    def run(_: object):
        def node(state):
            return state
        return node
    m = ModuleType("agents.chair")
    m.run = run  # type: ignore[attr-defined]
    return m


def test_build_graph_has_checkpointer(monkeypatch):
    monkeypatch.setitem(sys.modules, "agents.chair", _stub_chair())
    cfg = ProtocolConfig(id=ProtocolId.ONE_SHOT, roles=["chair"], comm_pattern="hierarchical")
    g = build_graph(cfg)
    from langgraph.checkpoint.memory import InMemorySaver
    assert isinstance(g.checkpointer, InMemorySaver)
