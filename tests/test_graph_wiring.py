import sys
from types import ModuleType, SimpleNamespace

import pytest

from core.protocols import ProtocolConfig, ProtocolId
from core.graph import build_graph


class RecorderGraph:
    """Minimal StateGraph test double recording wiring calls."""

    def __init__(self, state_spec):  # pragma: no cover - used only in tests
        self.edges: list[tuple[str, str]] = []
        self.conditional_edges: list[tuple[str, dict[str, str]]] = []

    def add_node(self, name, node):  # pragma: no cover - not asserted
        pass

    def add_edge(self, start, end):
        self.edges.append((start, end))

    def add_conditional_edges(self, name, cond, mapping):
        self.conditional_edges.append((name, mapping))

    def compile(self, **_):  # accept arbitrary kwargs like checkpointer
        return self


@pytest.fixture(autouse=True)
def stub_langgraph(monkeypatch):
    """Inject a stub ``langgraph.graph`` module."""

    graph_mod = ModuleType("langgraph.graph")
    graph_mod.StateGraph = RecorderGraph
    graph_mod.START = "START"
    graph_mod.END = "END"

    pkg = ModuleType("langgraph")
    pkg.graph = graph_mod

    monkeypatch.setitem(sys.modules, "langgraph", pkg)
    monkeypatch.setitem(sys.modules, "langgraph.graph", graph_mod)


@pytest.fixture
def stub_agents(monkeypatch):
    """Replace ``import_module`` to supply lightweight agent stubs."""

    def fake_import(name):
        role = name.split(".")[-1]
        if role == "chair":
            def run(toggles):
                def node(state):
                    return state
                return node
            return SimpleNamespace(run=run)
        else:
            def run(state):
                return state
            return SimpleNamespace(run=run)

    import core.graph as graph

    monkeypatch.setattr(graph, "import_module", fake_import)


def _expected_hierarchical(START, END):
    return [
        (START, "chair"),
        ("chair", "risk"),
        ("risk", END),
    ]


def _expected_round_table(START, END):
    return [
        (START, "chair"),
        ("chair", "risk"),
        ("risk", END),
    ]


def _expected_flat(START, END):
    return [
        (START, "chair"),
        ("chair", "risk"),
        ("risk", END),
    ]


def _expected_hybrid(START, END):
    return [
        (START, "chair"),
        ("chair", "risk"),
        ("risk", END),
    ]


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("hierarchical", _expected_hierarchical),
        ("round_table", _expected_round_table),
        ("flat", _expected_flat),
        ("parallel", _expected_flat),
        ("iterative_refinement", _expected_round_table),
        ("voting", _expected_flat),
        ("hybrid", _expected_hybrid),
    ],
)
def test_comm_patterns_wiring(pattern, expected, stub_agents):
    cfg = ProtocolConfig(
        id=ProtocolId.ONE_SHOT,
        roles=["fundamental", "technical", "risk", "chair"],
        comm_pattern=pattern,
    )

    g = build_graph(cfg)
    from langgraph.graph import START, END

    assert g.edges == expected(START, END)
    assert g.conditional_edges == []
