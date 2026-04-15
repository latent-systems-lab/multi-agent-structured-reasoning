import numpy as np

from core.memory import MemoryStore, encode_state


def test_add_increases_length():
    store = MemoryStore()
    store.add([1.0, 0.0], "a")
    store.add([0.0, 1.0], "b")
    assert len(store) == 2


def test_top_k_retrieval_returns_expected():
    store = MemoryStore()
    store.add([1.0, 0.0], "a")
    store.add([0.0, 1.0], "b")
    store.add([0.9, 0.1], "c")

    results = store.top_k([1.0, 0.0], k=2)
    payloads = [p for p, _ in results]
    assert payloads == ["a", "c"]


def test_encode_state_is_deterministic():
    s1 = {"b": 2.0, "a": 1.0}
    s2 = {"a": 1.0, "b": 2.0}
    np.testing.assert_array_equal(encode_state(s1), encode_state(s2))

