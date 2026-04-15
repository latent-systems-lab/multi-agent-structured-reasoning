"""Simple in-memory vector store.

Stores vectors alongside arbitrary payloads and supports cosine similarity
search. NumPy is used by default for similarity calculations. If FAISS is
installed at runtime, it will be leveraged for efficient similarity search.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import faiss

    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover - faiss is optional
    faiss = None
    _FAISS_AVAILABLE = False


def _normalize(v: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return ``v`` normalized to unit length.

    Parameters
    ----------
    v:
        Vector to normalize.
    """

    v = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm == 0:
        return v
    return v / norm


class MemoryStore:
    """A minimal vector store backed by NumPy/FAISS."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim
        self.vectors = np.empty((0, dim), dtype=np.float32) if dim else np.empty((0, 0), dtype=np.float32)
        self.payloads: List[Any] = []
        self._index = None
        if _FAISS_AVAILABLE and dim:
            self._index = faiss.IndexFlatIP(dim)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.payloads)

    def add(self, vector: Sequence[float] | np.ndarray, payload: Any) -> None:
        """Add a vector with associated payload to the store."""

        vec = _normalize(np.asarray(vector, dtype=np.float32))
        if self.dim is None:
            self.dim = vec.shape[0]
            self.vectors = np.empty((0, self.dim), dtype=np.float32)
            if _FAISS_AVAILABLE:
                self._index = faiss.IndexFlatIP(self.dim)
        elif vec.shape[0] != self.dim:
            msg = f"Vector has dimension {vec.shape[0]} but store expects {self.dim}"
            raise ValueError(msg)

        self.vectors = np.vstack([self.vectors, vec])
        self.payloads.append(payload)
        if self._index is not None:
            self._index.add(vec.reshape(1, -1))

    def top_k(self, query: Sequence[float] | np.ndarray, k: int = 5) -> List[Tuple[Any, float]]:
        """Return the ``k`` payloads with highest cosine similarity."""

        if len(self.payloads) == 0:
            return []

        q = _normalize(np.asarray(query, dtype=np.float32))
        if self._index is not None:
            scores, idxs = self._index.search(q.reshape(1, -1), k)
            idxs = idxs[0]
            scores = scores[0]
        else:
            scores = self.vectors @ q
            idxs = np.argsort(scores)[::-1][:k]
            scores = scores[idxs]

        results: List[Tuple[Any, float]] = []
        for i, score in zip(idxs, scores):
            if i == -1:
                continue
            results.append((self.payloads[int(i)], float(score)))
        return results


def encode_state(s_t: dict[str, float]) -> np.ndarray:
    """Encode a state dictionary into a deterministic vector.

    Keys are sorted alphabetically; missing values (for expected keys not
    present) are filled with ``0`` ensuring consistent ordering when the same
    set of keys is used across states.
    """

    keys = sorted(s_t.keys())
    return np.array([s_t.get(k, 0.0) for k in keys], dtype=np.float32)

