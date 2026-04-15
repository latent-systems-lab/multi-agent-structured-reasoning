"""Validation helpers for agent outputs."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from utils.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from core.schemas import AnalystProposal, ChairCandidate, ChairDecision


def validate_proposal(p: AnalystProposal) -> AnalystProposal:
    """Validate an ``AnalystProposal``.

    Raises
    ------
    ValueError
        If any field is out of bounds or otherwise inconsistent.
    """

    # Proposed absolute weights should be within [-1, 1].
    for sym, w in p.weights.items():
        if not -1.0 <= w <= 1.0:
            raise ValueError(f"weights[{sym}]={w} outside [-1, 1]")

    # Claim must not be empty
    if not p.claim:
        raise ValueError("claim cannot be empty")

    # Evidence and risk flags entries must be non-empty strings
    if any(not e for e in p.evidence):
        raise ValueError("evidence entries cannot be empty")
    if any(not f for f in p.risk_flags):
        raise ValueError("risk_flags entries cannot be empty")

    # Data references must have required fields and, if referencing a symbol,
    # that symbol must appear in the proposal's weights.
    for ref in p.data_refs:
        if not (ref.source and ref.as_of and ref.hash):
            raise ValueError("DataRef fields cannot be empty")
        if ref.symbol is not None and ref.symbol not in p.weights:
            raise ValueError(
                f"DataRef symbol '{ref.symbol}' missing from weights"
            )

    return p


def validate_candidate(c: ChairCandidate, universe: list[str]) -> ChairCandidate:
    """Validate a ``ChairCandidate`` against the trading universe."""

    weights = c.weights
    optimize_cash = os.getenv("OPTIMIZE_CASH", "0") not in {"0", "false", "False"}
    allowed_extra = {"CASH"} if optimize_cash else set()
    extra = (set(weights) - allowed_extra) - set(universe)
    if extra:
        raise ValueError(f"Unknown symbols in candidate: {sorted(extra)}")
    missing = set(universe) - set(weights)
    if missing:
        raise ValueError(f"Candidate missing weights for: {sorted(missing)}")

    return c


def validate_decision(
    d: ChairDecision, universe: list[str], gross_cap: float, pos_cap: float
) -> ChairDecision:
    """Validate the final ``ChairDecision`` under position limits."""

    weights = d.weights

    # Universe coverage (allow optional CASH when enabled)
    optimize_cash = os.getenv("OPTIMIZE_CASH", "0") not in {"0", "false", "False"}
    allowed_extra = {"CASH"} if optimize_cash else set()
    extra = (set(weights) - allowed_extra) - set(universe)
    if extra:
        raise ValueError(f"Unknown symbols in decision: {sorted(extra)}")
    missing = set(universe) - set(weights)
    if missing:
        raise ValueError(f"Decision missing weights for: {sorted(missing)}")

    # Per-asset exposure caps
    for sym, w in list(weights.items()):
        if abs(w) > pos_cap + 1e-12:
            capped_w = pos_cap if w > 0 else -pos_cap
            logger.warning(
                "Weight exceeds position cap; capping",
                symbol=sym,
                weight=w,
                cap=pos_cap,
                capped_weight=capped_w,
            )
            weights[sym] = capped_w

    gross = sum(abs(w) for w in weights.values())
    if gross > gross_cap + 1e-12:
        logger.warning(
            "Gross exposure exceeds cap",
            gross=gross,
            gross_cap=gross_cap,
        )

    net = sum(weights.values())
    if abs(net) > 1e-12:
        logger.warning("Decision not market-neutral", net=net)

    return d
