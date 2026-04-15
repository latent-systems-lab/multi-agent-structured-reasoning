"""Prompt generation utilities."""

from __future__ import annotations

from typing import Any, Dict, List
import os

import json

from core.protocols import ReasoningToggles


def _format_beliefs(beliefs: Dict[str, List[str]]) -> str:
    """Return a readable string representation of agent beliefs."""

    parts: list[str] = []
    for agent, items in beliefs.items():
        if items:
            joined = ", ".join(items)
            parts.append(f"{agent}: {joined}")
    return "; ".join(parts) if parts else "None"


def analyst_prompt(
    role: str,
    toggles: ReasoningToggles,
    schema_json: Dict[str, Any],
    beliefs: Dict[str, List[str]] | None = None,
) -> str:
    """Return a prompt for an analyst agent.

    Parameters
    ----------
    role:
        Name of the analyst role.
    toggles:
        Reasoning toggles for the current protocol.
    schema_json:
        JSON schema of the expected response model.
    """
    reasoning_bits: list[str] = [
        f"You are the {role} analyst."
    ]

    if toggles.cot:
        reasoning_bits.append(
            "Think through the problem step by step and include your reasoning."
        )
    if toggles.debate.enabled:
        reasoning_bits.append(
            "Prepare to defend your proposal in a debate with other analysts."
        )
    if toggles.self_consistency.enabled:
        reasoning_bits.append(
            f"Generate diverse hypotheses; the system will sample {toggles.self_consistency.M} times."
        )
    if toggles.theory_of_mind:
        reasoning_bits.append(
            "Reason about peer beliefs and intentions. Before finalizing your own weights, explicitly predict what EACH peer is "
            "likely to propose (top 2-3 overweights/underweights)."
            " Populate the `mind_reading` list with one entry per peer:"
            " {target: <peer_name>, predicted_weights: {SYM: w, ...}, confidence: [0,1], summary: <one line>}."
            " Keep predictions concise; skip a peer only if you truly have no signal."
        )

    if beliefs:
        others = {k: v for k, v in beliefs.items() if k != role}
        if others:
            reasoning_bits.append(
                "Peer beliefs: " + _format_beliefs(others)
            )

    # Multi-asset instruction: universe may mix asset classes.
    reasoning_bits.append(
        "Treat the universe as multi-asset: equities, FX/currencies, commodities, and bonds may all appear."
        " Always consider every symbol provided, regardless of asset class."
        " If a tool or metric is not applicable to a given asset (e.g., fundamentals for FX), use alternative"
        " signals such as price trends/momentum, volatility, correlations, macro context, or default to a neutral"
        " weight rather than excluding the symbol."
    )

    # Role-specific guidance: fundamental analyst should hedge/diversify when
    # fundamentals are absent for some assets (e.g., FX/commodities/bonds).
    if role.lower() == "fundamental":
        reasoning_bits.append(
            "When fundamentals are unavailable or not meaningful for a symbol (e.g., FX, commodities, or some bonds),"
            " treat that symbol as a hedge/diversifier and assign a neutral baseline weight rather than zero."
            " Use an equal-weight baseline across all symbols and apply small fundamental tilts only where data exists."
            " Do not apply equal-weight just because there is no data for a few symbols."
        )

    reasoning = " ".join(reasoning_bits)

    schema_str = json.dumps(schema_json, indent=2)

    extra = []
    include_cash = os.getenv("OPTIMIZE_CASH", "0") not in {"0", "false", "False"}
    cash_phrase = " and CASH" if include_cash else ""
    extra.append(
        f"When proposing allocations, include weights over symbols{cash_phrase} that sum to 1. "
        "Negative weights denote short positions and are permitted within risk limits."
    )
    # Reliability fallback for cases with limited/missing data or tool failures.
    extra.append(
        "If some tools or inputs are rate-limited or unavailable, still produce a best-effort portfolio:"
        " default to equal-weight across the provided symbols; optionally apply small tilts using any available"
        " signal (e.g., momentum, beta, headline sentiment) or modest adjustments from previous weights."
        " When CASH is allowed, allocate only the residual to CASH so the portfolio remains invested within"
        " constraints. Do not return an all-CASH portfolio and do not say you cannot proceed; always return"
        " a valid, usable proposal."
    )
    extra_msg = ("\n" + " ".join(extra) + "\n") if extra else "\n"

    return (
        f"{reasoning}\n"
        + extra_msg +
        "Return JSON following this schema:\n"
        f"{schema_str}\n"
    )


def chair_prompt(
    toggles: ReasoningToggles,
    schema_json: Dict[str, Any],
    beliefs: Dict[str, List[str]] | None = None,
) -> str:
    """Return a prompt for the chair agent."""
    reasoning_bits: list[str] = [
        "You are the chair responsible for synthesising analyst proposals."
    ]
    reasoning_bits.append(
        "Negative weights denote short positions and are permitted within risk limits."
    )
    if toggles.cot:
        reasoning_bits.append(
            "Explain your synthesis with clear chain-of-thought reasoning."
        )
    if toggles.debate.enabled:
        reasoning_bits.append(
            "Account for critiques exchanged during debate rounds."
        )
    if toggles.self_consistency.enabled:
        reasoning_bits.append(
            "Aggregate multiple candidate decisions using self-consistency."
        )
    if toggles.theory_of_mind:
        reasoning_bits.append(
            "Consider how each analyst's perspective might reflect their goals and information."
        )

    if beliefs:
        reasoning_bits.append(
            "Current analyst beliefs: " + _format_beliefs(beliefs)
        )

    # Multi-asset and fallback synthesis guidance to avoid exclusions/all-CASH outcomes.
    reasoning_bits.append(
        "Treat the universe as multi-asset: equities, FX/currencies, commodities, and bonds may all appear, and all"
        " symbols must be considered in the decision. If analyst proposals are incomplete, missing, or indecisive,"
        " synthesise a best-effort decision that assigns weights to every symbol: prefer equal-weight across the"
        " universe with small, clearly justified tilts when any signals are available. When CASH is allowed, allocate"
        " only the residual to CASH so that non-CASH weights remain meaningfully invested. Do not exclude symbols due"
        " to asset class or produce an all-CASH portfolio unless explicitly required by risk constraints."
    )

    reasoning = " ".join(reasoning_bits)

    schema_str = json.dumps(schema_json, indent=2)

    return (
        f"{reasoning}\n"
        "Return JSON following this schema:\n"
        f"{schema_str}\n"
    )


def critique_prompt(
    role: str,
    deltas_summary: str,
    violations: list[str],
    toggles: ReasoningToggles,
    beliefs: Dict[str, List[str]] | None = None,
) -> str:
    """Prompt used for critique/response rounds during debate."""
    reasoning_bits: list[str] = [
        f"As the {role} analyst, evaluate the latest proposal."
    ]

    if toggles.cot:
        reasoning_bits.append(
            "Provide your critique step by step before giving a final statement."
        )
    if toggles.debate.enabled:
        reasoning_bits.append(
            "Engage with prior arguments and offer constructive challenges."
        )
    if toggles.self_consistency.enabled:
        reasoning_bits.append(
            "Ensure your critique is consistent with your earlier reasoning."
        )
    if toggles.theory_of_mind:
        reasoning_bits.append(
            "Reflect on the motivations and assumptions of your peers."
        )

    if beliefs:
        others = {k: v for k, v in beliefs.items() if k != role}
        if others:
            reasoning_bits.append(
                "Peer beliefs: " + _format_beliefs(others)
            )

    violations_text = "; ".join(violations) if violations else "None"

    return (
        f"{ ' '.join(reasoning_bits) }\n"
        f"Summary of changes: {deltas_summary}\n"
        f"Violations so far: {violations_text}\n"
        "Offer a revised proposal or defend your previous stance."
    )
