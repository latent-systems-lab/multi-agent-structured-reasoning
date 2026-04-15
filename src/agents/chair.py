from __future__ import annotations
from typing import Callable, TYPE_CHECKING, Dict, Any, List, Tuple, Iterable
from dataclasses import dataclass
import os, math

from core.fallbacks import fallback_risk

if TYPE_CHECKING:
    from core.schemas import GraphState, ChairCandidate, ChairDecision, Proposal
    from core.protocols import ReasoningToggles

R_MAX_HARD = int(os.getenv("MAX_ROUNDS_PER_DAY", "5"))
EPSILON_L2 = float(os.getenv("DEBATE_EPSILON_L2", "1e-6"))
KEEP_AUDIT = int(os.getenv("CHAIR_CANDS_KEEP", "16"))
OPT_CASH = os.getenv("OPTIMIZE_CASH", "0").lower() not in {"0", "false"}

LAMBDA_RISK = float(os.getenv("CHAIR_LAMBDA_RISK", "1.0"))
LAMBDA_TC = float(os.getenv("CHAIR_LAMBDA_TC", "1.0"))
LAMBDA_LIQ = float(os.getenv("CHAIR_LAMBDA_LIQ", "1.0"))
LAMBDA_CONS = float(os.getenv("CHAIR_LAMBDA_CONS", "0.5"))
LAMBDA_SCENARIO = float(os.getenv("CHAIR_LAMBDA_SCENARIO", "0.5"))

@dataclass
class DebateCtx:
    round_no: int
    feedback: str
    last_proposals: Dict[str, "Proposal"]

def _l2(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    return math.sqrt(sum((float(a.get(k,0.0))-float(b.get(k,0.0)))**2 for k in keys))


def _mse(a: Dict[str,float], b: Dict[str,float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    return sum((float(a.get(k,0.0)) - float(b.get(k,0.0)))**2 for k in keys) / len(keys)


def _mind_metrics(props: Dict[str,"Proposal"]) -> tuple[dict[str,float], list[str]]:
    """Return (per-analyst MSE, feedback lines) comparing predictions vs. actual weights."""
    per_src: dict[str,float] = {}
    notes: list[str] = []
    for src, p in props.items():
        preds = getattr(p, "mind_reading", None) or []
        if not preds:
            per_src[src] = float("nan")
            continue
        losses = []
        for pred in preds:
            tgt = getattr(pred, "target", "")
            tgt_p = props.get(tgt)
            if not tgt_p or not getattr(tgt_p, "weights", None):
                continue
            losses.append(_mse(pred.predicted_weights or {}, tgt_p.weights or {}))
        per_src[src] = sum(losses)/len(losses) if losses else float("nan")
    bad = [(k,v) for k,v in per_src.items() if v==v]
    if bad:
        worst = max(bad, key=lambda kv: kv[1])
        notes.append(f"- ToM: largest mind-prediction MSE={worst[1]:.3f} ({worst[0]} misread a peer).")
    return per_src, notes

def _aggregate_abs_mean(abs_props: List[Dict[str, float]]) -> Dict[str, float]:
    keys = set().union(*abs_props) if abs_props else set()
    n = max(len(abs_props), 1)
    return {k: sum(float(p.get(k,0.0)) for p in abs_props)/n for k in keys}

def _aggregate_delta_mean(props: Dict[str,"Proposal"], prev_w: Dict[str,float], universe: List[str]) -> Dict[str,float]:
    combined = {s: 0.0 for s in universe}
    for p in props.values():
        if getattr(p, "weights", None):
            for s in universe:
                combined[s] += float(p.weights.get(s, 0.0)) - float(prev_w.get(s, 0.0))
    count = max(len(props), 1)
    w = {s: float(prev_w.get(s, 0.0)) + combined[s]/float(count) for s in universe}
    if universe and "CASH" not in w:
        net = sum(w.values())
        adj = net/float(len(universe))
        for s in universe:
            w[s] -= adj
    return w


def _consensus_score(props: Dict[str, "Proposal"], universe: Iterable[str]) -> float:
    if not props:
        return 0.0
    symbols = list(universe)
    if not symbols:
        return 0.0
    disagreements: List[float] = []
    for sym in symbols:
        vals = [
            float(getattr(p, "weights", {}).get(sym, 0.0))
            for p in props.values()
            if getattr(p, "weights", None)
        ]
        if len(vals) >= 2:
            disagreements.append(max(vals) - min(vals))
    if not disagreements:
        return 1.0
    avg_range = sum(disagreements) / len(disagreements)
    return max(0.0, 1.0 - avg_range / 2.0)


def _scenario_alignment(weights: Dict[str, float], props: Dict[str, "Proposal"]) -> float:
    scenario = props.get("scenario")
    if not scenario or not getattr(scenario, "weights", None):
        return 0.0
    target = scenario.weights
    confidence = float(getattr(scenario, "confidence", 0.5) or 0.5)
    diff = sum(
        abs(float(weights.get(sym, 0.0)) - float(target.get(sym, 0.0)))
        for sym in set(weights) | set(target)
    )
    return -confidence * (diff / max(1.0, len(target)))


def _transaction_cost(weights: Dict[str, float], prev_w: Dict[str, float]) -> float:
    keys = set(weights) | set(prev_w)
    return sum(abs(float(weights.get(sym, 0.0)) - float(prev_w.get(sym, 0.0))) for sym in keys)


def _liquidity_penalty(weights: Dict[str, float], features: Dict[str, Dict[str, float]]) -> float:
    penalty = 0.0
    for sym, w in weights.items():
        feats = features.get(sym, {}) if features else {}
        liquidity = float(feats.get("liquidity_proxy", 1.0) or 1.0)
        penalty += abs(float(w)) * (1.0 / max(1e-6, liquidity))
    return penalty


def _expected_return(weights: Dict[str, float], features: Dict[str, Dict[str, float]]) -> float:
    exp = 0.0
    for sym, w in weights.items():
        feats = features.get(sym, {}) if features else {}
        signal = float(feats.get("momentum_20", feats.get("return_1d", 0.0)) or 0.0)
        exp += float(w) * signal
    return exp


def _apply_comm_pattern(
    props: Dict[str, "Proposal"],
    prev_w: Dict[str, float],
    universe: List[str],
    comm_pattern: str,
) -> Dict[str, float]:
    pattern = (comm_pattern or "hierarchical").lower()
    if pattern == "flat":
        abs_props = [p.weights for p in props.values() if getattr(p, "weights", None)]
        if abs_props:
            return _aggregate_abs_mean(abs_props)
    if pattern == "parallel":
        groups = {
            "fundamental_stack": ["fundamental", "technical"],
            "context_stack": ["sentiment", "scenario"],
        }
        combined = {sym: 0.0 for sym in universe}
        active = 0
        for members in groups.values():
            member_props = [
                props[name].weights
                for name in members
                if name in props and getattr(props[name], "weights", None)
            ]
            if member_props:
                grp = _aggregate_abs_mean(member_props)
                active += 1
                for sym in universe:
                    combined[sym] += grp.get(sym, 0.0)
        if active:
            return {sym: combined[sym] / active for sym in universe}
    if pattern == "voting":
        votes = {sym: 0.0 for sym in universe}
        for proposal in props.values():
            weights = getattr(proposal, "weights", {}) or {}
            for sym in universe:
                val = float(weights.get(sym, 0.0))
                votes[sym] += 1.0 if val > 0 else -1.0 if val < 0 else 0.0
        total = max(1, len(props))
        return {sym: votes[sym] / total for sym in universe}
    if pattern == "hybrid":
        flat_w = _apply_comm_pattern(props, prev_w, universe, "flat")
        debate_w = _aggregate_delta_mean(props, prev_w, universe)
        return {
            sym: 0.5 * flat_w.get(sym, 0.0) + 0.5 * debate_w.get(sym, 0.0)
            for sym in universe
        }
    return _aggregate_delta_mean(props, prev_w, universe)


def _select_sc_candidate(
    samples: List["ChairCandidate"], aggregation: str
) -> "ChairCandidate":
    if not samples:
        raise ValueError("self-consistency requires at least one candidate")
    aggregation = (aggregation or "median_utility").lower()
    if aggregation == "mean_utility":
        target = sum(float(c.utility) for c in samples) / len(samples)
    else:
        ordered = sorted(samples, key=lambda c: float(c.utility))
        target = float(ordered[len(ordered) // 2].utility)

    def _key(c: "ChairCandidate") -> tuple[float, float, float]:
        cvar = float(getattr(getattr(c, "risk", None), "cvar_95", float("inf")))
        liq = float(getattr(c, "liquidity_penalty", float("inf")))
        return (abs(float(c.utility) - target), cvar, liq)

    return min(samples, key=_key)

def _ensure_invested(w: Dict[str,float], universe: List[str]) -> Dict[str,float]:
    noncash = [s for s in (w.keys() or universe) if s != "CASH"] or list(universe)
    if not noncash:
        return w
    if sum(abs(w.get(s,0.0)) for s in noncash) < 1e-12:
        eq = 1.0/float(len(noncash))
        w = {s: eq for s in noncash}
        if OPT_CASH:
            w["CASH"] = 0.0
    return w

def _synthesize(props: Dict[str,"Proposal"]) -> str:
    parts = []
    for name, p in props.items():
        claim = getattr(p, "claim", "") or ""
        rationale = getattr(p, "rationale", "") or ""
        parts.append(f"[{name}] {claim or rationale}")
    text = " ".join(parts)
    return text[:4000]

def _utility(cand: "ChairCandidate") -> float:
    base = float(getattr(cand, "base_expected_return", 0.0) or 0.0)
    scenario = float(getattr(cand, "scenario_alignment", 0.0) or 0.0)
    consensus = float(getattr(cand, "consensus_score", 0.0) or 0.0)
    turnover = float(getattr(cand, "turnover", 0.0) or 0.0)
    liquidity = float(getattr(cand, "liquidity_penalty", 0.0) or 0.0)
    risk_penalty = 0.0
    if getattr(cand, "risk", None) is not None:
        risk_penalty = float(cand.risk.cvar_95) + 0.5 * float(cand.risk.gross)
    return (
        base
        + LAMBDA_SCENARIO * scenario
        + LAMBDA_CONS * consensus
        - LAMBDA_RISK * risk_penalty
        - LAMBDA_TC * turnover
        - LAMBDA_LIQ * liquidity
    )

def _judge(
    round_no: int,
    r_max: int,
    curr_w: Dict[str, float],
    prev_w: Dict[str, float] | None,
    epsilon: float,
) -> Tuple[bool, str]:
    if round_no >= r_max:
        return False, f"Hit R_max={r_max}."
    if prev_w is None:
        return True, "First round; continue."
    d = _l2(curr_w, prev_w)
    if d <= epsilon:
        return False, f"Converged (L2={d:.2e} <= {epsilon})."
    return True, f"L2 change {d:.2e} > {epsilon}; continue."

def _build_feedback(props: Dict[str,"Proposal"]) -> str:
    sym_set = set().union(*(p.weights.keys() for p in props.values() if getattr(p,"weights",None))) if props else set()
    lines = []
    for s in list(sym_set)[:8]:
        vals = [float(props[name].weights.get(s,0.0)) for name in props if getattr(props[name],"weights",None)]
        if len(vals) >= 2:
            rng = (max(vals)-min(vals))
            if rng > 0.05:
                lines.append(f"- Large disagreement on {s} (range ~{rng:.2f}).")
    if not lines:
        lines.append("- Minor disagreements; refine top-2 convictions and risk hedges.")
    return "Chair feedback:\n" + "\n".join(lines)

def _to_decision(
    cand: "ChairCandidate",
    state: "GraphState",
    rounds: int,
    toggles: "ReasoningToggles",
) -> "ChairDecision":
    from core.schemas import ChairDecision
    return ChairDecision(
        date=state.inputs.date,
        weights=cand.weights,
        utility=cand.utility,
        synthesis=cand.synthesis,
        protocol_id=getattr(cand, "used_protocol", "debate"),
        rounds_taken=rounds,
        sc_M=(getattr(getattr(toggles,"self_consistency",object()),"M",1)
              if getattr(getattr(toggles,"self_consistency",object()),"enabled",False) else 1),
        token_in=0, token_out=0, latency_ms=0,
        data_refs=[ref for p in (cand.supporting or {}).values() for ref in getattr(p,"data_refs",[])],
        supporting=cand.supporting,
        dissenting=(cand.risk.violations if getattr(cand,"risk",None) else []),
    )

def run(
    toggles: "ReasoningToggles",
    agents: Dict[str, Callable[[Any, DebateCtx], "Proposal"]] | None = None,
    *,
    comm_pattern: str = "hierarchical",
) -> Callable[["GraphState"], Dict[str, Any]]:

    from core.schemas import ChairCandidate  # local to avoid cycles

    R_cfg = getattr(getattr(toggles, "debate", object()), "R_max", 3) or 3
    R_max = max(1, min(R_cfg, R_MAX_HARD))
    epsilon_cfg = float(getattr(getattr(toggles, "debate", object()), "epsilon_l2", EPSILON_L2) or EPSILON_L2)
    tom_on = bool(getattr(toggles, "theory_of_mind", False))
    lambda_tom = float(os.getenv("TOM_UTILITY_PENALTY", "0.05"))

    def _run_single(inputs, feedback_seed: str = "", seed_proposals=None):
        universe: List[str] = list(getattr(inputs, "universe", []) or [])
        prev_w: Dict[str, float] = dict(getattr(inputs, "prev_weights", {}) or {})
        features = getattr(inputs, "features", {}) or {}
        prices_window = getattr(inputs, "prices_window", {}) or {}
        stress = getattr(inputs, "stress_scenarios", {}) or {}

        local_cands: List[ChairCandidate] = []
        feedback = feedback_seed
        last_props: Dict[str, "Proposal"] = {}
        prev_weights_for_judge: Dict[str, float] | None = None
        mind_by_src: Dict[str, float] = {}
        avg_mind: float = 0.0

        for r in range(1, R_max + 1):
            ctx = DebateCtx(round_no=r, feedback=feedback, last_proposals=last_props)

            proposals: Dict[str, "Proposal"] = {}
            if agents:
                for name, fn in agents.items():
                    try:
                        proposals[name] = fn(inputs, ctx)
                    except Exception:
                        continue
            else:
                proposals = dict(seed_proposals or {})

            mind_by_src, mind_notes = _mind_metrics(proposals)
            avg_mind = (
                sum(v for v in mind_by_src.values() if v == v)
                / max(1, sum(1 for v in mind_by_src.values() if v == v))
            ) if mind_by_src else 0.0

            weights = _apply_comm_pattern(proposals, prev_w, universe, comm_pattern)
            if not weights:
                weights = _aggregate_delta_mean(proposals, prev_w, universe)
            weights = _ensure_invested(weights, universe)
            synthesis = _synthesize(proposals)

            cand = ChairCandidate(
                weights=weights,
                utility=0.0,
                synthesis=synthesis,
                used_protocol=f"debate/{comm_pattern}",
                supporting=proposals,
            )
            cand.base_expected_return = _expected_return(weights, features)
            cand.turnover = _transaction_cost(weights, prev_w)
            cand.liquidity_penalty = _liquidity_penalty(weights, features)
            cand.consensus_score = _consensus_score(proposals, universe)
            cand.scenario_alignment = _scenario_alignment(weights, proposals)
            cand.risk = fallback_risk(weights, prices_window, universe, stress)
            cand.utility = _utility(cand)
            cand.mind_loss = avg_mind
            if tom_on:
                cand.utility -= lambda_tom * avg_mind

            local_cands.append(cand)

            cont, _ = _judge(
                round_no=r,
                r_max=R_max,
                curr_w=weights,
                prev_w=prev_weights_for_judge,
                epsilon=epsilon_cfg,
            )
            if not cont:
                break
            feedback = _build_feedback(proposals)
            if mind_notes:
                feedback += "\n" + "\n".join(mind_notes)
            last_props = proposals
            prev_weights_for_judge = weights

        return local_cands, mind_by_src, avg_mind

    def node(state: "GraphState") -> Dict[str, Any]:
        if getattr(state, "decision", None) is not None:
            return {}

        inputs = getattr(state, "inputs", None)
        if inputs is None:
            return {}

        sc_cfg = getattr(toggles, "self_consistency", None)
        sc_enabled = bool(getattr(sc_cfg, "enabled", False))
        sc_M = max(1, int(getattr(sc_cfg, "M", 1) or 1)) if sc_enabled else 1
        aggregation = getattr(sc_cfg, "aggregation", "median_utility") if sc_enabled else "median_utility"

        samples: List[List[ChairCandidate]] = []
        sample_meta: List[tuple[Dict[str, float], float]] = []
        for idx in range(sc_M):
            cands, mind_by_src, avg_mind = _run_single(
                inputs, seed_proposals=getattr(state, "proposals", {})
            )
            samples.append(cands)
            sample_meta.append((mind_by_src, avg_mind))

        if sc_enabled:
            finals = [c[-1] for c in samples if c]
            chosen = _select_sc_candidate(finals, aggregation)
            chosen.used_protocol = f"debate/{comm_pattern}+sc"
            best_idx = finals.index(chosen)
            local_cands = samples[best_idx]
            mind_by_src, avg_mind = sample_meta[best_idx]
        else:
            local_cands = samples[0]
            mind_by_src, avg_mind = sample_meta[0]
            chosen = local_cands[-1]

        decision = _to_decision(chosen, state, rounds=len(local_cands), toggles=toggles)

        audit = [
            {
                "round": i + 1,
                "util": float(c.utility),
                "cvar": float(getattr(getattr(c, "risk", None), "cvar_95", float("nan"))),
            }
            for i, c in enumerate(local_cands[-KEEP_AUDIT:])
        ]
        flags: Dict[str, Any] = {"debate_audit": audit}
        if sc_enabled:
            flags["sc_samples"] = [
                {
                    "utility": float(c[-1].utility) if c else float("nan"),
                    "cvar": float(getattr(getattr(c[-1], "risk", None), "cvar_95", float("nan"))) if c else float("nan"),
                }
                for c in samples
            ]
        if mind_by_src:
            flags.update({"mind_by_src": mind_by_src, "mind_avg": avg_mind})

        return {
            "decision": decision,
            "chair_candidates": local_cands[-KEEP_AUDIT:],
            "flags": flags,
        }

    return node
