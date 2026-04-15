"""Runtime execution helpers for backtests."""

from __future__ import annotations

import time
import os
from pathlib import Path
from typing import Dict, TYPE_CHECKING, Any, Mapping
from dataclasses import dataclass

import asyncio
import json

import numpy as np

from data.alpha_vantage import (
    get_fundamentals,
    get_headlines,
    get_prices_window,
    set_throttle,
)
from data.features import compute_features
from data.vix import get_vix_close
from data.warehouse import load_calendar, write_artifact, append_positions_rows
from portfolio.constraints import feasible_bounds
from utils.tokens import account_tokens
from utils.rng import seed_everything
from core.validators import validate_decision
from core.memory import MemoryStore, encode_state
from core.schemas import ChairCandidate, GraphInputs, Experience, EpisodicSnippet
from core.protocols import ReasoningToggles

# LangGraph resumption helpers
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from eval.metrics import cvar_95, hit_rate, max_drawdown, sharpe
from eval.regimes import walk_forward_regimes
from eval.reporter import (
    write_ops_json,
    write_token_usage_json,
    write_metrics_csv,
    write_metrics_json,
    write_returns_json,
    append_decision_row,
    append_returns_row,
)
from utils.logging import get_logger
from utils.token_logging import append_daily_usage

if TYPE_CHECKING:  # pragma: no cover
    from core.schemas import ChairDecision
    from langgraph.graph import StateGraph
else:  # pragma: no cover
    StateGraph = object  # type: ignore


logger = get_logger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for episodic memory and reward shaping."""

    episodic: bool = True
    k: int = 5
    alpha_return: float = 1.0
    alpha_risk: float = 1.0
    alpha_tc: float = 1.0


def run_decision_day(
    graph: StateGraph,
    inputs: GraphInputs,
    resume: Command | None = None,
    *,
    thread_id: str | None = None,
    flags: Dict[str, Any] | None = None,
) -> ChairDecision:
    """Execute the graph for a single decision day.

    Parameters
    ----------
    graph:
        Compiled LangGraph decision graph.
    inputs:
        Data inputs for the trading day.
    resume:
        Optional :class:`Command` used to resume execution after an interrupt.
    thread_id:
        Identifier for the execution thread.  Reusing the same ``thread_id``
        across invocations allows resuming from checkpoints.

    Returns
    -------
    ChairDecision
        Validated decision enriched with token/latency accounting.
    """

    # Determine invocation payload and configuration.  When resuming from an
    # interrupt ``inputs`` will be a ``Command`` instance and is passed
    # directly.  Otherwise we wrap the ``GraphInputs`` under the ``inputs``
    # channel which LangGraph understands.
    start = time.perf_counter()
    state: Mapping[str, Any] | Any
    config = {"configurable": {"thread_id": thread_id or getattr(inputs, "date", "0")}}
    if resume is None:
        payload: Any = {"inputs": inputs}
        if flags:
            payload["flags"] = flags
    else:
        payload = resume

    # LangGraph graphs may expose an asynchronous ``ainvoke`` or require
    # streaming.  Attempt synchronous invocation first and fall back to
    # ``ainvoke`` when necessary.  Wrap in a lightweight retry loop to
    # withstand transient LLM backend errors (e.g., 500s from Gemini).

    def _invoke_once():
        try:
            return asyncio.run(
                graph.ainvoke(payload, {**config, "recursion_limit": 1000})
            )
        except TypeError:
            # only fall back if the graph doesn't accept config
            return asyncio.run(graph.ainvoke(payload))

    def _is_transient_error(exc: Exception) -> bool:
        msg = str(exc)
        return (
            "InternalServerError" in msg
            or "500" in msg
            or "internal error has occurred" in msg.lower()
            or "Rate limit" in msg
            or "429" in msg
            or "temporarily unavailable" in msg.lower()
        )

    attempts = max(1, int(os.getenv("LLM_TOPLEVEL_MAX_RETRIES", "1")))
    backoff_sec = float(os.getenv("LLM_TOPLEVEL_BACKOFF_SECONDS", "0"))
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            state = _invoke_once()
            break
        except Exception as exc:  # pragma: no cover - transient external failures
            last_exc = exc
            if i == attempts - 1 or not _is_transient_error(exc):
                logger.exception(
                    "Graph invocation failed (attempt %d/%d)", i + 1, attempts
                )
                raise
            delay = backoff_sec * (2**i)
            logger.warning(
                "Transient LLM error; retrying in %.1fs (attempt %d/%d): %s",
                delay,
                i + 1,
                attempts,
                str(exc),
            )
            time.sleep(delay)
    latency_ms = int((time.perf_counter() - start) * 1000)

    # Handle LangGraph interrupts by surfacing them to the caller so they can
    # supply resume values.  The checkpoint store allows the graph to pick up
    # where it left off once a ``Command`` is provided.
    if isinstance(state, Mapping) and "__interrupt__" in state:
        raise GraphInterrupt(state["__interrupt__"])

    decision = state["decision"] if isinstance(state, Mapping) else state.decision

    # Extract any usage metadata the graph or LLM nodes may have attached to the
    # state so token accounting can incorporate Gemini metrics.
    usage: Mapping[str, Any] | None = None
    if isinstance(state, Mapping):
        total_in = 0
        total_out = 0
        for message in state.get("messages", []):
            meta = getattr(message, "usage_metadata", None)
            if not meta:
                continue
            total_in += int(
                meta.get("prompt_token_count")
                or meta.get("input_tokens")
                or meta.get("prompt_tokens")
                or 0
            )
            total_out += int(
                meta.get("candidates_token_count")
                or meta.get("output_tokens")
                or meta.get("completion_tokens")
                or 0
            )
        if total_in or total_out:
            usage = {"input_tokens": total_in, "output_tokens": total_out}
        else:
            usage = state.get("usage") or state.get("usage_metadata")

    # Validate against portfolio constraints
    bounds = feasible_bounds(inputs.universe)
    decision = validate_decision(
        decision,
        inputs.universe,
        gross_cap=bounds["gross_cap"],
        pos_cap=bounds["max_weight"],
    )

    # Enrich with accounting info
    accounting = account_tokens(
        latency_ms, decision.token_in, decision.token_out, usage
    )
    decision.latency_ms = accounting["latency_ms"]
    decision.token_in = accounting["token_in"]
    decision.token_out = accounting["token_out"]
    if usage:
        try:
            setattr(decision, "usage_metadata", dict(usage))
        except Exception:
            pass

    return decision


def run_backtest_loop(
    graph: StateGraph,
    universe: list[str],
    start: str,
    end: str,
    costs: Dict[str, float],
    market_api: Dict[str, str],
    seeds: Dict[str, int] | None,
    outdir: str,
    stress_scenarios: Dict[str, float] | None = None,
    regimes: list[dict[str, str]] | None = None,
    initial_capital: float = 1_000_000.0,
    optimize_cash: bool = False,
    rebalance_every: int = 1,
    ref_symbol: str | None = "SPY",
    toggles: ReasoningToggles | None = None,
    memory: MemoryConfig | None = None,
) -> None:
    """Iterate over trading days and execute the decision graph.

    The loop fetches required data, calls :func:`run_decision_day`, projects the
    resulting weights into the feasible set and writes daily artefacts to
    ``outdir``.
    """

    toggles = toggles or ReasoningToggles()
    memory = memory or MemoryConfig()

    if seeds and "global" in seeds:
        seed_everything(seeds["global"])

    if "throttle_per_sec" in market_api:
        set_throttle(int(market_api["throttle_per_sec"]))

    # Create output directory with unique suffix if it already exists
    base_outdir = Path(outdir)
    if base_outdir.exists() and any(base_outdir.iterdir()):
        counter = 1
        while True:
            new_outdir = Path(f"{outdir}_{counter}")
            if not new_outdir.exists():
                outdir = str(new_outdir)
                break
            counter += 1

    Path(outdir).mkdir(parents=True, exist_ok=True)

    calendar = load_calendar(start, end)
    bounds = feasible_bounds(universe, optimize_cash=optimize_cash)
    logger.info(
        "Backtest config: optimize_cash=%s, bounds=%s, outdir=%s",
        optimize_cash,
        bounds,
        outdir,
    )
    prev_weights = {sym: 0.0 for sym in universe}
    daily_returns: list[float] = []
    returns_log: list[dict[str, float]] = []
    last_decision: ChairDecision | None = None
    tokens_latency: list[Dict[str, int | str]] = []
    token_usage: list[Dict[str, object]] = []
    cumulative = 1.0
    mem: MemoryStore | None = None
    last_experience: Experience | None = None
    debug_memory = os.getenv("BACKTEST_DEBUG") == "1"
    if debug_memory:
        import psutil

        proc = psutil.Process(os.getpid())

    def _build_state(
        market_ctx: Dict[str, float], feats: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        if feats:
            names = {name for f in feats.values() for name in f}
            for name in names:
                vals = [f[name] for f in feats.values() if name in f]
                if vals:
                    stats[f"mean_{name}"] = float(np.mean(vals))
        return {**market_ctx, **stats}

    def _write_rollups() -> None:
        """Persist summary artifacts for progress so far.

        Called after each day and on failure to ensure partial results are
        available even if the run does not complete.
        """
        try:
            # Keep JSON summaries current; CSVs are appended per day elsewhere
            write_ops_json(str(Path(outdir) / "ops.json"), tokens_latency)
            write_token_usage_json(str(Path(outdir) / "token_usage.json"), token_usage)
            write_returns_json(str(Path(outdir) / "returns.json"), returns_log)
            metrics = {
                "sharpe": sharpe(daily_returns),
                "max_drawdown": max_drawdown(daily_returns),
                "cvar_95": cvar_95(daily_returns),
                "hit_rate": hit_rate(daily_returns),
            }
            write_metrics_csv(str(Path(outdir) / "metrics.csv"), {"strategy": metrics})
            write_metrics_json(
                str(Path(outdir) / "metrics.json"), {"strategy": metrics}
            )
            if regimes:
                trunc_calendar = calendar[: len(daily_returns)]
                regime_metrics = walk_forward_regimes(
                    trunc_calendar, daily_returns, regimes
                )
                write_metrics_csv(str(Path(outdir) / "regimes.csv"), regime_metrics)
                write_metrics_json(str(Path(outdir) / "regimes.json"), regime_metrics)
        except Exception as exc:  # pragma: no cover - best effort persistence
            logger.warning("Failed to write rollup artifacts", error=str(exc))

    from portfolio.optimizer import project_to_feasible

    async def fetch_day_data(day: str) -> tuple[
        Dict[str, list[tuple[str, float]]],
        Dict[str, Dict[str, float]],
        Dict[str, Dict[str, float]],
        Dict[str, list[str]],
        Dict[str, float],
    ]:
        logger.debug("fetching_data", day=day)
        prices_task = asyncio.to_thread(
            get_prices_window, universe, end_date=day, lookback_days=60
        )
        fundamentals_task = asyncio.to_thread(get_fundamentals, universe, as_of=day)
        headlines_task = asyncio.to_thread(get_headlines, universe, as_of=day)
        try:
            prices_window, fundamentals, headlines = await asyncio.gather(
                prices_task, fundamentals_task, headlines_task
            )
        except Exception as exc:  # pragma: no cover - async failures
            logger.error("Failed to fetch market data", day=day, error=str(exc))
            raise
        try:
            features = await asyncio.to_thread(compute_features, prices_window)
        except Exception as exc:  # pragma: no cover - compute failure
            logger.error("Failed to compute features", day=day, error=str(exc))
            raise
        market_context: Dict[str, float] = {}
        try:
            vix = get_vix_close(day)
            if vix is not None:
                market_context["vix"] = float(vix)
        except Exception:  # pragma: no cover - non-critical enrichment
            pass
        return prices_window, features, fundamentals, headlines, market_context

    async def loop() -> None:
        nonlocal prev_weights, cumulative, last_decision, mem, last_experience
        if not calendar:
            return
        current_data = await fetch_day_data(calendar[0])
        for idx, day in enumerate(calendar):
            os.environ["TOKEN_LOG_DATE"] = day
            os.environ["TOKEN_LOG_OUTDIR"] = outdir
            try:
                logger.info("start_day", day=day, idx=idx)
                if debug_memory:
                    rss = proc.memory_info().rss
                    logger.debug("memory_usage", day=day, rss=rss)
                logger.info(
                    "portfolio_value", portfolio_value=initial_capital * cumulative
                )
                (
                    prices_window,
                    features,
                    fundamentals,
                    headlines,
                    market_context,
                ) = current_data
                s_t = _build_state(market_context, features)
                if mem is None and (
                    (memory.episodic and memory.k > 0)
                    or toggles.episodic_top_k > 0
                    or toggles.experience_replay
                ):
                    mem = MemoryStore(len(encode_state(s_t)))
                flags: Dict[str, Any] = {}
                if mem and (
                    (memory.episodic and memory.k > 0) or toggles.episodic_top_k > 0
                ):
                    k = memory.k if memory.episodic else toggles.episodic_top_k
                    topk = mem.top_k(encode_state(s_t), k=k)
                    snippets = [
                        EpisodicSnippet(
                            content=(
                                f"r={p.reward:.4f}"
                                if isinstance(p, Experience)
                                else str(p)
                            ),
                            score=score,
                        )
                        for p, score in topk
                    ]
                    flags["episodic_topk"] = snippets
                if toggles.experience_replay and last_experience is not None:
                    flags["replay_last"] = last_experience

                # ---- trading-day guard via ref_symbol (optional) ----
                if ref_symbol and ref_symbol in prices_window:
                    ref_series = prices_window.get(ref_symbol, [])
                    is_trading_day = bool(ref_series) and (ref_series[-1][0] == day)
                    if not is_trading_day:
                        # try to advance data to next calendar day if available
                        next_day = (
                            calendar[idx + 1] if idx + 1 < len(calendar) else None
                        )
                        current_data = (
                            await fetch_day_data(next_day) if next_day else current_data
                        )
                        continue

                # daily asset returns from window tail
                day_rets: Dict[str, float] = {}
                for sym in universe:
                    series = prices_window.get(sym, [])
                    if len(series) >= 2:
                        prev_p = series[-2][1]
                        curr_p = series[-1][1]
                        day_rets[sym] = curr_p / prev_p - 1.0
                    else:
                        day_rets[sym] = 0.0

                pre_cost = sum(
                    prev_weights.get(sym, 0.0) * day_rets[sym] for sym in universe
                )
                cash_weight = float(prev_weights.get("CASH", 0.0))
                cash_rate = costs.get("cash_rate_bps", 0.0) / 10000.0 / 252.0
                cash_yield = cash_weight * cash_rate
                borrow_rate = costs.get("borrow_bps_annual", 0.0) / 252.0 / 10000.0
                borrow_cost = (
                    sum(max(-prev_weights.get(sym, 0.0), 0.0) for sym in universe)
                    * borrow_rate
                )

                inputs = GraphInputs(
                    date=day,
                    universe=universe,
                    prices_window=prices_window,
                    features=features,
                    fundamentals=fundamentals,
                    headlines=headlines,
                    market_context=market_context,
                    prev_weights=prev_weights,
                    stress_scenarios=stress_scenarios or {},
                    chair_candidates=[],
                )

                # ---- rebalance cadence ----
                do_rebalance = idx % max(1, rebalance_every) == 0

                if do_rebalance:
                    logger.debug("running_decision", day=day)
                    next_day = calendar[idx + 1] if idx + 1 < len(calendar) else None
                    if next_day:
                        decision_task = asyncio.to_thread(
                            run_decision_day, graph, inputs, flags=flags
                        )
                        current_data_task = fetch_day_data(next_day)
                        decision, current_data = await asyncio.gather(
                            decision_task, current_data_task
                        )
                    else:
                        decision = await asyncio.to_thread(
                            run_decision_day, graph, inputs, flags=flags
                        )
                        current_data = None
                else:
                    # HOLD: reuse previous weights; make a lightweight ChairDecision
                    if last_decision is None:
                        logger.debug("running_decision", day=day)
                        # ensure first day always rebalances even if cadence says otherwise
                        decision = await asyncio.to_thread(
                            run_decision_day, graph, inputs, flags=flags
                        )
                    else:
                        prev_dec = last_decision
                        decision = prev_dec.model_copy(
                            update={
                                "date": day,
                                "synthesis": "HOLD (carry previous weights)",
                                "protocol_id": "HOLD",
                                "token_in": 0,
                                "token_out": 0,
                                "latency_ms": 0,
                            }
                        )
                    # keep current_data ready for next loop iteration
                    next_day = calendar[idx + 1] if idx + 1 < len(calendar) else None
                    if next_day:
                        current_data = await fetch_day_data(next_day)

                # project to feasible etc. (unchanged)
                candidate = ChairCandidate(
                    weights=decision.weights,
                    utility=decision.utility,
                    synthesis=decision.synthesis,
                    used_protocol=decision.protocol_id,
                    supporting={},
                )
                if optimize_cash and "CASH" not in candidate.weights:
                    cash_seed = max(0.0, 1.0 - sum(candidate.weights.values()))
                    candidate.weights["CASH"] = float(cash_seed)

                from portfolio.optimizer import project_to_feasible

                pre_proj = dict(candidate.weights)
                logger.debug("projecting_decision", day=day)
                decision.weights = project_to_feasible(candidate, bounds)

                turnover = sum(
                    abs(decision.weights.get(sym, 0.0) - prev_weights.get(sym, 0.0))
                    for sym in universe
                )
                trade_bps = costs.get("tc_bps", 0.0) + costs.get("slippage_bps", 0.0)
                trading_cost = turnover * trade_bps / 10000.0

                day_return = pre_cost + cash_yield - borrow_cost - trading_cost
                risk_proxy = (
                    float(
                        np.mean(
                            [f.get("volatility_20", 0.0) for f in features.values()]
                        )
                    )
                    if features
                    else 0.0
                )
                reward = (
                    memory.alpha_return * (pre_cost + cash_yield)
                    - memory.alpha_risk * risk_proxy
                    - memory.alpha_tc * trading_cost
                )
                next_s_t = (
                    _build_state(current_data[4], current_data[1])
                    if current_data is not None
                    else None
                )
                if mem and toggles.experience_replay:
                    exp = Experience(
                        obs=s_t,
                        action=json.dumps(decision.weights),
                        reward=float(reward),
                        next_obs=next_s_t,
                        done=False,
                    )
                    mem.add(encode_state(s_t), exp)
                    last_experience = exp
                cumulative *= 1.0 + day_return
                daily_returns.append(day_return)
                returns_log.append(
                    {"date": day, "return": day_return, "cumulative": cumulative - 1.0}
                )

                tokens_latency.append(
                    {
                        "date": day,
                        "token_in": decision.token_in,
                        "token_out": decision.token_out,
                        "latency_ms": decision.latency_ms,
                    }
                )

                usage_meta = getattr(decision, "usage_metadata", None)
                if usage_meta:
                    rec = {"date": day, **usage_meta}
                else:
                    rec = {
                        "date": day,
                        "input_tokens": decision.token_in,
                        "output_tokens": decision.token_out,
                        "total_tokens": decision.token_in + decision.token_out,
                    }
                token_usage.append(rec)
                append_daily_usage(outdir, day)

                # --- positions calc & persistence (unchanged) ---
                equity = initial_capital * cumulative
                day_positions: list[Dict[str, float | str]] = []
                long_notional = 0.0
                short_notional = 0.0
                for sym in universe:
                    series = prices_window.get(sym, [])
                    price = float(series[-1][1]) if series else 1.0
                    weight = float(decision.weights.get(sym, 0.0))
                    notional = weight * equity
                    units = notional / price if price else 0.0
                    rec: Dict[str, float | str] = {
                        "date": day,
                        "symbol": sym,
                        "price": price,
                        "weight": weight,
                        "notional": notional,
                        "units": units,
                    }
                    day_positions.append(rec)
                    if notional >= 0:
                        long_notional += notional
                    else:
                        short_notional += notional

                cash_notional = equity - long_notional + short_notional
                cash_weight = cash_notional / equity if equity else 0.0
                cash_rec: Dict[str, float | str] = {
                    "date": day,
                    "symbol": "CASH",
                    "price": 1.0,
                    "weight": cash_weight,
                    "notional": cash_notional,
                    "units": cash_notional,
                }
                day_positions.append(cash_rec)

                logger.debug("persisting_day", day=day)
                await asyncio.to_thread(
                    write_artifact,
                    Path(outdir) / f"decisions/decision_{day}.json",
                    decision.model_dump(),
                )
                await asyncio.to_thread(
                    write_artifact,
                    Path(outdir) / f"portfolio/portfolio_{day}.csv",
                    day_positions,
                )

                prev_weights = decision.weights

                symbols_for_header = sorted(
                    set(universe + (["CASH"] if optimize_cash else []))
                )
                await asyncio.to_thread(
                    append_decision_row,
                    str(Path(outdir) / "decisions.csv"),
                    decision,
                    symbols_for_header,
                )
                await asyncio.to_thread(
                    append_returns_row,
                    str(Path(outdir) / "returns.csv"),
                    day,
                    day_return,
                    cumulative - 1.0,
                )
                await asyncio.to_thread(
                    append_positions_rows,
                    str(Path(outdir) / "positions.csv"),
                    day_positions,
                )

                await asyncio.to_thread(_write_rollups)
                last_decision = decision
            except Exception:
                logger.exception("day failed", day=day, idx=idx)
                raise

    try:
        asyncio.run(loop())
    except Exception as exc:
        # Best-effort: write whatever we have so far before propagating
        logger.exception(
            "Backtest loop terminated early; writing partial results", exc_info=exc
        )
        _write_rollups()
        raise

    # Final JSON snapshots and metrics (CSVs already appended per day)
    write_ops_json(str(Path(outdir) / "ops.json"), tokens_latency)
    write_token_usage_json(str(Path(outdir) / "token_usage.json"), token_usage)
    metrics = {
        "sharpe": sharpe(daily_returns),
        "max_drawdown": max_drawdown(daily_returns),
        "cvar_95": cvar_95(daily_returns),
        "hit_rate": hit_rate(daily_returns),
    }
    write_returns_json(str(Path(outdir) / "returns.json"), returns_log)
    write_metrics_csv(str(Path(outdir) / "metrics.csv"), {"strategy": metrics})
    write_metrics_json(str(Path(outdir) / "metrics.json"), {"strategy": metrics})
    # CSVs are maintained incrementally; no final rewrite

    if regimes:
        regime_metrics = walk_forward_regimes(calendar, daily_returns, regimes)
        write_metrics_csv(str(Path(outdir) / "regimes.csv"), regime_metrics)
        write_metrics_json(str(Path(outdir) / "regimes.json"), regime_metrics)

    return None
