#!/usr/bin/env python
from __future__ import annotations
import os
import csv
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, Mapping

import yaml  # type: ignore[import-untyped]

from utils.logging import get_logger
from data.warehouse import load_calendar, write_artifact, append_positions_rows
from data.alpha_vantage import (
    get_fundamentals,
    get_headlines,
    get_prices_window,
    set_throttle,
)
from data.features import compute_features
from data.vix import get_vix_close
from eval.metrics import cvar_95, hit_rate, max_drawdown, sharpe
from eval.reporter import (
    write_ops_json,
    write_metrics_csv,
    write_metrics_json,
    write_returns_json,
    append_decision_row,
    append_returns_row,
)
from portfolio.constraints import feasible_bounds
from portfolio.optimizer import project_to_feasible
from core.schemas import GraphInputs, ChairCandidate
from core.validators import validate_decision
from utils.tokens import account_tokens
from utils.rng import seed_everything

from core.graph import build_graph
from core.protocols import ProtocolId, ReasoningToggles, protocol_presets

logger = get_logger(__name__)


def run_decision_day(
    graph, inputs: GraphInputs, resume=None, *, thread_id: str | None = None
):
    import time, asyncio

    start = time.perf_counter()
    config = {"configurable": {"thread_id": thread_id or getattr(inputs, "date", "0")}}
    payload = {"inputs": inputs} if resume is None else resume

    def _invoke_once():
        try:
            result = graph.invoke(payload, config)
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)
            return result
        except TypeError:
            result = graph.invoke(payload)
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)
            return result
        except Exception:
            try:
                return asyncio.run(graph.ainvoke(payload, config))
            except TypeError:
                return asyncio.run(graph.ainvoke(payload))

    def _is_transient(exc: Exception) -> bool:
        m = str(exc).lower()
        return (
            "500" in m
            or "internal" in m
            or "429" in m
            or "rate limit" in m
            or "temporarily unavailable" in m
        )

    attempts = max(1, int(os.getenv("LLM_TOPLEVEL_MAX_RETRIES", "3")))
    backoff = float(os.getenv("LLM_TOPLEVEL_BACKOFF_SECONDS", "2.0"))
    for i in range(attempts):
        try:
            state = _invoke_once()
            break
        except Exception as exc:
            if i == attempts - 1 or not _is_transient(exc):
                logger.exception(
                    "Graph invocation failed (attempt %d/%d)", i + 1, attempts
                )
                raise
            delay = backoff * (2**i)
            logger.warning(
                "Transient LLM error; retrying in %.1fs: %s", delay, str(exc)
            )
            import time as _t

            _t.sleep(delay)
    latency_ms = int((time.perf_counter() - start) * 1000)

    if isinstance(state, Mapping) and "__interrupt__" in state:
        from langgraph.errors import GraphInterrupt

        raise GraphInterrupt(state["__interrupt__"])

    decision = state["decision"] if isinstance(state, Mapping) else state.decision
    usage = (
        state.get("usage") or state.get("usage_metadata")
        if isinstance(state, Mapping)
        else None
    )

    try:
        acc = account_tokens(
            latency_ms,
            getattr(decision, "token_in", 0),
            getattr(decision, "token_out", 0),
            usage,
        )
        decision.latency_ms = acc["latency_ms"]
        decision.token_in = acc["token_in"]
        decision.token_out = acc["token_out"]
    except Exception:
        pass
    return decision


def _read_last_progress(outdir: Path) -> tuple[str | None, float, dict[str, float]]:
    returns_csv = outdir / "returns.csv"
    last_day = None
    cumulative = 1.0
    if returns_csv.exists():
        with open(returns_csv, "r", encoding="utf-8") as f:
            rdr = csv.reader(f)
            next(rdr, None)
            rows = list(rdr)
            if rows:
                last_row = rows[-1]
                if last_row and len(last_row) >= 3:
                    last_day = last_row[0]
                    try:
                        cumulative = 1.0 + float(last_row[2])
                    except Exception:
                        pass

    last_weights: dict[str, float] = {}
    if last_day:
        dec_path = outdir / f"decisions/decision_{last_day}.json"
        if dec_path.exists():
            try:
                with open(dec_path, "r", encoding="utf-8") as f:
                    j = json.load(f)
                last_weights = j.get("weights", {}) or {}
            except Exception:
                pass
    else:
        decs = sorted(outdir.glob("decision_*.json"))
        if decs:
            last = decs[-1]
            last_day = last.stem.replace("decision_", "")
            try:
                with open(last, "r", encoding="utf-8") as f:
                    j = json.load(f)
                last_weights = j.get("weights", {}) or {}
            except Exception:
                pass
            if returns_csv.exists():
                with open(returns_csv, "r", encoding="utf-8") as f:
                    rdr = csv.DictReader(f)
                    rows = list(rdr)
                    if rows:
                        cumulative = 1.0 + float(rows[-1]["cumulative"])
    return last_day, cumulative, last_weights


async def _fetch_day_data(universe: list[str], day: str):
    prices_window = await asyncio.to_thread(get_prices_window, universe, day, 60)
    fundamentals = await asyncio.to_thread(get_fundamentals, universe, day)
    headlines = await asyncio.to_thread(get_headlines, universe, day)
    features = await asyncio.to_thread(
        lambda: __import__("data.features").features.compute_features(prices_window)
    )
    market_ctx: Dict[str, float] = {}
    try:
        vix = get_vix_close(day)
        if vix is not None:
            market_ctx["vix"] = float(vix)
    except Exception:
        pass
    return prices_window, features, fundamentals, headlines, market_ctx


def resume_backtest_loop(
    graph,
    universe: list[str],
    start: str,
    end: str,
    costs: Dict[str, float],
    market_api: Dict[str, str],
    outdir: str,
    *,
    optimize_cash: bool = False,
    rebalance_every: int = 1,
    ref_symbol: str | None = "SPY",
    seeds: Dict[str, int] | None = None,
) -> None:
    if seeds and "global" in seeds:
        seed_everything(seeds["global"])
    if "throttle_per_sec" in market_api:
        # tolerate floats (e.g., 0.1)
        try:
            set_throttle(float(market_api["throttle_per_sec"]))
        except Exception:
            pass
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    calendar = load_calendar(start, end)
    if not calendar:
        logger.warning("Empty calendar between %s and %s", start, end)
        return

    last_day, cumulative, prev_weights = _read_last_progress(out)
    if last_day:
        try:
            start_idx = calendar.index(last_day) + 1
        except ValueError:
            start_idx = 0
            cumulative = 1.0
            prev_weights = {s: 0.0 for s in universe}
        logger.info(
            "Resuming after %s (start_idx=%d), cumulative=%.6f",
            last_day,
            start_idx,
            cumulative,
        )
    else:
        start_idx = 0
        prev_weights = {s: 0.0 for s in universe}
        cumulative = 1.0
        logger.info("No prior progress found; starting from %s", calendar[0])

    bounds = feasible_bounds(universe, optimize_cash=optimize_cash)
    daily_returns: list[float] = []
    returns_log: list[dict[str, float]] = []
    tokens_latency: list[Dict[str, int | str]] = []
    decisions_cache: list[Any] = []

    async def loop():
        nonlocal prev_weights, cumulative
        if start_idx >= len(calendar):
            logger.info("Nothing to resume; all days processed.")
            return

        current_data = await _fetch_day_data(universe, calendar[start_idx])
        for idx in range(start_idx, len(calendar)):
            day = calendar[idx]
            prices_window, features, fundamentals, headlines, market_context = (
                current_data
            )

            if ref_symbol:
                ref_series = prices_window.get(ref_symbol, [])
                is_trading = bool(ref_series) and (ref_series[-1][0] == day)
                if not is_trading:
                    next_day = calendar[idx + 1] if idx + 1 < len(calendar) else None
                    current_data = (
                        await _fetch_day_data(universe, next_day)
                        if next_day
                        else current_data
                    )
                    continue

            day_rets: Dict[str, float] = {}
            for sym in universe:
                series = prices_window.get(sym, [])
                if len(series) >= 2:
                    p0 = series[-2][1]
                    p1 = series[-1][1]
                    day_rets[sym] = (p1 / p0) - 1.0
                else:
                    day_rets[sym] = 0.0

            pre_cost = sum(
                prev_weights.get(sym, 0.0) * day_rets[sym] for sym in universe
            )
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
                stress_scenarios={},
            )

            do_rebalance = (idx - start_idx) % max(1, rebalance_every) == 0

            if do_rebalance or not decisions_cache:
                next_day = calendar[idx + 1] if idx + 1 < len(calendar) else None
                if next_day:
                    decision_task = asyncio.to_thread(run_decision_day, graph, inputs)
                    current_data_task = _fetch_day_data(universe, next_day)
                    decision, current_data = await asyncio.gather(
                        decision_task, current_data_task
                    )
                else:
                    decision = await asyncio.to_thread(run_decision_day, graph, inputs)
                    current_data = None
            else:
                prev_dec = decisions_cache[-1]
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
                next_day = calendar[idx + 1] if idx + 1 < len(calendar) else None
                if next_day:
                    current_data = await _fetch_day_data(universe, next_day)

            decision = validate_decision(
                decision,
                universe,
                gross_cap=bounds["gross_cap"],
                pos_cap=bounds["max_weight"],
            )
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
            decision.weights = project_to_feasible(candidate, bounds)

            turnover = sum(
                abs(decision.weights.get(sym, 0.0) - prev_weights.get(sym, 0.0))
                for sym in universe
            )
            trade_bps = costs.get("tc_bps", 0.0) + costs.get("slippage_bps", 0.0)
            trading_cost = turnover * trade_bps / 10000.0

            day_return = pre_cost - borrow_cost - trading_cost
            cumulative *= 1.0 + day_return
            daily_returns.append(day_return)
            returns_log.append(
                {"date": day, "return": day_return, "cumulative": cumulative - 1.0}
            )

            decisions_cache.append(decision)
            tokens_latency.append(
                {
                    "date": day,
                    "token_in": decision.token_in,
                    "token_out": decision.token_out,
                    "latency_ms": decision.latency_ms,
                }
            )

            equity = cumulative
            day_positions: list[Dict[str, float | str]] = []
            long_notional = 0.0
            short_notional = 0.0
            for sym in universe:
                series = prices_window.get(sym, [])
                price = float(series[-1][1]) if series else 1.0
                weight = float(decision.weights.get(sym, 0.0))
                notional = weight * equity
                units = notional / price if price else 0.0
                rec = {
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
            cash_rec = {
                "date": day,
                "symbol": "CASH",
                "price": 1.0,
                "weight": (cash_notional / equity if equity else 0.0),
                "notional": cash_notional,
                "units": cash_notional,
            }
            day_positions.append(cash_rec)

            await asyncio.to_thread(
                write_artifact,
                Path(out) / f"decisions/decision_{day}.json",
                decision.model_dump(),
            )
            await asyncio.to_thread(
                write_artifact,
                Path(out) / f"portfolio/portfolio_{day}.csv",
                day_positions,
            )

            prev_weights = decision.weights

            symbols_for_header = sorted(set(universe + ["CASH"]))
            await asyncio.to_thread(
                append_decision_row,
                str(Path(out) / "decisions.csv"),
                decision,
                symbols_for_header,
            )
            await asyncio.to_thread(
                append_returns_row,
                str(Path(out) / "returns.csv"),
                day,
                day_return,
                cumulative - 1.0,
            )
            await asyncio.to_thread(
                append_positions_rows, str(Path(out) / "positions.csv"), day_positions
            )

            await asyncio.to_thread(
                write_ops_json, str(Path(out) / "ops.json"), tokens_latency
            )
            await asyncio.to_thread(
                write_returns_json, str(Path(out) / "returns.json"), returns_log
            )
            metrics = {
                "sharpe": sharpe(daily_returns),
                "max_drawdown": max_drawdown(daily_returns),
                "cvar_95": cvar_95(daily_returns),
                "hit_rate": hit_rate(daily_returns),
            }
            await asyncio.to_thread(
                write_metrics_csv, str(Path(out) / "metrics.csv"), {"strategy": metrics}
            )
            await asyncio.to_thread(
                write_metrics_json,
                str(Path(out) / "metrics.json"),
                {"strategy": metrics},
            )

    asyncio.run(loop())


# -------------------------- CLI (config only) --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/run_cot_sc_off.yaml",
        help="path to the YAML configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # env knobs to keep behavior identical to main.py
    os.environ["LLM_MODEL"] = cfg.get("llm_model", "gemini-2.5-flash-lite")
    os.environ["LLM_TEMPERATURE"] = str(cfg.get("llm_temperature", 0.0))
    if cfg.get("llm_max_retries") is not None:
        os.environ["LLM_MAX_RETRIES"] = str(cfg["llm_max_retries"])
    if cfg.get("llm_toplevel_max_retries") is not None:
        os.environ["LLM_TOPLEVEL_MAX_RETRIES"] = str(cfg["llm_toplevel_max_retries"])
    if cfg.get("llm_toplevel_backoff_seconds") is not None:
        os.environ["LLM_TOPLEVEL_BACKOFF_SECONDS"] = str(
            cfg["llm_toplevel_backoff_seconds"]
        )
    os.environ["ENABLE_TOOLS"] = "1" if cfg.get("enable_tools", True) else "0"
    if cfg.get("optimize_cash") is not None:
        os.environ["OPTIMIZE_CASH"] = "1" if cfg.get("optimize_cash") else "0"

    protocol = protocol_presets(ProtocolId(cfg["protocol"]))
    if cfg.get("toggles"):
        toggles_cfg = dict(cfg["toggles"])
        tom_cfg = toggles_cfg.pop("tom", None)
        if isinstance(tom_cfg, dict) and tom_cfg.get("enabled") is not None:
            toggles_cfg["theory_of_mind"] = bool(tom_cfg["enabled"])
        protocol.toggles = ReasoningToggles(
            **{**protocol.toggles.model_dump(), **toggles_cfg}
        )
    graph = build_graph(protocol)

    # Defaults if not present in YAML
    rebalance_every = int(cfg.get("rebalance_every", 30))
    ref_symbol = cfg.get("ref_symbol", "SPY")

    # Throttle env key wiring if needed
    market_api = cfg.get("market_api", {})
    key_env = market_api.get("key_env")
    if key_env and key_env not in os.environ:
        logger.warning("Market API key_env %s not set in environment.", key_env)

    resume_backtest_loop(
        graph=graph,
        universe=cfg["universe"],
        start=cfg["start"],
        end=cfg["end"],
        costs=cfg["costs"],
        market_api=market_api,
        outdir=cfg["outdir"],
        optimize_cash=bool(cfg.get("optimize_cash", False)),
        rebalance_every=rebalance_every,
        ref_symbol=ref_symbol,
        seeds=cfg.get("seeds"),
    )
