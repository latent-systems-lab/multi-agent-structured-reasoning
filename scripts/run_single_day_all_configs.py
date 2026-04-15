#!/usr/bin/env python
"""Run one trading day for every experiment configuration."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict

import yaml  # type: ignore[import-untyped]

from core.graph import build_graph
from core.protocols import ProtocolId, ReasoningToggles, protocol_presets
from core.runtime import run_backtest_loop, MemoryConfig

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
SUMMARY_CSV = Path("single_day_summary.csv")


def _unique_outdir(base: str) -> str:
    path = Path(base)
    if not path.exists():
        return base
    counter = 1
    while True:
        candidate = Path(f"{base}_{counter}")
        if not candidate.exists():
            return str(candidate)
        counter += 1


def process_config(cfg_path: Path, day: str) -> Dict[str, float]:
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Environment setup mirroring main.py
    os.environ["LLM_MODEL"] = cfg.get("llm_model", "gemini-2.5-flash-lite")
    os.environ["LLM_TEMPERATURE"] = str(cfg.get("llm_temperature", 0.0))
    if cfg.get("llm_max_retries") is not None:
        os.environ["LLM_MAX_RETRIES"] = str(cfg["llm_max_retries"])
    if cfg.get("llm_toplevel_max_retries") is not None:
        os.environ["LLM_TOPLEVEL_MAX_RETRIES"] = str(cfg["llm_toplevel_max_retries"])
    if cfg.get("llm_toplevel_backoff_seconds") is not None:
        os.environ["LLM_TOPLEVEL_BACKOFF_SECONDS"] = str(cfg["llm_toplevel_backoff_seconds"])
    if cfg.get("optimize_cash") is not None:
        os.environ["OPTIMIZE_CASH"] = "1" if cfg.get("optimize_cash") else "0"
    os.environ["ENABLE_TOOLS"] = "1" if cfg.get("enable_tools", True) else "0"

    protocol = protocol_presets(ProtocolId(cfg["protocol"]))
    if cfg.get("toggles"):
        toggles_cfg = dict(cfg["toggles"])
        tom_cfg = toggles_cfg.pop("tom", None)
        if isinstance(tom_cfg, dict) and tom_cfg.get("enabled") is not None:
            toggles_cfg["theory_of_mind"] = bool(tom_cfg["enabled"])
        protocol.toggles = ReasoningToggles(
            **{**protocol.toggles.model_dump(), **toggles_cfg}
        )

    memory_cfg = MemoryConfig(**cfg.get("memory", {}))
    graph = build_graph(protocol)

    run_name = Path(cfg["outdir"])
    if run_name.parts and run_name.parts[0] == "runs":
        run_name = Path(*run_name.parts[1:])
    outdir = _unique_outdir(str(Path("runs") / "single_day" / run_name))

    start_time = time.time()
    run_backtest_loop(
        graph=graph,
        universe=cfg["universe"],
        start=day,
        end=day,
        costs=cfg["costs"],
        market_api=cfg["market_api"],
        seeds=cfg.get("seeds"),
        outdir=outdir,
        stress_scenarios=cfg.get("stress_scenarios"),
        regimes=cfg.get("regimes"),
        initial_capital=float(cfg.get("initial_capital", 1_000_000.0)),
        optimize_cash=bool(cfg.get("optimize_cash", False)),
        rebalance_every=1,
        ref_symbol="SPY",
        toggles=protocol.toggles,
        memory=memory_cfg,
    )
    end_time = time.time()
    run_latency_ms = (end_time - start_time) * 1000

    csv_path = Path(outdir) / "daily_token_usage.csv"
    usage_path = Path(outdir) / "token_usage.json"
    ops_path = Path(outdir) / "ops.json"
    token_in = token_out = latency = 0.0
    latency_count = 0
    tokens_found = False

    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("date") == day:
                    token_in += float(row.get("input_tokens", 0))
                    token_out += float(row.get("output_tokens", 0))
                    tokens_found = True
                    break

    if not tokens_found and usage_path.exists():
        with usage_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for row in data:
            token_in += float(row.get("input_tokens", 0))
            token_out += float(row.get("output_tokens", 0))
            if "latency_ms" in row:
                latency += float(row.get("latency_ms", 0))
                latency_count += 1
        tokens_found = True
    elif not tokens_found and ops_path.exists():
        with ops_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for row in data:
            token_in += float(row.get("token_in", 0))
            token_out += float(row.get("token_out", 0))
            latency += float(row.get("latency_ms", 0))
            latency_count += 1
        tokens_found = True

    if latency_count == 0 and ops_path.exists():
        with ops_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for row in data:
            latency += float(row.get("latency_ms", 0))
            latency_count += 1

    return {
        "config": cfg_path.name,
        "token_in": token_in,
        "token_out": token_out,
        "avg_latency_ms": run_latency_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-day backtests for all configs")
    parser.add_argument(
        "--summary", default=str(SUMMARY_CSV), help="Path to summary CSV output"
    )
    args = parser.parse_args()
    date = "2020-01-03"
    rows = []
    cfg_files = sorted(CONFIG_DIR.glob("*.yaml"))
    for cfg_file in cfg_files:
        print(f"Running file: {cfg_file}")
        try:
            result = process_config(cfg_file, date)
            rows.append(result)
            print(
                f"{cfg_file.name}: in={result['token_in']}, out={result['token_out']}, avg_latency={result['avg_latency_ms']:.2f}ms"
            )
        except Exception as exc:  # pragma: no cover - best effort
            print(f"{cfg_file.name}: failed ({exc})")

    if rows:
        with open(args.summary, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["config", "token_in", "token_out", "avg_latency_ms"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Summary written to {args.summary}")


if __name__ == "__main__":
    main()
