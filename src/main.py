"""Command-line entry point for the backtesting system."""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml  # type: ignore[import-untyped]
import os

from core.graph import build_graph
from core.protocols import ProtocolId, ReasoningToggles, ProtocolConfig, protocol_presets
from core.runtime import run_backtest_loop, MemoryConfig
import time

from utils.logging import get_logger
logger = get_logger(__name__)
REBALANCE_EVERY = 1

BASELINE_PROTOCOLS = {
    "single_prompt": ProtocolId.ONE_SHOT,
    "chair_only": ProtocolId.JUDGE_ONLY,
    "flat": ProtocolId.FLAT,
    "voting": ProtocolId.VOTING,
}
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/run_sc.yaml",
        help="path to the YAML configuration file"
    )
    parser.add_argument(
        "--regimes",
        default=None,
        help="optional path to YAML file specifying evaluation regimes",
    )
    parser.add_argument(
        "--baseline-suite",
        action="store_true",
        help="run the predefined baseline suite (single prompt, chair-only, flat, voting)",
    )
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.regimes:    
        with open(args.regimes, "r", encoding="utf-8") as f:
            cfg["regimes"] = yaml.safe_load(f)
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
    def _build_protocol(pid: ProtocolId) -> ProtocolConfig:
        proto = protocol_presets(pid)
        if cfg.get("toggles"):
            toggles_cfg = dict(cfg["toggles"])
            tom_cfg = toggles_cfg.pop("tom", None)
            if isinstance(tom_cfg, dict) and tom_cfg.get("enabled") is not None:
                toggles_cfg["theory_of_mind"] = bool(tom_cfg["enabled"])
            proto.toggles = ReasoningToggles(
                **{**proto.toggles.model_dump(), **toggles_cfg}
            )
        return proto

    memory_cfg = MemoryConfig(**cfg.get("memory", {}))

    if args.baseline_suite:
        base_outdir = Path(cfg["outdir"])
        base_outdir.mkdir(parents=True, exist_ok=True)
        for label, pid in BASELINE_PROTOCOLS.items():
            protocol = _build_protocol(pid)
            outdir = str(base_outdir / label)
            logger.info("Running baseline", name=label, protocol=pid.value, outdir=outdir)
            graph = build_graph(protocol)
            run_backtest_loop(
                graph=graph,
                universe=cfg["universe"],
                start=cfg["start"],
                end=cfg["end"],
                costs=cfg["costs"],
                market_api=cfg["market_api"],
                seeds=cfg.get("seeds"),
                outdir=outdir,
                stress_scenarios=cfg.get("stress_scenarios"),
                regimes=cfg.get("regimes"),
                initial_capital=float(cfg.get("initial_capital", 1_000_000.0)),
                optimize_cash=bool(cfg.get("optimize_cash", False)),
                rebalance_every=REBALANCE_EVERY,
                ref_symbol="SPY",
                toggles=protocol.toggles,
                memory=memory_cfg,
            )
        logger.info(
            "Baseline suite complete", config=args.config, outputs=str(base_outdir)
        )
    else:
        protocol = _build_protocol(ProtocolId(cfg["protocol"]))
        graph = build_graph(protocol)
        run_backtest_loop(
            graph=graph,
            universe=cfg["universe"],
            start=cfg["start"],
            end=cfg["end"] ,
            costs=cfg["costs"],
            market_api=cfg["market_api"],
            seeds=cfg.get("seeds"),
            outdir=cfg["outdir"],
            stress_scenarios=cfg.get("stress_scenarios"),
            regimes=cfg.get("regimes"),
            initial_capital=float(cfg.get("initial_capital", 1_000_000.0)),
            optimize_cash=bool(cfg.get("optimize_cash", False)),
            rebalance_every=REBALANCE_EVERY,
            ref_symbol="SPY",
            toggles=protocol.toggles,
            memory=memory_cfg,
        )
        logger.info(f"Done. Config: {args.config}. Results saved to: {cfg['outdir']}")


if __name__ == "__main__":
    main()
