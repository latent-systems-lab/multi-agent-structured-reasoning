"""Aggregate performance metrics across multiple runs.

Usage:
    python analysis/aggregate_metrics.py [--baseline NAME] [--outdir DIR] <run_dir1> [<run_dir2> ...]

Each ``run_dir`` must contain a ``returns.csv`` or ``returns.json`` file with a
``return`` field for each day.  The script loads the daily returns, builds a
``{protocol: [returns]}`` mapping and passes it to
:func:`eval.comparison.comparison_report` which writes ``comparison.csv`` and
``comparison.json`` in ``outdir``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List
import sys

# Ensure ``src`` is on ``sys.path`` when executing from repository root.
ROOT = Path(__file__).resolve().parents[1]
SYS_PATH = ROOT / "src"
if str(SYS_PATH) not in sys.path:
    sys.path.insert(0, str(SYS_PATH))

from eval.comparison import comparison_report


def load_returns(run_dir: Path) -> List[float]:
    """Load daily returns from ``run_dir``.

    The function looks for ``returns.csv`` first and falls back to
    ``returns.json``.  The return values are expected to be in a field named
    ``return``.
    """

    csv_path = run_dir / "returns.csv"
    if csv_path.exists():
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            return [float(row["return"]) for row in reader]

    json_path = run_dir / "returns.json"
    if json_path.exists():
        with json_path.open() as f:
            data = json.load(f)
        return [float(entry["return"]) for entry in data]

    raise FileNotFoundError(f"returns.csv or returns.json not found in {run_dir}")


def aggregate(run_dirs: List[str], baseline: str | None, outdir: str | None) -> None:
    """Aggregate returns and generate comparison report."""

    returns: Dict[str, List[float]] = {}
    for run in run_dirs:
        path = Path(run)
        returns[path.name] = load_returns(path)

    base = baseline or Path(run_dirs[0]).name
    out = Path(outdir) if outdir else Path(run_dirs[0])
    comparison_report(returns, baseline=base, outdir=str(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate performance metrics across multiple runs."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more run directories containing returns.csv or returns.json",
    )
    parser.add_argument(
        "--baseline",
        help="Name of baseline protocol (defaults to the first run directory name)",
    )
    parser.add_argument(
        "--outdir",
        help="Directory for consolidated comparison reports (defaults to first run)",
    )
    args = parser.parse_args()
    aggregate(args.run_dirs, baseline=args.baseline, outdir=args.outdir)
