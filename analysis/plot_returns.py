"""Plot cumulative returns from run directories.

Usage:
    python analysis/plot_returns.py <run_dir1> [<run_dir2> ...]

Each run directory must contain a `returns.csv` file with columns
`date`, `return`, and `cumulative`. For every provided run directory a
`cumulative_returns.png` plot is stored in the same folder. If multiple
runs are supplied, an additional `comparison.png` overlay plot is saved in the first run's directory.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import polars as pl


def load_returns(run_dir: Path) -> pl.DataFrame:
    """Load `returns.csv` from ``run_dir``."""
    csv_path = run_dir / "returns.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"returns.csv not found in {run_dir}")
    return pl.read_csv(csv_path, try_parse_dates=True)


def plot_single(df: pl.DataFrame, run_dir: Path) -> None:
    """Plot cumulative returns for a single run."""
    pdf = df.to_pandas()
    fig, ax = plt.subplots()
    ax.plot(pdf["date"], pdf["cumulative"], label="cumulative")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(f"Cumulative Return - {run_dir.name}")
    ax.grid(True)
    fig.autofmt_xdate()
    out_path = run_dir / "cumulative_returns.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(data: list[tuple[pl.DataFrame, Path]]) -> None:
    """Overlay cumulative returns from multiple runs."""
    fig, ax = plt.subplots()
    for df, run_dir in data:
        pdf = df.to_pandas()
        ax.plot(pdf["date"], pdf["cumulative"], label=run_dir.name)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Cumulative Return Comparison")
    ax.grid(True)
    ax.legend()
    fig.autofmt_xdate()
    out_path = data[0][1] / "comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(run_dirs: list[str]) -> None:
    data: list[tuple[pl.DataFrame, Path]] = []
    for run in run_dirs:
        run_dir = Path(run)
        df = load_returns(run_dir)
        plot_single(df, run_dir)
        data.append((df, run_dir))
    if len(data) > 1:
        plot_overlay(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot cumulative returns from run directories."
    )
    parser.add_argument(
        "run_dirs", nargs="+", help="One or more run directories containing returns.csv"
    )
    args = parser.parse_args()
    main(args.run_dirs)
