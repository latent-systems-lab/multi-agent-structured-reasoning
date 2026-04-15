import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("runs")
BLOCK_SIZE = 5


def load_returns(root=ROOT) -> pd.DataFrame:
    series_ret = {}
    for sub in sorted(root.iterdir()):
        csv = sub / "returns.csv"
        if not csv.exists():
            continue
        name = sub.name
        df = pd.read_csv(csv)
        if "date" not in df.columns or "return" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates("date", keep="last")
        series_ret[name] = df.set_index("date")["return"]
    if not series_ret:
        raise RuntimeError("No returns found.")
    return pd.DataFrame(series_ret).sort_index()


def cvar(arr, alpha=0.95):
    if arr.size == 0:
        return np.nan
    var = np.quantile(arr, 1 - alpha)
    tail = arr[arr <= var]
    return np.mean(tail) if tail.size else np.nan


def annualized_sharpe(arr, freq=252):
    sd = np.std(arr, ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return (np.mean(arr) / sd) * np.sqrt(freq)


def bootstrap_diff_paired_block(
    X: np.ndarray,
    Y: np.ndarray,
    func,
    n_iter=10000,
    block_size=BLOCK_SIZE,
    random_state=7,
):
    """Paired five-day moving-block bootstrap on aligned dates."""
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    obs = func(X) - func(Y)
    diffs = np.empty(n_iter, dtype=float)
    starts = np.arange(max(n - block_size + 1, 1))

    for i in range(n_iter):
        sampled = []
        while len(sampled) < n:
            start = int(rng.choice(starts))
            sampled.extend(range(start, min(start + block_size, n)))
        b = np.array(sampled[:n], dtype=int)
        diffs[i] = func(X[b]) - func(Y[b])

    p = np.sum(np.abs(diffs) >= np.abs(obs)) / n_iter
    return obs, p


def main():
    ret_df = load_returns()
    strategies = ret_df.columns.tolist()
    base = strategies[0]

    results_boot = []
    for other in strategies[1:]:
        pair = ret_df[[base, other]].dropna()
        x = pair[base].values
        y = pair[other].values
        if len(pair) < 50:
            results_boot.append(
                {
                    "Comparison": f"{base} vs {other}",
                    "MeanDiff": "NA",
                    "p-value Mean": "NA",
                    "CVaR Diff": "NA",
                    "p-value CVaR": "NA",
                    "N_overlap": len(pair),
                }
            )
            continue

        mean_diff, p_mean = bootstrap_diff_paired_block(
            x, y, func=np.mean, n_iter=10000
        )
        cvar_diff, p_cvar = bootstrap_diff_paired_block(
            x, y, func=cvar, n_iter=6000
        )

        results_boot.append(
            {
                "Comparison": f"{base} vs {other}",
                "MeanDiff": f"{mean_diff:.5f}",
                "p-value Mean": f"{p_mean:.4f}",
                "CVaR Diff": f"{cvar_diff:.5f}",
                "p-value CVaR": f"{p_cvar:.4f}",
                "N_overlap": len(pair),
            }
        )

    print(
        "\nBootstrap significance tests (Mean & CVaR) - "
        "paired five-day moving-block, overlap-aligned:"
    )
    print("=" * 98)
    print(pd.DataFrame(results_boot).to_string(index=False))
    print("=" * 98)

    results_sharpe = []
    for other in strategies[1:]:
        pair = ret_df[[base, other]].dropna()
        x = pair[base].values
        y = pair[other].values
        if len(pair) < 50:
            results_sharpe.append(
                {
                    "Comparison": f"{base} vs {other}",
                    "Sharpe_base": "NA",
                    "Sharpe_other": "NA",
                    "Diff": "NA",
                    "p-value": "NA",
                    "N_overlap": len(pair),
                }
            )
            continue

        sr_base = annualized_sharpe(x)
        sr_other = annualized_sharpe(y)
        diff, p = bootstrap_diff_paired_block(
            x, y, func=annualized_sharpe, n_iter=10000
        )

        results_sharpe.append(
            {
                "Comparison": f"{base} vs {other}",
                "Sharpe_base": f"{sr_base:.3f}" if pd.notna(sr_base) else "NA",
                "Sharpe_other": f"{sr_other:.3f}" if pd.notna(sr_other) else "NA",
                "Diff": f"{diff:.3f}" if pd.notna(diff) else "NA",
                "p-value": f"{p:.4f}" if pd.notna(p) else "NA",
                "N_overlap": len(pair),
            }
        )

    print(
        "\nSharpe ratio difference (annualized) - "
        "paired five-day moving-block bootstrap, overlap-aligned:"
    )
    print("=" * 98)
    print(pd.DataFrame(results_sharpe).to_string(index=False))
    print("=" * 98)


if __name__ == "__main__":
    main()
