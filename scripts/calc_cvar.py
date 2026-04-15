import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("runs")

def load_returns(root: Path):
    """Load daily returns from all runs."""
    series_ret = {}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        csv_path = sub / "returns.csv"
        if not csv_path.exists():
            continue
        name = sub.name
        df = pd.read_csv(csv_path)
        if "date" not in df.columns or "return" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates("date", keep="last")
        series_ret[name] = df.set_index("date")["return"]
    return series_ret

def calculate_var_cvar(returns: pd.Series, alpha=0.95):
    """Compute parametric-free historical VaR and CVaR at (1-alpha) tail."""
    if returns.empty:
        return np.nan, np.nan
    # 5% quantile (for 95% confidence)
    var_level = 1 - alpha
    var = returns.quantile(var_level)
    cvar = returns[returns <= var].mean()
    return var, cvar

def main():
    series_ret = load_returns(ROOT)
    metrics = {}
    for name, r in series_ret.items():
        var95, cvar95 = calculate_var_cvar(r, alpha=0.95)
        metrics[name] = {
            "VaR 95%": f"{var95:.4f}",
            "CVaR 95%": f"{cvar95:.4f}",
        }

    df = pd.DataFrame(metrics).T
    print("\nRisk Metrics (95% confidence):")
    print("=" * 60)
    print(df.to_string())
    print("=" * 60)

if __name__ == "__main__":
    main()
