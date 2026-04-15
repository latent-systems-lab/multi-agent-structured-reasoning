
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple

import pandas as pd

PRICING_INPUT_PER_1K   = float(os.getenv("PRICING_INPUT_PER_1K", "0.10"))
PRICING_OUTPUT_PER_1K  = float(os.getenv("PRICING_OUTPUT_PER_1K", "0.00"))
PRICING_THINK_PER_1K   = float(os.getenv("PRICING_THINK_PER_1K",  "0.40"))

BILL_THINK_AS_OUTPUT   = os.getenv("BILL_THINK_AS_OUTPUT", "0").lower() in {"1", "true", "yes"}

BASELINE_NAME = os.getenv("BASELINE_NAME", "")

ROOT_DIR = Path(os.getenv("RUNS_ROOT", "runs")).resolve()

def load_op_json(path: Path) -> List[Dict[str, Any]]:
    """Load op.json which can be a JSON list or JSONL file."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    records: List[Dict[str, Any]] = []
    try:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        if len(records) == 1 and isinstance(records[0], list):
            records = records[0]
        return records
    except Exception:
        pass

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except Exception:
        pass

    return []

def iter_runs(root: Path) -> Iterable[Tuple[str, Path]]:
    """Yield (run_name, path_to_op_json) for each run folder that has op.json."""
    if not root.exists():
        return
    for op in root.rglob("ops.json"):
        run_name = op.parent.name
        yield run_name, op

def safe_get(d: Dict[str, Any], key: str, default=0.0) -> float:
    try:
        v = d.get(key, default)
        return float(v if v is not None else default)
    except Exception:
        return float(default)

def compute_cost(in_tokens: float, out_tokens: float, think_tokens: float) -> float:
    in_cost   = (in_tokens   / 1000.0) * PRICING_INPUT_PER_1K
    if BILL_THINK_AS_OUTPUT:
        out_cost  = ((out_tokens + think_tokens) / 1000.0) * PRICING_OUTPUT_PER_1K
        think_cost = 0.0
    else:
        out_cost  = (out_tokens  / 1000.0) * PRICING_OUTPUT_PER_1K
        think_cost = (think_tokens / 1000.0) * PRICING_THINK_PER_1K
    return in_cost + out_cost + think_cost

def median(lst: List[float]) -> float:
    if not lst:
        return 0.0
    s = sorted(lst)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])

def main() -> None:
    rows = []
    for run_name, op_path in iter_runs(ROOT_DIR):
        recs = load_op_json(op_path)
        if not recs:
            continue

        tokens_total: List[float] = []
        lat_s: List[float] = []
        costs: List[float] = []

        for r in recs:
            tin   = safe_get(r, "token_in", 0.0)
            tout  = safe_get(r, "token_out", 0.0)
            tthink= safe_get(r, "thinking_tokens", 0.0)
            latms = safe_get(r, "latency_ms", 0.0)

            tokens_total.append(tin + tout + tthink)
            lat_s.append(latms / 1000.0 if latms else 0.0)
            costs.append(compute_cost(tin, tout, tthink))

        rows.append({
            "run": run_name,
            "n_records": len(recs),
            "Tokens/Decision (Median)": median(tokens_total),
            "Latency (s, Median)": median(lat_s),
            "Cost/Decision (USD, Median)": median(costs),
        })

    if not rows:
        print(f"No op.json files found under {ROOT_DIR}")
        return

    df = pd.DataFrame(rows).sort_values("Cost/Decision (USD, Median)").reset_index(drop=True)

    if BASELINE_NAME:
        base_rows = df[df["run"].str.contains(BASELINE_NAME)]
        if not base_rows.empty:
            base_cost = float(base_rows.iloc[0]["Cost/Decision (USD, Median)"])
        else:
            base_cost = float(df["Cost/Decision (USD, Median)"].median())
    else:
        base_cost = float(df["Cost/Decision (USD, Median)"].median())

    if base_cost <= 0:
        df["Relative Cost (Index)"] = 1.0
    else:
        df["Relative Cost (Index)"] = df["Cost/Decision (USD, Median)"] / base_cost

    out_csv = Path("runs") / "ops_cost_latency_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved: {out_csv.resolve()}")

if __name__ == "__main__":
    main()
