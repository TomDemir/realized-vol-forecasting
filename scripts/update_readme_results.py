#!/usr/bin/env python3
"""Copy results/metrics.csv and results/dm_tests.csv into README.md verbatim
(between the results markers), so the README table is exactly the code output."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- results:start -->", "<!-- results:end -->"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def main():
    metrics = pd.read_csv(ROOT / "results" / "metrics.csv", dtype=str)
    dm = pd.read_csv(ROOT / "results" / "dm_tests.csv", dtype=str)
    conc = pd.read_csv(ROOT / "results" / "mse_concentration.csv", dtype=str)
    seeds = pd.read_csv(ROOT / "results" / "seed_qlike.csv", dtype=str)
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    verdict = json.loads((ROOT / "results" / "verdict.json").read_text())
    if verdict["mlp_declared_better"]:
        verdict_line = ("**Verdict (rule above): MLP-QLIKE is declared better: it beats HAR and "
                        "Linear-QLIKE on QLIKE with p < 0.05.**")
    else:
        missing = [b for b in ["HAR", "Linear-QLIKE"] if not verdict[f"beats_{b}"]]
        verdict_line = ("**Verdict (rule above): negative result. MLP-QLIKE is not declared better: "
                        f"it does not beat {' and '.join(missing)} on QLIKE with p < 0.05.**")
    sr = verdict["seed_qlike_range"]
    seed_line = "Single-seed QLIKE range (min, max): " + "; ".join(
        f"{m} {v[0]:.6e} to {v[1]:.6e}" for m, v in sr.items())
    block = "\n".join([
        START,
        "`results/metrics.csv`",
        "",
        md_table(metrics),
        "",
        "`results/dm_tests.csv` (d = loss(model) - loss(benchmark))",
        "",
        md_table(dm),
        "",
        verdict_line,
        "",
        "`results/seed_qlike.csv`",
        "",
        md_table(seeds),
        "",
        seed_line,
        "",
        "`results/mse_concentration.csv` (share of the total squared error due to the 10 worst days)",
        "",
        md_table(conc),
        "",
        "`results/summary.json`",
        "",
        "| item | value |",
        "|---|---|",
        f"| out-of-sample target days | {summary['n_targets']} |",
        f"| flagged target days (excluded) | {summary['n_flagged_targets']} |",
        f"| evaluable days common to all models | {summary['n_evaluable_common']} |",
        f"| HAR forecasts floored at {summary['har_eps_floor']} | {summary['har_n_floored']} |",
        f"| GARCH re-estimations | {summary['garch_n_refits']} |",
        f"| GARCH fits with non-zero convergence flag | {summary['garch_nonzero_convergence_flags']} |",
        "",
        "![HAR and GARCH forecasts vs realized RV, 2024](results/forecast_vs_realized_2024.png)",
        END,
    ])
    readme = ROOT / "README.md"
    text = readme.read_text()
    i, j = text.index(START), text.index(END) + len(END)
    readme.write_text(text[:i] + block + text[j:])


if __name__ == "__main__":
    main()
