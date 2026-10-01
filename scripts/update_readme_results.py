#!/usr/bin/env python3
"""Copy results/metrics.csv and results/dm_tests.csv into README.md verbatim
(between the results markers), so the README table is exactly the code output."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- results:start -->", "<!-- results:end -->"
LSTART, LEND = "<!-- limits:start -->", "<!-- limits:end -->"
ALPHA = 0.05


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
    abl = pd.read_csv(ROOT / "results" / "ablation.csv")
    abl_str = pd.read_csv(ROOT / "results" / "ablation.csv", dtype=str).fillna("")
    avg = pd.read_csv(ROOT / "results" / "adam_vs_glm.csv", dtype=str)
    over = pd.read_csv(ROOT / "results" / "overforecast.csv", dtype=str)
    sig_lines = []
    for loss in ["QLIKE", "MSE"]:
        steps = abl[(abl[f"p_{loss}"] < ALPHA)]
        if len(steps):
            items = "; ".join(f"{r.ingredient.strip()} ({r.model} vs {r.vs}, DM {r[f'dm_{loss}']:.3f}, "
                              f"p = {r[f'p_{loss}']:.3g})" for _, r in steps.iterrows())
            sig_lines.append(f"- {loss}: consecutive steps with p < 0.05: {items}.")
        else:
            sig_lines.append(f"- {loss}: no consecutive step with p < 0.05.")
    not_sig = abl[(abl["step"] > 0) & (abl["p_QLIKE"] >= ALPHA)]
    sig_lines.append("- QLIKE: consecutive steps with p >= 0.05: " + "; ".join(
        f"{r.ingredient.strip()} ({r.model} vs {r.vs}, p = {r.p_QLIKE:.3g})" for _, r in not_sig.iterrows()) + ".")
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
        "### Ablation",
        "",
        "`results/ablation.csv` (DM against the previous step, d = loss(step) - loss(previous))",
        "",
        md_table(abl_str),
        "",
        "What the ablation shows (p < 0.05 between consecutive steps only):",
        "",
        *sig_lines,
        "",
        "`results/adam_vs_glm.csv` (Linear-QLIKE trained by Adam vs the same model solved by the Gamma "
        "GLM; d = loss(Adam) - loss(GLM); relative difference = |h_Adam / h_GLM - 1|)",
        "",
        md_table(avg),
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
    conc_num = pd.read_csv(ROOT / "results" / "mse_concentration.csv")["top10_share"]
    conc_min, conc_max = conc_num.min(), conc_num.max()
    limits = "\n".join([
        LSTART,
        "- **Over-forecasts after 2020-03-13.** Ratio forecast / realized `rv5` per model on the target "
        "days 2020-03-13 to 2020-03-20, and the largest ratio over all evaluable days "
        "(`results/overforecast.csv`):",
        "",
        md_table(over),
        "",
        f"- **MSE concentration.** The 10 worst days account for between {conc_min:.1%} and {conc_max:.1%} "
        "of each model's total squared error (`results/mse_concentration.csv`, table above).",
        "- **Market.** Crypto spot data only, traded 24/7: no market closures, overnight gaps or "
        "weekends of the kind found in equity or futures markets.",
        "- **One asset.** BTCUSDT spot on one venue only.",
        "- **Ablation order.** Each ingredient is measured in the single order of the chain above; "
        "its contribution in another order is not measured.",
        LEND,
    ])
    readme = ROOT / "README.md"
    text = readme.read_text()
    i, j = text.index(START), text.index(END) + len(END)
    text = text[:i] + block + text[j:]
    i, j = text.index(LSTART), text.index(LEND) + len(LEND)
    readme.write_text(text[:i] + limits + text[j:])


if __name__ == "__main__":
    main()
