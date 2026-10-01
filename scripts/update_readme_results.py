#!/usr/bin/env python3
"""Copy results/metrics.csv and results/dm_tests.csv into README.md verbatim
(between the results markers), so the README table is exactly the code output."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- results:start -->", "<!-- results:end -->"
LSTART, LEND = "<!-- limits:start -->", "<!-- limits:end -->"
SSTART, SEND = "<!-- summary:start -->", "<!-- summary:end -->"
DSTART, DEND = "<!-- diag:start -->", "<!-- diag:end -->"
DIAG_DAY = "2023-08-12"
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
    feats = pd.read_csv(ROOT / "results" / "ablation_features.csv")
    feats_str = pd.read_csv(ROOT / "results" / "ablation_features.csv", dtype=str).fillna("")

    def holm_lines(table, label_col, fmt):
        lines = []
        for loss in ["QLIKE", "MSE"]:
            sig = table[table[f"p_holm_{loss}"] < ALPHA]
            non = table[table[f"p_holm_{loss}"] >= ALPHA]
            if len(sig):
                lines.append(f"- {loss}, significant (Holm p < 0.05): " + "; ".join(fmt(r, loss) for _, r in sig.iterrows()) + ".")
            else:
                lines.append(f"- {loss}: no comparison is significant (Holm p < 0.05).")
            if len(non):
                lines.append(f"- {loss}, not significant: " + "; ".join(fmt(r, loss) for _, r in non.iterrows()) + ".")
        return lines

    def fmt_chain(r, loss):
        return (f"{r.ingredient.strip()} ({r.model} vs {r.vs}, DM {r[f'dm_{loss}']:.3f}, "
                f"Holm p = {r[f'p_holm_{loss}']:.3g})")

    def fmt_feat(r, loss):
        return (f"{r.comparison}: {r.model} vs {r.vs} ({r.ingredient}), DM {r[f'dm_{loss}']:.3f}, "
                f"Holm p = {r[f'p_holm_{loss}']:.3g}")

    chain = abl[abl["step"] > 0]
    sig_lines = holm_lines(chain, "ingredient", fmt_chain)
    feat_lines = holm_lines(feats, "comparison", fmt_feat)
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
        "Main chain, Holm-adjusted within the family of its 8 DM tests:",
        "",
        *sig_lines,
        "",
        "`results/ablation_features.csv` (decomposition of step 3; d = loss(model) - loss(vs); "
        "Holm within the family of its 8 DM tests)",
        "",
        md_table(feats_str),
        "",
        *feat_lines,
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
    met = pd.read_csv(ROOT / "results" / "metrics.csv").set_index("model")
    best = met["QLIKE"].idxmin()
    dmt = pd.read_csv(ROOT / "results" / "dm_tests.csv")
    mlp_lin = dmt[(dmt.model == "MLP-QLIKE") & (dmt.benchmark == "Linear-QLIKE") & (dmt.loss == "QLIKE")].iloc[0]
    sig_chain = chain[chain["p_holm_QLIKE"] < ALPHA]
    non_chain = chain[chain["p_holm_QLIKE"] >= ALPHA]
    sig_mse = int((chain["p_holm_MSE"] < ALPHA).sum() + (feats["p_holm_MSE"] < ALPHA).sum())
    fa = feats.set_index("comparison")
    over_num = pd.read_csv(ROOT / "results" / "overforecast.csv").set_index("model")
    diag = pd.read_csv(ROOT / "results" / f"diagnostic_{DIAG_DAY}.csv", parse_dates=["date"]).set_index("date")
    diag_fc = pd.read_csv(ROOT / "results" / f"diagnostic_{DIAG_DAY}_forecasts.csv")
    diag_raw = pd.read_csv(ROOT / "results" / f"diagnostic_{DIAG_DAY}_raw.csv", parse_dates=["date"]).set_index("date")
    d0 = diag.loc[DIAG_DAY]
    n_oos = summary["n_targets"]

    summary_lines = [
        SSTART,
        f"- Over {verdict['n_evaluable']} evaluable out-of-sample days (2020-01-01 to 2026-08-31), mean QLIKE "
        f"ranges from {met['QLIKE'].min():.4f} ({best}) to {met['QLIKE'].max():.4f} ({met['QLIKE'].idxmax()}); "
        f"HAR has {met.loc['HAR', 'QLIKE']:.4f}.",
        "- In the ablation chain from HAR to MLP-QLIKE, the only step with a Holm-adjusted p-value below 0.05 "
        "on QLIKE is " + "; ".join(f"{r.ingredient.strip()} (Holm p = {r.p_holm_QLIKE:.2g})" for _, r in sig_chain.iterrows())
        + "; " + ", ".join(f"{r.ingredient.strip()} (Holm p = {r.p_holm_QLIKE:.2g})" for _, r in non_chain.iterrows())
        + " are not significant.",
        f"- Within that step, adding the weekday of t+1 is significant on QLIKE, alone (Holm p = "
        f"{fa.loc['a', 'p_holm_QLIKE']:.2g}) and on top of the returns (Holm p = {fa.loc['c vs b', 'p_holm_QLIKE']:.2g}); "
        f"adding r_t and min(r_t, 0) is not, alone (Holm p = {fa.loc['b', 'p_holm_QLIKE']:.2g}) or on top of the "
        f"weekday (Holm p = {fa.loc['c vs a', 'p_holm_QLIKE']:.2g}).",
        f"- {sig_mse} of the 16 ablation comparisons on MSE are significant after Holm.",
        f"- Under the decision rule, the MLP result is negative: MLP-QLIKE does not beat Linear-QLIKE on QLIKE "
        f"(DM {mlp_lin.dm_stat:.2f}, p = {mlp_lin.p_value:.2f}).",
        f"- Limits: one asset (BTCUSDT spot) traded 24/7; the 10 worst days account for {conc_min:.1%} to "
        f"{conc_max:.1%} of each model's squared error; the largest forecast / realized ratio is "
        f"{over_num['max_ratio'].max():.1f} ({over_num['max_ratio'].idxmax()}, {DIAG_DAY}).",
        SEND,
    ]
    prev_day = diag_raw.index[diag_raw.index < pd.Timestamp(DIAG_DAY)].max()
    diag_lines = [
        DSTART,
        f"Observed for target day {DIAG_DAY} ({d0.weekday}), from `results/diagnostic_{DIAG_DAY}*.csv`:",
        "",
        f"- `rv5` = {d0.rv5:.4g}, the lowest of the {n_oos} out-of-sample days "
        f"(rank {int(d0.rv5_rank_in_oos_lowest_first)}); `n_obs` = {int(d0.n_obs)}, `n_missing` = "
        f"{int(d0.n_missing)}, `flag` = {d0.flag}.",
        f"- 1m bars: {int(diag_raw.loc[DIAG_DAY, 'n_1m_bars'])}; high-low range "
        f"{diag_raw.loc[DIAG_DAY, 'high_low_range_pct']:.2f} %; volume {diag_raw.loc[DIAG_DAY, 'volume_btc']:.0f} BTC "
        f"({diag_raw.loc[prev_day, 'volume_btc']:.0f} BTC on {prev_day:%Y-%m-%d}); "
        f"{int(diag_raw.loc[DIAG_DAY, 'n_zero_5m_close_changes'])} of the 287 within-day changes between "
        "consecutive 5-min grid closes are zero.",
        f"- Every model's forecast exceeds `rv5`: ratios from {diag_fc.ratio.min():.1f} "
        f"({diag_fc.loc[diag_fc.ratio.idxmin(), 'model']}) to {diag_fc.ratio.max():.1f} "
        f"({diag_fc.loc[diag_fc.ratio.idxmax(), 'model']}).",
        "",
        "Daily values, day -5 to day +5:",
        "",
        md_table(pd.read_csv(ROOT / "results" / f"diagnostic_{DIAG_DAY}.csv", dtype=str)),
        "",
        "Forecasts for the day:",
        "",
        md_table(pd.read_csv(ROOT / "results" / f"diagnostic_{DIAG_DAY}_forecasts.csv", dtype=str)),
        "",
        "1m-bar statistics, day -2 to day +2:",
        "",
        md_table(pd.read_csv(ROOT / "results" / f"diagnostic_{DIAG_DAY}_raw.csv", dtype=str)),
        DEND,
    ]

    readme = ROOT / "README.md"
    text = readme.read_text()
    i, j = text.index(START), text.index(END) + len(END)
    text = text[:i] + block + text[j:]
    i, j = text.index(LSTART), text.index(LEND) + len(LEND)
    text = text[:i] + limits + text[j:]
    for a, b, block_lines in ((SSTART, SEND, summary_lines), (DSTART, DEND, diag_lines)):
        i, j = text.index(a), text.index(b) + len(b)
        text = text[:i] + "\n".join(block_lines) + text[j:]
    readme.write_text(text)


if __name__ == "__main__":
    main()
