#!/usr/bin/env python3
"""Evaluate every model on the common evaluable days of session 2.

Inputs: results/forecasts.csv (baselines, with the `evaluable` mask) and
results/forecasts_MLP-QLIKE.csv, results/forecasts_Linear-QLIKE.csv.
Outputs: results/metrics.csv, results/dm_tests.csv, results/mse_concentration.csv,
results/seed_qlike.csv, results/verdict.json, results/ablation.csv,
results/adam_vs_glm.csv, results/overforecast.csv, results/ablation_features.csv.

Holm adjustment: one family per table (main chain; feature decomposition),
each family holding all of its DM tests on both losses (QLIKE and MSE).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import (diebold_mariano, dm_table, holm, metrics_table,  # noqa: E402
                            mse_concentration, mse_loss, qlike_loss)

BASE = ["Naive", "EWMA", "GARCH", "HAR"]
ML = ["MLP-QLIKE", "Linear-QLIKE"]
PAIRS = [("Naive", "HAR"), ("EWMA", "HAR"), ("GARCH", "HAR"),
         ("MLP-QLIKE", "HAR"), ("MLP-QLIKE", "Linear-QLIKE"), ("Linear-QLIKE", "HAR")]
ALPHA = 0.05
ABL = ["HAR-log-OLS", "HAR-log-QLIKE", "Linear-QLIKE-GLM", "Linear-QLIKE-GLM-22",
       "HAR-log-QLIKE+dow", "HAR-log-QLIKE+ret"]
FEATURE_STEPS = [
    ("a", "HAR-log-QLIKE+dow", "+ weekday of t+1 only", "HAR-log-QLIKE"),
    ("b", "HAR-log-QLIKE+ret", "+ r_t and min(r_t, 0) only", "HAR-log-QLIKE"),
    ("c vs a", "Linear-QLIKE-GLM", "both (adds r_t, min(r_t, 0) to a)", "HAR-log-QLIKE+dow"),
    ("c vs b", "Linear-QLIKE-GLM", "both (adds weekday to b)", "HAR-log-QLIKE+ret"),
]


def add_holm(table: pd.DataFrame) -> pd.DataFrame:
    """Holm over all DM p-values of the table (QLIKE and MSE together = one family)."""
    out = table.copy()
    p = np.concatenate([out["p_QLIKE"].to_numpy(float), out["p_MSE"].to_numpy(float)])
    adj = holm(p)
    n = len(out)
    out.insert(out.columns.get_loc("p_QLIKE") + 1, "p_holm_QLIKE", adj[:n])
    out.insert(out.columns.get_loc("p_MSE") + 1, "p_holm_MSE", adj[n:])
    return out


def feature_table(F: pd.DataFrame, rv: pd.Series) -> pd.DataFrame:
    rows = []
    for label, m, ingredient, prev in FEATURE_STEPS:
        row = {"comparison": label, "model": m, "ingredient": ingredient, "vs": prev,
               "QLIKE": float(qlike_loss(rv, F[m]).mean()), "MSE": float(mse_loss(rv, F[m]).mean()),
               "QLIKE_vs": float(qlike_loss(rv, F[prev]).mean()), "MSE_vs": float(mse_loss(rv, F[prev]).mean())}
        for name, fn in (("QLIKE", qlike_loss), ("MSE", mse_loss)):
            r = diebold_mariano(fn(rv, F[m]), fn(rv, F[prev]))
            row[f"dm_{name}"], row[f"p_{name}"] = r["dm_stat"], r["p_value"]
            row["T"], row["nw_lag"] = r["T"], r["nw_lag"]
        rows.append(row)
    cols = ["comparison", "model", "ingredient", "vs", "QLIKE", "QLIKE_vs", "MSE", "MSE_vs",
            "dm_QLIKE", "p_QLIKE", "dm_MSE", "p_MSE", "T", "nw_lag"]
    return add_holm(pd.DataFrame(rows)[cols])
CHAIN = [("HAR", "HAR (levels, OLS)"), ("HAR-log-OLS", "+ log"), ("HAR-log-QLIKE", "+ QLIKE loss"),
         ("Linear-QLIKE-GLM", "+ inputs (r, min(r,0), weekday)"), ("MLP-QLIKE", "+ non-linearity")]
OVER_DAYS = ("2020-03-13", "2020-03-20")


def ablation_table(F: pd.DataFrame, rv: pd.Series) -> pd.DataFrame:
    rows = []
    for i, (m, ingredient) in enumerate(CHAIN):
        row = {"step": i, "model": m, "ingredient": ingredient,
               "QLIKE": float(qlike_loss(rv, F[m]).mean()), "MSE": float(mse_loss(rv, F[m]).mean())}
        if i > 0:
            prev = CHAIN[i - 1][0]
            row["vs"] = prev
            for name, fn in (("QLIKE", qlike_loss), ("MSE", mse_loss)):
                r = diebold_mariano(fn(rv, F[m]), fn(rv, F[prev]))
                row[f"dm_{name}"], row[f"p_{name}"] = r["dm_stat"], r["p_value"]
                row["T"], row["nw_lag"] = r["T"], r["nw_lag"]
        rows.append(row)
    cols = ["step", "model", "ingredient", "QLIKE", "MSE", "vs", "dm_QLIKE", "p_QLIKE", "dm_MSE", "p_MSE",
            "T", "nw_lag"]
    return pd.DataFrame(rows).reindex(columns=cols)


def adam_vs_glm(F: pd.DataFrame, rv: pd.Series, seeds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    adam = {"Linear-QLIKE (Adam, 5-seed mean)": F["Linear-QLIKE"]}
    adam.update({f"Linear-QLIKE seed {c.rsplit('seed', 1)[1]}": seeds[c] for c in seeds.columns})
    for glm in ["Linear-QLIKE-GLM-22", "Linear-QLIKE-GLM"]:
        for a_name, a in adam.items():
            rel = (a / F[glm] - 1).abs()
            dq = diebold_mariano(qlike_loss(rv, a), qlike_loss(rv, F[glm]))
            rows.append({"adam": a_name, "glm": glm, "QLIKE_adam": float(qlike_loss(rv, a).mean()),
                         "QLIKE_glm": float(qlike_loss(rv, F[glm]).mean()),
                         "QLIKE_diff": float(qlike_loss(rv, a).mean() - qlike_loss(rv, F[glm]).mean()),
                         "dm_QLIKE": dq["dm_stat"], "p_QLIKE": dq["p_value"],
                         "median_abs_rel_diff": float(rel.median()), "max_abs_rel_diff": float(rel.max()),
                         "max_diff_date": rel.idxmax().strftime("%Y-%m-%d")})
    return pd.DataFrame(rows)


def overforecast(F: pd.DataFrame, rv: pd.Series) -> pd.DataFrame:
    ratio = F.div(rv, axis=0)
    window = ratio.loc[OVER_DAYS[0]:OVER_DAYS[1]].T
    window.columns = [f"ratio_{d:%Y-%m-%d}" for d in window.columns]
    window.insert(0, "max_ratio", ratio.max())
    window.insert(1, "max_ratio_date", ratio.idxmax().dt.strftime("%Y-%m-%d"))
    window.index.name = "model"
    return window.reset_index()


def main() -> int:
    res = ROOT / "results"
    base = pd.read_csv(res / "forecasts.csv", parse_dates=["date"], index_col="date")
    mask = base["evaluable"].astype(bool)
    fc = base[BASE].copy()
    seeds = {}
    for m in ML:
        f = pd.read_csv(res / f"forecasts_{m}.csv", parse_dates=["date"], index_col="date")
        if not f.index.equals(fc.index):
            raise ValueError(f"{m}: forecast dates differ from the baselines")
        fc[m] = f[m]
        seeds[m] = f[[c for c in f.columns if c.startswith(f"{m}_seed")]]
    abl = pd.read_csv(res / "forecasts_ablation.csv", parse_dates=["date"], index_col="date")
    if not abl.index.equals(fc.index):
        raise ValueError("ablation forecast dates differ from the baselines")
    for m in ABL:
        fc[m] = abl[m]
    bad = ~np.isfinite(fc[mask]).all(axis=1) | ~(fc[mask] > 0).all(axis=1)
    if bad.any():
        raise ValueError(f"non-evaluable forecasts on session-2 evaluable days: {list(fc[mask].index[bad])[:10]}")

    rv = base.loc[mask, "rv5"]
    F = fc[mask]
    metrics = metrics_table(F, rv)
    metrics.to_csv(res / "metrics.csv", index=False, float_format="%.6e")
    dm = dm_table(F, rv, pairs=PAIRS)
    dm.to_csv(res / "dm_tests.csv", index=False, float_format="%.6g")
    conc = mse_concentration(F, rv, k=10)
    conc.to_csv(res / "mse_concentration.csv", index=False, float_format="%.6g")

    seed_rows = []
    for m in ML:
        for c in seeds[m].columns:
            h = seeds[m].loc[mask, c]
            seed_rows.append({"model": m, "seed": int(c.rsplit("seed", 1)[1]),
                              "QLIKE": float(qlike_loss(rv, h).mean()), "MSE": float(mse_loss(rv, h).mean())})
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(res / "seed_qlike.csv", index=False, float_format="%.6e")

    ablation = add_holm(ablation_table(F, rv))
    feats = feature_table(F, rv)
    feats.to_csv(res / "ablation_features.csv", index=False, float_format="%.6g")
    ablation.to_csv(res / "ablation.csv", index=False, float_format="%.6g")
    avg = adam_vs_glm(F, rv, seeds["Linear-QLIKE"].loc[mask])
    avg.to_csv(res / "adam_vs_glm.csv", index=False, float_format="%.6g")
    over = overforecast(F, rv)
    over.to_csv(res / "overforecast.csv", index=False, float_format="%.4g")

    q = dm[dm["loss"] == "QLIKE"].set_index(["model", "benchmark"])
    beats = {b: bool(q.loc[("MLP-QLIKE", b), "dm_stat"] < 0 and q.loc[("MLP-QLIKE", b), "p_value"] < ALPHA)
             for b in ["HAR", "Linear-QLIKE"]}
    verdict = {"rule": "MLP-QLIKE is declared better only if it beats HAR AND Linear-QLIKE on QLIKE "
                       "(DM statistic < 0, p < 0.05)",
               "beats_HAR": beats["HAR"], "beats_Linear-QLIKE": beats["Linear-QLIKE"],
               "mlp_declared_better": all(beats.values()), "n_evaluable": int(mask.sum()),
               "seed_qlike_range": {m: [float(seed_df.loc[seed_df.model == m, "QLIKE"].min()),
                                        float(seed_df.loc[seed_df.model == m, "QLIKE"].max())] for m in ML}}
    (res / "verdict.json").write_text(json.dumps(verdict, indent=2))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 200)
    print("metrics.csv\n" + metrics.to_string(index=False))
    print("\ndm_tests.csv\n" + dm.to_string(index=False))
    print("\nmse_concentration.csv\n" + conc.to_string(index=False))
    print("\nseed_qlike.csv\n" + seed_df.to_string(index=False))
    print("\n" + json.dumps(verdict, indent=2))
    print("\nablation.csv\n" + ablation.to_string(index=False))
    print("\nablation_features.csv\n" + feats.to_string(index=False))
    print("\nadam_vs_glm.csv\n" + avg.to_string(index=False))
    print("\noverforecast.csv\n" + over.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
