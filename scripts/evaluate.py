#!/usr/bin/env python3
"""Evaluate every model on the common evaluable days of session 2.

Inputs: results/forecasts.csv (baselines, with the `evaluable` mask) and
results/forecasts_MLP-QLIKE.csv, results/forecasts_Linear-QLIKE.csv.
Outputs: results/metrics.csv, results/dm_tests.csv, results/mse_concentration.csv,
results/seed_qlike.csv, results/verdict.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import dm_table, metrics_table, mse_concentration, mse_loss, qlike_loss  # noqa: E402

BASE = ["Naive", "EWMA", "GARCH", "HAR"]
ML = ["MLP-QLIKE", "Linear-QLIKE"]
PAIRS = [("Naive", "HAR"), ("EWMA", "HAR"), ("GARCH", "HAR"),
         ("MLP-QLIKE", "HAR"), ("MLP-QLIKE", "Linear-QLIKE"), ("Linear-QLIKE", "HAR")]
ALPHA = 0.05


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
