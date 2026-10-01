#!/usr/bin/env python3
"""Diagnostic of one target day (default 2023-08-12).

Writes results/diagnostic_<day>.csv (daily table, day +/- 5), and
results/diagnostic_<day>_forecasts.csv (each model's forecast for the day, ratio
to realized rv5), and results/diagnostic_<day>_raw.csv (1m-bar statistics of the
day +/- 2, from data/raw).
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_daily  # noqa: E402

MODELS = ["Naive", "EWMA", "GARCH", "HAR", "HAR-log-OLS", "HAR-log-QLIKE", "HAR-log-QLIKE+dow",
          "HAR-log-QLIKE+ret", "Linear-QLIKE-GLM", "Linear-QLIKE", "MLP-QLIKE"]


def raw_day_stats(day: pd.Timestamp) -> dict:
    path = ROOT / "data" / "raw" / f"BTCUSDT-1m-{day:%Y-%m}.zip"
    with zipfile.ZipFile(path) as z:
        b = pd.read_csv(z.open(z.namelist()[0]), header=None)
    unit = "us" if b[0].iloc[0] >= 10**14 else "ms"
    b.index = pd.to_datetime(b[0], unit=unit)
    x = b.loc[f"{day:%Y-%m-%d}"]
    close = x[4]
    grid = close[close.index.minute % 5 == 4]  # closes at the 5-min grid points of the day
    return {"date": day.date(), "n_1m_bars": len(x), "volume_btc": float(x[5].sum()),
            "n_trades": int(x[8].sum()), "high_low_range_pct": float((x[2].max() / x[3].min() - 1) * 100),
            "n_zero_1m_close_changes": int((close.diff() == 0).sum()),
            "n_zero_5m_close_changes": int((grid.diff() == 0).sum())}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--day", default="2023-08-12")
    day = pd.Timestamp(p.parse_args().day)
    res = ROOT / "results"
    df = load_daily()
    oos = df.loc["2020-01-01":"2026-08-31", "rv5"]
    win = df.loc[day - pd.Timedelta(days=5):day + pd.Timedelta(days=5),
                 ["rv5", "n_obs", "n_missing", "flag", "ret"]].copy()
    win.insert(0, "weekday", win.index.day_name())
    win["rv5_rank_in_oos_lowest_first"] = oos.rank(method="min").reindex(win.index).astype("Int64")
    win.index.name = "date"
    win.to_csv(res / f"diagnostic_{day:%Y-%m-%d}.csv", float_format="%.6g")

    fc = pd.read_csv(res / "forecasts.csv", parse_dates=["date"], index_col="date")
    for name in ["MLP-QLIKE", "Linear-QLIKE"]:
        fc[name] = pd.read_csv(res / f"forecasts_{name}.csv", parse_dates=["date"], index_col="date")[name]
    abl = pd.read_csv(res / "forecasts_ablation.csv", parse_dates=["date"], index_col="date")
    fc = fc.join(abl)
    rv = df.loc[day, "rv5"]
    rows = [{"model": m, "forecast": float(fc.loc[day, m]), "realized_rv5": float(rv),
             "ratio": float(fc.loc[day, m] / rv)} for m in MODELS]
    pd.DataFrame(rows).to_csv(res / f"diagnostic_{day:%Y-%m-%d}_forecasts.csv", index=False, float_format="%.6g")

    raw = pd.DataFrame([raw_day_stats(day + pd.Timedelta(days=k)) for k in range(-2, 3)])
    raw.to_csv(res / f"diagnostic_{day:%Y-%m-%d}_raw.csv", index=False, float_format="%.6g")
    print(win.to_string(), "\n", pd.DataFrame(rows).to_string(index=False), "\n", raw.to_string(index=False))
    print(f"\nrv5 OOS: min {oos.min():.6g} on {oos.idxmin().date()}, median {oos.median():.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
