#!/usr/bin/env python3
"""Plot HAR and GARCH one-day-ahead forecasts vs realized RV over 2024.

Reads results/forecasts.csv, writes results/forecast_vs_realized_2024.png.
Needs matplotlib (requirements-notebook.txt).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
fc = pd.read_csv(ROOT / "results" / "forecasts.csv", parse_dates=["date"], index_col="date")
fc = fc.loc["2024-01-01":"2024-12-31"]

REALIZED, FORECAST = "#8a8a8a", "#2a6fb0"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e3e3e3", "grid.linewidth": 0.6, "axes.edgecolor": "#888888",
                     "font.size": 10})
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, sharey=True)
for ax, model in zip(axes, ["HAR", "GARCH"]):
    ax.plot(fc.index, fc["rv5"], color=REALIZED, linewidth=0.9, label="Realized RV (rv5)")
    ax.plot(fc.index, fc[model], color=FORECAST, linewidth=1.4, label=f"{model} forecast")
    ax.set_yscale("log")
    ax.set_ylabel("daily variance")
    ax.set_title(f"{model}: one-day-ahead forecast vs realized, 2024", loc="left")
    ax.legend(frameon=False, loc="upper right", ncol=2)
fig.tight_layout()
out = ROOT / "results" / "forecast_vs_realized_2024.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
