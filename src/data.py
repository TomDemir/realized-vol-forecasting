"""Load the derived daily table and build model inputs.

Flagged-day policy
------------------
* Explanatory variable: rv_adj = rv5 * 1440 / (1440 - n_missing);
  rv_adj = NaN when n_missing = 1440. A forecast whose explanatory window
  contains a NaN is not evaluable.
* Target: rv5 of day t+1. Flagged days are excluded from the loss
  computation, identically for every model (see evaluation.common_mask).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 1440
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "derived" / "btcusdt_rv5_daily.csv"


def add_rv_adj(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    present = MINUTES_PER_DAY - out["n_missing"]
    out["rv_adj"] = np.where(present > 0, out["rv5"] * MINUTES_PER_DAY / present.where(present > 0), np.nan)
    return out


def load_daily(path: Path | str = DEFAULT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["flag"] = df["flag"].astype(bool)
    df = df.set_index("date").sort_index()
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("dates must be unique and increasing")
    if not (df.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("daily grid has gaps")
    return add_rv_adj(df)
