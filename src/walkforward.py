"""Expanding-window walk-forward, one-day-ahead.

At the end of each origin day t, every model is fitted on a copy of the rows
with date <= t and asked for the forecast of day t+1. Target days run from
`oos_start` to `oos_end` inclusive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import BaseModel

OOS_START = "2020-01-01"
OOS_END = "2026-08-31"


def walk_forward(df: pd.DataFrame, models: list[BaseModel], oos_start: str = OOS_START,
                 oos_end: str = OOS_END) -> pd.DataFrame:
    """Return a long table: target_date, origin, data_end, model, forecast."""
    targets = df.loc[oos_start:oos_end].index
    if len(targets) == 0:
        raise ValueError("empty out-of-sample range")
    rows = []
    for target in targets:
        origin = target - pd.Timedelta(days=1)
        hist = df.loc[:origin].copy()  # only data <= t is handed to the models
        if hist.index.max() != origin:
            raise ValueError(f"missing origin row {origin}")
        for m in models:
            m.fit(hist)
            f = m.predict(target)
            rows.append((target, origin, m.last_date, m.name, f))
    out = pd.DataFrame(rows, columns=["target_date", "origin", "data_end", "model", "forecast"])
    return out


def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    return long.pivot(index="target_date", columns="model", values="forecast")
