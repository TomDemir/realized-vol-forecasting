"""Integrity of the daily RV table: unique and strictly increasing dates."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from build_rv import daily_rv, dedupe

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "derived" / "btcusdt_rv5_daily.csv"


def _synthetic_table():
    idx = pd.date_range("2024-12-30", "2025-01-03", freq="min", inclusive="left", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(idx))))
    bars = pd.DataFrame({"close": close}, index=idx)
    return daily_rv(bars)


def _real_table():
    if not CSV.exists():
        pytest.skip(f"{CSV} not built")
    return pd.read_csv(CSV, parse_dates=["date"])


@pytest.fixture(params=["synthetic", "real"])
def table(request):
    return _synthetic_table() if request.param == "synthetic" else _real_table()


def test_no_duplicates(table):
    assert not table["date"].duplicated().any()


def test_monotonic_dates(table):
    d = pd.to_datetime(table["date"])
    assert d.is_monotonic_increasing
    # Fixed daily grid: consecutive calendar days, no gap.
    assert (d.diff().dropna() == pd.Timedelta(days=1)).all()


def test_columns(table):
    assert list(table.columns) == ["date", "rv5", "n_obs", "n_missing", "flag", "ret"]


def test_dedupe_exact_and_conflicting():
    t = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 00:00", "2020-01-01 00:01"], tz="UTC")
    bars, n = dedupe(pd.DataFrame({"close": [1.0, 1.0, 2.0]}, index=t))
    assert n == 1 and not bars.index.has_duplicates
    with pytest.raises(ValueError):
        dedupe(pd.DataFrame({"close": [1.0, 1.5, 2.0]}, index=t))
