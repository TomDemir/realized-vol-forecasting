#!/usr/bin/env python3
"""Build daily 5-minute realized variance for BTCUSDT spot from Binance 1m klines.

Conventions
-----------
* Timestamps: Binance Vision spot klines use milliseconds before 2025-01-01 and
  microseconds from 2025-01-01. Each value is normalized individually
  (value >= 1e14 -> microseconds, else milliseconds). A millisecond epoch only
  reaches 1e14 in the year 5138, and a microsecond epoch is below 1e14 only
  before 1973, so the threshold is unambiguous for this dataset.
* Price at grid point t (t on the fixed UTC 5-minute grid, t = HH:00, HH:05, ...)
  is the close of the 1m bar that opened at t - 1 min, i.e. the last trade price
  of the minute ending at t.
* 5-min log return r_t = log P_t - log P_{t-5min}. Day D uses the 288 returns
  whose end points fall in (D 00:00, D+1 00:00]; the first one starts from the
  close of the last bar of D-1.
* 1m bars whose open_time is not a whole minute (one block in 2018-02, see
  read_month_zip) are dropped, not snapped, and count as missing minutes.
* No forward fill: if either end point is missing, the return is missing and not
  counted. rv5 = sum of squared available returns; n_obs = number of such returns.
* n_missing = number of the 1440 one-minute bars of day D (open times D 00:00 to
  D 23:59) absent from the raw data. flag = n_missing > 5 % of 1440 (= 72).
  Flagged days are the excluded days; they are listed in a separate file.

Usage:
    python scripts/build_rv.py [--raw data/raw] [--out data/derived]
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 1440
RETURNS_PER_DAY = 288
MISSING_THRESHOLD = 0.05  # fraction of the 1440 minutes
US_THRESHOLD = 10**14  # >= this -> microseconds, else milliseconds

OUT_NAME = "btcusdt_rv5_daily.csv"
EXCLUDED_NAME = "btcusdt_rv5_excluded_days.csv"
ZIP_RE = re.compile(r"BTCUSDT-1m-(\d{4})-(\d{2})\.zip$")


# --------------------------------------------------------------------------- #
# Timestamp handling
# --------------------------------------------------------------------------- #
def normalize_open_time(raw: pd.Series | np.ndarray) -> pd.DatetimeIndex:
    """Convert raw Binance open_time values (ms or µs, mixed allowed) to UTC datetimes."""
    v = np.asarray(raw, dtype=np.int64)
    if (v < 0).any():
        raise ValueError("negative timestamps in raw data")
    ms = np.where(v >= US_THRESHOLD, v // 1000, v)
    if (v >= US_THRESHOLD).any() and (v[v >= US_THRESHOLD] % 1000 != 0).any():
        raise ValueError("microsecond timestamps with sub-millisecond part: unexpected format")
    return pd.DatetimeIndex(pd.to_datetime(ms, unit="ms", utc=True)).as_unit("ns")


# --------------------------------------------------------------------------- #
# Raw loading
# --------------------------------------------------------------------------- #
def read_month_zip(path: Path) -> pd.DataFrame:
    """Read one monthly zip -> DataFrame(index=open_time UTC, column=close)."""
    m = ZIP_RE.search(path.name)
    if not m:
        raise ValueError(f"unexpected file name {path.name}")
    year, month = int(m.group(1)), int(m.group(2))
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{path.name}: expected 1 csv, found {names}")
        with z.open(names[0]) as f:
            df = pd.read_csv(f, header=None, usecols=[0, 4], names=["open_time", "close"],
                             dtype=str)
    # Some Binance files may carry a header row: drop non-numeric first rows explicitly.
    numeric = df["open_time"].str.fullmatch(r"\d+")
    if not numeric.all():
        bad = df.loc[~numeric, "open_time"].tolist()
        if len(bad) > 1 or numeric.iloc[0]:
            raise ValueError(f"{path.name}: non-numeric open_time values {bad[:5]}")
        df = df.loc[numeric]
    ts = normalize_open_time(df["open_time"].astype(np.int64).to_numpy())
    close = df["close"].astype(np.float64).to_numpy()
    out = pd.DataFrame({"close": close}, index=ts)
    out.index.name = "open_time"
    # Every bar must belong to the month named by the file.
    lo = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    hi = lo + pd.offsets.MonthBegin(1)
    outside = (out.index < lo) | (out.index >= hi)
    if outside.any():
        raise ValueError(f"{path.name}: {outside.sum()} bars outside {lo:%Y-%m}")
    # Bars whose open_time is not on the 1m UTC grid are NOT snapped to the
    # minute: they are dropped and therefore counted as missing minutes.
    # Known case: 2018-02-09 09:59:14.789 .. 2018-02-10 05:59:14.789 (1201 bars
    # offset by +14.789 s after the 2018-02-08 exchange halt).
    off_grid = out.index != out.index.floor("min")
    if off_grid.any():
        offs = sorted(set((out.index[off_grid] - out.index[off_grid].floor("min")).total_seconds()))
        print(f"[warn] {path.name}: dropping {int(off_grid.sum())} off-grid bars "
              f"({out.index[off_grid].min()} .. {out.index[off_grid].max()}, "
              f"offsets s={offs[:5]}); counted as missing minutes", file=sys.stderr)
        out = out.loc[~off_grid]
    if not (out["close"] > 0).all():
        raise ValueError(f"{path.name}: non-positive close prices")
    return out


def dedupe(bars: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate minutes. Conflicting duplicates raise."""
    dup = bars.index.duplicated(keep=False)
    if not dup.any():
        return bars, 0
    grp = bars[dup].groupby(level=0)["close"].nunique()
    conflicting = grp[grp > 1]
    if len(conflicting):
        raise ValueError(f"conflicting duplicate bars at {list(conflicting.index[:10])}")
    n_dropped = int(bars.index.duplicated(keep="first").sum())
    return bars[~bars.index.duplicated(keep="first")], n_dropped


def load_raw(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("BTCUSDT-1m-*.zip"))
    if not files:
        raise FileNotFoundError(f"no BTCUSDT-1m-*.zip in {raw_dir}")
    frames = [read_month_zip(f) for f in files]
    bars = pd.concat(frames).sort_index()
    bars, n_dup = dedupe(bars)
    if n_dup:
        print(f"[info] dropped {n_dup} exact duplicate 1m bars")
    return bars


# --------------------------------------------------------------------------- #
# Realized variance
# --------------------------------------------------------------------------- #
def daily_rv(bars: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Compute the daily RV table from 1m bars (index = open_time UTC, column close).

    start / end are inclusive UTC dates ('YYYY-MM-DD'); by default the first and
    last dates present in `bars`. Every calendar day in [start, end] appears in
    the output, including days with no data at all.
    """
    idx = bars.index
    if idx.has_duplicates:
        raise ValueError("duplicate 1m bars: call dedupe() first")
    d0 = pd.Timestamp(start, tz="UTC") if start else idx.min().floor("D")
    d1 = pd.Timestamp(end, tz="UTC") if end else idx.max().floor("D")
    days = pd.date_range(d0, d1, freq="D")

    # --- missing-minute count on the full fixed 1m grid of each day ---------
    minute_grid = pd.date_range(d0, d1 + pd.Timedelta(days=1), freq="min", inclusive="left")
    present = pd.Series(minute_grid.isin(idx), index=minute_grid)
    n_present = present.groupby(present.index.floor("D")).sum().reindex(days, fill_value=0)
    n_missing = MINUTES_PER_DAY - n_present.astype(int)

    # --- 5-min prices on the fixed grid, no forward fill ---------------------
    # Price at grid point t = close of bar opened at t - 1 min.
    grid = pd.date_range(d0, d1 + pd.Timedelta(days=1), freq="5min")  # includes D1+1 00:00
    price = bars["close"].reindex(grid - pd.Timedelta(minutes=1))
    price.index = grid
    logp = np.log(price)
    r = logp.diff()  # r at t = log P_t - log P_{t-5}; NaN if either end is missing
    r = r.iloc[1:]  # drop d0 00:00, which has no predecessor on the grid
    # A return ending at t belongs to the day of (t - 5 min): ends in (D 00:00, D+1 00:00].
    day_of_r = (r.index - pd.Timedelta(minutes=5)).floor("D")
    rv5 = np.square(r).groupby(day_of_r).sum(min_count=1)  # NaN if no valid return
    n_obs = r.groupby(day_of_r).count()

    out = pd.DataFrame(index=days)
    out["rv5"] = rv5.reindex(days)
    out["n_obs"] = n_obs.reindex(days, fill_value=0).astype(int)
    out["n_missing"] = n_missing.astype(int)
    out["flag"] = out["n_missing"] > MISSING_THRESHOLD * MINUTES_PER_DAY
    out.index = out.index.date
    out.index.name = "date"
    return out.reset_index()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/derived")
    p.add_argument("--start", default=None, help="first UTC date (default: first date in data)")
    p.add_argument("--end", default=None, help="last UTC date (default: last date in data)")
    args = p.parse_args(argv)

    bars = load_raw(Path(args.raw))
    print(f"[info] {len(bars):,} 1m bars from {bars.index.min()} to {bars.index.max()}")
    table = daily_rv(bars, args.start, args.end)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / OUT_NAME, index=False, float_format="%.10e")
    excluded = table.loc[table["flag"], ["date", "n_missing", "n_obs"]]
    excluded.to_csv(out_dir / EXCLUDED_NAME, index=False)

    print(f"\nWrote {out_dir / OUT_NAME}")
    print(f"Total days      : {len(table)}")
    print(f"Flagged days    : {int(table['flag'].sum())} (n_missing > {int(MISSING_THRESHOLD * MINUTES_PER_DAY)})")
    print(f"Days with any missing minute: {int((table['n_missing'] > 0).sum())}")
    print("\nExcluded (flagged) days:")
    print(excluded.to_string(index=False) if len(excluded) else "  none")
    print("\nTop 10 days by missing minutes:")
    top = table.sort_values(["n_missing", "date"], ascending=[False, True]).head(10)
    print(top[["date", "n_missing", "n_obs", "flag"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
