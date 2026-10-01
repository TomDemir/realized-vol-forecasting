"""ms -> µs switch of Binance Vision spot klines at 2025-01-01."""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from build_rv import normalize_open_time, read_month_zip

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

LAST_MS = 1735689540000  # 2024-12-31 23:59:00 UTC, milliseconds
FIRST_US = 1735689600000000  # 2025-01-01 00:00:00 UTC, microseconds


def test_known_epochs():
    ts = normalize_open_time(np.array([LAST_MS, FIRST_US]))
    assert ts[0] == pd.Timestamp("2024-12-31 23:59:00", tz="UTC")
    assert ts[1] == pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    assert ts[1] - ts[0] == pd.Timedelta(minutes=1)


def test_mixed_units_are_contiguous_across_boundary():
    ms = LAST_MS - 60_000 * np.arange(5)[::-1]  # 23:55 .. 23:59 in ms
    us = (FIRST_US + 60_000_000 * np.arange(5))  # 00:00 .. 00:04 in µs
    ts = normalize_open_time(np.concatenate([ms, us]))
    assert ts.is_monotonic_increasing
    assert (ts[1:] - ts[:-1] == pd.Timedelta(minutes=1)).all()
    assert str(ts.tz) == "UTC"


def test_rejects_sub_millisecond_microseconds():
    with pytest.raises(ValueError):
        normalize_open_time(np.array([FIRST_US + 1]))


def _make_zip(tmp_path: Path, month: str, open_times: list[int]) -> Path:
    rows = "".join(f"{t},1,1,1,100.0,1,{t},1,1,1,1,0\n" for t in open_times)
    p = tmp_path / f"BTCUSDT-1m-{month}.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(f"BTCUSDT-1m-{month}.csv", rows)
    return p


def test_read_month_zip_both_units(tmp_path):
    dec = read_month_zip(_make_zip(tmp_path, "2024-12", [LAST_MS]))
    jan = read_month_zip(_make_zip(tmp_path, "2025-01", [FIRST_US]))
    both = pd.concat([dec, jan])
    assert list(both.index) == [pd.Timestamp("2024-12-31 23:59", tz="UTC"),
                                pd.Timestamp("2025-01-01 00:00", tz="UTC")]


def test_read_month_zip_rejects_wrong_unit(tmp_path):
    # A µs value misread as ms would land far outside the month -> must fail loudly.
    bad = _make_zip(tmp_path, "2025-01", [FIRST_US // 1000 * 10**6])
    with pytest.raises(ValueError):
        read_month_zip(bad)


@pytest.mark.skipif(not (RAW / "BTCUSDT-1m-2024-12.zip").exists()
                    or not (RAW / "BTCUSDT-1m-2025-01.zip").exists(),
                    reason="raw files for 2024-12 / 2025-01 not downloaded")
def test_real_files_boundary():
    dec = read_month_zip(RAW / "BTCUSDT-1m-2024-12.zip")
    jan = read_month_zip(RAW / "BTCUSDT-1m-2025-01.zip")
    assert dec.index.max() == pd.Timestamp("2024-12-31 23:59", tz="UTC")
    assert jan.index.min() == pd.Timestamp("2025-01-01 00:00", tz="UTC")
    with zipfile.ZipFile(RAW / "BTCUSDT-1m-2025-01.zip") as z:
        first = z.open(z.namelist()[0]).readline().split(b",")[0]
    assert int(first) >= 10**15  # raw file really is in microseconds


def test_off_grid_bars_dropped_not_snapped(tmp_path):
    base = 1518170400000  # 2018-02-09 10:00:00 UTC, ms
    aligned = [base - 60_000, base + 60_000]
    off = [base + 14_789]  # 10:00:14.789 -> off the 1m grid
    df = read_month_zip(_make_zip(tmp_path, "2018-02", aligned + off))
    assert list(df.index) == [pd.Timestamp("2018-02-09 09:59", tz="UTC"),
                              pd.Timestamp("2018-02-09 10:01", tz="UTC")]
