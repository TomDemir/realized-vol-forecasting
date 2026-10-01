"""RV on a synthetic day, checked against a hand computation."""
import numpy as np
import pandas as pd

from build_rv import daily_rv

DAY = "2021-03-10"


def _bars_for_day(prices_at_minute: dict[int, float], day=DAY, prev_close=None):
    """1m bars: key = minute index of the bar open within the day (0..1439)."""
    d = pd.Timestamp(day, tz="UTC")
    idx = [d + pd.Timedelta(minutes=m) for m in prices_at_minute]
    closes = list(prices_at_minute.values())
    if prev_close is not None:
        idx.insert(0, d - pd.Timedelta(minutes=1))
        closes.insert(0, prev_close)
    return pd.DataFrame({"close": closes}, index=pd.DatetimeIndex(idx))


def test_constant_price_gives_zero_rv():
    bars = _bars_for_day({m: 100.0 for m in range(1440)}, prev_close=100.0)
    out = daily_rv(bars, DAY, DAY)
    row = out.iloc[0]
    assert row["rv5"] == 0.0
    assert row["n_obs"] == 288
    assert row["n_missing"] == 0
    assert not row["flag"]


def test_hand_computed_rv():
    # Flat at 100 except:
    #  bar opened 00:04 (price at grid 00:05) closes at 110
    #  bar opened 12:29 (price at grid 12:30) closes at 90
    # Other 1m bars move intra-bucket but the grid price stays 100: the
    # intermediate minutes must NOT enter the 5-min RV.
    prices = {m: 100.0 for m in range(1440)}
    prices[4] = 110.0
    prices[749] = 90.0
    prices[2] = 250.0  # inside a bucket, not on the grid -> ignored
    bars = _bars_for_day(prices, prev_close=100.0)
    out = daily_rv(bars, DAY, DAY)

    # Returns: 00:00->00:05 log(110/100), 00:05->00:10 log(100/110),
    #          12:25->12:30 log(90/100), 12:30->12:35 log(100/90); all others 0.
    expected = 2 * np.log(1.1) ** 2 + 2 * np.log(0.9) ** 2
    assert np.isclose(out.iloc[0]["rv5"], expected, rtol=0, atol=1e-15)
    assert out.iloc[0]["n_obs"] == 288


def test_overnight_return_belongs_to_next_day():
    # Previous close 100, first grid price of the day (00:05) = 105.
    prices = {m: 105.0 for m in range(1440)}
    bars = _bars_for_day(prices, prev_close=100.0)
    out = daily_rv(bars, DAY, DAY)
    # Grid 00:00 price = close of bar opened 23:59 the day before = 100.
    assert np.isclose(out.iloc[0]["rv5"], np.log(1.05) ** 2)


def test_missing_minutes_no_forward_fill():
    prices = {m: 100.0 * np.exp(0.001 * (m % 2)) for m in range(1440)}
    # Remove bars opened at 00:09 (grid 00:10) and 00:10..00:19 (11 minutes).
    for m in range(9, 20):
        del prices[m]
    bars = _bars_for_day(prices, prev_close=100.0)
    out = daily_rv(bars, DAY, DAY).iloc[0]
    assert out["n_missing"] == 11
    # Grid points affected: 00:10 (bar 00:09) and 00:15 (bar 00:14) are missing.
    # 00:20 needs bar 00:19 -> missing too. Returns lost: ending at 00:10, 00:15,
    # 00:20, 00:25 -> 4 returns.
    assert out["n_obs"] == 288 - 4
    assert not out["flag"]  # 11 <= 72


def test_flag_threshold():
    prices = {m: 100.0 for m in range(1440)}
    for m in range(1000, 1073):  # 73 missing minutes > 72
        del prices[m]
    out = daily_rv(_bars_for_day(prices, prev_close=100.0), DAY, DAY).iloc[0]
    assert out["n_missing"] == 73 and out["flag"]
    prices[1000] = 100.0  # 72 missing -> not flagged
    out = daily_rv(_bars_for_day(prices, prev_close=100.0), DAY, DAY).iloc[0]
    assert out["n_missing"] == 72 and not out["flag"]


def test_fully_missing_day_is_present_and_flagged():
    d = pd.Timestamp(DAY, tz="UTC")
    idx = pd.DatetimeIndex([d - pd.Timedelta(days=1) + pd.Timedelta(minutes=m) for m in range(1440)]
                           + [d + pd.Timedelta(days=1) + pd.Timedelta(minutes=m) for m in range(1440)])
    bars = pd.DataFrame({"close": 100.0}, index=idx)
    out = daily_rv(bars).set_index("date")
    row = out.loc[pd.Timestamp(DAY).date()]
    assert row["n_missing"] == 1440 and row["n_obs"] == 0
    assert np.isnan(row["rv5"]) and row["flag"]


def test_daily_return_equals_sum_of_5min_returns():
    rng = np.random.default_rng(7)
    d0 = pd.Timestamp("2021-03-09 23:59", tz="UTC")
    idx = pd.date_range(d0, periods=1 + 2 * 1440, freq="min")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(idx))))
    bars = pd.DataFrame({"close": close}, index=idx)
    out = daily_rv(bars, "2021-03-10", "2021-03-11").set_index("date")
    c = pd.Series(close, index=idx)
    for day in ["2021-03-10", "2021-03-11"]:
        d = pd.Timestamp(day, tz="UTC")
        expected = np.log(c[d + pd.Timedelta(minutes=1439)] / c[d - pd.Timedelta(minutes=1)])
        assert np.isclose(out.loc[d.date(), "ret"], expected, rtol=1e-12)


def test_daily_return_nan_without_ffill():
    prices = {m: 100.0 for m in range(1439)}  # bar 23:59 missing
    out = daily_rv(_bars_for_day(prices, prev_close=100.0), DAY, DAY).iloc[0]
    assert np.isnan(out["ret"])
