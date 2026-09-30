"""Anti-leakage tests: forecasts must depend only on data <= origin."""
import numpy as np
import pandas as pd
import pytest

from src.data import DEFAULT_CSV, add_rv_adj, load_daily
from src.models import EWMA, GARCH11, HAR, Naive
from src.walkforward import walk_forward

MODEL_FACTORIES = {
    "Naive": lambda: Naive(),
    "EWMA": lambda: EWMA(0.94),
    "GARCH": lambda: GARCH11(refit_every=22),
    "HAR": lambda: HAR(),
}


def synthetic_daily(n=520, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    logv = np.empty(n)
    logv[0] = np.log(1e-3)
    for i in range(1, n):
        logv[i] = 0.95 * logv[i - 1] + 0.05 * np.log(1e-3) + 0.3 * rng.standard_normal()
    rv = np.exp(logv)
    ret = np.sqrt(rv) * rng.standard_normal(n)
    n_missing = np.zeros(n, dtype=int)
    n_missing[[30, 100, 460]] = [100, 1440, 10]
    rv[100] = np.nan
    ret[[100, 101]] = np.nan
    df = pd.DataFrame({"rv5": rv, "n_obs": 288, "n_missing": n_missing,
                       "flag": n_missing > 72, "ret": ret}, index=dates)
    df.index.name = "date"
    return add_rv_adj(df)


def perturb_after(df, t, seed=1):
    rng = np.random.default_rng(seed)
    out = df.copy()
    after = out.index > t
    k = after.sum()
    out.loc[after, "rv5"] = out.loc[after, "rv5"] * rng.uniform(0.1, 10, k)
    out.loc[after, "ret"] = rng.standard_normal(k) * 0.2
    out.loc[after, "n_missing"] = rng.integers(0, 1441, k)
    out.loc[after, "flag"] = out.loc[after, "n_missing"] > 72
    return add_rv_adj(out.drop(columns="rv_adj"))


def _run(df, name, start, end):
    long = walk_forward(df, [MODEL_FACTORIES[name]()], start, end)
    return long.set_index("target_date")["forecast"]


@pytest.mark.parametrize("name", list(MODEL_FACTORIES))
@pytest.mark.parametrize("cut", ["2020-01-20", "2020-03-10"])
def test_no_leakage_synthetic(name, cut):
    df = synthetic_daily()
    t = pd.Timestamp(cut)
    base = _run(df, name, "2020-01-01", "2020-04-30")
    pert = _run(perturb_after(df, t), name, "2020-01-01", "2020-04-30")
    upto = base.index <= t + pd.Timedelta(days=1)
    assert upto.sum() > 0
    assert np.isfinite(base.to_numpy()).all()  # the test is not vacuous
    np.testing.assert_array_equal(base[upto].to_numpy(), pert[upto].to_numpy())
    # sanity: the perturbation does change later forecasts
    assert not np.allclose(base[~upto].to_numpy(), pert[~upto].to_numpy(), equal_nan=True)


@pytest.mark.skipif(not DEFAULT_CSV.exists(), reason="derived CSV not built")
@pytest.mark.parametrize("name", list(MODEL_FACTORIES))
def test_no_leakage_real(name):
    df = load_daily()
    t = pd.Timestamp("2024-01-20")
    base = _run(df, name, "2024-01-01", "2024-02-15")
    pert = _run(perturb_after(df, t), name, "2024-01-01", "2024-02-15")
    upto = base.index <= t + pd.Timedelta(days=1)
    assert np.isfinite(base.to_numpy()).all()
    np.testing.assert_array_equal(base[upto].to_numpy(), pert[upto].to_numpy())


@pytest.mark.parametrize("name", list(MODEL_FACTORIES))
def test_forecast_dates_strictly_after_data(name):
    df = synthetic_daily()
    long = walk_forward(df, [MODEL_FACTORIES[name]()], "2020-01-01", "2020-02-29")
    assert (long["target_date"] > long["data_end"]).all()
    assert (long["target_date"] - long["origin"] == pd.Timedelta(days=1)).all()
    assert (long["data_end"] == long["origin"]).all()


def test_predict_rejects_non_future_target():
    df = synthetic_daily()
    m = Naive().fit(df.loc[:"2020-01-10"])
    for bad in ["2020-01-10", "2020-01-09"]:
        with pytest.raises(ValueError):
            m.predict(pd.Timestamp(bad))
    with pytest.raises(ValueError):
        m.predict(pd.Timestamp("2020-01-12"))  # not one day ahead
    assert np.isfinite(m.predict(pd.Timestamp("2020-01-11")))
