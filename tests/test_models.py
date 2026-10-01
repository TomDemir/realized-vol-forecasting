"""Model mechanics checked against direct computations."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data import add_rv_adj
from src.models import EPS_FLOOR, EWMA, GARCH11, HAR, Naive
from test_walkforward_leakage import synthetic_daily


def test_rv_adj():
    df = pd.DataFrame({"rv5": [1.0, 2.0, np.nan, 3.0], "n_missing": [0, 72, 1440, 720]})
    out = add_rv_adj(df)
    np.testing.assert_allclose(out["rv_adj"].to_numpy()[[0, 1, 3]],
                               [1.0, 2.0 * 1440 / 1368, 6.0])
    assert np.isnan(out["rv_adj"].iloc[2])


def test_naive_uses_rv_adj_of_origin():
    df = synthetic_daily()
    hist = df.loc[:"2019-01-31"]
    f = Naive().fit(hist).predict(pd.Timestamp("2019-02-01"))
    assert f == hist["rv_adj"].iloc[-1]


def test_ewma_manual_recursion():
    r = np.array([0.01, -0.02, 0.015, 0.0, 0.03])
    df = pd.DataFrame({"ret": r}, index=pd.date_range("2020-01-01", periods=5))
    lam = 0.94
    s2 = r[0] ** 2
    for x in r[1:]:
        s2 = lam * s2 + (1 - lam) * x ** 2
    f = EWMA(lam).fit(df).predict(pd.Timestamp("2020-01-06"))
    assert np.isclose(f, s2, rtol=1e-14)


def test_ewma_starts_after_last_nan():
    r = np.array([0.5, np.nan, 0.01, 0.02])
    df = pd.DataFrame({"ret": r}, index=pd.date_range("2020-01-01", periods=4))
    f = EWMA(0.94).fit(df).predict(pd.Timestamp("2020-01-05"))
    assert np.isclose(f, 0.94 * 0.01**2 + 0.06 * 0.02**2)
    df.iloc[-1, 0] = np.nan
    assert np.isnan(EWMA(0.94).fit(df).predict(pd.Timestamp("2020-01-05")))


def test_garch_forecast_matches_recursion():
    df = synthetic_daily()
    hist = df.loc[:"2020-05-01"]
    m = GARCH11()
    f = m.fit(hist).predict(pd.Timestamp("2020-05-02"))
    mu, omega, alpha, beta = m.params[["mu", "omega", "alpha[1]", "beta[1]"]]
    tail = hist["ret"].iloc[102:]  # NaN-free run starts after row 101 (last NaN return)
    assert tail.notna().all() and np.isnan(hist["ret"].iloc[101])
    r = tail * 100
    res = m._model(tail).fix(m.params.to_numpy())
    s2_t = res.conditional_volatility[-1] ** 2
    manual = (omega + alpha * (r.iloc[-1] - mu) ** 2 + beta * s2_t) / 1e4
    assert np.isclose(f, manual, rtol=1e-10)


def test_garch_refit_schedule():
    df = synthetic_daily()
    m = GARCH11(refit_every=22)
    for t in pd.date_range("2020-01-01", periods=45):  # >= 250 clean obs from here
        m.fit(df.loc[:t])
    assert m.n_refits == 3  # calls 0, 22, 44


def test_har_matches_statsmodels_ols():
    df = synthetic_daily()
    hist = df.loc[:"2020-04-30"]
    m = HAR().fit(hist)
    X = HAR.features(hist["rv_adj"])
    y = hist["rv5"].shift(-1)
    fl = hist["flag"].shift(-1)
    ok = X.notna().all(axis=1) & y.notna() & (fl == False)  # noqa: E712
    ols = sm.OLS(y[ok], sm.add_constant(X[ok])).fit()
    np.testing.assert_allclose(m.coef, ols.params.to_numpy(), rtol=1e-8)
    f = m.predict(pd.Timestamp("2020-05-01"))
    assert np.isclose(f, ols.params["const"] + X.iloc[-1] @ ols.params[["rv_d", "rv_w", "rv_m"]])


def test_har_floor_is_counted():
    # Mean-reverting series with a negative day-to-day slope; a large last value
    # pushes the linear forecast below zero.
    rng = np.random.default_rng(8)
    n = 300
    x = np.empty(n)
    x[0] = 2.0
    for i in range(1, n):
        x[i] = 3.8 - 0.9 * x[i - 1] + 0.05 * rng.standard_normal()
    x[-1] = 30.0
    idx = pd.date_range("2020-01-01", periods=n)
    df = add_rv_adj(pd.DataFrame({"rv5": x, "n_missing": 0, "flag": False}, index=idx))
    m = HAR()
    f = m.fit(df).predict(idx[-1] + pd.Timedelta(days=1))
    raw = m.coef[0] + HAR.features(df["rv_adj"]).iloc[-1].to_numpy() @ m.coef[1:]
    assert raw <= 0
    assert f == EPS_FLOOR
    assert m.n_floored == 1
    assert m.floored_dates == [idx[-1] + pd.Timedelta(days=1)]
