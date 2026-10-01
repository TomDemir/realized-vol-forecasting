"""Ablation models: Gamma deviance = 2 x QLIKE, GLM optimum = QLIKE optimum, HAR-log-OLS."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

from src.ablation import GLM_DROP, HARLogOLS, HARLogQLIKE, LinearQLIKEGLM, _rows, fit_gamma_log, har_log_features
from src.evaluation import qlike_loss
from src.ml import make_features
from test_walkforward_leakage import synthetic_daily

GAMMA_LOG = sm.families.Gamma(link=sm.families.links.Log())


def test_gamma_deviance_equals_twice_qlike():
    rng = np.random.default_rng(0)
    y = np.exp(rng.normal(-7, 1, 500))
    mu = np.exp(rng.normal(-7, 1, 500))
    dev = GAMMA_LOG.deviance(y, mu)
    assert np.isclose(dev, 2 * qlike_loss(y, mu).sum(), rtol=1e-12)
    # per observation: unit deviance of each point = 2 x its QLIKE
    unit = np.array([GAMMA_LOG.deviance(y[i:i + 1], mu[i:i + 1]) for i in range(len(y))])
    np.testing.assert_allclose(unit, 2 * qlike_loss(y, mu), rtol=1e-10)
    # hand case: y = 2, mu = 1 -> 2 (2 - ln 2 - 1)
    assert np.isclose(GAMMA_LOG.deviance(np.array([2.0]), np.array([1.0])), 2 * (1 - np.log(2)))


def _qlike_opt(X, y, b0):
    Xc = np.column_stack([np.ones(len(y)), X])

    def f(b):
        return qlike_loss(y, np.exp(Xc @ b)).mean()

    res = minimize(f, b0, method="BFGS", options={"gtol": 1e-12, "maxiter": 10000})
    return res, f


def test_glm_solves_the_qlike_problem():
    df = synthetic_daily()
    hist = df.loc[:"2020-04-30"]
    X = har_log_features(hist)
    Xtr, ytr, _ = _rows(hist, X)
    glm = fit_gamma_log(Xtr, ytr)
    assert glm.converged
    # GLM deviance is exactly 2 x the summed QLIKE at its own estimate.
    mu = np.exp(np.column_stack([np.ones(len(ytr)), Xtr]) @ glm.params)
    assert np.isclose(glm.deviance, 2 * qlike_loss(ytr, mu).sum(), rtol=1e-12)
    # A generic optimizer on mean QLIKE, started away from the GLM solution, finds the same point.
    opt, f = _qlike_opt(Xtr, ytr, np.r_[np.log(ytr.mean()), np.zeros(Xtr.shape[1])])
    assert f(glm.params) <= opt.fun + 1e-12
    np.testing.assert_allclose(glm.params, opt.x, rtol=0, atol=1e-5)


def test_dropping_reference_dummy_keeps_the_qlike_optimum():
    df = synthetic_daily()
    hist = df.loc[:"2020-04-30"]
    X = make_features(hist)
    Xtr, ytr, _ = _rows(hist, X)
    mu_, sd = Xtr.mean(0), Xtr.std(0)
    Xs = (Xtr - mu_) / np.where(sd > 0, sd, 1)
    keep = [c != GLM_DROP for c in X.columns]
    glm = fit_gamma_log(Xs[:, keep], ytr)
    assert glm.converged
    full_opt, f_full = _qlike_opt(Xs, ytr, np.r_[np.log(ytr.mean()), np.zeros(Xs.shape[1])])
    Xc = np.column_stack([np.ones(len(ytr)), Xs[:, keep]])
    q_glm = qlike_loss(ytr, np.exp(Xc @ glm.params)).mean()
    assert q_glm <= full_opt.fun + 1e-10  # same function class: no loss from the reparametrization


def test_har_log_ols_matches_statsmodels():
    df = synthetic_daily()
    hist = df.loc[:"2020-04-30"]
    m = HARLogOLS().fit(hist)
    X = har_log_features(hist)
    Xtr, ytr, _ = _rows(hist, X)
    ols = sm.OLS(np.log(ytr), sm.add_constant(Xtr)).fit()
    np.testing.assert_allclose(m.coef, ols.params, rtol=1e-9)
    assert np.isclose(m.s2, ols.ssr / ols.df_resid, rtol=1e-12)
    x_t = X.iloc[-1].to_numpy()
    expected = np.exp(ols.params[0] + x_t @ ols.params[1:] + ols.scale / 2)
    assert np.isclose(m.predict(pd.Timestamp("2020-05-01")), expected, rtol=1e-12)


def test_har_log_qlike_forecast_formula():
    df = synthetic_daily()
    hist = df.loc[:"2020-04-30"]
    m = HARLogQLIKE().fit(hist)
    x_t = har_log_features(hist).iloc[-1].to_numpy()
    assert np.isclose(m.predict(pd.Timestamp("2020-05-01")), np.exp(m.coef[0] + x_t @ m.coef[1:]))
    assert m.n_not_converged == 0


def test_linear_glm_standardization_and_refit():
    df = synthetic_daily()
    m = LinearQLIKEGLM(refit_every=22)
    mus = []
    for t in pd.date_range("2020-01-01", periods=23):
        m.fit(df.loc[:t].copy())
        mus.append(m.mu.copy())
    hist = df.loc[:"2020-01-01"]
    Xtr, _, rows = _rows(hist, make_features(hist).drop(columns=GLM_DROP))
    np.testing.assert_array_equal(mus[0], Xtr.mean(axis=0))
    assert rows.max() + pd.Timedelta(days=1) <= pd.Timestamp("2020-01-01")
    np.testing.assert_array_equal(mus[0], mus[21])
    assert not np.array_equal(mus[21], mus[22])
    assert m.n_refits == 2 and m.n_not_converged == 0


def test_feature_subsets_use_the_right_columns():
    from src.ablation import FEATURE_SETS, feature_subset_models
    df = synthetic_daily()
    hist = df.loc[:"2020-04-30"]
    for m in feature_subset_models():
        m.fit(hist)
        cols = FEATURE_SETS[m.name]
        assert len(m.coef) == 1 + len(cols) and len(m.mu) == len(cols)
        X = make_features(hist)[cols]
        Xtr, _, _ = _rows(hist, X)
        np.testing.assert_array_equal(m.mu, Xtr.mean(axis=0))
    assert FEATURE_SETS["HAR-log-QLIKE+dow"][:3] == ["log_rv_d", "log_rv_w", "log_rv_m"]
    assert "dow_6" not in FEATURE_SETS["HAR-log-QLIKE+dow"]
    assert FEATURE_SETS["HAR-log-QLIKE+ret"][3:] == ["r", "r_neg"]


def test_har_log_inputs_identical_in_both_feature_builders():
    df = synthetic_daily()
    a = har_log_features(df).to_numpy()
    b = make_features(df)[["log_rv_d", "log_rv_w", "log_rv_m"]].to_numpy()
    np.testing.assert_array_equal(a, b)
