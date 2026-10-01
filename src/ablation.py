"""Ablation models between HAR (levels, OLS) and Linear-QLIKE.

HAR-log-OLS     : OLS of log rv5(t+1) on [1, log RV_d, log RV_w, log RV_m];
                  h = exp(x b + s^2 / 2), s^2 = SSR / (n - k) of the current fit.
HAR-log-QLIKE   : log h = x b with the same x, b minimizing mean QLIKE, solved by a
                  Gamma GLM with log link (statsmodels, IRLS). The Gamma unit
                  deviance is 2 (y/mu - log(y/mu) - 1) = 2 x QLIKE, so the GLM
                  optimum is the QLIKE optimum.
Linear-QLIKE-GLM: the 12 inputs of Linear-QLIKE (src.ml.make_features),
                  standardized with the training window of each refit, same Gamma
                  GLM. The 7 weekday dummies plus the intercept are collinear, and
                  statsmodels IRLS does not converge on that rank-deficient design
                  (tested: 1000 iterations on the 2023-12-31 window). The GLM
                  therefore drops dow_6 (Sunday of t+1) as reference category:
                  intercept + 6 dummies spans exactly the same set of functions,
                  so the QLIKE optimum and the information used are unchanged.

RV_d, RV_w, RV_m are those of HAR (rv_adj, 1 / 5 / 22-day means). Training rows:
days s with s+1 <= t, complete inputs, target day s+1 not flagged, rv5(s+1) > 0.
HAR-log-OLS and HAR-log-QLIKE are re-estimated daily; Linear-QLIKE-GLM takes
`refit_every` (1 = daily for the ablation chain, 22 = Linear-QLIKE schedule).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .ml import make_features

GLM_DROP = "dow_6"  # reference category for the Gamma GLM (see module docstring)
from .models import HAR, BaseModel


def har_log_features(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(HAR.features(df["rv_adj"]))


def _rows(df: pd.DataFrame, X: pd.DataFrame):
    y_next = df["rv5"].shift(-1)
    flag_next = df["flag"].shift(-1).astype("boolean")
    ok = (np.isfinite(X).all(axis=1) & y_next.notna() & (y_next > 0) & (flag_next == False))  # noqa: E712
    ok = ok.fillna(False).to_numpy(dtype=bool)
    return X.to_numpy()[ok], y_next.to_numpy()[ok], X.index[ok]


def fit_gamma_log(X: np.ndarray, y: np.ndarray):
    """Gamma GLM, log link, intercept added. Returns the statsmodels results."""
    Xc = np.column_stack([np.ones(len(y)), X])
    start = np.zeros(Xc.shape[1])
    start[0] = np.log(y.mean())
    model = sm.GLM(y, Xc, family=sm.families.Gamma(link=sm.families.links.Log()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(start_params=start, maxiter=200, tol=1e-10)
    return res


class HARLogOLS(BaseModel):
    name = "HAR-log-OLS"

    def __init__(self):
        super().__init__()
        self.coef = None
        self.s2 = None

    def _fit(self, hist):
        X = har_log_features(hist)
        Xtr, ytr, _ = _rows(hist, X)
        A = np.column_stack([np.ones(len(ytr)), Xtr])
        ly = np.log(ytr)
        coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
        resid = ly - A @ coef
        self.coef = coef
        self.s2 = float(resid @ resid / (len(ly) - A.shape[1]))
        x_t = X.iloc[-1].to_numpy()
        self._forecast = np.exp(coef[0] + x_t @ coef[1:] + self.s2 / 2) if np.isfinite(x_t).all() else np.nan


class HARLogQLIKE(BaseModel):
    name = "HAR-log-QLIKE"

    def __init__(self):
        super().__init__()
        self.coef = None
        self.n_not_converged = 0

    def _fit(self, hist):
        X = har_log_features(hist)
        Xtr, ytr, _ = _rows(hist, X)
        res = fit_gamma_log(Xtr, ytr)
        if not res.converged:
            self.n_not_converged += 1
        self.coef = np.asarray(res.params)
        x_t = X.iloc[-1].to_numpy()
        self._forecast = np.exp(self.coef[0] + x_t @ self.coef[1:]) if np.isfinite(x_t).all() else np.nan


class LinearQLIKEGLM(BaseModel):
    def __init__(self, refit_every: int = 1, name: str | None = None):
        super().__init__()
        self.refit_every = refit_every
        self.name = name or ("Linear-QLIKE-GLM" if refit_every == 1 else f"Linear-QLIKE-GLM-{refit_every}")
        self.n_calls = 0
        self.coef = None
        self.mu = self.sd = None
        self.n_not_converged = 0
        self.n_refits = 0

    def _fit(self, hist):
        X = make_features(hist).drop(columns=GLM_DROP)
        if self.n_calls % self.refit_every == 0 or self.coef is None:
            Xtr, ytr, _ = _rows(hist, X)
            self.mu = Xtr.mean(axis=0)
            sd = Xtr.std(axis=0)
            self.sd = np.where(sd > 0, sd, 1.0)
            res = fit_gamma_log((Xtr - self.mu) / self.sd, ytr)
            if not res.converged:
                self.n_not_converged += 1
            self.coef = np.asarray(res.params)
            self.n_refits += 1
        self.n_calls += 1
        x_t = X.iloc[-1].to_numpy()
        if not np.isfinite(x_t).all():
            self._forecast = np.nan
            return
        self._forecast = float(np.exp(self.coef[0] + ((x_t - self.mu) / self.sd) @ self.coef[1:]))
