"""Baseline one-day-ahead variance forecasters.

Interface: fit(hist) receives only rows with date <= t (the forecast origin);
predict(target_date) returns the forecast for t+1 (NaN = not evaluable).
predict refuses any target date that is not strictly after the last date seen
in fit.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

EPS_FLOOR = 1e-8  # HAR floor for non-positive forecasts


class BaseModel:
    name = "base"

    def __init__(self):
        self.last_date: pd.Timestamp | None = None
        self._forecast = np.nan

    def fit(self, hist: pd.DataFrame) -> "BaseModel":
        self.last_date = hist.index.max()
        self._fit(hist)
        return self

    def predict(self, target_date: pd.Timestamp) -> float:
        if self.last_date is None:
            raise RuntimeError("fit() must be called before predict()")
        if not target_date > self.last_date:
            raise ValueError(f"target {target_date} not strictly after data end {self.last_date}")
        if target_date != self.last_date + pd.Timedelta(days=1):
            raise ValueError("only one-day-ahead forecasts are supported")
        return float(self._forecast)

    def _fit(self, hist: pd.DataFrame) -> None:  # pragma: no cover
        raise NotImplementedError


def clean_return_tail(r: pd.Series) -> pd.Series:
    """Longest NaN-free run of returns ending at the last date (empty if last is NaN)."""
    na = r.isna().to_numpy()
    if na.any():
        last_na = np.flatnonzero(na)[-1]
        return r.iloc[last_na + 1:]
    return r


class Naive(BaseModel):
    """RV_hat(t+1) = rv_adj(t)."""
    name = "Naive"

    def _fit(self, hist):
        self._forecast = hist["rv_adj"].iloc[-1]


class EWMA(BaseModel):
    """RiskMetrics on daily squared returns, fixed lambda = 0.94.

    sigma2(t+1) = lambda * sigma2(t) + (1 - lambda) * r(t)^2, started at
    sigma2 = r(s)^2 on the first day s of the NaN-free return run ending at t.
    """
    name = "EWMA"

    def __init__(self, lam: float = 0.94):
        super().__init__()
        self.lam = lam

    def _fit(self, hist):
        r = clean_return_tail(hist["ret"])
        if len(r) == 0:
            self._forecast = np.nan
            return
        s2 = np.square(r).ewm(alpha=1 - self.lam, adjust=False).mean()
        self._forecast = s2.iloc[-1]


class GARCH11(BaseModel):
    """GARCH(1,1), constant mean, normal errors (arch package).

    Parameters re-estimated on the first call and then every `refit_every`
    calls (one call per forecast origin); every day the conditional variance is
    filtered with the latest parameters on data <= t only. Returns are scaled by
    100 for numerical stability; the forecast is scaled back.
    """
    name = "GARCH"
    SCALE = 100.0

    def __init__(self, refit_every: int = 22, min_obs: int = 250):
        super().__init__()
        self.refit_every = refit_every
        self.min_obs = min_obs
        self.params: pd.Series | None = None
        self.n_calls = 0
        self.n_refits = 0
        self.fit_log: list[dict] = []

    def _model(self, r: pd.Series):
        from arch import arch_model
        return arch_model(r.to_numpy() * self.SCALE, mean="Constant", vol="GARCH", p=1, q=1,
                          dist="normal", rescale=False)

    def _fit(self, hist):
        r = clean_return_tail(hist["ret"])
        refit_due = self.n_calls % self.refit_every == 0
        self.n_calls += 1
        if len(r) < self.min_obs:
            self._forecast = np.nan
            return
        am = self._model(r)
        if refit_due or self.params is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = am.fit(disp="off", show_warning=False)
            self.params = res.params.copy()
            self.n_refits += 1
            self.fit_log.append({"origin": hist.index.max(), "nobs": len(r),
                                 "convergence_flag": int(res.convergence_flag),
                                 **{k: float(v) for k, v in res.params.items()}})
        fixed = am.fix(self.params.to_numpy())
        f = fixed.forecast(horizon=1, reindex=False)
        self._forecast = float(f.variance.to_numpy()[-1, 0]) / self.SCALE**2


class HAR(BaseModel):
    """HAR-RV in levels (Corsi 2009), OLS, re-estimated every day.

    RV(t+1) = b0 + bd RV(t) + bw mean(RV t-4..t) + bm mean(RV t-21..t)
    Regressors use rv_adj; the left-hand side is the target, rv5, and rows whose
    target day is flagged are dropped (same rule as the loss). Rows whose
    regressor window contains a NaN are dropped. Forecasts <= 0 are floored at
    EPS_FLOOR and counted in n_floored.
    """
    name = "HAR"

    def __init__(self, eps: float = EPS_FLOOR):
        super().__init__()
        self.eps = eps
        self.n_floored = 0
        self.floored_dates: list[pd.Timestamp] = []
        self.coef: np.ndarray | None = None

    @staticmethod
    def features(rv_adj: pd.Series) -> pd.DataFrame:
        # rolling(min_periods=window) returns NaN if any value in the window is NaN
        return pd.DataFrame({
            "rv_d": rv_adj,
            "rv_w": rv_adj.rolling(5, min_periods=5).mean(),
            "rv_m": rv_adj.rolling(22, min_periods=22).mean(),
        })

    def _fit(self, hist):
        X = self.features(hist["rv_adj"])
        y_next = hist["rv5"].shift(-1)  # target of row s is day s+1 (<= t by construction)
        flag_next = hist["flag"].shift(-1).astype("boolean")
        ok = X.notna().all(axis=1) & y_next.notna() & (flag_next == False)  # noqa: E712
        ok = ok.fillna(False).to_numpy(dtype=bool)
        Xm = np.column_stack([np.ones(ok.sum()), X.to_numpy()[ok]])
        coef, *_ = np.linalg.lstsq(Xm, y_next.to_numpy()[ok], rcond=None)
        self.coef = coef
        x_t = X.iloc[-1].to_numpy()
        if np.isnan(x_t).any():
            self._forecast = np.nan
            return
        f = coef[0] + x_t @ coef[1:]
        if f <= 0:
            self.n_floored += 1
            self.floored_dates.append(hist.index.max() + pd.Timedelta(days=1))
            f = self.eps
        self._forecast = f


def default_models() -> list[BaseModel]:
    return [Naive(), EWMA(0.94), GARCH11(refit_every=22), HAR()]
