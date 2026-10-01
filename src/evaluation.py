"""Losses, common evaluation sample and Diebold-Mariano test."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def mse_loss(rv, h):
    rv, h = np.asarray(rv, float), np.asarray(h, float)
    return (rv - h) ** 2


def qlike_loss(rv, h):
    rv, h = np.asarray(rv, float), np.asarray(h, float)
    x = rv / h
    return x - np.log(x) - 1.0


LOSSES = {"MSE": mse_loss, "QLIKE": qlike_loss}


def common_mask(forecasts: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    """Target days evaluable for every model: target not flagged, target rv5 > 0,
    and a finite, positive forecast from each model."""
    tgt = df.loc[forecasts.index]
    ok = (~tgt["flag"]) & tgt["rv5"].notna() & (tgt["rv5"] > 0)
    ok &= np.isfinite(forecasts).all(axis=1) & (forecasts > 0).all(axis=1)
    return ok


def metrics_table(forecasts: pd.DataFrame, rv: pd.Series) -> pd.DataFrame:
    rows = []
    for m in forecasts.columns:
        rows.append({"model": m, "N": len(rv),
                     "MSE": float(mse_loss(rv, forecasts[m]).mean()),
                     "QLIKE": float(qlike_loss(rv, forecasts[m]).mean())})
    return pd.DataFrame(rows)


def nw_lag(T: int) -> int:
    return int(math.floor(4 * (T / 100) ** (2 / 9)))


def newey_west_lrv(d: np.ndarray, lag: int) -> float:
    """Newey-West (Bartlett) long-run variance of d."""
    d = np.asarray(d, float)
    T = len(d)
    u = d - d.mean()
    lrv = u @ u / T
    for k in range(1, lag + 1):
        gamma = u[k:] @ u[:-k] / T
        lrv += 2 * (1 - k / (lag + 1)) * gamma
    return lrv


def diebold_mariano(loss_model: np.ndarray, loss_bench: np.ndarray) -> dict:
    """d = loss_model - loss_bench. DM = mean(d) / sqrt(LRV/T), N(0,1), two-sided.
    Positive DM: the benchmark has the lower loss."""
    d = np.asarray(loss_model, float) - np.asarray(loss_bench, float)
    T = len(d)
    lag = nw_lag(T)
    lrv = newey_west_lrv(d, lag)
    stat = d.mean() / math.sqrt(lrv / T)
    p = 2 * stats.norm.sf(abs(stat))  # survival function: no underflow to 0 for large |stat|
    return {"T": T, "nw_lag": lag, "mean_d": d.mean(), "dm_stat": stat, "p_value": p,
            "sign": int(np.sign(stat))}


def dm_table(forecasts: pd.DataFrame, rv: pd.Series, bench: str = "HAR",
             pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """DM tests for (model, benchmark) pairs; default: every model against `bench`."""
    if pairs is None:
        pairs = [(m, bench) for m in forecasts.columns if m != bench]
    rows = []
    for loss_name, fn in LOSSES.items():
        for m, b in pairs:
            res = diebold_mariano(fn(rv, forecasts[m]), fn(rv, forecasts[b]))
            rows.append({"model": m, "benchmark": b, "loss": loss_name, **res})
    return pd.DataFrame(rows)


def mse_concentration(forecasts: pd.DataFrame, rv: pd.Series, k: int = 10) -> pd.DataFrame:
    """Share of the total squared error due to the k days with the largest squared error."""
    rows = []
    for m in forecasts.columns:
        se = pd.Series(mse_loss(rv, forecasts[m]), index=forecasts.index)
        top = se.nlargest(k)
        rows.append({"model": m, "N": len(se), "MSE": float(se.mean()),
                     f"top{k}_share": float(top.sum() / se.sum()),
                     f"top{k}_dates": " ".join(d.strftime("%Y-%m-%d") for d in top.index)})
    return pd.DataFrame(rows)
