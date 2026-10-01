"""QLIKE-trained neural forecasters: MLP-QLIKE and its linear control.

Inputs at origin t (all known at the end of day t, computed on rv_adj):
  log RV_d = log rv_adj(t)
  log RV_w = log mean(rv_adj t-4..t)
  log RV_m = log mean(rv_adj t-21..t)
  r_t, min(r_t, 0)
  day-of-week of t+1, one-hot (7 columns)
Output: log h(t+1). Loss: mean(RV * exp(-out) + out - log RV - 1), RV = rv5(t+1).

Training rows at a refit on origin t: every day s with s+1 <= t whose inputs
are complete and whose target day s+1 is not flagged (same rule as HAR).
Standardization: mean and standard deviation (ddof=0) of the training rows of
the current refit only (zero std -> 1). Early stopping on the last 10 % of the
training rows in chronological order; no shuffling (full-batch Adam).

Fixed hyperparameters, not tuned on any out-of-sample data:
  hidden (32, 32) ReLU for MLP-QLIKE, none for Linear-QLIKE; Adam lr 1e-3;
  full batch; max 20000 epochs; patience 100 epochs; best validation weights
  restored; output bias initialized at log(mean RV) of the fitting rows;
  refit every 22 origins; seeds 0..4; forecast = mean of the 5 seeds' h.
The only value set by looking at data is max_epochs: 5000 was first tried and
the linear control hit that cap on the refit at origin 2019-12-31 (validation
rows 2019-10-23..2019-12-30); without a cap it stopped at epochs 7143-7582, so
the cap was raised to 20000. No out-of-sample data (targets >= 2020-01-01) was
used. Torch runs on one CPU thread with deterministic algorithms so that a
re-run reproduces the forecasts exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from .models import BaseModel

FEATURES = ["log_rv_d", "log_rv_w", "log_rv_m", "r", "r_neg"] + [f"dow_{k}" for k in range(7)]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature row for each origin date s (uses data <= s only)."""
    rv = df["rv_adj"]
    out = pd.DataFrame(index=df.index)
    out["log_rv_d"] = np.log(rv)
    out["log_rv_w"] = np.log(rv.rolling(5, min_periods=5).mean())
    out["log_rv_m"] = np.log(rv.rolling(22, min_periods=22).mean())
    out["r"] = df["ret"]
    out["r_neg"] = np.minimum(df["ret"], 0.0)
    dow_next = (df.index + pd.Timedelta(days=1)).dayofweek  # weekday of t+1, Monday = 0
    for k in range(7):
        out[f"dow_{k}"] = (dow_next == k).astype(float)
    return out[FEATURES]


def training_rows(df: pd.DataFrame, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    y_next = df["rv5"].shift(-1)
    flag_next = df["flag"].shift(-1).astype("boolean")
    ok = (X.notna().all(axis=1) & np.isfinite(X).all(axis=1) & y_next.notna() & (y_next > 0)
          & (flag_next == False))  # noqa: E712
    ok = ok.fillna(False).to_numpy(dtype=bool)
    return X.to_numpy()[ok], y_next.to_numpy()[ok], X.index[ok]


def qlike_log_loss(out: torch.Tensor, rv: torch.Tensor) -> torch.Tensor:
    return (rv * torch.exp(-out) + out - torch.log(rv) - 1.0).mean()


def build_net(n_in: int, hidden: tuple[int, ...]) -> nn.Module:
    layers: list[nn.Module] = []
    d = n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def train_net(Xs: np.ndarray, y: np.ndarray, hidden: tuple[int, ...], seed: int, lr: float,
              max_epochs: int, patience: int, val_frac: float) -> tuple[nn.Module, dict]:
    n = len(y)
    n_val = max(1, int(round(val_frac * n)))
    n_fit = n - n_val
    Xt = torch.tensor(Xs, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64).unsqueeze(1)
    Xf, yf, Xv, yv = Xt[:n_fit], yt[:n_fit], Xt[n_fit:], yt[n_fit:]

    torch.manual_seed(seed)
    net = build_net(Xs.shape[1], hidden).double()
    with torch.no_grad():
        net[-1].bias.fill_(float(np.log(y[:n_fit].mean())))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best, best_state, best_epoch, wait = np.inf, None, -1, 0
    for epoch in range(max_epochs):
        net.train()
        opt.zero_grad()
        loss = qlike_log_loss(net(Xf), yf)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            v = float(qlike_log_loss(net(Xv), yv))
        if v < best:
            best, best_epoch, wait = v, epoch, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, {"best_epoch": best_epoch, "epochs_run": epoch + 1, "val_qlike": best,
                 "n_fit": n_fit, "n_val": n_val}


class QLIKENet(BaseModel):
    """Walk-forward wrapper: refit every `refit_every` origins, predict daily."""

    def __init__(self, name: str, hidden: tuple[int, ...], seeds=(0, 1, 2, 3, 4), lr: float = 1e-3,
                 max_epochs: int = 20000, patience: int = 100, val_frac: float = 0.10,
                 refit_every: int = 22):
        super().__init__()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.name = name
        self.hidden = tuple(hidden)
        self.seeds = tuple(seeds)
        self.lr, self.max_epochs, self.patience, self.val_frac = lr, max_epochs, patience, val_frac
        self.refit_every = refit_every
        self.n_calls = 0
        self.nets: list[nn.Module] = []
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.train_last_row: pd.Timestamp | None = None
        self.fit_log: list[dict] = []
        self.seed_forecasts: np.ndarray = np.full(len(self.seeds), np.nan)

    def _refit(self, hist: pd.DataFrame, X: pd.DataFrame) -> None:
        Xtr, ytr, rows = training_rows(hist, X)
        self.mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)
        Xs = (Xtr - self.mu) / self.sd
        self.nets = []
        for s in self.seeds:
            net, info = train_net(Xs, ytr, self.hidden, s, self.lr, self.max_epochs, self.patience,
                                  self.val_frac)
            self.nets.append(net)
            self.fit_log.append({"origin": hist.index.max(), "seed": s, "n_rows": len(ytr),
                                 "last_row": rows.max(), **info})
        self.train_last_row = rows.max()

    def _fit(self, hist: pd.DataFrame) -> None:
        X = make_features(hist)
        if self.n_calls % self.refit_every == 0 or not self.nets:
            self._refit(hist, X)
        self.n_calls += 1
        x_t = X.iloc[-1].to_numpy()
        if not np.isfinite(x_t).all():
            self.seed_forecasts = np.full(len(self.seeds), np.nan)
            self._forecast = np.nan
            return
        xs = torch.tensor(((x_t - self.mu) / self.sd)[None, :], dtype=torch.float64)
        with torch.no_grad():
            h = np.array([float(torch.exp(net(xs))[0, 0]) for net in self.nets])
        self.seed_forecasts = h
        self._forecast = float(h.mean())


def mlp_qlike(**kw) -> QLIKENet:
    return QLIKENet("MLP-QLIKE", hidden=(32, 32), **kw)


def linear_qlike(**kw) -> QLIKENet:
    return QLIKENet("Linear-QLIKE", hidden=(), **kw)
