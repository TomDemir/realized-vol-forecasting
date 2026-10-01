# realized-vol-forecasting
Daily realized volatility forecasting on crypto from 5-minute returns: naive, EWMA, GARCH(1,1), HAR-RV and one ML model, walk-forward evaluation with QLIKE, MSE and Diebold-Mariano tests.

## Reproduce

```bash
pip install -r requirements.txt            # + requirements-notebook.txt for plots / notebook
python scripts/download.py                 # BTCUSDT spot 1m klines 2018-01..2026-08 -> data/raw/ (SHA-256 checked, git-ignored)
python scripts/build_rv.py                 # -> data/derived/btcusdt_rv5_daily.csv (+ btcusdt_rv5_excluded_days.csv)
python scripts/run_baselines.py            # -> results/*.csv, results/summary.json
python scripts/plot_2024.py                # -> results/forecast_vs_realized_2024.png
python scripts/update_readme_results.py    # copies results/ tables into this README
pytest
```

## Data

`data/derived/btcusdt_rv5_daily.csv`, one row per UTC day from 2018-01-01 to 2026-08-31:

| column | definition |
|---|---|
| `date` | UTC day |
| `rv5` | sum of squared 5-min log returns on a fixed UTC grid (price at grid point t = close of the 1m bar opened at t - 1 min), no forward fill |
| `n_obs` | number of valid 5-min returns (max 288) |
| `n_missing` | missing 1m bars out of 1440 |
| `flag` | `n_missing > 72` (more than 5 % missing) |
| `ret` | close-to-close log return of the UTC day: close of the bar opened at D 23:59 vs the same bar of D-1; NaN if either bar is missing (no forward fill) |

Conventions are documented at the top of `scripts/build_rv.py`.

### Off-grid bars in 2018-02

`BTCUSDT-1m-2018-02.zip` contains 1201 contiguous 1m bars whose open time is offset by +14.789 s
from the minute grid, from 2018-02-09 09:59:14.789 to 2018-02-10 05:59:14.789 UTC. They are
dropped, not snapped to the minute, and counted as missing minutes. Impact: one day changes status.
2018-02-10 is flagged with 375 missing minutes; with snapping it would have 15 and would not be
flagged. 2018-02-09 is flagged under both choices.

## Flagged-day policy

- **Explanatory variable**: `rv_adj = rv5 * 1440 / (1440 - n_missing)`; `rv_adj = NaN` when
  `n_missing = 1440`. A forecast whose explanatory-variable window contains a NaN is marked not
  evaluable.
- **Target**: `rv5` of day t+1. Flagged target days are excluded from the loss computation,
  identically for every model. Losses are computed on the days evaluable for all models.
- **Return-based models (EWMA, GARCH)**: a variance recursion cannot run through a missing
  return, so they use the longest NaN-free run of `ret` ending at t. Missing returns are
  2018-01-01, 2018-02-08, 2018-02-09 and 2018-02-10, so this run starts on 2018-02-11 for every
  out-of-sample origin. If `ret(t)` is NaN the forecast is not evaluable.

## Walk-forward protocol (`src/walkforward.py`)

- One-day-ahead: at the end of day t, forecast the RV of day t+1 using only rows with date <= t.
  The engine hands each model a copy of `df.loc[:t]`; `predict` refuses any target date that is
  not exactly t+1.
- Out-of-sample target days: 2020-01-01 to 2026-08-31, expanding window.
- Each model exposes `fit(data <= t)` and `predict(t+1)` (`src/models.py`).

| model | specification |
|---|---|
| Naive | `RV_hat(t+1) = rv_adj(t)` |
| EWMA | RiskMetrics on squared daily returns, lambda = 0.94 fixed; `s2(t+1) = 0.94 s2(t) + 0.06 ret(t)^2`, started at `ret^2` of the first day of the NaN-free run |
| GARCH | GARCH(1,1), constant mean, normal errors (`arch`), returns x100; parameters re-estimated on the first origin and every 22 origins; conditional variance filtered daily with the latest parameters on data <= t; minimum 250 returns |
| HAR | HAR-RV in levels (Corsi 2009) by OLS, re-estimated daily: `RV(t+1) = b0 + bd RV(t) + bw mean(RV t-4..t) + bm mean(RV t-21..t)`; regressors use `rv_adj`, left-hand side is `rv5` with flagged target days dropped; forecasts <= 0 are floored at 1e-8 and counted |

Metrics: `MSE = mean((RV - h)^2)`, `QLIKE = mean(RV/h - log(RV/h) - 1)`.
Diebold-Mariano: each model against HAR, for QLIKE and MSE, `d = loss(model) - loss(HAR)`,
Newey-West (Bartlett) HAC variance with lag `floor(4 (T/100)^(2/9))`, two-sided normal p-value,
`sign` = sign of the statistic.

## Results

<!-- results:start -->
`results/metrics.csv`

| model | N | MSE | QLIKE |
|---|---|---|---|
| Naive | 2423 | 1.262669e-05 | 4.836047e-01 |
| EWMA | 2423 | 9.070281e-06 | 4.269562e-01 |
| GARCH | 2423 | 8.465005e-06 | 4.220103e-01 |
| HAR | 2423 | 1.458162e-05 | 3.763463e-01 |

`results/dm_tests.csv` (d = loss(model) - loss(HAR))

| model | benchmark | loss | T | nw_lag | mean_d | dm_stat | p_value | sign |
|---|---|---|---|---|---|---|---|---|
| Naive | HAR | MSE | 2423 | 8 | -1.95493e-06 | -0.506445 | 0.612544 | -1 |
| EWMA | HAR | MSE | 2423 | 8 | -5.51134e-06 | -0.905194 | 0.365362 | -1 |
| GARCH | HAR | MSE | 2423 | 8 | -6.11661e-06 | -0.971277 | 0.33141 | -1 |
| Naive | HAR | QLIKE | 2423 | 8 | 0.107258 | 3.14872 | 0.00163986 | 1 |
| EWMA | HAR | QLIKE | 2423 | 8 | 0.0506099 | 1.69166 | 0.0907113 | 1 |
| GARCH | HAR | QLIKE | 2423 | 8 | 0.0456641 | 4.82099 | 1.4285e-06 | 1 |

`results/summary.json`

| item | value |
|---|---|
| out-of-sample target days | 2435 |
| flagged target days (excluded) | 12 |
| evaluable days common to all models | 2423 |
| HAR forecasts floored at 1e-08 | 0 |
| GARCH re-estimations | 111 |
| GARCH fits with non-zero convergence flag | 0 |

![HAR and GARCH forecasts vs realized RV, 2024](results/forecast_vs_realized_2024.png)
<!-- results:end -->

## Licenses

Code under MIT (`LICENSE`). Derived data under CC BY-NC-SA 4.0 (`data/derived/LICENSE-DATA`).
Source: Binance Vision (data.binance.vision).
