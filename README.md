# realized-vol-forecasting
Daily realized volatility forecasting on crypto from 5-minute returns: naive, EWMA, GARCH(1,1), HAR-RV and one ML model, walk-forward evaluation with QLIKE, MSE and Diebold-Mariano tests.

## Reproduce

```bash
pip install -r requirements.txt            # + requirements-notebook.txt for plots / notebook
python scripts/download.py                 # BTCUSDT spot 1m klines 2018-01..2026-08 -> data/raw/ (SHA-256 checked, git-ignored)
python scripts/build_rv.py                 # -> data/derived/btcusdt_rv5_daily.csv (+ btcusdt_rv5_excluded_days.csv)
python scripts/run_baselines.py            # -> results/forecasts.csv, results/summary.json
python scripts/run_ml.py --model MLP-QLIKE     # -> results/forecasts_MLP-QLIKE.csv
python scripts/run_ml.py --model Linear-QLIKE  # -> results/forecasts_Linear-QLIKE.csv
python scripts/run_ablation.py             # -> results/forecasts_ablation.csv (HAR-log-OLS, HAR-log-QLIKE, Linear-QLIKE-GLM)
python scripts/evaluate.py                 # -> metrics, DM tests, MSE concentration, seed QLIKE, verdict
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

### QLIKE-trained models (`src/ml.py`, PyTorch on CPU)

| model | specification |
|---|---|
| MLP-QLIKE | 12 inputs, 2 hidden layers of 32 units with ReLU, output `log h(t+1)` |
| Linear-QLIKE | same inputs, loss and protocol, no hidden layer (control) |

- Inputs at origin t, computed on `rv_adj`: `log RV_d`, `log RV_w` (mean of 5 days), `log RV_m`
  (mean of 22 days), `r_t`, `min(r_t, 0)`, day of week of t+1 (one-hot, 7 columns).
- Standardization with the mean and standard deviation of the training window of the current
  refit only, recomputed at every refit.
- Loss: `mean(RV exp(-out) + out - log RV - 1)` with `RV = rv5(t+1)`; training rows with a flagged
  target day are dropped (same rule as HAR).
- Adam, lr 1e-3, full batch (no shuffling), early stopping on the last 10 % of the training rows
  in chronological order (patience 100 epochs, best weights restored), maximum 20000 epochs,
  output bias initialized at `log(mean RV)` of the fitting rows.
- Refit every 22 origins, expanding window. Seeds 0 to 4; the forecast is the mean of the five
  seeds' `h`. One CPU thread and deterministic algorithms: a re-run gives identical forecasts.
- Hyperparameters are fixed and were not tuned on the out-of-sample period. The only value set
  from data is the epoch cap: with a cap of 5000 the linear control did not stop on the refit at
  origin 2019-12-31 (its early-stopping rows are 2019-10-23 to 2019-12-30); without a cap it
  stopped between epochs 7143 and 7582, so the cap was set to 20000.

### Ablation models (`src/ablation.py`)

| model | specification |
|---|---|
| HAR-log-OLS | OLS of `log rv5(t+1)` on `log RV_d, log RV_w, log RV_m` (HAR variables on `rv_adj`); `h = exp(x b + s^2/2)`, `s^2 = SSR/(n-k)` of the current fit; re-estimated daily |
| HAR-log-QLIKE | same inputs, `log h = x b`, `b` minimizes mean QLIKE, solved by a Gamma GLM with log link (statsmodels IRLS, tolerance 1e-10); the Gamma deviance is 2 x the summed QLIKE; re-estimated daily |
| Linear-QLIKE-GLM | the 12 inputs of Linear-QLIKE, standardized on each training window, same Gamma GLM; `dow_6` (Sunday of t+1) is dropped as reference category because statsmodels IRLS does not converge with the intercept plus 7 collinear dummies; intercept + 6 dummies spans the same set of functions; re-estimated daily |
| Linear-QLIKE-GLM-22 | identical, re-estimated every 22 origins (the Linear-QLIKE schedule), used only for the Adam vs GLM comparison |

The ablation chain adds one ingredient per step: HAR (levels, OLS) -> HAR-log-OLS (+ log) ->
HAR-log-QLIKE (+ QLIKE loss) -> Linear-QLIKE-GLM (+ inputs) -> MLP-QLIKE (+ non-linearity).
Each step is tested against the previous one with the same Diebold-Mariano test (Newey-West,
lag `floor(4 (T/100)^(2/9))`). The last step also changes the estimation procedure: MLP-QLIKE uses
Adam with early stopping, a 5-seed mean and a refit every 22 origins, while Linear-QLIKE-GLM is an
exact daily optimum.

### Decision rule

MLP-QLIKE is declared better only if it beats HAR **and** Linear-QLIKE on QLIKE with p < 0.05
(Diebold-Mariano statistic < 0). Otherwise the result is reported as negative.

Metrics: `MSE = mean((RV - h)^2)`, `QLIKE = mean(RV/h - log(RV/h) - 1)`.
Evaluation days: the days evaluable for all session-2 models (the `evaluable` column of
`results/forecasts.csv`); the QLIKE-trained models are evaluated on the same days.
Diebold-Mariano for QLIKE and MSE: Naive, EWMA, GARCH, MLP-QLIKE and Linear-QLIKE against HAR,
and MLP-QLIKE against Linear-QLIKE; `d = loss(model) - loss(benchmark)`,
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
| MLP-QLIKE | 2423 | 1.746373e-05 | 2.887384e-01 |
| Linear-QLIKE | 2423 | 4.444917e-05 | 2.860165e-01 |
| HAR-log-OLS | 2423 | 8.766126e-06 | 3.314852e-01 |
| HAR-log-QLIKE | 2423 | 8.711536e-06 | 3.314109e-01 |
| Linear-QLIKE-GLM | 2423 | 3.107330e-05 | 2.826707e-01 |
| Linear-QLIKE-GLM-22 | 2423 | 1.221348e-04 | 2.832493e-01 |

`results/dm_tests.csv` (d = loss(model) - loss(benchmark))

| model | benchmark | loss | T | nw_lag | mean_d | dm_stat | p_value | sign |
|---|---|---|---|---|---|---|---|---|
| Naive | HAR | MSE | 2423 | 8 | -1.95493e-06 | -0.506445 | 0.612544 | -1 |
| EWMA | HAR | MSE | 2423 | 8 | -5.51134e-06 | -0.905194 | 0.365362 | -1 |
| GARCH | HAR | MSE | 2423 | 8 | -6.11661e-06 | -0.971277 | 0.33141 | -1 |
| MLP-QLIKE | HAR | MSE | 2423 | 8 | 2.88211e-06 | 0.589381 | 0.555606 | 1 |
| MLP-QLIKE | Linear-QLIKE | MSE | 2423 | 8 | -2.69854e-05 | -1.00799 | 0.31346 | -1 |
| Linear-QLIKE | HAR | MSE | 2423 | 8 | 2.98675e-05 | 0.968934 | 0.332578 | 1 |
| Naive | HAR | QLIKE | 2423 | 8 | 0.107258 | 3.14872 | 0.00163986 | 1 |
| EWMA | HAR | QLIKE | 2423 | 8 | 0.0506099 | 1.69166 | 0.0907113 | 1 |
| GARCH | HAR | QLIKE | 2423 | 8 | 0.0456641 | 4.82099 | 1.4285e-06 | 1 |
| MLP-QLIKE | HAR | QLIKE | 2423 | 8 | -0.0876079 | -5.8425 | 5.14242e-09 | -1 |
| MLP-QLIKE | Linear-QLIKE | QLIKE | 2423 | 8 | 0.00272192 | 0.48129 | 0.63031 | 1 |
| Linear-QLIKE | HAR | QLIKE | 2423 | 8 | -0.0903298 | -7.98457 | 1.41017e-15 | -1 |

**Verdict (rule above): negative result. MLP-QLIKE is not declared better: it does not beat Linear-QLIKE on QLIKE with p < 0.05.**

### Ablation

`results/ablation.csv` (DM against the previous step, d = loss(step) - loss(previous))

| step | model | ingredient | QLIKE | MSE | vs | dm_QLIKE | p_QLIKE | dm_MSE | p_MSE | T | nw_lag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | HAR | HAR (levels, OLS) | 0.376346 | 1.45816e-05 |  |  |  |  |  |  |  |
| 1 | HAR-log-OLS | + log | 0.331485 | 8.76613e-06 | HAR | -2.09695 | 0.0359976 | -1.02434 | 0.305673 | 2423 | 8 |
| 2 | HAR-log-QLIKE | + QLIKE loss | 0.331411 | 8.71154e-06 | HAR-log-OLS | -0.00843944 | 0.993266 | -0.622569 | 0.533568 | 2423 | 8 |
| 3 | Linear-QLIKE-GLM | + inputs (r, min(r,0), weekday) | 0.282671 | 3.10733e-05 | HAR-log-QLIKE | -9.8852 | 4.82606e-23 | 1.01447 | 0.310359 | 2423 | 8 |
| 4 | MLP-QLIKE | + non-linearity | 0.288738 | 1.74637e-05 | Linear-QLIKE-GLM | 1.49253 | 0.135562 | -1.03032 | 0.302861 | 2423 | 8 |

What the ablation shows (p < 0.05 between consecutive steps only):

- QLIKE: consecutive steps with p < 0.05: + log (HAR-log-OLS vs HAR, DM -2.097, p = 0.036); + inputs (r, min(r,0), weekday) (Linear-QLIKE-GLM vs HAR-log-QLIKE, DM -9.885, p = 4.83e-23).
- MSE: no consecutive step with p < 0.05.
- QLIKE: consecutive steps with p >= 0.05: + QLIKE loss (HAR-log-QLIKE vs HAR-log-OLS, p = 0.993); + non-linearity (MLP-QLIKE vs Linear-QLIKE-GLM, p = 0.136).

`results/adam_vs_glm.csv` (Linear-QLIKE trained by Adam vs the same model solved by the Gamma GLM; d = loss(Adam) - loss(GLM); relative difference = |h_Adam / h_GLM - 1|)

| adam | glm | QLIKE_adam | QLIKE_glm | QLIKE_diff | dm_QLIKE | p_QLIKE | median_abs_rel_diff | max_abs_rel_diff | max_diff_date |
|---|---|---|---|---|---|---|---|---|---|
| Linear-QLIKE (Adam, 5-seed mean) | Linear-QLIKE-GLM-22 | 0.286016 | 0.283249 | 0.00276718 | 1.00744 | 0.313723 | 0.0396482 | 0.880431 | 2021-12-26 |
| Linear-QLIKE seed 0 | Linear-QLIKE-GLM-22 | 0.286933 | 0.283249 | 0.00368347 | 1.50068 | 0.13344 | 0.0425997 | 0.801876 | 2021-10-27 |
| Linear-QLIKE seed 1 | Linear-QLIKE-GLM-22 | 0.286678 | 0.283249 | 0.00342918 | 0.889688 | 0.373633 | 0.0491226 | 0.933565 | 2021-12-26 |
| Linear-QLIKE seed 2 | Linear-QLIKE-GLM-22 | 0.287285 | 0.283249 | 0.00403552 | 1.30194 | 0.192938 | 0.0492591 | 1.37107 | 2021-12-25 |
| Linear-QLIKE seed 3 | Linear-QLIKE-GLM-22 | 0.28988 | 0.283249 | 0.006631 | 2.19669 | 0.0280428 | 0.0457219 | 1.27673 | 2020-06-24 |
| Linear-QLIKE seed 4 | Linear-QLIKE-GLM-22 | 0.285922 | 0.283249 | 0.00267287 | 0.964628 | 0.334731 | 0.041072 | 0.785689 | 2022-01-21 |
| Linear-QLIKE (Adam, 5-seed mean) | Linear-QLIKE-GLM | 0.286016 | 0.282671 | 0.00334574 | 1.19075 | 0.23375 | 0.0410732 | 0.889757 | 2021-12-26 |
| Linear-QLIKE seed 0 | Linear-QLIKE-GLM | 0.286933 | 0.282671 | 0.00426202 | 1.69571 | 0.089941 | 0.0445508 | 0.80231 | 2021-10-27 |
| Linear-QLIKE seed 1 | Linear-QLIKE-GLM | 0.286678 | 0.282671 | 0.00400773 | 1.01865 | 0.308367 | 0.0512342 | 0.943155 | 2021-12-26 |
| Linear-QLIKE seed 2 | Linear-QLIKE-GLM | 0.287285 | 0.282671 | 0.00461408 | 1.4903 | 0.136146 | 0.0513128 | 1.37121 | 2021-12-25 |
| Linear-QLIKE seed 3 | Linear-QLIKE-GLM | 0.28988 | 0.282671 | 0.00720955 | 2.30116 | 0.0213824 | 0.0465459 | 1.32818 | 2020-06-24 |
| Linear-QLIKE seed 4 | Linear-QLIKE-GLM | 0.285922 | 0.282671 | 0.00325143 | 1.14284 | 0.253106 | 0.0429654 | 0.786397 | 2022-01-21 |

`results/seed_qlike.csv`

| model | seed | QLIKE | MSE |
|---|---|---|---|
| MLP-QLIKE | 0 | 3.009313e-01 | 4.998032e-06 |
| MLP-QLIKE | 1 | 2.865596e-01 | 4.759032e-06 |
| MLP-QLIKE | 2 | 2.965676e-01 | 3.639070e-05 |
| MLP-QLIKE | 3 | 2.915580e-01 | 6.593107e-06 |
| MLP-QLIKE | 4 | 2.924768e-01 | 2.078860e-04 |
| Linear-QLIKE | 0 | 2.869328e-01 | 9.993784e-05 |
| Linear-QLIKE | 1 | 2.866785e-01 | 9.999398e-05 |
| Linear-QLIKE | 2 | 2.872848e-01 | 6.991528e-06 |
| Linear-QLIKE | 3 | 2.898803e-01 | 1.897179e-05 |
| Linear-QLIKE | 4 | 2.859222e-01 | 9.992859e-05 |

Single-seed QLIKE range (min, max): MLP-QLIKE 2.865596e-01 to 3.009313e-01; Linear-QLIKE 2.859222e-01 to 2.898803e-01

`results/mse_concentration.csv` (share of the total squared error due to the 10 worst days)

| model | N | MSE | top10_share | top10_dates |
|---|---|---|---|---|
| Naive | 2423 | 1.26267e-05 | 0.85011 | 2020-03-14 2021-05-19 2020-03-13 2021-05-20 2020-03-12 2021-01-11 2021-09-07 2021-01-12 2021-09-08 2020-03-17 |
| EWMA | 2423 | 9.07028e-06 | 0.814526 | 2020-03-13 2021-05-19 2020-03-12 2021-01-11 2021-09-07 2021-02-23 2021-05-21 2021-05-23 2024-08-05 2021-01-29 |
| GARCH | 2423 | 8.46501e-06 | 0.844586 | 2020-03-13 2021-05-19 2020-03-12 2021-01-11 2021-09-07 2021-02-23 2021-05-21 2021-05-23 2024-08-05 2021-01-29 |
| HAR | 2423 | 1.45816e-05 | 0.909621 | 2020-03-14 2020-03-13 2021-05-19 2020-03-12 2021-01-11 2021-09-07 2021-05-20 2021-02-23 2020-03-16 2024-08-05 |
| MLP-QLIKE | 2423 | 1.74637e-05 | 0.939087 | 2020-03-13 2021-05-19 2020-03-12 2021-01-11 2020-03-14 2021-09-07 2021-02-23 2020-03-16 2020-05-10 2024-08-05 |
| Linear-QLIKE | 2423 | 4.44492e-05 | 0.974071 | 2020-03-13 2021-05-19 2020-03-12 2021-01-11 2021-05-20 2021-09-07 2021-02-23 2020-03-14 2024-08-05 2020-05-10 |
| HAR-log-OLS | 2423 | 8.76613e-06 | 0.861303 | 2020-03-13 2021-05-19 2020-03-12 2020-03-14 2021-01-11 2021-09-07 2021-02-23 2020-03-17 2024-08-05 2021-01-29 |
| HAR-log-QLIKE | 2423 | 8.71154e-06 | 0.860003 | 2020-03-13 2021-05-19 2020-03-12 2020-03-14 2021-01-11 2021-09-07 2021-02-23 2021-01-29 2024-08-05 2020-05-10 |
| Linear-QLIKE-GLM | 2423 | 3.10733e-05 | 0.960432 | 2020-03-13 2021-05-19 2020-03-12 2021-05-20 2021-01-11 2021-09-07 2020-03-14 2021-02-23 2020-03-17 2020-05-10 |
| Linear-QLIKE-GLM-22 | 2423 | 0.000122135 | 0.989929 | 2020-03-13 2021-05-19 2020-03-12 2021-05-20 2021-01-11 2021-09-07 2020-03-14 2021-02-23 2020-05-10 2024-08-05 |

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

## Limits

<!-- limits:start -->
- **Over-forecasts after 2020-03-13.** Ratio forecast / realized `rv5` per model on the target days 2020-03-13 to 2020-03-20, and the largest ratio over all evaluable days (`results/overforecast.csv`):

| model | max_ratio | max_ratio_date | ratio_2020-03-13 | ratio_2020-03-14 | ratio_2020-03-15 | ratio_2020-03-16 | ratio_2020-03-17 | ratio_2020-03-18 | ratio_2020-03-19 | ratio_2020-03-20 |
|---|---|---|---|---|---|---|---|---|---|---|
| Naive | 20.68 | 2023-07-01 | 0.4435 | 14.61 | 0.8232 | 0.4475 | 3.208 | 1.03 | 1.033 | 0.5165 |
| EWMA | 19.57 | 2025-03-22 | 0.1464 | 2.189 | 1.731 | 0.732 | 2.246 | 2.204 | 2.143 | 1.132 |
| GARCH | 81.61 | 2023-08-12 | 0.1711 | 2.296 | 1.622 | 0.6115 | 1.689 | 1.495 | 1.302 | 0.6747 |
| HAR | 64.19 | 2023-08-12 | 0.2153 | 17.7 | 0.1363 | 0.3102 | 2.609 | 1.272 | 1.159 | 0.595 |
| MLP-QLIKE | 14.08 | 2023-08-12 | 2.587 | 4.178 | 1.328 | 0.4104 | 2.138 | 1.661 | 1.125 | 0.807 |
| Linear-QLIKE | 17.47 | 2023-08-12 | 3.804 | 3.003 | 1.142 | 0.5281 | 2.521 | 1.842 | 0.7782 | 0.975 |
| HAR-log-OLS | 17.18 | 2023-08-12 | 0.1374 | 5.001 | 1.296 | 0.6435 | 3.098 | 1.785 | 1.282 | 0.6492 |
| HAR-log-QLIKE | 24.8 | 2023-08-12 | 0.1233 | 4.111 | 1.245 | 0.6039 | 2.742 | 1.608 | 0.95 | 0.4766 |
| Linear-QLIKE-GLM | 15.89 | 2023-08-12 | 3.271 | 3.375 | 1.346 | 0.6195 | 2.984 | 2.163 | 1.025 | 0.9177 |
| Linear-QLIKE-GLM-22 | 16.54 | 2023-08-12 | 5.819 | 3.341 | 1.357 | 0.5629 | 2.756 | 2.066 | 0.7499 | 1.118 |

- **MSE concentration.** The 10 worst days account for between 81.5% and 99.0% of each model's total squared error (`results/mse_concentration.csv`, table above).
- **Market.** Crypto spot data only, traded 24/7: no market closures, overnight gaps or weekends of the kind found in equity or futures markets.
- **One asset.** BTCUSDT spot on one venue only.
- **Ablation order.** Each ingredient is measured in the single order of the chain above; its contribution in another order is not measured.
<!-- limits:end -->

## Licenses

Code under MIT (`LICENSE`). Derived data under CC BY-NC-SA 4.0 (`data/derived/LICENSE-DATA`).
Source: Binance Vision (data.binance.vision).
