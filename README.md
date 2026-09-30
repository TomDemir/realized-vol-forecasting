# realized-vol-forecasting
Daily realized volatility forecasting on crypto from 5-minute returns: naive, EWMA, GARCH(1,1), HAR-RV and one ML model, walk-forward evaluation with QLIKE, MSE and Diebold-Mariano tests.

## Data layer (session 1)

```bash
pip install -r requirements.txt
python scripts/download.py      # BTCUSDT spot 1m klines 2018-01..2026-08 -> data/raw/ (SHA-256 checked, git-ignored)
python scripts/build_rv.py      # -> data/derived/btcusdt_rv5_daily.csv (+ btcusdt_rv5_excluded_days.csv)
pytest
```

`btcusdt_rv5_daily.csv` columns: `date` (UTC day), `rv5` (sum of squared 5-min log returns on a
fixed UTC grid, no forward fill), `n_obs` (valid 5-min returns, max 288), `n_missing` (missing 1m
bars out of 1440), `flag` (`n_missing > 72`, i.e. > 5 %). Conventions are documented at the top of
`scripts/build_rv.py`. The notebook needs `requirements-notebook.txt`.

Licenses: code under MIT (`LICENSE`); derived data under CC BY-NC-SA 4.0
(`data/derived/LICENSE-DATA`). Source: Binance Vision (data.binance.vision).
