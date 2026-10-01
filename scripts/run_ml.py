#!/usr/bin/env python3
"""Walk-forward for one QLIKE-trained model; writes results/forecasts_<name>.csv
(ensemble forecast + one column per seed) and results/fits_<name>.csv."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_daily  # noqa: E402
from src.ml import linear_qlike, mlp_qlike  # noqa: E402
from src.walkforward import OOS_END, OOS_START  # noqa: E402

FACTORIES = {"MLP-QLIKE": mlp_qlike, "Linear-QLIKE": linear_qlike}


def run(name: str, df: pd.DataFrame, start=OOS_START, end=OOS_END, **kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same engine contract as src.walkforward.walk_forward, plus per-seed forecasts."""
    model = FACTORIES[name](**kw)
    rows = []
    for target in df.loc[start:end].index:
        origin = target - pd.Timedelta(days=1)
        hist = df.loc[:origin].copy()
        model.fit(hist)
        f = model.predict(target)
        rows.append({"date": target, "origin": origin, "data_end": model.last_date, name: f,
                     **{f"{name}_seed{s}": v for s, v in zip(model.seeds, model.seed_forecasts)}})
    return pd.DataFrame(rows).set_index("date"), pd.DataFrame(model.fit_log)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(FACTORIES), required=True)
    args = p.parse_args()
    t0 = time.time()
    fc, fits = run(args.model, load_daily())
    assert (fc.index > fc["data_end"]).all()
    out = ROOT / "results"
    fc.drop(columns=["origin", "data_end"]).to_csv(out / f"forecasts_{args.model}.csv", float_format="%.10e")
    fits.to_csv(out / f"fits_{args.model}.csv", index=False)
    print(f"{args.model}: {len(fc)} forecasts, {fits['origin'].nunique()} refits, "
          f"{int((fits['epochs_run'] >= 20000).sum())} fits at the epoch cap, {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
