#!/usr/bin/env python3
"""Run the baseline walk-forward and write results/.

Outputs: results/forecasts.csv, results/metrics.csv, results/dm_tests.csv,
results/garch_fits.csv, results/summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_daily  # noqa: E402
from src.evaluation import common_mask, dm_table, metrics_table  # noqa: E402
from src.models import EWMA, GARCH11, HAR, Naive  # noqa: E402
from src.walkforward import OOS_END, OOS_START, to_wide, walk_forward  # noqa: E402

MODEL_ORDER = ["Naive", "EWMA", "GARCH", "HAR"]


def main() -> int:
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df = load_daily()
    har, garch = HAR(), GARCH11(refit_every=22)
    models = [Naive(), EWMA(0.94), garch, har]
    long = walk_forward(df, models, OOS_START, OOS_END)
    assert (long["target_date"] > long["data_end"]).all()
    fc = to_wide(long)[MODEL_ORDER]

    mask = common_mask(fc, df)
    rv = df.loc[fc.index, "rv5"]
    fc_out = fc.copy()
    fc_out.insert(0, "rv5", rv)
    fc_out.insert(1, "flag", df.loc[fc.index, "flag"])
    fc_out["evaluable"] = mask
    fc_out.index.name = "date"
    fc_out.to_csv(out / "forecasts.csv", float_format="%.10e")

    metrics = metrics_table(fc[mask], rv[mask])
    metrics.to_csv(out / "metrics.csv", index=False, float_format="%.6e")
    dm = dm_table(fc[mask], rv[mask], bench="HAR")
    dm.to_csv(out / "dm_tests.csv", index=False, float_format="%.6g")
    pd.DataFrame(garch.fit_log).to_csv(out / "garch_fits.csv", index=False)

    non_eval = {m: int((~fc[m].notna()).sum()) for m in fc.columns}
    summary = {
        "oos_start": OOS_START, "oos_end": OOS_END,
        "n_targets": int(len(fc)),
        "n_flagged_targets": int(df.loc[fc.index, "flag"].sum()),
        "n_not_evaluable_forecasts": non_eval,
        "n_evaluable_common": int(mask.sum()),
        "har_n_floored": har.n_floored,
        "har_floored_dates": [d.strftime("%Y-%m-%d") for d in har.floored_dates],
        "har_eps_floor": har.eps,
        "garch_n_refits": garch.n_refits,
        "garch_nonzero_convergence_flags": int(sum(r["convergence_flag"] != 0 for r in garch.fit_log)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200)
    print("metrics.csv\n", metrics.to_string(index=False), sep="")
    print("\ndm_tests.csv\n", dm.to_string(index=False), sep="")
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
