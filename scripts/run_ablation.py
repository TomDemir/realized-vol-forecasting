#!/usr/bin/env python3
"""Walk-forward for the ablation models -> results/forecasts_ablation.csv,
results/ablation_fit_summary.json."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ablation import HARLogOLS, HARLogQLIKE, LinearQLIKEGLM  # noqa: E402
from src.data import load_daily  # noqa: E402
from src.walkforward import OOS_END, OOS_START, to_wide, walk_forward  # noqa: E402


def main() -> int:
    t0 = time.time()
    df = load_daily()
    models = [HARLogOLS(), HARLogQLIKE(), LinearQLIKEGLM(refit_every=1), LinearQLIKEGLM(refit_every=22)]
    long = walk_forward(df, models, OOS_START, OOS_END)
    assert (long["target_date"] > long["data_end"]).all()
    fc = to_wide(long)[[m.name for m in models]]
    fc.index.name = "date"
    out = ROOT / "results"
    fc.to_csv(out / "forecasts_ablation.csv", float_format="%.10e")
    summary = {m.name: {"n_not_converged": getattr(m, "n_not_converged", None),
                        "n_refits": getattr(m, "n_refits", len(fc))} for m in models}
    (out / "ablation_fit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), f"\n{time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
