"""Losses and DM test on hand-computed cases."""
import math

import numpy as np
import pandas as pd

from src.evaluation import (common_mask, diebold_mariano, mse_loss, newey_west_lrv, nw_lag,
                            qlike_loss)


def test_mse_hand():
    rv = [2.0, 1.0, 4.0]
    h = [1.0, 1.0, 2.0]
    # (2-1)^2 = 1, 0, (4-2)^2 = 4 -> mean 5/3
    assert np.allclose(mse_loss(rv, h), [1.0, 0.0, 4.0])
    assert math.isclose(mse_loss(rv, h).mean(), 5 / 3)


def test_qlike_hand():
    rv = [2.0, 1.0, 1.0]
    h = [1.0, 1.0, 2.0]
    # RV/h - log(RV/h) - 1: 2 - ln2 - 1 = 1 - ln2 ; 0 ; 0.5 + ln2 - 1 = ln2 - 0.5
    expected = [1 - math.log(2), 0.0, math.log(2) - 0.5]
    assert np.allclose(qlike_loss(rv, h), expected, rtol=0, atol=1e-15)
    assert math.isclose(qlike_loss(rv, h).mean(), 0.5 / 3, rel_tol=1e-12)


def test_qlike_zero_iff_perfect():
    assert qlike_loss([3.0], [3.0])[0] == 0.0
    assert (qlike_loss([1.0, 1.0], [0.5, 2.0]) > 0).all()


def test_nw_lag_formula():
    assert nw_lag(100) == 4
    assert nw_lag(2423) == math.floor(4 * (24.23) ** (2 / 9))
    assert nw_lag(2423) == 8


def test_newey_west_hand():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    # u = [-1.5,-0.5,0.5,1.5]; g0 = (2.25+0.25+0.25+2.25)/4 = 1.25
    # g1 = (u1u0 + u2u1 + u3u2)/4 = (0.75 - 0.25 + 0.75)/4 = 0.3125
    # lag 1 (Bartlett weight 1/2): g0 + 2*(1/2)*g1 = 1.5625
    assert math.isclose(newey_west_lrv(d, 1), 1.5625)
    assert math.isclose(newey_west_lrv(d, 0), 1.25)


def test_dm_hand():
    lm = np.array([2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0])
    lb = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    d = lm - lb
    out = diebold_mariano(lm, lb)
    lag = nw_lag(8)
    expected = d.mean() / math.sqrt(newey_west_lrv(d, lag) / 8)
    assert out["nw_lag"] == lag and math.isclose(out["dm_stat"], expected)
    assert out["sign"] == 1 and 0 <= out["p_value"] <= 1


def test_common_mask_excludes_flagged_for_all():
    idx = pd.date_range("2020-01-01", periods=4)
    df = pd.DataFrame({"rv5": [1.0, 1.0, 1.0, 1.0], "flag": [False, True, False, False]}, index=idx)
    fc = pd.DataFrame({"A": [1.0, 1.0, np.nan, 1.0], "B": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    assert common_mask(fc, df).tolist() == [True, False, False, True]
