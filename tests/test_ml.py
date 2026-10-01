"""MLP-QLIKE and Linear-QLIKE: leakage, standardization, reproducibility, inputs, loss."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.data import DEFAULT_CSV, load_daily
from src.evaluation import qlike_loss
from src.ml import FEATURES, linear_qlike, make_features, mlp_qlike, qlike_log_loss, training_rows
from test_walkforward_leakage import perturb_after, synthetic_daily

# Small settings keep the tests fast; the code path is the production one.
FAST = dict(seeds=(0, 1), max_epochs=300, patience=20)
FACTORIES = {"MLP-QLIKE": mlp_qlike, "Linear-QLIKE": linear_qlike}


def run(df, name, start, end, **kw):
    model = FACTORIES[name](**{**FAST, **kw})
    out = {}
    for target in df.loc[start:end].index:
        origin = target - pd.Timedelta(days=1)
        model.fit(df.loc[:origin].copy())
        assert model.last_date == origin and target > model.last_date
        out[target] = model.predict(target)
    return pd.Series(out), model


@pytest.mark.parametrize("name", list(FACTORIES))
@pytest.mark.parametrize("cut", ["2020-01-20", "2020-03-10"])
def test_no_leakage_synthetic(name, cut):
    df = synthetic_daily()
    t = pd.Timestamp(cut)
    base, _ = run(df, name, "2020-01-01", "2020-04-30")
    pert, _ = run(perturb_after(df, t), name, "2020-01-01", "2020-04-30")
    upto = base.index <= t + pd.Timedelta(days=1)
    assert np.isfinite(base.to_numpy()).all()
    np.testing.assert_array_equal(base[upto].to_numpy(), pert[upto].to_numpy())
    assert not np.allclose(base[~upto].to_numpy(), pert[~upto].to_numpy())


@pytest.mark.skipif(not DEFAULT_CSV.exists(), reason="derived CSV not built")
@pytest.mark.parametrize("name", list(FACTORIES))
def test_no_leakage_real(name):
    df = load_daily()
    t = pd.Timestamp("2024-01-20")
    base, _ = run(df, name, "2024-01-01", "2024-02-15")
    pert, _ = run(perturb_after(df, t), name, "2024-01-01", "2024-02-15")
    upto = base.index <= t + pd.Timedelta(days=1)
    assert np.isfinite(base.to_numpy()).all()
    np.testing.assert_array_equal(base[upto].to_numpy(), pert[upto].to_numpy())


@pytest.mark.parametrize("name", list(FACTORIES))
def test_standardization_uses_only_data_up_to_t(name):
    df = synthetic_daily()
    t = pd.Timestamp("2020-02-15")
    model = FACTORIES[name](**FAST)
    model.fit(df.loc[:t].copy())
    # Independent recomputation from rows whose target day is <= t.
    hist = df.loc[:t]
    X = make_features(hist)
    Xtr, _, rows = training_rows(hist, X)
    assert rows.max() + pd.Timedelta(days=1) <= t
    assert model.train_last_row == rows.max()
    np.testing.assert_allclose(model.mu, Xtr.mean(axis=0), rtol=0, atol=0)
    sd = Xtr.std(axis=0)
    np.testing.assert_allclose(model.sd, np.where(sd > 0, sd, 1.0), rtol=0, atol=0)
    # Statistics from a longer window differ, and future data cannot change them.
    longer = make_features(df.loc[:"2020-05-31"])
    assert not np.allclose(model.mu, training_rows(df.loc[:"2020-05-31"], longer)[0].mean(axis=0))
    m2 = FACTORIES[name](**FAST)
    m2.fit(perturb_after(df, t).loc[:t].copy())
    np.testing.assert_array_equal(model.mu, m2.mu)
    np.testing.assert_array_equal(model.sd, m2.sd)


def test_standardization_refreshed_at_each_refit():
    df = synthetic_daily()
    model = mlp_qlike(**FAST, refit_every=22)
    mus = []
    for t in pd.date_range("2020-01-01", periods=45):
        model.fit(df.loc[:t].copy())
        mus.append(model.mu.copy())
    np.testing.assert_array_equal(mus[0], mus[21])  # no refit in between
    assert not np.array_equal(mus[21], mus[22])  # refit at call 22
    assert not np.array_equal(mus[22], mus[44])  # refit at call 44


@pytest.mark.parametrize("name", list(FACTORIES))
def test_fixed_seeds_identical_rerun(name):
    df = synthetic_daily()
    a, ma = run(df, name, "2020-02-01", "2020-03-15")
    b, mb = run(df, name, "2020-02-01", "2020-03-15")
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    np.testing.assert_array_equal(ma.seed_forecasts, mb.seed_forecasts)


def test_ensemble_is_mean_of_seeds():
    df = synthetic_daily()
    _, m = run(df, "MLP-QLIKE", "2020-02-01", "2020-02-01")
    assert m.predict(pd.Timestamp("2020-02-01")) == pytest.approx(m.seed_forecasts.mean(), rel=1e-15)
    assert len(set(m.seed_forecasts)) == len(m.seeds)  # seeds really differ


def test_architectures():
    df = synthetic_daily()
    mlp = mlp_qlike(**FAST).fit(df.loc[:"2020-01-31"].copy())
    lin = linear_qlike(**FAST).fit(df.loc[:"2020-01-31"].copy())
    shapes = [tuple(p.shape) for p in mlp.nets[0].parameters()]
    assert shapes == [(32, 12), (32,), (32, 32), (32,), (1, 32), (1,)]
    assert [tuple(p.shape) for p in lin.nets[0].parameters()] == [(1, 12), (1,)]
    assert any(isinstance(m, torch.nn.ReLU) for m in mlp.nets[0])


def test_features_hand():
    idx = pd.date_range("2024-01-01", periods=30)  # 2024-01-01 is a Monday
    rv = np.arange(1, 31, dtype=float) * 1e-4
    ret = np.where(np.arange(30) % 2 == 0, 0.01, -0.02)
    df = pd.DataFrame({"rv_adj": rv, "ret": ret}, index=idx)
    X = make_features(df)
    assert list(X.columns) == FEATURES
    row = X.loc["2024-01-30"]  # index 29: rv = 30e-4, ret = -0.02; t+1 = Wednesday
    assert np.isclose(row["log_rv_d"], np.log(30e-4))
    assert np.isclose(row["log_rv_w"], np.log(np.mean(rv[25:30])))
    assert np.isclose(row["log_rv_m"], np.log(np.mean(rv[8:30])))
    assert row["r"] == -0.02 and row["r_neg"] == -0.02
    assert X.loc["2024-01-29", "r_neg"] == 0.0  # positive return -> 0
    dow = row[[f"dow_{k}" for k in range(7)]].to_numpy()
    assert dow.tolist() == [0, 0, 1, 0, 0, 0, 0]  # Wednesday = 2
    assert X.loc["2024-01-20":"2024-01-21", "log_rv_m"].isna().tolist() == [True, True]


def test_qlike_log_loss_matches_evaluation_qlike():
    rv = np.array([2e-4, 5e-4, 1e-3])
    out = np.log(np.array([3e-4, 5e-4, 4e-4]))
    torch_loss = float(qlike_log_loss(torch.tensor(out), torch.tensor(rv)))
    assert np.isclose(torch_loss, qlike_loss(rv, np.exp(out)).mean(), rtol=1e-12)
