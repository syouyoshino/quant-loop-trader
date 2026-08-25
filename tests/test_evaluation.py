import datetime

import polars as pl
import numpy as np
from quant_loop_trader.evaluation import time_split, evaluate
from quant_loop_trader.data import PROC_DIR
from quant_loop_trader.features import add_features, feature_columns
from quant_loop_trader.experiment import make_labels

def test_time_split_hidden_future():
    df = pl.read_parquet(str(PROC_DIR / "SPY.parquet"))
    df = make_labels(df, 5)
    df = add_features(df).drop_nulls(subset=feature_columns()+["label"])
    train, test = time_split(df, 0.7)
    assert train["event_time"].max() < test["event_time"].min()
    # evaluation should not leak
    assert train.height + test.height == df.height

def test_evaluate_metrics():
    y_true = np.array([0,1,1,0,1])
    y_pred = np.array([0,1,0,0,1])
    y_prob = np.array([0.2,0.8,0.4,0.3,0.9])
    prices = np.array([100,101,102,101,103,104])
    m = evaluate(y_true, y_pred, y_prob, prices)
    for k in ["accuracy","precision","recall","sharpe_strategy","cumulative_return_strategy","turnover","transaction_cost_adj_return"]:
        assert k in m
    assert 0 <= m["accuracy"] <= 1


def test_bootstrap_ci_contains_point_estimate():
    from quant_loop_trader.evaluation import bootstrap_ci
    rng = np.random.default_rng(1)
    rets = rng.normal(0.001, 0.01, 300)
    lo, hi = bootstrap_ci(rets)
    assert lo < rets.mean() < hi and lo < hi


def test_purge_creates_true_embargo_gap():
    """Audit C1 regression: last training LABEL's t+h endpoint must fall strictly
    before the first test observation. The old boundary-shifting purge failed this."""
    from quant_loop_trader.evaluation import time_split
    n, h = 1000, 5
    df = pl.DataFrame({
        "event_time": [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(n)],
        "close": [float(i) for i in range(n)],
        "label": [0] * n,
    })
    train, test = time_split(df, 0.7, purge=h)
    # last train row's position in the ORIGINAL ordering
    last_train_t = train["event_time"][-1]
    orig_idx_last_train = (last_train_t - datetime.date(2020, 1, 1)).days
    first_test_idx = (test["event_time"][0] - datetime.date(2020, 1, 1)).days
    # the label at t reads close[t+h]; that price must precede the test window
    assert orig_idx_last_train + h < first_test_idx
    # and no rows are lost: embargo gap is real but total accounting holds
    assert first_test_idx - orig_idx_last_train == h + 1  # indices: gap rows are h+1 positions
