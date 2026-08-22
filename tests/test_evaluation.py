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
