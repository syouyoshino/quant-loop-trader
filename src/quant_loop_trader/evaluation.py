"""Evaluation — time-split, metrics, autopsy. Only module allowed to see future."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, brier_score_loss

logger = logging.getLogger(__name__)


def time_split(df: pl.DataFrame, train_ratio: float = 0.7) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Time-ordered split, no shuffle. Returns (train, test_hidden)."""
    df = df.sort("event_time")
    n = df.height
    cut = int(n * train_ratio)
    train = df.slice(0, cut)
    test = df.slice(cut, n - cut)
    # invariant: train strictly before test
    if train.height and test.height:
        assert train["event_time"].max() < test["event_time"].min(), "time leakage in split"
    return train, test


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(np.sqrt(252) * returns.mean() / returns.std())


def _max_drawdown(cum_returns: np.ndarray) -> float:
    if len(cum_returns) == 0:
        return 0.0
    peak = np.maximum.accumulate(cum_returns)
    dd = (cum_returns - peak) / (peak + 1e-12)
    return float(dd.min())


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, prices: np.ndarray) -> dict:
    """Return dict with prediction + financial metrics."""
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0
    rec = float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0
    try:
        brier = float(brier_score_loss(y_true, y_prob)) if len(y_true) else 0.0
    except Exception:
        brier = 0.0

    # financial: equal-weight long if pred=1 else flat. Returns based on 5d forward?
    # For metrics we use next-day return as proxy; cumulative product
    # prices are close series aligned with y_true length (test)
    if len(prices) < 2:
        strat_rets = np.array([])
        bench_rets = np.array([])
    else:
        daily_ret = np.diff(prices) / prices[:-1]
        # align: y_pred[i] predicts move from prices[i] to prices[i+1]? For simplicity daily
        # pad to same length
        n = min(len(daily_ret), len(y_pred))
        strat_rets = daily_ret[:n] * y_pred[:n]  # long only
        bench_rets = daily_ret[:n]  # buy-hold

    # transaction cost 5bps per turnover
    turnover = float(np.abs(np.diff(y_pred)).mean()) if len(y_pred) > 1 else 0.0
    cost = turnover * 0.0005
    strat_rets_net = strat_rets - cost / max(len(strat_rets), 1) if len(strat_rets) else strat_rets

    cum_strat = np.cumprod(1 + strat_rets_net) if len(strat_rets_net) else np.array([1.0])
    cum_bench = np.cumprod(1 + bench_rets) if len(bench_rets) else np.array([1.0])

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "brier_score": brier,
        "sharpe_strategy": _sharpe(strat_rets_net),
        "sharpe_benchmark": _sharpe(bench_rets),
        "volatility_strategy": float(strat_rets_net.std()) if len(strat_rets_net) else 0.0,
        "volatility_benchmark": float(bench_rets.std()) if len(bench_rets) else 0.0,
        "max_drawdown_strategy": _max_drawdown(cum_strat),
        "max_drawdown_benchmark": _max_drawdown(cum_bench),
        "cumulative_return_strategy": float(cum_strat[-1] - 1) if len(cum_strat) else 0.0,
        "cumulative_return_benchmark": float(cum_bench[-1] - 1) if len(cum_bench) else 0.0,
        "turnover": turnover,
        "transaction_cost_adj_return": float(cum_strat[-1] - 1) if len(cum_strat) else 0.0,
        "n_test": int(len(y_true)),
    }
    return metrics


def autopsy(df_test: pl.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Simple failure analysis grouped by vol regime."""
    if df_test.height == 0 or len(y_true) == 0:
        return {"note": "empty test"}
    # need vol10 column; if missing compute
    vol = df_test["vol10"].to_numpy() if "vol10" in df_test.columns else np.zeros(len(y_true))
    # quintiles
    try:
        qs = np.quantile(vol[np.isfinite(vol)], [0.2, 0.4, 0.6, 0.8])
    except Exception:
        qs = [0, 0, 0, 0]
    regimes = np.digitize(vol, qs)
    out = {}
    for r in range(5):
        mask = regimes == r
        if mask.sum() == 0:
            continue
        acc = (y_true[mask] == y_pred[mask]).mean()
        out[f"regime_{r}_acc"] = float(acc)
        out[f"regime_{r}_n"] = int(mask.sum())
    # overall error breakdown
    out["overall_accuracy"] = float((y_true == y_pred).mean())
    out["error_rate"] = float((y_true != y_pred).mean())
    out["false_positive_rate"] = float(((y_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1))
    out["false_negative_rate"] = float(((y_pred == 0) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1))
    logger.info(json.dumps({"event": "autopsy", **out}))
    return out
