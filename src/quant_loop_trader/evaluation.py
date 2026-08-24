"""Evaluation — time-split, metrics, autopsy. Only module allowed to see future."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, brier_score_loss

logger = logging.getLogger(__name__)


def time_split(df: pl.DataFrame, train_ratio: float = 0.7, purge: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Time-ordered split, no shuffle. Purge drops last `purge` train rows whose
    labels read prices inside the hidden test window."""
    df = df.sort("event_time")
    n = df.height
    cut = int(n * train_ratio)
    cut = max(0, cut - max(0, purge))
    train = df.slice(0, cut)
    test = df.slice(cut, n - cut)
    # invariant: train strictly before test
    if train.height and test.height:
        assert train["event_time"].max() < test["event_time"].min(), "time leakage in split"
    return train, test


def _sharpe(returns: np.ndarray, periods_per_year: float = 252) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std())


def _max_drawdown(cum_returns: np.ndarray) -> float:
    if len(cum_returns) == 0:
        return 0.0
    peak = np.maximum.accumulate(cum_returns)
    dd = (cum_returns - peak) / (peak + 1e-12)
    return float(dd.min())


def _downside_dev(returns: np.ndarray, mar: float = 0.0) -> float:
    downside = np.minimum(returns - mar, 0)
    return float(np.sqrt((downside**2).mean())) if len(returns) else 0.0


def _trade_quality(strat_rets: np.ndarray) -> dict:
    """Per-decision quality: distinguishes better decisions from merely more trades."""
    if len(strat_rets) == 0:
        return {"win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "n_trades": 0}
    active = strat_rets[strat_rets != 0]  # decisions where we held a position
    wins = active[active > 0]
    losses = active[active <= 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    return {
        "win_rate": float(len(wins) / len(active)) if len(active) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
        "expectancy": float(active.mean()) if len(active) else 0.0,
        "n_trades": int(len(active)),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, prices: np.ndarray, horizon: int = 1) -> dict:
    """Return dict with prediction + financial metrics."""
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0
    rec = float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0
    try:
        brier = float(brier_score_loss(y_true, y_prob)) if len(y_true) else 0.0
    except Exception:
        brier = 0.0

    # financial: position held for `horizon` days (matches label horizon), long if pred=1 else flat.
    # Non-overlapping horizon buckets: signal from bucket start, return over the bucket.
    h = max(1, int(horizon))
    n_buckets = min(len(y_pred), max(0, (len(prices) - 1) // h))
    strat_rets = np.array([])
    bench_rets = np.array([])
    pos = np.array([])
    if len(prices) >= 2 and n_buckets > 0:
        bucket_ret = prices[h::h][:n_buckets] / prices[:-h:h][:n_buckets] - 1
        pos = y_pred[: n_buckets * h].reshape(n_buckets, h)[:, 0]
        bench_rets = bucket_ret
        strat_rets = bucket_ret * pos

    # transaction cost: 5bps per position change, charged to the switching bucket
    changes = np.abs(np.diff(pos)) if n_buckets > 1 else np.array([])
    turnover = float(changes.mean()) if len(changes) else 0.0
    strat_rets_net = strat_rets.copy()
    if len(strat_rets_net):
        # change between bucket i-1 and i costs entry at bucket i
        strat_rets_net[1:] -= changes * 0.0005

    ppy = 252 / h  # h-day buckets → annualization factor
    cum_strat = np.cumprod(1 + strat_rets_net) if len(strat_rets_net) else np.array([1.0])
    cum_bench = np.cumprod(1 + bench_rets) if len(bench_rets) else np.array([1.0])

    # extended risk: downside deviation, Sortino, VaR/ES, Calmar
    dd_dev = _downside_dev(strat_rets_net)
    mean_p = float(strat_rets_net.mean()) * ppy if len(strat_rets_net) else 0.0
    sortino = mean_p / (dd_dev * ppy) if dd_dev > 0 else 0.0
    var_95 = float(np.quantile(strat_rets_net, 0.05)) if len(strat_rets_net) else 0.0
    tail = strat_rets_net[strat_rets_net <= var_95] if len(strat_rets_net) else np.array([])
    es_95 = float(tail.mean()) if len(tail) else 0.0
    mdd = _max_drawdown(cum_strat)
    calmar = mean_p / abs(mdd) if mdd < 0 else 0.0

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "brier_score": brier,
        "sharpe_strategy": _sharpe(strat_rets_net, periods_per_year=ppy),
        "sharpe_benchmark": _sharpe(bench_rets, periods_per_year=ppy),
        "volatility_strategy": float(strat_rets_net.std()) if len(strat_rets_net) else 0.0,
        "volatility_benchmark": float(bench_rets.std()) if len(bench_rets) else 0.0,
        "max_drawdown_strategy": mdd,
        "max_drawdown_benchmark": _max_drawdown(cum_bench),
        "cumulative_return_strategy": float(cum_strat[-1] - 1) if len(cum_strat) else 0.0,
        "cumulative_return_benchmark": float(cum_bench[-1] - 1) if len(cum_bench) else 0.0,
        "turnover": turnover,
        "transaction_cost_adj_return": float(strat_rets.sum()) if len(strat_rets) else 0.0,
        # risk-adjusted extras
        "sortino_ratio": sortino,
        "downside_deviation": dd_dev,
        "var_95": var_95,
        "expected_shortfall_95": es_95,
        "calmar_ratio": calmar,
        # trade quality
        **_trade_quality(strat_rets),
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
