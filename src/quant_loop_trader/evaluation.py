"""Evaluation — time-split, metrics, autopsy. Only module allowed to see future."""
from __future__ import annotations

import json
import logging

import polars as pl
import math

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, brier_score_loss

from quant_loop_trader.market import calendar_days, periods_per_year

logger = logging.getLogger(__name__)

DEFAULT_COST_PER_SIDE = 0.0005
COST_SENSITIVITY_BPS = (5, 10, 25, 50)


def time_split(df: pl.DataFrame, train_ratio: float = 0.7, purge: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Time-ordered split with a TRUE embargo gap.

    Audit C1: the previous version moved the TEST boundary along with the train
    boundary, so the gap never existed and last-train labels read hidden-test
    prices. Correct semantics: the test window starts at the ORIGINAL cut; training
    ends `purge` rows earlier, guaranteeing label_t+h < first test observation."""
    df = df.sort("event_time")
    n = df.height
    cut = int(n * train_ratio)
    h = max(0, int(purge))
    train_end = max(0, cut - h)
    train = df.slice(0, train_end)
    test = df.slice(cut, n - cut)
    if train.height and test.height:
        assert train["event_time"].max() < test["event_time"].min(), "time leakage in split"
    return train, test


def _sharpe(returns: np.ndarray, periods_per_year: float = 252) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std())


def _max_drawdown(cum_returns: np.ndarray) -> float:
    """Max drawdown of a wealth path, including initial capital 1.0."""
    if len(cum_returns) == 0:
        return 0.0
    curve = np.concatenate([[1.0], cum_returns])
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / (peak + 1e-12)
    return float(dd.min())


def _downside_dev(returns: np.ndarray, mar: float = 0.0) -> float:
    downside = np.minimum(returns - mar, 0)
    return float(np.sqrt((downside**2).mean())) if len(returns) else 0.0


def _trade_quality(strat_rets: np.ndarray) -> dict:
    """Per-decision quality: distinguishes better decisions from merely more trades."""
    if len(strat_rets) == 0:
        return {"win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "n_trades": 0}
    active = strat_rets[strat_rets != 0]
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


def _apply_costs(strat_rets: np.ndarray, pos: np.ndarray, cost: float,
                 liquidate: bool = False) -> np.ndarray:
    out = strat_rets.copy()
    if not len(out):
        return out
    changes = np.abs(np.diff(pos)) if len(pos) > 1 else np.array([])
    if len(changes):
        out[1:] -= changes * cost
    if len(pos) and pos[0] > 0:
        out[0] -= cost
    if liquidate and len(pos) and pos[-1] > 0:
        out[-1] -= cost
    return out


def _cost_sensitivity(strat_rets: np.ndarray, pos: np.ndarray) -> dict[str, float]:
    out = {}
    for bps in COST_SENSITIVITY_BPS:
        net = _apply_costs(strat_rets, pos, bps / 10_000.0, liquidate=True)
        compounded = np.cumprod(1 + net) if len(net) else np.array([1.0])
        out[str(bps)] = float(compounded[-1] - 1)
    return out


def _phase_cost_sensitivity(prices: np.ndarray, y_pred: np.ndarray,
                            horizon: int) -> dict[str, dict[str, float]]:
    """Evaluate every non-overlapping h-day entry phase, not only phase zero.

    For a 5-day model there are five legitimate daily schedules: enter on test
    index 0, 1, 2, 3, or 4 and then every five days. The legacy strategy series
    uses phase zero for chart continuity; this diagnostic exposes whether its
    economics depend on that arbitrary starting day.
    """
    h = max(1, int(horizon))
    out: dict[str, dict[str, float]] = {}
    last_start = min(len(y_pred), max(0, len(prices) - h))
    for phase in range(h):
        starts = np.arange(phase, last_start, h, dtype=int)
        if len(starts) == 0:
            continue
        phase_rets = prices[starts + h] / prices[starts] - 1
        phase_pos = np.asarray(y_pred)[starts]
        strat_rets = phase_rets * phase_pos
        out[str(phase)] = _cost_sensitivity(strat_rets, phase_pos)
    return out


def _worst_phase_cost_sensitivity(
    phase_costs: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Conservative canonical stress: the worst legitimate entry phase per fee."""
    out: dict[str, float] = {}
    for bps in COST_SENSITIVITY_BPS:
        values = [
            float(row[str(bps)])
            for row in phase_costs.values()
            if str(bps) in row
        ]
        out[str(bps)] = min(values) if values else 0.0
    return out


def _phase_robustness(phase_costs: dict[str, dict[str, float]], bps: int = 25) -> dict:
    values = [float(row[str(bps)]) for row in phase_costs.values() if str(bps) in row]
    if not values:
        return {
            "bps_per_side": bps,
            "n_phases": 0,
            "min_compounded_return": 0.0,
            "median_compounded_return": 0.0,
            "max_compounded_return": 0.0,
            "all_positive": False,
        }
    return {
        "bps_per_side": bps,
        "n_phases": len(values),
        "min_compounded_return": float(np.min(values)),
        "median_compounded_return": float(np.median(values)),
        "max_compounded_return": float(np.max(values)),
        "all_positive": bool(all(v > 0 for v in values)),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, prices: np.ndarray,
             horizon: int = 1, ticker: str = "SPY",
             transaction_cost: float = DEFAULT_COST_PER_SIDE) -> dict:
    """Return prediction and financial metrics using the market's real calendar.

    Existing research-window metrics preserve the historical phase-zero chart
    convention. Canonical promotion economics use the additional fully-liquidated
    fields and worst-phase cost stress, so an h-day result is not promoted merely
    because the test window happened to begin on a favorable entry day.
    """
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0
    rec = float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0
    try:
        brier = float(brier_score_loss(y_true, y_prob)) if len(y_true) else float("nan")
    except Exception:
        brier = float("nan")

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

    changes = np.abs(np.diff(pos)) if n_buckets > 1 else np.array([])
    turnover = float(changes.mean()) if len(changes) else 0.0
    strat_rets_net = _apply_costs(strat_rets, pos, transaction_cost, liquidate=False)

    # A completed historical backtest must liquidate an open terminal position.
    # Keep the historical research-window convention above for backward-compatible
    # chart reconciliation, but expose and use this fully liquidated series for
    # promotion economics and explicit cost-adjusted reporting.
    strat_rets_liquidated = _apply_costs(strat_rets, pos, transaction_cost, liquidate=True)
    exit_cost_applied = bool(len(strat_rets_liquidated) and len(pos) and pos[-1] > 0)

    primary_phase_costs = _cost_sensitivity(strat_rets, pos)
    phase_costs = _phase_cost_sensitivity(prices, y_pred, h)
    canonical_costs = _worst_phase_cost_sensitivity(phase_costs)
    phase_25 = _phase_robustness(phase_costs, bps=25)

    ppy = periods_per_year(ticker, h)
    cum_strat = np.cumprod(1 + strat_rets_net) if len(strat_rets_net) else np.array([1.0])
    cum_strat_liq = (
        np.cumprod(1 + strat_rets_liquidated)
        if len(strat_rets_liquidated) else np.array([1.0])
    )
    cum_bench = np.cumprod(1 + bench_rets) if len(bench_rets) else np.array([1.0])

    dd_dev = _downside_dev(strat_rets_net)
    mean_p = float(strat_rets_net.mean()) * ppy if len(strat_rets_net) else 0.0
    sortino = mean_p / (dd_dev * np.sqrt(ppy)) if dd_dev > 0 else 0.0
    var_95 = float(np.quantile(strat_rets_net, 0.05)) if len(strat_rets_net) else 0.0
    tail = strat_rets_net[strat_rets_net <= var_95] if len(strat_rets_net) else np.array([])
    es_95 = float(tail.mean()) if len(tail) else 0.0
    mdd = _max_drawdown(cum_strat)
    n_periods = max(len(cum_strat), 1)
    cagr = (float(cum_strat[-1]) ** (ppy / n_periods) - 1) if len(cum_strat) and cum_strat[-1] > 0 else -1.0
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "brier_score": brier,
        "ticker": ticker,
        "calendar_days": calendar_days(ticker),
        "periods_per_year": ppy,
        "transaction_cost_per_side": float(transaction_cost),
        "execution_policy": "non_overlapping_h_day_blocks_phase0",
        "primary_phase_offset": 0,
        "sharpe_strategy": _sharpe(strat_rets_net, periods_per_year=ppy),
        "sharpe_strategy_liquidated": _sharpe(strat_rets_liquidated, periods_per_year=ppy),
        "sharpe_benchmark": _sharpe(bench_rets, periods_per_year=ppy),
        "volatility_strategy": float(strat_rets_net.std()) if len(strat_rets_net) else 0.0,
        "volatility_benchmark": float(bench_rets.std()) if len(bench_rets) else 0.0,
        "max_drawdown_strategy": mdd,
        "max_drawdown_benchmark": _max_drawdown(cum_bench),
        "cumulative_return_strategy": float(cum_strat[-1] - 1) if len(cum_strat) else 0.0,
        "cumulative_return_strategy_liquidated": (
            float(cum_strat_liq[-1] - 1) if len(cum_strat_liq) else 0.0
        ),
        "cumulative_return_benchmark": float(cum_bench[-1] - 1) if len(cum_bench) else 0.0,
        "turnover": turnover,
        # Legacy alias retained for old consumers. It is an arithmetic sum, not a
        # compounded return; new code should use arithmetic_net_return_sum or the
        # explicit compounded liquidated field below.
        "transaction_cost_adj_return": float(strat_rets_net.sum()) if len(strat_rets_net) else 0.0,
        "arithmetic_net_return_sum": float(strat_rets_net.sum()) if len(strat_rets_net) else 0.0,
        "transaction_cost_adj_return_compounded": (
            float(cum_strat_liq[-1] - 1) if len(cum_strat_liq) else 0.0
        ),
        # Canonical cost stress is conservative across every possible h-day phase.
        # Keep phase zero separately so old charts and reports remain reconcilable.
        "cost_sensitivity_compounded": canonical_costs,
        "phase0_cost_sensitivity_compounded": primary_phase_costs,
        "phase_cost_sensitivity_compounded": phase_costs,
        "phase_robustness_25bps": phase_25,
        "exit_cost_applied": exit_cost_applied,
        "gross_return": float(strat_rets.sum()) if len(strat_rets) else 0.0,
        "sortino_ratio": sortino,
        "downside_deviation": dd_dev,
        "var_95": var_95,
        "expected_shortfall_95": es_95,
        "calmar_ratio": calmar,
        **_trade_quality(strat_rets_net),
        "return_ci95": bootstrap_ci(strat_rets_net),
        "n_return_buckets": int(len(strat_rets_net)),
        "n_test": int(len(y_true)),
    }
    return metrics


def autopsy(df_test: pl.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Simple failure analysis grouped by vol regime."""
    if df_test.height == 0 or len(y_true) == 0:
        return {"note": "empty test"}
    vol = df_test["vol10"].to_numpy() if "vol10" in df_test.columns else np.zeros(len(y_true))
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
    out["overall_accuracy"] = float((y_true == y_pred).mean())
    out["error_rate"] = float((y_true != y_pred).mean())
    out["false_positive_rate"] = float(((y_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1))
    out["false_negative_rate"] = float(((y_pred == 0) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1))
    logger.info(json.dumps({"event": "autopsy", **out}))
    return out


def bootstrap_ci(returns: np.ndarray, n_boot: int = 1000, seed: int = 42,
                 alpha: float = 0.05, block: int = 5) -> tuple[float, float]:
    """Moving-block bootstrap CI on mean return."""
    n = len(returns)
    if n < 2:
        return (0.0, 0.0)
    block = max(1, min(int(block), n))
    n_blocks = math.ceil(n / block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    means = []
    for row in starts:
        sample = np.concatenate([returns[s:s + block] for s in row])[:n]
        means.append(float(sample.mean()))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)