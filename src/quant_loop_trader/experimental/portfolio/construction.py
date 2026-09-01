"""Deferred multi-asset position sizing and risk controls."""
from __future__ import annotations

import numpy as np


def equal_weight(n_positions: int) -> np.ndarray:
    return np.full(n_positions, 1.0 / max(n_positions, 1))


def volatility_weight(returns: np.ndarray, lookback: int = 20) -> np.ndarray:
    r = returns[-lookback:]
    vol = r.std(axis=0)
    inv = 1.0 / np.where(vol > 1e-12, vol, np.inf)
    total = inv.sum()
    if not np.isfinite(total) or total <= 0:
        return equal_weight(returns.shape[1])
    return np.nan_to_num(inv / total)


def apply_max_position(weights: np.ndarray, max_weight: float = 0.25) -> np.ndarray:
    if not 0 < max_weight <= 1:
        raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
    w = np.clip(weights, 0.0, None).astype(float)
    cap = max_weight
    frozen = np.zeros(len(w), dtype=bool)
    for _ in range(10):
        over = (w > cap + 1e-12) & ~frozen
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        frozen |= over
        under = ~frozen
        s = w[under].sum()
        if s <= 0 or excess <= 0:
            break
        w[under] += excess * (w[under] / s)
    return w


def drawdown_stop(cumulative_returns: np.ndarray, limit: float = -0.15) -> bool:
    if len(cumulative_returns) == 0:
        return False
    peak = np.maximum.accumulate(cumulative_returns)
    dd = ((cumulative_returns - peak) / (peak + 1e-12)).min()
    return bool(dd <= limit)


def size_positions(returns_window: np.ndarray, scheme: str = "equal",
                   max_weight: float = 0.25) -> np.ndarray:
    if returns_window.ndim != 2 or returns_window.shape[1] == 0:
        raise ValueError("returns_window must be 2-D with at least one asset")
    if scheme == "equal":
        w = equal_weight(returns_window.shape[1])
    elif scheme == "volatility":
        w = volatility_weight(returns_window)
    elif scheme == "risk":
        w = volatility_weight(returns_window) ** 2
        w = w / w.sum()
    else:
        raise ValueError(f"unknown sizing scheme '{scheme}'")
    return apply_max_position(w, max_weight)
