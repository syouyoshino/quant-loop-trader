"""Portfolio construction (Phase 8): position sizing + risk controls.

Separates prediction quality from portfolio mechanics: given a frame of
per-period predictions/returns, produce weights under each sizing scheme and
enforce hard limits (max position, drawdown stop).
"""
from __future__ import annotations

import numpy as np
import polars as pl


def equal_weight(n_positions: int) -> np.ndarray:
    return np.full(n_positions, 1.0 / max(n_positions, 1))


def volatility_weight(returns: np.ndarray, lookback: int = 20) -> np.ndarray:
    """Inverse-volatility weighting per asset column. Higher risk → smaller weight."""
    r = returns[-lookback:]
    vol = r.std(axis=0)
    inv = 1.0 / np.where(vol > 1e-12, vol, np.inf)
    w = inv / inv.sum()
    return np.nan_to_num(w)


def apply_max_position(weights: np.ndarray, max_weight: float = 0.25) -> np.ndarray:
    """Cap and renormalise so exposure stays fully invested but never concentrated."""
    w = np.clip(weights, 0.0, max_weight)
    s = w.sum()
    return w / s if s > 0 else w


def drawdown_stop(cumulative_returns: np.ndarray, limit: float = -0.15) -> bool:
    """True when the drawdown limit is breached — portfolio-level kill switch."""
    if len(cumulative_returns) == 0:
        return False
    peak = np.maximum.accumulate(cumulative_returns)
    dd = ((cumulative_returns - peak) / (peak + 1e-12)).min()
    return bool(dd <= limit)


def size_positions(returns_window: np.ndarray, scheme: str = "equal",
                   max_weight: float = 0.25) -> np.ndarray:
    """Unified entry point. returns_window shape = (lookback, n_assets)."""
    if scheme == "equal":
        w = equal_weight(returns_window.shape[1])
    elif scheme == "volatility":
        w = volatility_weight(returns_window)
    elif scheme == "risk":
        # risk weighting: same inverse-vol family, squared to penalise variance harder
        w = volatility_weight(returns_window) ** 2
        w = w / w.sum()
    else:
        raise ValueError(f"unknown sizing scheme '{scheme}'")
    return apply_max_position(w, max_weight)
