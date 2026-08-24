import datetime

import numpy as np
import pytest

from quant_loop_trader.portfolio.construction import (
    apply_max_position, drawdown_stop, equal_weight, size_positions, volatility_weight,
)


def test_equal_weight_sums_to_one():
    w = equal_weight(4)
    assert len(w) == 4 and abs(w.sum() - 1.0) < 1e-12


def test_volatility_weight_prefers_stable_assets():
    rng = np.random.default_rng(0)
    rets = np.column_stack([rng.normal(0, 0.01, 100), rng.normal(0, 0.05, 100)])
    w = volatility_weight(rets)
    assert w[0] > w[1] * 3  # stable asset gets far more weight


def test_max_position_cap_enforced_and_renormalised():
    raw = np.array([0.8, 0.1, 0.1])
    w = apply_max_position(raw, max_weight=0.25)
    assert (w <= 0.25 + 1e-12).all()
    assert abs(w.sum() - 1.0) < 1e-9


def test_drawdown_stop_trips():
    up = np.cumprod(1 + np.full(50, 0.001))
    assert not drawdown_stop(up, limit=-0.15)
    crash = np.cumprod(1 + np.concatenate([np.full(40, 0.001), np.full(10, -0.05)]))
    assert drawdown_stop(crash, limit=-0.15)


def test_size_positions_dispatch():
    rets = np.random.default_rng(1).normal(0, 0.02, (30, 5))
    for scheme in ["equal", "volatility", "risk"]:
        w = size_positions(rets, scheme=scheme, max_weight=0.4)
        assert abs(w.sum() - 1.0) < 1e-6 and (w <= 0.4 + 1e-12).all()
    with pytest.raises(ValueError, match="unknown sizing"):
        size_positions(rets, scheme="yolo")
