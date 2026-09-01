import datetime

import polars as pl
import pytest

from quant_loop_trader.experimental.strategies.framework import (
    MomentumStrategy,
    STRATEGY_REGISTRY,
    build_strategy,
)
from quant_loop_trader.replay import ReplayEngine


def _snapshot(tmp_path):
    rows = []
    for i in range(60):
        d = datetime.date(2020, 1, 1) + datetime.timedelta(days=i)
        close = 100 + i * (1 if i < 30 else -0.5)
        rows.append({
            "event_time": d,
            "available_time": d,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1000,
        })
    p = tmp_path / "TEST.parquet"
    pl.DataFrame(rows).write_parquet(str(p))
    return ReplayEngine(p)


def test_strategy_registry_and_metadata():
    assert "momentum_reference" in STRATEGY_REGISTRY
    s = build_strategy("momentum_reference")
    assert s.metadata()["horizon"] == 5


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        build_strategy("martingale")


def test_momentum_reference_produces_frozen_predictions_from_pit_only(tmp_path):
    engine = _snapshot(tmp_path)
    snap = engine.get_snapshot("TEST", "2020-02-29")
    preds = MomentumStrategy().generate_predictions(snap, "TEST")
    assert len(preds) > 10
    for p in preds:
        assert p.timestamp <= "2020-02-29"
        assert 0 <= p.prediction <= 1
        assert (p.confidence > 0.5) == (p.prediction == 1)
