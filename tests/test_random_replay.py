from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_loop_trader.random_replay import (
    build_replay_window,
    eligible_start_indices,
    prepare_replay_frame,
    run_random_replay,
    select_balanced_starts,
)


def _btc_frame(start: str = "2019-01-01", days: int = 1096) -> pl.DataFrame:
    start_date = date.fromisoformat(start)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    x = np.arange(days, dtype=float)
    close = 100.0 + 0.02 * x + 5.0 * np.sin(x / 7.0) + 2.0 * np.sin(x / 31.0)
    return pl.DataFrame(
        {
            "event_time": dates,
            "available_time": dates,
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 1000.0 + (x % 50.0),
        }
    ).with_columns(
        pl.col("event_time").cast(pl.Date),
        pl.col("available_time").cast(pl.Date),
    )


def test_replay_window_purges_horizon_before_start():
    frame = prepare_replay_frame(_btc_frame(days=700), horizon=5)
    start_idx = 250
    train, test = build_replay_window(
        frame,
        start_idx,
        horizon=5,
        trade_days=60,
        min_training_days=100,
    )

    assert train.height == start_idx - 5
    assert test.height == 60
    assert train["event_time"].max() < test["event_time"].min()
    purged = frame.slice(train.height, 5)
    assert purged.height == 5
    assert purged["event_time"].max() < test["event_time"].min()


def test_seeded_sampling_is_unique_reproducible_and_year_balanced():
    frame = prepare_replay_frame(_btc_frame(days=1096), horizon=5)
    candidates = eligible_start_indices(
        frame,
        horizon=5,
        trade_days=45,
        min_training_days=100,
        sample_start=date(2019, 6, 1),
        sample_end=date(2021, 10, 1),
    )

    first = select_balanced_starts(frame, candidates, runs=18, seed=123)
    second = select_balanced_starts(frame, candidates, runs=18, seed=123)
    assert first == second
    assert len(first) == len(set(first)) == 18

    years = [frame["event_time"][idx].year for idx in first]
    counts = {year: years.count(year) for year in set(years)}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_random_replay_never_loads_or_trades_into_campaign_holdout(tmp_path, monkeypatch):
    monkeypatch.setenv("QLT_CRYPTO_CAMPAIGN_ID", "test_random_replay_v1")
    monkeypatch.setenv("QLT_CRYPTO_HOLDOUT_START", "2022-01-01")

    snapshot = tmp_path / "btc.parquet"
    _btc_frame(days=1096).write_parquet(snapshot)

    report = run_random_replay(
        ticker="BTCUSD",
        horizon=5,
        runs=8,
        trade_days=60,
        min_training_days=120,
        data_start="2019-01-01",
        seed=77,
        source_snapshot=snapshot,
        root=tmp_path,
    )

    assert report["config"]["data_end"] == "2021-12-31"
    assert report["config"]["campaign_holdout_start"] == "2022-01-01"
    assert report["summary"]["runs"] == 8
    assert report["summary"]["unique_start_dates"] == 8
    assert report["summary"]["statistical_independence"] == "not_assumed"

    runs = pl.read_csv(report["artifacts"]["runs_csv"])
    assert all(date.fromisoformat(d) < date(2022, 1, 1) for d in runs["trade_end"])
    assert Path(report["artifacts"]["summary_json"]).exists()
    assert Path(report["artifacts"]["config_json"]).exists()


def test_random_replay_rejects_data_end_that_reaches_holdout(monkeypatch):
    monkeypatch.setenv("QLT_CRYPTO_CAMPAIGN_ID", "test_random_replay_v1")
    monkeypatch.setenv("QLT_CRYPTO_HOLDOUT_START", "2022-01-01")

    with pytest.raises(ValueError, match="random_replay_data_end_reaches_holdout"):
        run_random_replay(
            ticker="BTCUSD",
            runs=1,
            trade_days=60,
            min_training_days=100,
            data_start="2019-01-01",
            data_end="2022-01-01",
        )
