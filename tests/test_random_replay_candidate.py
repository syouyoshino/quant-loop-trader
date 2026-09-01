from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

import quant_loop_trader.random_replay as rr


def _btc_frame(start: str = "2019-01-01", days: int = 1096) -> pl.DataFrame:
    start_date = date.fromisoformat(start)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    x = np.arange(days, dtype=float)
    close = 100.0 + 0.02 * x + 4.0 * np.sin(x / 11.0)
    return pl.DataFrame({
        "event_time": dates,
        "available_time": dates,
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": 1000.0 + (x % 30.0),
    }).with_columns(
        pl.col("event_time").cast(pl.Date),
        pl.col("available_time").cast(pl.Date),
    )


def test_replay_snapshot_requires_full_pit_ohlcv_schema():
    bad = _btc_frame(days=30).drop("available_time")
    with pytest.raises(ValueError, match="source_snapshot_missing_columns:available_time"):
        rr._validate_source_frame(bad)


def test_candidate_replay_uses_candidate_model_features_and_indexes_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("QLT_CRYPTO_CAMPAIGN_ID", "candidate_replay_test_v1")
    monkeypatch.setenv("QLT_CRYPTO_HOLDOUT_START", "2022-01-01")
    snapshot = tmp_path / "candidate_btc.parquet"
    _btc_frame().write_parquet(snapshot)

    spec = rr.CandidateReplaySpec(
        experiment_id="candidate_001",
        ticker="BTCUSD",
        horizon=5,
        model_seed=777,
        model_type="majority",
        model_params={},
        feature_columns=("ret_1", "ret_5"),
        dataset_snapshot=snapshot,
        dataset_id="candidate_dataset",
        dataset_checksum="abc123",
        spec_fingerprint="spec123",
        model_version="majority-v1",
        feature_version="baseline-subset",
        dataset_snapshot_sha256="snapshotsha",
        research_start="2019-01-01",
        research_end="2021-11-30",
    )
    monkeypatch.setattr(rr, "_load_candidate_spec", lambda experiment_id, root: spec)

    report = rr.run_random_replay(
        experiment_id="candidate_001",
        runs=4,
        trade_days=60,
        min_training_days=120,
        seed=123,
        root=tmp_path,
    )

    cfg = report["config"]
    assert cfg["candidate_experiment_id"] == "candidate_001"
    assert cfg["model_type"] == "majority"
    assert cfg["model_seed"] == 777
    assert cfg["sampling_seed"] == 123
    assert cfg["feature_set"] == ["ret_1", "ret_5"]
    assert cfg["data_end"] == "2021-11-30"
    assert report["summary"]["runs"] == 4

    evidence = rr.latest_replay_for_experiment("candidate_001", root=tmp_path)
    assert evidence is not None
    assert evidence["random_replay_id"] == report["random_replay_id"]
    assert evidence["summary"]["runs"] == 4


def test_candidate_replay_rejects_strategy_overrides(tmp_path, monkeypatch):
    spec = rr.CandidateReplaySpec(
        experiment_id="candidate_002",
        ticker="BTCUSD",
        horizon=5,
        model_seed=42,
        model_type="logistic",
        model_params={},
        feature_columns=tuple(rr.improved_feature_columns()),
        dataset_snapshot=tmp_path / "unused.parquet",
        dataset_id=None,
        dataset_checksum=None,
        spec_fingerprint=None,
        model_version=None,
        feature_version=None,
        dataset_snapshot_sha256=None,
        research_start="2019-01-01",
        research_end="2021-12-31",
    )
    monkeypatch.setattr(rr, "_load_candidate_spec", lambda experiment_id, root: spec)
    with pytest.raises(ValueError, match="candidate_ticker_override_forbidden"):
        rr.run_random_replay(experiment_id="candidate_002", ticker="SPY", root=tmp_path)


def test_candidate_replay_rejects_data_end_after_sealed_window(tmp_path, monkeypatch):
    spec = rr.CandidateReplaySpec(
        experiment_id="candidate_003",
        ticker="BTCUSD",
        horizon=5,
        model_seed=42,
        model_type="logistic",
        model_params={},
        feature_columns=tuple(rr.improved_feature_columns()),
        dataset_snapshot=tmp_path / "unused.parquet",
        dataset_id=None,
        dataset_checksum=None,
        spec_fingerprint=None,
        model_version=None,
        feature_version=None,
        dataset_snapshot_sha256=None,
        research_start="2019-01-01",
        research_end="2021-06-30",
    )
    monkeypatch.setattr(rr, "_load_candidate_spec", lambda experiment_id, root: spec)
    monkeypatch.setenv("QLT_CRYPTO_CAMPAIGN_ID", "candidate_replay_test_v1")
    monkeypatch.setenv("QLT_CRYPTO_HOLDOUT_START", "2022-01-01")

    with pytest.raises(ValueError, match="candidate_data_end_after_snapshot"):
        rr.run_random_replay(
            experiment_id="candidate_003",
            data_end="2021-07-01",
            root=tmp_path,
        )
