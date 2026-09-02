import json
from types import SimpleNamespace

import polars as pl
import pytest

from quant_loop_trader.agents import _campaign_from_config
from quant_loop_trader.candidate import CandidateSpec
from quant_loop_trader.core import PIPELINE_VERSION
from quant_loop_trader.data import dataset_metadata, seal_dataset_snapshot
from quant_loop_trader.experiment import _runtime_environment
from quant_loop_trader.features import improved_feature_columns
from quant_loop_trader.market import (
    DEFAULT_CRYPTO_CAMPAIGN_ID,
    DEFAULT_CRYPTO_HOLDOUT_START,
)


def _tiny_frame(close: float = 100.0) -> pl.DataFrame:
    return pl.DataFrame({
        "event_time": [__import__("datetime").date(2024, 1, 1)],
        "available_time": [__import__("datetime").date(2024, 1, 1)],
        "open": [close - 1],
        "high": [close + 1],
        "low": [close - 2],
        "close": [close],
        "volume": [10.0],
    })


def test_pipeline_version_requires_explicit_candidate_identity(tmp_path, monkeypatch):
    monkeypatch.delenv("QLT_HOLDOUT_START", raising=False)
    monkeypatch.delenv("QLT_CAMPAIGN_ID", raising=False)
    assert PIPELINE_VERSION >= 4

    snap = tmp_path / "dataset.parquet"
    _tiny_frame().write_parquet(snap)
    cfg = {
        "pipeline_version": PIPELINE_VERSION,
        "ticker": "SPY",
        "horizon": 5,
        "seed": 42,
        "start": "2018-01-01",
        "end": "2024-12-31",
        "campaign_id": "default",
        "campaign_holdout_start": None,
        "feature_version_improved": "v1+vol_regime_ret5_x_vol10",
        "feature_columns": improved_feature_columns(),
        "model_version": "sklearn-LogReg-C1.0-scaled",
        "model_type": "logistic",
        "model_params": {},
        "dataset_id": "dataset",
        "dataset_checksum": "checksum",
        "spec_fingerprint": "fingerprint",
    }
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()
    bundle = SimpleNamespace(
        config=cfg,
        report={},
        exp_dir=exp_dir,
        dataset_snapshot=snap,
        lock={"dataset_snapshot_sha256": "sha"},
    )

    candidate = CandidateSpec.from_bundle(bundle)
    assert candidate.campaign == "default"
    assert candidate.model_type == "logistic"
    assert candidate.model_params == {}
    assert candidate.feature_columns == tuple(improved_feature_columns())

    for key, error in (
        ("campaign_id", "candidate_campaign_id_missing"),
        ("feature_columns", "candidate_feature_columns_missing"),
        ("model_type", "candidate_model_type_missing"),
        ("model_params", "candidate_model_params_missing"),
    ):
        broken = dict(cfg)
        broken.pop(key)
        bad_bundle = SimpleNamespace(**{**bundle.__dict__, "config": broken})
        with pytest.raises(ValueError, match=error):
            CandidateSpec.from_bundle(bad_bundle)


def test_dataset_identity_uses_full_checksum_and_128bit_id(tmp_path):
    frame = _tiny_frame()
    meta = dataset_metadata(frame, "BTCUSD", "test")
    assert len(meta["checksum"]) == 64
    suffix = meta["dataset_id"].rsplit("_", 1)[-1]
    assert len(suffix) == 32

    path = tmp_path / f"{meta['dataset_id']}.parquet"
    seal_dataset_snapshot(frame, path, meta["checksum"])
    seal_dataset_snapshot(frame, path, meta["checksum"])

    other = _tiny_frame(101.0)
    other_meta = dataset_metadata(other, "BTCUSD", "test")
    with pytest.raises(RuntimeError, match="dataset_snapshot_collision"):
        seal_dataset_snapshot(other, path, other_meta["checksum"])


def test_campaign_resolution_keeps_multiple_testing_populations_separate():
    assert _campaign_from_config(
        json.dumps({"campaign_id": "btc_2026_v1"}), "BTCUSD"
    ) == "btc_2026_v1"
    assert _campaign_from_config(
        json.dumps({"campaign_holdout_start": DEFAULT_CRYPTO_HOLDOUT_START}),
        "BTCUSD",
    ) == DEFAULT_CRYPTO_CAMPAIGN_ID
    assert _campaign_from_config(
        json.dumps({"campaign_holdout_start": "2026-01-01"}), "BTCUSD"
    ) is None
    assert _campaign_from_config(json.dumps({}), "SPY") == "default"


def test_runtime_environment_has_stable_fingerprint_fields():
    env = _runtime_environment()
    assert len(env["fingerprint"]) == 64
    assert env["python_version"]
    assert env["python_implementation"]
    assert set(env["packages"]) >= {
        "numpy", "polars", "duckdb", "scikit-learn", "scipy", "pyarrow"
    }
