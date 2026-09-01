from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant_loop_trader.candidate import CandidateSpec, validate_feature_columns


def _bundle(tmp_path, *, feature_columns=None, model_type="majority", model_params=None):
    snapshot = tmp_path / "candidate.parquet"
    snapshot.write_bytes(b"sealed-test-snapshot")
    cfg = {
        "ticker": "BTCUSD",
        "horizon": 5,
        "seed": 777,
        "start": "2019-01-01",
        "end": "2023-12-31",
        "campaign_holdout_start": "2024-01-01",
        "campaign_id": "btc_pre2024_v1",
        "dataset_id": "btc_dataset_1",
        "dataset_checksum": "abc123",
        "spec_fingerprint": "spec123",
        "model_version": "model-v1",
        "feature_version_improved": "features-v1",
        "model_type": model_type,
        "model_params": model_params or {},
        "feature_columns": feature_columns or ["ret_1", "ret_5"],
    }
    report = {"parameters": {}}
    return SimpleNamespace(
        config=cfg,
        report=report,
        exp_dir=tmp_path / "candidate_001",
        dataset_snapshot=snapshot,
        lock={"dataset_snapshot_sha256": "snapshotsha"},
    )


def test_candidate_spec_binds_exact_strategy_and_btc_campaign(tmp_path, monkeypatch):
    monkeypatch.delenv("QLT_CRYPTO_HOLDOUT_START", raising=False)
    monkeypatch.delenv("QLT_CRYPTO_CAMPAIGN_ID", raising=False)
    spec = CandidateSpec.from_bundle(
        _bundle(
            tmp_path,
            feature_columns=["ret_1", "ret_5"],
            model_type="majority",
            model_params={"example": 1},
        )
    )

    assert spec.ticker == "BTCUSD"
    assert spec.horizon == 5
    assert spec.model_seed == 777
    assert spec.model_type == "majority"
    assert spec.model_params == {"example": 1}
    assert spec.feature_columns == ("ret_1", "ret_5")
    assert spec.campaign == "btc_pre2024_v1"
    assert spec.holdout_start == "2024-01-01"


def test_candidate_spec_rejects_changed_btc_campaign(tmp_path, monkeypatch):
    monkeypatch.setenv("QLT_CRYPTO_HOLDOUT_START", "2025-01-01")
    monkeypatch.setenv("QLT_CRYPTO_CAMPAIGN_ID", "btc_pre2025_v1")
    with pytest.raises(ValueError, match="candidate_campaign_holdout_mismatch"):
        CandidateSpec.from_bundle(_bundle(tmp_path))


def test_candidate_features_fail_closed_when_builder_is_unsupported():
    with pytest.raises(ValueError, match="unsupported_candidate_features"):
        validate_feature_columns(["ret_1", "future_magic_feature"])
