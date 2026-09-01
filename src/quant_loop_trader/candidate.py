"""Canonical strategy identity for validation, replay, and final holdout.

A CandidateSpec is derived only from a verified experiment bundle. It binds the
model, features, immutable dataset snapshot, horizon, seed, and BTC campaign
boundary so downstream checks cannot silently evaluate a different strategy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_loop_trader.features import improved_feature_columns
from quant_loop_trader.market import campaign_holdout_start, campaign_id

LEGACY_FEATURE_VERSION = "v1+vol_regime_ret5_x_vol10"
LEGACY_MODEL_VERSION = "sklearn-LogReg-C1.0-scaled"


@dataclass(frozen=True)
class CandidateSpec:
    experiment_id: str
    ticker: str
    horizon: int
    model_seed: int
    model_type: str
    model_params: dict
    feature_columns: tuple[str, ...]
    dataset_snapshot: Path
    dataset_id: str | None
    dataset_checksum: str | None
    spec_fingerprint: str | None
    model_version: str | None
    feature_version: str | None
    dataset_snapshot_sha256: str | None
    research_start: str
    research_end: str
    campaign: str | None = None
    holdout_start: str | None = None

    @property
    def seed(self) -> int:
        return self.model_seed

    @classmethod
    def from_bundle(cls, bundle) -> "CandidateSpec":
        cfg = bundle.config
        report = bundle.report
        experiment_id = safe_experiment_id(bundle.exp_dir.name)
        ticker = str(cfg["ticker"]).upper()

        active_boundary = campaign_holdout_start(ticker)
        candidate_boundary = cfg.get("campaign_holdout_start")
        if candidate_boundary != active_boundary:
            raise ValueError(
                "candidate_campaign_holdout_mismatch:"
                f"candidate={candidate_boundary}:active={active_boundary}"
            )

        active_campaign = campaign_id(ticker)
        candidate_campaign = cfg.get("campaign_id")
        if candidate_campaign is not None and candidate_campaign != active_campaign:
            raise ValueError(
                "candidate_campaign_id_mismatch:"
                f"candidate={candidate_campaign}:active={active_campaign}"
            )

        feature_version = cfg.get("feature_version_improved")
        feature_cols = cfg.get("feature_columns")
        if not feature_cols:
            feature_cols = report.get("parameters", {}).get("feature_columns")
        if not feature_cols:
            # Existing sealed experiments predate explicit feature_columns. Their
            # exact feature identity is still recoverable from this frozen version.
            if feature_version != LEGACY_FEATURE_VERSION:
                raise ValueError(
                    "candidate_feature_columns_missing_for_version:"
                    f"{feature_version or 'missing'}"
                )
            feature_cols = improved_feature_columns()
        feature_cols = tuple(str(c) for c in feature_cols)
        validate_feature_columns(feature_cols)

        model_version = cfg.get("model_version")
        model_type = cfg.get("model_type") or report.get("parameters", {}).get("model_type")
        if not model_type:
            # Same compatibility rule for the pre-CandidateSpec experiment format:
            # never infer logistic for an unknown model version.
            if model_version != LEGACY_MODEL_VERSION:
                raise ValueError(
                    "candidate_model_type_missing_for_version:"
                    f"{model_version or 'missing'}"
                )
            model_type = "logistic"
        model_params = (
            cfg.get("model_params")
            or report.get("parameters", {}).get("model_params")
            or {}
        )

        snapshot = Path(bundle.dataset_snapshot)
        if not snapshot.exists():
            raise FileNotFoundError(f"candidate_dataset_snapshot_missing:{snapshot}")

        return cls(
            experiment_id=experiment_id,
            ticker=ticker,
            horizon=int(cfg["horizon"]),
            model_seed=int(cfg["seed"]),
            model_type=str(model_type),
            model_params=dict(model_params),
            feature_columns=feature_cols,
            dataset_snapshot=snapshot,
            dataset_id=cfg.get("dataset_id"),
            dataset_checksum=cfg.get("dataset_checksum"),
            spec_fingerprint=cfg.get("spec_fingerprint"),
            model_version=model_version,
            feature_version=feature_version,
            dataset_snapshot_sha256=bundle.lock.get("dataset_snapshot_sha256"),
            research_start=str(cfg.get("start", "2018-01-01")),
            research_end=str(cfg["end"]),
            campaign=active_campaign,
            holdout_start=active_boundary,
        )

    @classmethod
    def load(cls, experiment_id: str, exp_root: str | Path) -> "CandidateSpec":
        from quant_loop_trader.bundle import ExperimentBundle

        experiment_id = safe_experiment_id(experiment_id)
        bundle = ExperimentBundle.open_verified(experiment_id, Path(exp_root))
        return cls.from_bundle(bundle)


def safe_experiment_id(experiment_id: str) -> str:
    if not experiment_id or Path(experiment_id).name != experiment_id:
        raise ValueError("invalid_candidate_experiment_id")
    return experiment_id


def validate_feature_columns(feature_cols) -> tuple[str, ...]:
    """Fail closed until another causal feature builder is explicitly supported."""
    cols = tuple(str(c) for c in feature_cols)
    supported = set(improved_feature_columns())
    unsupported = [c for c in cols if c not in supported]
    if unsupported:
        raise ValueError(f"unsupported_candidate_features:{','.join(unsupported)}")
    if not cols:
        raise ValueError("candidate_feature_columns_empty")
    return cols
