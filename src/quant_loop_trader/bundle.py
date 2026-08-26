"""Verified experiment bundle.

Trusted consumers open experiments through :meth:`ExperimentBundle.open_verified`.
The bundle verifies sealed artifacts and the immutable content-addressed dataset
snapshot before exposing evidence to validation, reproduction, or holdout.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from quant_loop_trader.data import PROC_DIR


class BundleIntegrityError(RuntimeError):
    """Raised when authoritative experiment evidence is missing or tampered."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ExperimentBundle:
    exp_dir: Path
    experiment_id: str
    config: dict
    report: dict
    metrics: dict
    lock: dict
    _predictions: pl.DataFrame | None = None

    @property
    def spec(self) -> dict:
        return self.config

    @property
    def dataset_snapshot(self) -> Path:
        """Immutable content-addressed input for all post-acquisition work."""
        return PROC_DIR.parent / "datasets" / f"{self.config['dataset_id']}.parquet"

    @property
    def predictions(self) -> pl.DataFrame:
        if self._predictions is None:
            raise BundleIntegrityError("predictions_unavailable_on_unverified_bundle")
        return self._predictions

    @classmethod
    def open_verified(cls, experiment_id: str, exp_root: Path) -> "ExperimentBundle":
        exp_dir = Path(exp_root) / experiment_id
        lock_path = exp_dir / "predictions.lock"
        if not lock_path.exists():
            raise BundleIntegrityError(f"missing_predictions_lock:{experiment_id}")

        required = ("config.json", "report.json", "metrics.json")
        missing = [name for name in required if not (exp_dir / name).exists()]
        if missing:
            raise BundleIntegrityError(json.dumps([f"artifact_missing:{name}" for name in missing]))

        lock = json.loads(lock_path.read_text())
        cfg = json.loads((exp_dir / "config.json").read_text())
        report = json.loads((exp_dir / "report.json").read_text())
        metrics = json.loads((exp_dir / "metrics.json").read_text())
        issues: list[str] = []

        # Verify sealed experiment-local artifacts. Dataset anchors are handled
        # separately because the snapshot lives outside the experiment directory.
        for name, expected in lock.items():
            if name in ("locked_at", "dataset_parquet", "dataset_snapshot_sha256"):
                continue
            f = exp_dir / name
            if not f.exists():
                issues.append(f"locked_artifact_missing:{name}")
            elif _sha(f) != expected:
                issues.append(f"artifact_tampered:{name}")

        snapshot = PROC_DIR.parent / "datasets" / f"{cfg['dataset_id']}.parquet"
        snapshot_expected = lock.get("dataset_snapshot_sha256")
        if snapshot_expected:
            if not snapshot.exists():
                issues.append("locked_dataset_snapshot_missing")
            elif _sha(snapshot) != snapshot_expected:
                issues.append("artifact_tampered:dataset_snapshot")
        else:
            # Backward compatibility for pre-snapshot-lock experiments. New runs
            # never depend on this mutable-cache anchor.
            cache_expected = lock.get("dataset_parquet")
            cache = PROC_DIR / f"{cfg.get('ticker', 'SPY')}.parquet"
            if cache_expected and cache.exists() and _sha(cache) != cache_expected:
                issues.append("dataset_drift:legacy_input_data_changed_since_experiment")

        if issues:
            raise BundleIntegrityError(json.dumps(issues))

        predictions = pl.read_parquet(str(exp_dir / "predictions_improved.parquet"))
        return cls(exp_dir=exp_dir, experiment_id=experiment_id, config=cfg,
                   report=report, metrics=metrics, lock=lock,
                   _predictions=predictions)

    @classmethod
    def open_unverified(cls, experiment_id: str, exp_root: Path) -> "ExperimentBundle":
        """Explicit escape hatch for non-authoritative listing/UI consumers."""
        exp_dir = Path(exp_root) / experiment_id
        return cls(exp_dir=exp_dir, experiment_id=experiment_id,
                   config=json.loads((exp_dir / "config.json").read_text()),
                   report=json.loads((exp_dir / "report.json").read_text()),
                   metrics=json.loads((exp_dir / "metrics.json").read_text()),
                   lock={})
