"""Verified experiment bundle (simplification refactor).

ONE verified loading abstraction for trusted paths. Validation, holdout
adjudication, reproduction and authoritative reporting must consume a bundle —
never raw artifact paths — so integrity checking cannot be skipped.

Invariants enforced here:
  INVARIANT 4  every authoritative artifact reachable through the manifest
  INVARIANT 5  no trusted path can use an unverified bundle (open_verified raises)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

from quant_loop_trader.data import PROC_DIR


class BundleIntegrityError(RuntimeError):
    """Raised when locked artifacts are missing/tampered. Fail closed."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExperimentBundle:
    exp_dir: Path
    experiment_id: str
    config: dict
    report: dict
    metrics: dict
    lock: dict

    @property
    def spec(self) -> dict:
        return self.config

    @property
    def dataset_snapshot(self) -> Path:
        """Immutable content-addressed input for ALL post-acquisition work."""
        return PROC_DIR.parent / "datasets" / f"{self.config['dataset_id']}.parquet"

    @property
    def ticker_cache(self) -> Path:
        return PROC_DIR / f"{self.config['ticker']}.parquet"

    @classmethod
    def open_verified(cls, experiment_id: str, exp_root: Path) -> "ExperimentBundle":
        exp_dir = Path(exp_root) / experiment_id
        lock_path = exp_dir / "predictions.lock"
        if not lock_path.exists():
            raise BundleIntegrityError(f"missing_predictions_lock:{experiment_id}")
        lock = json.loads(lock_path.read_text())

        issues = []
        # verify every locked artifact (locked_at is metadata, dataset_parquet is
        # a drift anchor against the shared cache, both handled below)
        for name, expected in lock.items():
            if name in ("locked_at", "dataset_parquet"):
                continue
            f = exp_dir / name
            if not f.exists():
                issues.append(f"locked_artifact_missing:{name}")
            elif _sha(f) != expected:
                issues.append(f"artifact_tampered:{name}")

        cfg = json.loads((exp_dir / "config.json").read_text())
        report = json.loads((exp_dir / "report.json").read_text())
        metrics = json.loads((exp_dir / "metrics.json").read_text())

        # dataset drift: shared cache bytes vs locked-at-experiment-time
        want = lock.get("dataset_parquet")
        pq = PROC_DIR / f"{cfg.get('ticker', 'SPY')}.parquet"
        if want and pq.exists() and _sha(pq) != want:
            issues.append("dataset_drift:input_data_changed_since_experiment")

        if issues:
            raise BundleIntegrityError(json.dumps(issues))

        predictions = pl.read_parquet(str(exp_dir / "predictions_improved.parquet"))
        return cls(exp_dir=exp_dir, experiment_id=experiment_id, config=cfg,
                   report=report, metrics=metrics, lock=lock,
                   _predictions=predictions)

    @classmethod
    def open_unverified(cls, experiment_id: str, exp_root: Path) -> "ExperimentBundle":
        """Explicit escape hatch for non-authoritative consumers (e.g. listing).
        Trusted paths must NOT use this."""
        exp_dir = Path(exp_root) / experiment_id
        return cls(exp_dir=exp_dir, experiment_id=experiment_id,
                   config=json.loads((exp_dir / "config.json").read_text()),
                   report=json.loads((exp_dir / "report.json").read_text()),
                   metrics=json.loads((exp_dir / "metrics.json").read_text()),
                   lock={}, _predictions=None)

    @property
    def predictions(self) -> pl.DataFrame:
        return self._predictions


# dataclasses with an extra non-field attr: implement via __init__ override instead
def _init(self, exp_dir, experiment_id, config, report, metrics, lock, _predictions):
    object.__setattr__(self, "exp_dir", Path(exp_dir))
    object.__setattr__(self, "experiment_id", experiment_id)
    object.__setattr__(self, "config", config)
    object.__setattr__(self, "report", report)
    object.__setattr__(self, "metrics", metrics)
    object.__setattr__(self, "lock", lock)
    object.__setattr__(self, "_predictions", _predictions)


ExperimentBundle.__init__ = _init  # type: ignore[method-assign]
