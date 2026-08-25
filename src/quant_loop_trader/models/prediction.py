"""Prediction objects — the immutable record of one model output (charter STEP 3).

A Prediction is created once, hashed, and never modified; outcomes are attached
by the evaluation system, never by the predictor.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Prediction:
    timestamp: str            # prediction time T — only info <= T may be used
    ticker: str
    horizon: int              # days ahead the label resolves
    prediction: int           # 1 = up, 0 = down/flat
    confidence: float         # P(up) from the model
    features_used: list[str] = field(default_factory=list)
    model_version: str = "unknown"
    strategy_version: str = "unknown"
    experiment_id: str = ""

    def sha256(self) -> str:
        """Content hash — the lock used by predictions.lock / validation gates."""
        canonical = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_row(self) -> dict:
        return asdict(self)


SUPPORTED_HORIZONS = [1, 3, 5, 10, 20]
