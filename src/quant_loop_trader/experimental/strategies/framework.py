"""Deferred generalized strategy interfaces.

The active BTC research path is experiment -> CandidateSpec -> validation/replay/
holdout. This framework is preserved for a future multi-strategy activation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from quant_loop_trader.models.prediction import Prediction


class Strategy(ABC):
    name: str = "base"
    version: str = "v0"
    horizon: int = 5

    @abstractmethod
    def generate_predictions(self, snapshot, ticker: str) -> list[Prediction]:
        """Convert a PIT snapshot into predictions without accessing future data."""

    def metadata(self) -> dict:
        return {"name": self.name, "version": self.version, "horizon": self.horizon}


class MomentumStrategy(Strategy):
    name = "momentum_reference"
    version = "v1"
    horizon = 5

    def generate_predictions(self, snapshot, ticker: str) -> list[Prediction]:
        from quant_loop_trader.features.technical import add_features, feature_columns
        feat = add_features(snapshot).drop_nulls(subset=feature_columns())
        preds = []
        for row in feat.iter_rows(named=True):
            ret = row["ret_5"] or 0
            score = 1 if ret > 0 else 0
            preds.append(Prediction(
                timestamp=str(row["event_time"]),
                ticker=ticker.upper(),
                horizon=self.horizon,
                prediction=score,
                confidence=(
                    0.5 + min(abs(ret), 0.05)
                    if score == 1
                    else 0.5 - min(abs(ret), 0.05)
                ),
                features_used=feature_columns(),
                model_version=f"{self.name}-{self.version}",
                strategy_version=self.version,
            ))
        return preds


STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    MomentumStrategy.name: MomentumStrategy,
}


def build_strategy(name: str) -> Strategy:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"unknown strategy '{name}'; available: {sorted(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name]()
