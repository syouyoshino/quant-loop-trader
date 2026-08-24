"""Strategy framework (Phase 9). INTERFACES ONLY — no strategies run, no discovery.

Contract every strategy must honour:
- predictions come from ReplayEngine snapshots (PIT enforced upstream)
- outputs are frozen Prediction objects
- nothing here can promote itself: only validation/agents.validate_experiment moves status

Future strategy families register via STRATEGY_REGISTRY; none are active by default.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from quant_loop_trader.models.prediction import Prediction


class Strategy(ABC):
    """A research strategy converts a PIT snapshot into predictions. That is all."""

    name: str = "base"
    version: str = "v0"
    horizon: int = 5

    @abstractmethod
    def generate_predictions(self, snapshot, ticker: str) -> list[Prediction]:
        """snapshot: ReplayEngine.get_snapshot output — information available at T.
        Must NOT access evaluate_future() or any post-T data."""

    def metadata(self) -> dict:
        return {"name": self.name, "version": self.version, "horizon": self.horizon}


class MomentumStrategy(Strategy):
    """Reference implementation of the interface (NOT auto-run anywhere).
    Long when trailing momentum is positive — the baseline every strategy beats or dies."""

    name = "momentum_reference"
    version = "v1"
    horizon = 5

    def generate_predictions(self, snapshot, ticker: str) -> list[Prediction]:
        from quant_loop_trader.features.technical import add_features, feature_columns
        feat = add_features(snapshot).drop_nulls(subset=feature_columns())
        preds = []
        for row in feat.iter_rows(named=True):
            score = 1 if (row["ret_5"] or 0) > 0 else 0
            preds.append(Prediction(
                timestamp=str(row["event_time"]), ticker=ticker.upper(), horizon=self.horizon,
                prediction=score,
                confidence=0.5 + min(abs(row["ret_5"] or 0), 0.05),  # bounded, honest confidence
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
