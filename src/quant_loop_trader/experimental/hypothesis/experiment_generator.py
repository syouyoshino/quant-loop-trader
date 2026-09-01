"""Deferred ResearchHypothesis -> experiment-config adapter."""
from __future__ import annotations

from quant_loop_trader.experimental.hypothesis.hypothesis_engine import ResearchHypothesis


def to_experiment_config(h: ResearchHypothesis, start: str, end: str, seed: int = 42) -> dict:
    all_features = [c for cols in h.feature_groups.values() for c in cols]
    return {
        "hypothesis_id": h.hypothesis_id,
        "ticker": "SPY",
        "horizon": h.prediction_horizon,
        "start": start,
        "end": end,
        "seed": seed,
        "model_type": h.model_type,
        "feature_columns": all_features,
        "target_variable": h.target_variable,
        "research_question": h.research_question,
        "economic_reasoning": h.reasoning,
        "expected_mechanism": h.expected_mechanism,
        "risk_factors": h.risk_factors,
        "requires_walk_forward": True,
    }


def validate_config(cfg: dict) -> list[str]:
    errs = []
    for key in (
        "hypothesis_id", "ticker", "horizon", "start", "end", "seed",
        "model_type", "feature_columns",
    ):
        if key not in cfg:
            errs.append(f"missing_field:{key}")
    if "end" in cfg and "start" in cfg and cfg["end"] <= cfg["start"]:
        errs.append("end_before_start")
    if not cfg.get("feature_columns"):
        errs.append("no_features")
    return errs
