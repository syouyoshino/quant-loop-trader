"""Hypothesis ranker (Task 1c): research value score before spending compute.

score = information_gain + novelty + mechanism_quality - cost - duplicate_risk
"""
from __future__ import annotations

from quant_loop_trader.research.hypothesis_engine import ResearchHypothesis
from quant_loop_trader.research_memory import search_memory

MECHANISM_QUALITY = {  # charter: prefer understandable mechanisms
    "information_delay": 0.8, "risk_compensation": 0.8, "market_structure": 0.7,
    "behavioural": 0.6, "under_research": 0.3,
}


def score_hypothesis(h: ResearchHypothesis) -> dict:
    mem = search_memory(h.title[:60])
    fails = sum(1 for m in mem if m["memory_type"] in ("failure", "partial"))
    successes = sum(1 for m in mem if m["memory_type"] == "success")

    novelty = 1.0 if not h.similar_previous_experiments else max(0.2, 1.0 - 0.2 * len(h.similar_previous_experiments))
    info_gain = min(1.0, 0.4 + 0.3 * len(h.feature_groups))   # more families → more to learn
    mechanism = MECHANISM_QUALITY.get(h.expected_mechanism, 0.5)
    duplicate_risk = min(1.0, fails * 0.25)
    cost = 0.2 + 0.05 * len(h.feature_groups)                 # compute is cheap here but nonzero

    total = round(info_gain + novelty + mechanism - cost - duplicate_risk, 3)
    return {
        "hypothesis_id": h.hypothesis_id,
        "title": h.title,
        "score": total,
        "components": {"info_gain": info_gain, "novelty": novelty, "mechanism": mechanism,
                       "cost": cost, "duplicate_risk": duplicate_risk,
                       "prior_failures": fails, "prior_successes": successes},
    }


def rank(hypotheses: list[ResearchHypothesis], min_score: float = 0.0) -> list[dict]:
    ranked = sorted((score_hypothesis(h) for h in hypotheses), key=lambda s: -s["score"])
    return [s for s in ranked if s["score"] >= min_score]
