"""Deferred structured hypothesis generation from feature families and memory."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResearchHypothesis:
    hypothesis_id: str
    title: str
    research_question: str
    reasoning: str
    feature_groups: dict[str, list[str]]
    target_variable: str
    prediction_horizon: int
    model_type: str
    expected_mechanism: str
    risk_factors: list[str] = field(default_factory=list)
    similar_previous_experiments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _hyp_id(title: str) -> str:
    return "hyp_" + hashlib.sha256(title.encode()).hexdigest()[:10]


FEATURE_FAMILIES: dict[str, list[str]] = {
    "technical_momentum": ["ret_1", "ret_5", "ma10_gap"],
    "technical_volatility": ["vol10", "rsi14"],
    "macro_rates": ["fed_funds", "high_rate_regime"],
    "macro_inflation": ["inflation_yoy"],
    "macro_labor": ["unemployment"],
    "fundamental_quality": ["net_margin", "return_on_equity"],
    "fundamental_growth": ["revenue_growth", "earnings_growth"],
}

ALFRED_REQUIRED = {"macro_rates", "macro_inflation", "macro_labor"}

MECHANISMS = {
    "technical_momentum": "information_delay — prices under-react to news",
    "macro_rates": "risk_compensation — discount-rate regime shifts reprice assets",
    "macro_inflation": "market_structure — inflation surprises move risk premia",
    "macro_labor": "behavioural — labour data shifts growth expectations",
    "fundamental_quality": "risk_compensation — quality carries a persistent premium",
    "fundamental_growth": "information_delay — fundamental trends diffuse slowly",
}


@dataclass
class NoveltyReport:
    is_novel: bool
    similarity_scores: dict[str, float]
    reasons: list[str]


def token_similarity(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def novelty_check(hypothesis_text: str, previous_texts: list[str],
                  threshold: float = 0.55) -> NoveltyReport:
    scores = {
        prev[:60]: round(token_similarity(hypothesis_text, prev), 3)
        for prev in previous_texts
    }
    max_sim = max(scores.values()) if scores else 0.0
    reasons = []
    if max_sim >= threshold:
        reasons.append(f"too_similar_to_tested_hypothesis:{max_sim:.2f}")
    return NoveltyReport(
        is_novel=max_sim < threshold,
        similarity_scores=dict(sorted(scores.items(), key=lambda kv: -kv[1])[:5]),
        reasons=reasons,
    )


def generate_candidates(previous_texts: list[str], horizons: list[int] = (5,),
                        max_candidates: int = 10) -> list[ResearchHypothesis]:
    from quant_loop_trader.research_memory import search_memory

    allow_revised_macro = os.getenv("QLT_ALLOW_REVISED_MACRO", "").lower() == "true"
    families = sorted(
        f for f in FEATURE_FAMILIES
        if allow_revised_macro or f not in ALFRED_REQUIRED
    )
    out: list[ResearchHypothesis] = []
    for i, fa in enumerate(families):
        for fb in families[i + 1:]:
            mech_a = MECHANISMS.get(fa, "under_research")
            mech_b = MECHANISMS.get(fb, "under_research")
            title = f"{fa} x {fb} interaction at {'/'.join(str(h) for h in horizons)}d horizon"
            mem = search_memory(f"{fa} {fb}")
            fails = [m for m in mem if m["memory_type"] == "failure"]
            if len(fails) >= 3:
                logger.info(json.dumps({
                    "event": "hypothesis_skipped_failed_family",
                    "family": f"{fa}+{fb}",
                    "prior_failures": len(fails),
                }))
                continue
            hyp = ResearchHypothesis(
                hypothesis_id=_hyp_id(title),
                title=title,
                research_question=f"Does combining {fa} and {fb} features improve directional prediction?",
                reasoning=f"{mech_a}; combined with: {mech_b}",
                feature_groups={fa: FEATURE_FAMILIES[fa], fb: FEATURE_FAMILIES[fb]},
                target_variable="sign(close[t+h]/close[t]-1)",
                prediction_horizon=horizons[0],
                model_type="logistic",
                expected_mechanism=mech_a.split(" — ")[0],
                risk_factors=[
                    "regime_dependence: interaction may hold only in one volatility state",
                    "multiple_testing: one of many family combinations tested",
                ],
                similar_previous_experiments=[m["experiment_id"] or "" for m in mem[:3]],
            )
            if not novelty_check(title, previous_texts).is_novel:
                continue
            out.append(hyp)
            if len(out) >= max_candidates:
                return out
    return out
