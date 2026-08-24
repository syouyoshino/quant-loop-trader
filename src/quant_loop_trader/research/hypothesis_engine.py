"""Hypothesis engine (Task 1): structured research questions from feature families
+ research memory. Generates candidates, checks novelty, ranks by expected value.

Every hypothesis still enters the full experiment + validation pipeline — nothing
here claims an idea works.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResearchHypothesis:
    hypothesis_id: str
    title: str
    research_question: str
    reasoning: str                    # economic mechanism — why should this exist?
    feature_groups: dict[str, list[str]]
    target_variable: str              # e.g. "sign(fwd_ret)"
    prediction_horizon: int
    model_type: str                   # registry model name
    expected_mechanism: str           # behavioural | market_structure | risk_compensation | information_delay
    risk_factors: list[str] = field(default_factory=list)
    similar_previous_experiments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _hyp_id(title: str) -> str:
    return "hyp_" + hashlib.sha256(title.encode()).hexdigest()[:10]


# --- Feature families available for combination ------------------------------
FEATURE_FAMILIES: dict[str, list[str]] = {
    "technical_momentum": ["ret_1", "ret_5", "ma10_gap"],
    "technical_volatility": ["vol10", "rsi14"],
    "macro_rates": ["fed_funds", "high_rate_regime"],
    "macro_inflation": ["inflation_yoy"],
    "macro_labor": ["unemployment"],
    "fundamental_quality": ["net_margin", "return_on_equity"],
    "fundamental_growth": ["revenue_growth", "earnings_growth"],
}

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
    """Jaccard similarity over lowercase word tokens."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def novelty_check(hypothesis_text: str, previous_texts: list[str],
                  threshold: float = 0.55) -> NoveltyReport:
    """Duplicate guard: reject hypotheses too similar to already-tested ideas."""
    scores = {prev[:60]: round(token_similarity(hypothesis_text, prev), 3)
              for prev in previous_texts}
    max_sim = max(scores.values()) if scores else 0.0
    reasons = []
    if max_sim >= threshold:
        reasons.append(f"too_similar_to_tested_hypothesis:{max_sim:.2f}")
    return NoveltyReport(is_novel=max_sim < threshold,
                         similarity_scores=dict(sorted(scores.items(), key=lambda kv: -kv[1])[:5]),
                         reasons=reasons)


def generate_candidates(previous_texts: list[str], horizons: list[int] = (5,),
                        max_candidates: int = 10) -> list[ResearchHypothesis]:
    """Combine untested feature-family pairs × mechanisms into structured hypotheses.
    Memory-driven: pairs whose mechanism already failed repeatedly get skipped."""
    from quant_loop_trader.research_memory import search_memory

    families = sorted(FEATURE_FAMILIES)
    out: list[ResearchHypothesis] = []
    for i, fa in enumerate(families):
        for fb in families[i + 1:]:
            mech_a = MECHANISMS.get(fa, "under_research")
            mech_b = MECHANISMS.get(fb, "under_research")
            title = f"{fa} x {fb} interaction at {'/'.join(str(h) for h in horizons)}d horizon"
            # memory-driven skip: identical title family already rejected hard
            mem = search_memory(f"{fa} {fb}")
            fails = [m for m in mem if m["memory_type"] == "failure"]
            if len(fails) >= 3:
                logger.info(json.dumps({"event": "hypothesis_skipped_failed_family",
                                        "family": f"{fa}+{fb}", "prior_failures": len(fails)}))
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
            nov = novelty_check(title, previous_texts)
            if not nov.is_novel:
                continue
            out.append(hyp)
            if len(out) >= max_candidates:
                return out
    return out
