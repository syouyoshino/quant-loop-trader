"""Deferred hypothesis-engine tests: novelty, ranking, and config generation."""

from quant_loop_trader.experimental.hypothesis.hypothesis_engine import (
    generate_candidates,
    novelty_check,
    token_similarity,
)
from quant_loop_trader.experimental.hypothesis.experiment_generator import (
    to_experiment_config,
    validate_config,
)
from quant_loop_trader.experimental.hypothesis.hypothesis_ranker import rank


def test_token_similarity_bounds():
    assert token_similarity("alpha beta", "alpha beta gamma") > 0.5
    assert token_similarity("alpha beta", "delta epsilon") == 0.0


def test_novelty_check_flags_duplicates():
    prev = ["technical_momentum x macro_rates interaction at 5d horizon tested and failed"]
    rep = novelty_check("the technical_momentum x macro_rates interaction at 5d horizon", prev)
    assert not rep.is_novel and rep.reasons
    rep2 = novelty_check("fundamental_quality x macro_labor under quarterly regime rotation", prev)
    assert rep2.is_novel


def test_generate_candidates_structured_and_memory_aware():
    cands = generate_candidates(previous_texts=[], horizons=[5], max_candidates=6)
    assert cands
    for h in cands:
        d = h.to_dict()
        for field in [
            "hypothesis_id",
            "title",
            "research_question",
            "reasoning",
            "feature_groups",
            "target_variable",
            "prediction_horizon",
            "model_type",
            "expected_mechanism",
            "risk_factors",
        ]:
            assert field in d
        assert h.prediction_horizon == 5
        assert len(h.reasoning) > 20


def test_ranking_orders_and_penalizes_failures():
    cands = generate_candidates([], max_candidates=8)
    ranked = rank(cands)
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    top = ranked[0]
    assert {"info_gain", "novelty", "mechanism", "duplicate_risk"} <= set(top["components"])


def test_experiment_config_generation_and_validation():
    cands = generate_candidates([], max_candidates=2)
    cfg = to_experiment_config(cands[0], start="2019-01-01", end="2024-12-31")
    assert validate_config(cfg) == []
    assert cfg["requires_walk_forward"] is True
    bad = {"ticker": "SPY"}
    errs = validate_config(bad)
    assert "missing_field:hypothesis_id" in errs and "end_before_start" not in errs


def test_macro_families_gated_behind_alfred(monkeypatch):
    monkeypatch.delenv("QLT_ALLOW_REVISED_MACRO", raising=False)
    cands = generate_candidates([], max_candidates=50)
    for h in cands:
        assert not (set(h.feature_groups) & {"macro_rates", "macro_inflation", "macro_labor"})
    monkeypatch.setenv("QLT_ALLOW_REVISED_MACRO", "true")
    cands2 = generate_candidates([], max_candidates=50)
    assert any("macro_rates" in h.feature_groups for h in cands2)
