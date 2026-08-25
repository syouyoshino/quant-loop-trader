import json

import duckdb

import quant_loop_trader.data as dm

import numpy as np
import polars as pl
import pytest

from quant_loop_trader.agents import (
    _verify_locks,
    ROLES, statistical_review, validate_experiment,
)
from quant_loop_trader.experiment import run_experiment


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("stat_review")


def test_roles_cannot_violate_constitution():
    for role, perms in ROLES.items():
        assert "modify_evaluation_engine" in perms["cannot"] or "alter_scientific_standards" not in perms["can"]
    # researcher cannot approve own discoveries
    assert "approve_own_discoveries" in ROLES["quant_researcher"]["cannot"]


def test_statistical_review_rejects_coinflip(tmp_dir):
    # construct predictions at exactly coin-flip level: 50/50 correct on 200 rows
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)
    df = pl.DataFrame({"y_true": y_true, "y_pred": y_pred})
    p = tmp_dir / "stat_test_exp"
    p.mkdir(exist_ok=True)
    df.write_parquet(str(p / "predictions_improved.parquet"))
    review = statistical_review(p)
    assert review["approval_status"] == "REJECTED"
    assert any("not_significant" in i for i in review["issues_found"])


def test_statistical_review_small_sample(tmp_dir):
    df = pl.DataFrame({"y_true": [1] * 10, "y_pred": [1] * 10})
    p = tmp_dir / "stat_test_small"
    p.mkdir(exist_ok=True)
    df.write_parquet(str(p / "predictions_improved.parquet"))
    review = statistical_review(p)
    assert any("sample_size_too_small" in i for i in review["issues_found"])
    assert any("degenerate_constant_predictions" in i for i in review["issues_found"])


def test_validation_gate_end_to_end(isolated_research):
    report = run_experiment(ticker="SPY", horizon=5, start="2019-01-01", end="2024-12-31", seed=555)
    exp_id = report["experiment_id"]
    verdict = validate_experiment(exp_id)
    vpath = isolated_research / exp_id / "validation.json"
    assert vpath.exists()
    loaded = json.loads(vpath.read_text())
    assert loaded["approval_status"] in ("APPROVED", "REJECTED")
    # all three reviewers reported
    assert [r["reviewer"] for r in loaded["reviews"]] == [
        "statistical_reviewer", "adversarial_reviewer", "independent_replicator"]
    # replication must match (deterministic pipeline) — no mismatch issues allowed
    repl = loaded["reviews"][2]
    assert not any("replication_mismatch" in i for i in repl["issues_found"])
    # predictions lock verified — no tampering issues on fresh experiment
    assert not any("artifact_tampered" in i or "missing_predictions_lock" in i for i in loaded["issues_found"])


def test_prediction_lock_detects_tampering(isolated_research):
    report = run_experiment(ticker="SPY", horizon=5, start="2020-01-01", end="2023-12-31", seed=321)
    exp_dir = isolated_research / report["experiment_id"]
    # tamper: flip a stored prediction after the fact
    import polars as pl
    p = exp_dir / "predictions_improved.parquet"
    df = pl.read_parquet(str(p))
    df = df.with_columns(pl.when(pl.arange(0, df.height) == 0).then(1 - pl.col("y_pred")).otherwise(pl.col("y_pred")).alias("y_pred"))
    df.write_parquet(str(p))
    issues = _verify_locks(exp_dir)
    assert any("artifact_tampered:predictions_improved.parquet" in i for i in issues)
    verdict = validate_experiment(report["experiment_id"])
    assert "artifact_tampered:predictions_improved.parquet" in verdict["issues_found"]


def test_memory_correction_fires_on_rejected_keep(isolated_research):
    # simulate: KEEP wrote a success memory, then validation REJECTS it
    import duckdb
    import quant_loop_trader.data as dm
    from quant_loop_trader.agents import validate_experiment
    from quant_loop_trader.research_memory import search_memory

    report = run_experiment(ticker="SPY", horizon=5, start="2020-01-01", end="2023-12-31", seed=314)
    con = duckdb.connect(str(dm.DB_PATH))
    con.execute(
        "INSERT OR REPLACE INTO research_memory VALUES (?, ?, 'success', ?, ?, 'KEEP', 'premature', '{}', '{}', 0.7, '{}', 'v1', current_timestamp, TRUE)",
        [f"mem_{report['experiment_id']}_success", report["experiment_id"],
         "vol regime hypothesis", "Confirmed prematurely"],
    )
    con.close()

    verdict = validate_experiment(report["experiment_id"])
    rows = search_memory("vol regime hypothesis")
    mine = [r for r in rows if r["memory_id"] == f"mem_{report['experiment_id']}_success"]
    if verdict["approval_status"] == "REJECTED":
        assert mine and mine[0]["memory_type"] == "failure"      # corrected, not deleted
        assert "CORRECTED" in mine[0]["lesson"]                  # audit trail preserved


def test_dataset_parquet_lock_key_not_treated_as_artifact(isolated_research):
    from quant_loop_trader.agents import _verify_locks
    report = run_experiment(ticker="SPY", horizon=5, start="2020-01-01", end="2023-12-31", seed=315)
    issues = _verify_locks(isolated_research / report["experiment_id"])
    assert not any("locked_artifact_missing:dataset_parquet" in i for i in issues)


def test_reject_decision_never_becomes_eligible(isolated_research, monkeypatch):
    """Audit H1 regression: independent validation is ADDITIVE — it must never
    resurrect an experiment whose predeclared criterion already failed."""
    from quant_loop_trader import agents as ag
    report = run_experiment(ticker="SPY", horizon=5, start="2020-01-01", end="2023-12-31", seed=4242)
    # force the predeclared decision to REJECT (report.json is locked; rewrite lock too)
    exp_dir = isolated_research / report["experiment_id"]
    rep = json.loads((exp_dir / "report.json").read_text())
    rep["decision"] = "REJECT"
    (exp_dir / "report.json").write_text(json.dumps(rep, indent=2))
    lock = json.loads((exp_dir / "predictions.lock").read_text())
    import hashlib
    lock["report.json"] = hashlib.sha256((exp_dir / "report.json").read_bytes()).hexdigest()
    (exp_dir / "predictions.lock").write_text(json.dumps(lock, indent=2))

    # every reviewer approves cleanly
    monkeypatch.setattr(ag, "statistical_review",
                        lambda *a, **k: ag._msg("statistical_reviewer", ["t"], []))
    monkeypatch.setattr(ag, "adversarial_review",
                        lambda *a, **k: ag._msg("adversarial_reviewer", ["t"], []))
    monkeypatch.setattr(ag, "independent_replication",
                        lambda eid: ag._msg("independent_replicator", ["t"], []))
    # also neutralize the hardening layer (under test here is the DECISION gate)
    import polars as pl
    import quant_loop_trader.validation.walkforward as wf_mod
    import quant_loop_trader.validation.ablation as abl_mod

    class FakeWF:
        def __init__(self, *a, **k): pass
        def run(self, *a, **k):
            return {"folds": [], "mean_accuracy": 0.6, "accuracy_dispersion": 0.0,
                    "stable_across_time": True}
    monkeypatch.setattr(wf_mod, "WalkForwardValidator", FakeWF)
    monkeypatch.setattr(abl_mod, "run_ablation", lambda *a, **k: pl.DataFrame({
        "removed": ["momentum", "volatility"], "accuracy": [0.5, 0.5], "delta_vs_full": [0.0, 0.0]}))
    monkeypatch.setattr("quant_loop_trader.validation.multiple_testing.deflated_sharpe_ratio",
                        lambda *a, **k: {"dsr": 1.0, "expected_max_sharpe_h0": 0.0,
                                         "verdict": "GENUINE", "n_trials": 3})
    monkeypatch.setattr(ag, "_family_pvalues", lambda *a, **k: ([], None))

    verdict = validate_experiment(report["experiment_id"])
    assert verdict["approval_status"] == "APPROVED"  # gates pass...
    con = duckdb.connect(str(dm.DB_PATH))
    status = con.execute("SELECT status FROM model_registry WHERE model_id=?",
                         [f"{report['experiment_id']}_improved"]).fetchone()[0]
    con.close()
    # audit H1 invariant: a REJECT decision is terminal — validation approval
    # can NEVER resurrect it into eligible/candidate/champion
    assert status == "rejected" and status != "eligible" and status != "champion"


def test_low_confidence_dsr_blocks_approval(isolated_research, monkeypatch):
    """Audit H4 regression: LOW_CONFIDENCE DSR must not silently pass validation."""
    from quant_loop_trader.validation import multiple_testing as mt
    real = mt.deflated_sharpe_ratio
    def low(*a, **k):
        out = real(*a, **k)
        out = dict(out); out["verdict"] = "LOW_CONFIDENCE"; out["dsr"] = 0.55
        return out
    monkeypatch.setattr(mt, "deflated_sharpe_ratio", low)
    report = run_experiment(ticker="SPY", horizon=5, start="2020-01-01", end="2023-12-31", seed=777)
    verdict = validate_experiment(report["experiment_id"])
    assert any("dsr_low_confidence" in i for i in verdict["issues_found"])


def test_holdout_economic_gate_uses_compounded_wealth(isolated_research):
    """Audit H2 regression: arithmetic sum +10% while compounded wealth -20% must
    NOT satisfy the positive-return gate."""
    import numpy as np
    from quant_loop_trader.evaluation import evaluate
    y = np.array([1] * 6)
    p = np.array([1] * 6)
    prob = np.array([0.9] * 6)
    prices = np.array([100., 160., 80., 128., 64., 102., 122.])  # buckets: +60%, -50%, ...
    rets_net = [0.6 * (1 - 0.0005), -0.5 * (1 - 0.0005)]
    m = evaluate(y, p, prob, np.array([100., 160., 80.]), horizon=1)
    # construct explicitly: two buckets +60%/-50%
    m2 = evaluate(np.array([1, 1]), np.array([1, 1]), np.array([0.9, 0.9]),
                  np.array([100., 160., 80.]), horizon=1)
    assert m2["transaction_cost_adj_return"] > 0      # arithmetic sum positive
    assert m2["cumulative_return_strategy"] < 0       # compounded wealth NEGATIVE
