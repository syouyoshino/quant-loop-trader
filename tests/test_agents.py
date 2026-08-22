import json
from pathlib import Path

import numpy as np
import polars as pl

from quant_loop_trader.agents import (
    ROLES, statistical_review, adversarial_review,
    independent_replication, validate_experiment, _msg,
)
from quant_loop_trader.experiment import run_experiment


def test_roles_cannot_violate_constitution():
    for role, perms in ROLES.items():
        assert "modify_evaluation_engine" in perms["cannot"] or "alter_scientific_standards" not in perms["can"]
    # researcher cannot approve own discoveries
    assert "approve_own_discoveries" in ROLES["quant_researcher"]["cannot"]


def test_statistical_review_rejects_coinflip():
    # construct predictions at exactly coin-flip level: 50/50 correct on 200 rows
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)
    df = pl.DataFrame({"y_true": y_true, "y_pred": y_pred})
    p = Path("/tmp") / "stat_test_exp"
    p.mkdir(exist_ok=True)
    df.write_parquet(str(p / "predictions_improved.parquet"))
    review = statistical_review(p)
    assert review["approval_status"] == "REJECTED"
    assert any("not_significant" in i for i in review["issues_found"])


def test_statistical_review_small_sample():
    df = pl.DataFrame({"y_true": [1] * 10, "y_pred": [1] * 10})
    p = Path("/tmp") / "stat_test_small"
    p.mkdir(exist_ok=True)
    df.write_parquet(str(p / "predictions_improved.parquet"))
    review = statistical_review(p)
    assert any("sample_size_too_small" in i for i in review["issues_found"])


def test_validation_gate_end_to_end():
    report = run_experiment(ticker="SPY", horizon=5, start="2019-01-01", end="2024-12-31", seed=555)
    exp_id = report["experiment_id"]
    verdict = validate_experiment(exp_id)
    vpath = Path("data/experiments") / exp_id / "validation.json"
    assert vpath.exists()
    loaded = json.loads(vpath.read_text())
    assert loaded["approval_status"] in ("APPROVED", "REJECTED")
    # all three reviewers reported
    assert [r["reviewer"] for r in loaded["reviews"]] == [
        "statistical_reviewer", "adversarial_reviewer", "independent_replicator"]
    # replication must match (deterministic pipeline) — no mismatch issues allowed
    repl = loaded["reviews"][2]
    assert not any("replication_mismatch" in i for i in repl["issues_found"])
