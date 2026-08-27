"""Regression tests for the 2026-08-27 hostile audit fixes."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pytest

from quant_loop_trader.evaluation import evaluate
from quant_loop_trader.experiment import run_experiment
from quant_loop_trader.validation.holdout import (
    HoldoutIntegrityError,
    adjudicate_holdout,
    release_holdout_claim,
    verify_holdout_evidence,
)


def _make_eligible():
    report = run_experiment(
        ticker="SPY", horizon=5, start="2019-01-01", end="2024-12-31", seed=919
    )
    exp_id = report["experiment_id"]
    import quant_loop_trader.data as dm

    con = duckdb.connect(str(dm.DB_PATH))
    con.execute(
        "UPDATE model_registry SET status='eligible' WHERE model_id=?",
        [f"{exp_id}_improved"],
    )
    con.close()
    return exp_id


def test_final_holdout_is_sealed_and_db_committed(isolated_research):
    exp_id = _make_eligible()
    result = adjudicate_holdout(exp_id)
    assert "holdout_commit_error" not in str(result.get("reason", ""))

    from quant_loop_trader.experiment import EXP_ROOT
    import quant_loop_trader.data as dm

    assert (EXP_ROOT / exp_id / "holdout_report.json").exists()
    assert (EXP_ROOT / exp_id / "holdout.lock").exists()
    assert verify_holdout_evidence(exp_id) == result

    con = duckdb.connect(str(dm.DB_PATH))
    state, promoted, stored = con.execute(
        "SELECT state, promoted, result_json FROM holdout_claims WHERE experiment_id=?",
        [exp_id],
    ).fetchone()
    registry = con.execute(
        "SELECT status FROM model_registry WHERE model_id=?", [f"{exp_id}_improved"]
    ).fetchone()[0]
    con.close()
    assert state == "COMPLETE"
    assert bool(promoted) == bool(result["promoted"])
    assert json.loads(stored) == result
    assert (registry == "champion") == bool(result["promoted"])


def test_tampered_holdout_is_rejected_by_verifier_and_dashboard(
    isolated_research, monkeypatch
):
    exp_id = _make_eligible()
    adjudicate_holdout(exp_id)

    from quant_loop_trader.experiment import EXP_ROOT

    p = EXP_ROOT / exp_id / "holdout_report.json"
    raw = json.loads(p.read_text())
    raw["promoted"] = not bool(raw.get("promoted"))
    p.write_text(json.dumps(raw, indent=2))

    with pytest.raises(HoldoutIntegrityError, match="holdout_report_tampered"):
        verify_holdout_evidence(exp_id)

    from quant_loop_trader.dashboard import queries as q

    root = Path(isolated_research).parent
    monkeypatch.setenv("QLT_ROOT", str(root))
    q.clear_caches()
    art = q.artifacts(exp_id)
    assert art["holdout_report"] is None
    assert art["holdout_integrity"]["status"] == "FAIL"
    assert "holdout_report_tampered" in art["holdout_integrity"]["reason"]


def test_completed_holdout_claim_cannot_be_reopened(isolated_research):
    exp_id = _make_eligible()
    adjudicate_holdout(exp_id)
    assert release_holdout_claim(exp_id) == "refused:COMPLETE"


def test_liquidated_metrics_include_terminal_exit_cost():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 1])
    y_prob = np.array([0.9, 0.9, 0.9])
    prices = np.array([100.0, 101.0, 102.0, 103.0])
    m = evaluate(y_true, y_pred, y_prob, prices, horizon=1)

    assert m["exit_cost_applied"] is True
    assert m["cumulative_return_strategy_liquidated"] < m["cumulative_return_strategy"]
    assert m["transaction_cost_adj_return_compounded"] == pytest.approx(
        m["cumulative_return_strategy_liquidated"]
    )
    assert m["transaction_cost_adj_return"] == pytest.approx(m["arithmetic_net_return_sum"])
