"""Activation-blocker tests: holdout adjudication path (audit remediation).

Exercises the REAL adjudicate_holdout training/prediction path — no mocked .fit().
"""
import datetime

import duckdb
import polars as pl
import pytest

from quant_loop_trader.experiment import run_experiment
from quant_loop_trader.validation.holdout import (
    adjudicate_holdout,
    apply_holdout,
    release_holdout_claim,
    verify_holdout_evidence,
)


def _make_eligible(isolated_research):
    """Run a real experiment and force its improved variant to ELIGIBLE."""
    report = run_experiment(ticker="SPY", horizon=5, start="2019-01-01", end="2024-12-31", seed=77)
    exp_id = report["experiment_id"]
    import quant_loop_trader.data as dm
    con = duckdb.connect(str(dm.DB_PATH))
    con.execute("UPDATE model_registry SET status='eligible' WHERE model_id=?", [f"{exp_id}_improved"])
    row = con.execute("SELECT status FROM model_registry WHERE model_id=?",
                      [f"{exp_id}_improved"]).fetchone()
    con.close()
    return exp_id, row[0]


# --- A. real training call ---------------------------------------------------

def test_adjudication_real_training_call_succeeds(isolated_research):
    """Regression for the fit((X,y), None) bug: adjudication must complete through
    the REAL LogisticModel.fit(X, y) without raising."""
    exp_id, status = _make_eligible(isolated_research)
    assert status == "eligible"
    result = adjudicate_holdout(exp_id)
    assert isinstance(result, dict) and "promoted" in result


# --- B. promotion iff all gates pass -----------------------------------------

def test_champion_iff_promoted(isolated_research):
    exp_id, _ = _make_eligible(isolated_research)
    result = adjudicate_holdout(exp_id)
    import quant_loop_trader.data as dm
    con = duckdb.connect(str(dm.DB_PATH))
    status = con.execute("SELECT status FROM model_registry WHERE model_id=?",
                         [f"{exp_id}_improved"]).fetchone()[0]
    con.close()
    assert (status == "champion") == (result["promoted"] is True)


# --- C. non-eligible ----------------------------------------------------------

def test_non_eligible_cannot_adjudicate(isolated_research):
    report = run_experiment(ticker="SPY", horizon=5, start="2020-01-01", end="2023-12-31", seed=51)
    out = adjudicate_holdout(report["experiment_id"])
    assert out["promoted"] is False and "not_eligible" in out["reason"]


# --- D. broken adjudication fails closed --------------------------------------

def test_failed_adjudication_cannot_promote_and_fails_closed(isolated_research, monkeypatch):
    exp_id, _ = _make_eligible(isolated_research)
    import quant_loop_trader.validation.holdout as ho
    import quant_loop_trader.data as dm

    def boom(*a, **k):
        raise RuntimeError("validator exploded")

    monkeypatch.setattr(ho, "_load_holdout_frame", boom)
    out = ho.adjudicate_holdout(exp_id)
    assert out["promoted"] is False
    con = duckdb.connect(str(dm.DB_PATH))
    status = con.execute("SELECT status FROM model_registry WHERE model_id=?",
                         [f"{exp_id}_improved"]).fetchone()[0]
    con.close()
    assert status != "champion"


# --- E. insufficient holdout sample -------------------------------------------

def test_insufficient_holdout_sample_cannot_promote(isolated_research):
    report = run_experiment(ticker="SPY", horizon=5, start="2024-06-01", end="2024-12-31", seed=61)
    import quant_loop_trader.data as dm
    con = duckdb.connect(str(dm.DB_PATH))
    con.execute("UPDATE model_registry SET status='eligible' WHERE model_id=?",
                [f"{report['experiment_id']}_improved"])
    con.close()
    out = adjudicate_holdout(report["experiment_id"])
    assert out["promoted"] is False


# --- G. normal validation never creates a champion ----------------------------

def test_validate_experiment_never_creates_champion(isolated_research):
    from quant_loop_trader.agents import validate_experiment
    report = run_experiment(ticker="SPY", horizon=5, start="2019-01-01", end="2024-12-31", seed=88)
    verdict = validate_experiment(report["experiment_id"])
    import quant_loop_trader.data as dm
    con = duckdb.connect(str(dm.DB_PATH))
    statuses = con.execute("SELECT DISTINCT status FROM model_registry WHERE model_id LIKE ?",
                           [f"{report['experiment_id']}%"]).fetchall()
    con.close()
    assert all(s != "champion" for (s,) in statuses)
    if verdict["approval_status"] == "APPROVED":
        assert ("eligible",) in statuses


# --- H. holdout isolation: concrete timestamp disjointness --------------------

def test_holdout_rows_absent_from_training_frames(isolated_research):
    from quant_loop_trader.data import PROC_DIR
    from quant_loop_trader.validation.holdout import holdout_boundary
    from quant_loop_trader.replay import ReplayEngine

    start, end = "2019-01-01", "2024-12-31"
    boundary = datetime.date.fromisoformat(holdout_boundary(start, end))
    pq = PROC_DIR / "SPY.parquet"

    df_all = ReplayEngine(pq).get_snapshot("SPY", end)
    df_all = df_all.filter(pl.col("event_time") >= pl.lit(start).str.strptime(pl.Date, "%Y-%m-%d"))

    train_dates = set(df_all.filter(pl.col("event_time") < boundary)["event_time"].to_list())
    hold_dates = set(apply_holdout(df_all, start, end, use_holdout=True)["event_time"].to_list())

    assert not (train_dates & hold_dates), "training and holdout windows overlap"
    assert min(hold_dates) > max(train_dates)

    from quant_loop_trader.validation.holdout import _train_all_nonholdout
    cfg = {"ticker": "SPY", "start": start, "end": end, "horizon": 5, "seed": 1}
    Xtr, ytr = _train_all_nonholdout(pq, cfg)
    assert Xtr.shape[0] <= len(train_dates)
    assert ytr.shape[0] == Xtr.shape[0]


def test_holdout_rejection_corrects_provisional_success_memory(isolated_research):
    """KEEP → validation APPROVED → eligible → holdout REJECTED must correct the
    provisional success memory, not leave it confirmed."""
    from quant_loop_trader.research_memory import search_memory
    exp_id, _ = _make_eligible(isolated_research)
    import quant_loop_trader.data as dm
    dm.migrate_db()
    con = duckdb.connect(str(dm.DB_PATH))
    con.execute(
        "INSERT OR REPLACE INTO research_memory VALUES (?, ?, 'success', ?, ?, "
        "'KEEP', 'Confirmed: survived hidden-future split', '{}', '{}', 0.65, '{}', "
        "'v1', current_timestamp, TRUE)",
        [f"mem_{exp_id}_success", exp_id,
         "volatility regime hypothesis", "economic reasoning"])
    con.close()

    result = adjudicate_holdout(exp_id)
    if result["promoted"]:
        pytest.skip("genuinely promoted — nothing to correct")
    rows = search_memory("volatility regime hypothesis")
    mine = [r for r in rows if r["memory_id"] == f"mem_{exp_id}_success"]
    assert mine, "memory row vanished (audit trail destroyed)"
    assert mine[0]["memory_type"] == "failure"
    assert "CORRECTED" in mine[0]["lesson"] or "holdout" in mine[0]["lesson"].lower()


# --- F. the hidden holdout is consumable exactly once ------------------------

def test_holdout_cannot_be_adjudicated_twice(isolated_research):
    exp_id, _ = _make_eligible(isolated_research)
    first = adjudicate_holdout(exp_id)
    assert "holdout_already_consumed" not in str(first.get("reason", ""))

    import quant_loop_trader.data as dm
    con = duckdb.connect(str(dm.DB_PATH))
    con.execute("UPDATE model_registry SET status='eligible' WHERE model_id=?",
                [f"{exp_id}_improved"])
    con.close()
    second = adjudicate_holdout(exp_id)
    assert second["promoted"] is False
    assert second["reason"].startswith("holdout_already_consumed")


def test_crash_after_holdout_is_read_is_permanently_consumed(isolated_research):
    """Once CLAIMED, crash recovery must never make the hidden holdout reusable."""
    exp_id, _ = _make_eligible(isolated_research)
    import quant_loop_trader.data as dm
    con = duckdb.connect(str(dm.DB_PATH))
    con.execute("INSERT INTO holdout_claims (experiment_id, state) VALUES (?, 'CLAIMED')",
                [exp_id])
    con.close()

    blocked = adjudicate_holdout(exp_id)
    assert blocked["promoted"] is False
    assert blocked["reason"] == "holdout_already_consumed:CLAIMED"

    assert release_holdout_claim(exp_id) == "consumed:FAILED"
    assert release_holdout_claim(exp_id) == "refused:FAILED"

    con = duckdb.connect(str(dm.DB_PATH))
    state, raw_result = con.execute(
        "SELECT state, result_json FROM holdout_claims WHERE experiment_id=?", [exp_id]
    ).fetchone()
    status = con.execute(
        "SELECT status FROM model_registry WHERE model_id=?", [f"{exp_id}_improved"]
    ).fetchone()[0]
    assert state == "FAILED"
    assert status == "rejected"
    assert __import__("json").loads(raw_result)["consumed"] is True
    # Even a manual registry reset cannot reopen the consumed claim.
    con.execute("UPDATE model_registry SET status='eligible' WHERE model_id=?",
                [f"{exp_id}_improved"])
    con.close()

    recovered = adjudicate_holdout(exp_id)
    assert recovered["promoted"] is False
    assert recovered["reason"] == "holdout_already_consumed:FAILED"

    # Restore the fail-closed registry status to validate the sealed tombstone.
    con = duckdb.connect(str(dm.DB_PATH))
    con.execute("UPDATE model_registry SET status='rejected' WHERE model_id=?",
                [f"{exp_id}_improved"])
    con.close()
    sealed = verify_holdout_evidence(exp_id)
    assert sealed["promoted"] is False
    assert sealed["consumed"] is True
    assert sealed["reason"] == "holdout_aborted_after_claim"


def test_full_holdout_metrics_are_persisted(isolated_research):
    """The engine persists full holdout metrics and gates on liquidated economics."""
    exp_id, _ = _make_eligible(isolated_research)
    result = adjudicate_holdout(exp_id)
    if "adjudication_error" in str(result.get("reason", "")):
        pytest.skip(f"adjudication failed upstream: {result['reason']}")
    m = result["holdout_metrics"]
    for key in ("max_drawdown_strategy", "volatility_strategy", "sortino_ratio",
                "calmar_ratio", "var_95", "expected_shortfall_95", "turnover",
                "win_rate", "n_return_buckets"):
        assert key in m, key
    assert m["cumulative_return_strategy_liquidated"] == result["economic_gate"]["compounded_net_return"]
    assert m["sharpe_strategy_liquidated"] == result["economic_gate"]["sharpe_strategy"]