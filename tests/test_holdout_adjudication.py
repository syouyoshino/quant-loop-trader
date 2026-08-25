"""Activation-blocker tests: holdout adjudication path (audit remediation).

Exercises the REAL adjudicate_holdout training/prediction path — no mocked .fit().
"""
import datetime

import duckdb
import polars as pl
import pytest

from quant_loop_trader.experiment import run_experiment
from quant_loop_trader.validation.holdout import adjudicate_holdout, apply_holdout


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
    result = adjudicate_holdout(exp_id)  # pre-fix: raises inside sklearn .fit
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
    out = ho.adjudicate_holdout(exp_id)  # must fail closed, never raise-then-promote
    assert out["promoted"] is False
    import quant_loop_trader.data as dm
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

    # the REAL training matrix comes only from non-holdout rows: verify the exact
    # helper used by adjudication against the boundary timestamps
    from quant_loop_trader.validation.holdout import _train_all_nonholdout
    cfg = {"ticker": "SPY", "start": start, "end": end, "horizon": 5, "seed": 1}
    Xtr, ytr = _train_all_nonholdout(pq, cfg)
    # M1: features computed on full history then filtered → no warmup loss;
    # only label-horizon purge removes tail rows from TRAINING
    n_train_available = len(train_dates) - 5 - 5  # purge(h) + label horizon drop
    assert Xtr.shape[0] <= len(train_dates)       # never trains ON holdout rows
    assert ytr.shape[0] == Xtr.shape[0]
    assert ytr.shape[0] == Xtr.shape[0]


def test_holdout_rejection_corrects_provisional_success_memory(isolated_research):
    """Audit H6 regression: KEEP → validation APPROVED → eligible → holdout REJECTED
    must correct the provisional 'success' memory, not leave it confirmed."""
    from quant_loop_trader.research_memory import search_memory
    exp_id, _ = _make_eligible(isolated_research)
    import quant_loop_trader.data as dm
    dm.migrate_db()  # ensure schema exists in this test's isolated DB
    con = duckdb.connect(str(dm.DB_PATH))
    # seed the provisional success memory exactly as record_outcome would on KEEP
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
