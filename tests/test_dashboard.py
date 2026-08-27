"""The dashboard observes Quant Loop; it must never become a second source of truth.

These tests prove three things: the dashboard cannot write research state, its
charts are computed from the sealed artifacts (not invented), and it stays
functional in every real operating state — empty lab, stopped autonomy, and an
experiment mid-flight.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest

from quant_loop_trader.dashboard import api, queries as q, service as svc
from quant_loop_trader.dashboard.schemas import periods_per_year

REPO = Path(__file__).resolve().parents[1]

ENDPOINTS = ["/api/overview", "/api/cycles", "/api/cycles/current", "/api/experiments",
             "/api/hypotheses", "/api/champions", "/api/validation", "/api/system",
             "/api/activity", "/api/market"]


# --- fixtures ---------------------------------------------------------------
@pytest.fixture(autouse=True)
def no_cache(monkeypatch):
    monkeypatch.setattr(q, "CACHE_TTL", 0)
    q.clear_caches()


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """A miniature Quant Loop data root the dashboard can read."""
    monkeypatch.setenv("QLT_ROOT", str(tmp_path))
    monkeypatch.delenv("QLT_AUTONOMOUS_ENABLED", raising=False)
    for sub in ("experiments", "logs", "processed", "datasets"):
        (tmp_path / "data" / sub).mkdir(parents=True, exist_ok=True)
    _make_db(tmp_path / "data" / "research.duckdb")
    return tmp_path


def _make_db(path: Path, experiments=(), models=()):
    con = duckdb.connect(str(path))
    con.execute("""CREATE OR REPLACE TABLE experiments (experiment_id VARCHAR, dataset_id VARCHAR,
        ticker VARCHAR, horizon_days INTEGER, version VARCHAR, hypothesis VARCHAR,
        economic_reasoning VARCHAR, research_question VARCHAR, model_version VARCHAR,
        feature_version VARCHAR, seed INTEGER, config_json VARCHAR, metrics_json VARCHAR,
        decision VARCHAR, parent_experiment_id VARCHAR, provenance_json VARCHAR,
        created_at TIMESTAMP, authoritative BOOLEAN)""")
    con.execute("""CREATE OR REPLACE TABLE model_registry (model_id VARCHAR, parent_model_id VARCHAR,
        training_data_version VARCHAR, feature_version VARCHAR, parameters_json VARCHAR,
        performance_history_json VARCHAR, failure_modes VARCHAR, research_lineage VARCHAR,
        status VARCHAR, provenance_json VARCHAR, version VARCHAR, created_at TIMESTAMP)""")
    con.execute("""CREATE OR REPLACE TABLE research_memory (memory_id VARCHAR, experiment_id VARCHAR,
        memory_type VARCHAR, hypothesis VARCHAR, economic_reasoning VARCHAR, outcome VARCHAR,
        lesson VARCHAR, conditions VARCHAR, evidence_json VARCHAR, confidence FLOAT,
        provenance_json VARCHAR, version VARCHAR, created_at TIMESTAMP, authoritative BOOLEAN)""")
    con.execute("""CREATE OR REPLACE TABLE datasets (dataset_id VARCHAR, ticker VARCHAR, start_date DATE,
        end_date DATE, source VARCHAR, version VARCHAR, checksum VARCHAR, row_count INTEGER,
        validation_status VARCHAR, snapshot_definition VARCHAR, provenance_json VARCHAR,
        created_at TIMESTAMP)""")
    con.execute("""CREATE OR REPLACE TABLE tasks (task_id VARCHAR, task_type VARCHAR, payload_json VARCHAR,
        status VARCHAR, priority INTEGER, claimed_by VARCHAR, attempts INTEGER,
        result_json VARCHAR, provenance_json VARCHAR, version VARCHAR, created_at TIMESTAMP,
        updated_at TIMESTAMP)""")
    for row in models:
        con.execute("INSERT INTO model_registry VALUES (?,NULL,'v1','v1','{}','{}','','',?,'{}','v1',current_timestamp)", list(row))
    con.close()


def _register(root: Path, exp_id: str, *, authoritative: bool, status: str | None = None,
              hypothesis: str = "H1: test hypothesis"):
    """Insert the baseline + improved DB records a real run writes, and the
    matching model_registry rows (which carry no authoritative column)."""
    con = duckdb.connect(str(root / "data" / "research.duckdb"))
    for variant, hyp in (("baseline", "baseline momentum"), ("improved", hypothesis)):
        con.execute(
            "INSERT INTO experiments VALUES (?,'DS_1','SPY',5,'v1',?,'','','sklearn-LogReg',"
            "'v1',42,'{}','{}','KEEP',NULL,'{}',current_timestamp,?)",
            [f"{exp_id}_{variant}", hyp, authoritative])
        if status:
            con.execute(
                "INSERT INTO model_registry VALUES (?,NULL,'v1','v1','{}','{}','','',?,"
                "'{}','v1',current_timestamp)", [f"{exp_id}_{variant}", status])
    con.close()
    q.clear_caches()


def _seal_experiment(root: Path, exp_id: str, *, prices, preds, horizon=5,
                     decision="KEEP", ticker="SPY", validation=None, holdout=None,
                     sealed=True):
    """Write an experiment bundle the way experiment.run_experiment does."""
    d = root / "data" / "experiments" / exp_id
    d.mkdir(parents=True)
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE p (event_time DATE, available_time DATE, close DOUBLE, "
        "y_true TINYINT, y_pred BIGINT, y_prob DOUBLE)")
    day = datetime(2024, 1, 1)
    for i, (price, pred) in enumerate(zip(prices, preds)):
        t = (day + timedelta(days=i)).date()
        con.execute("INSERT INTO p VALUES (?,?,?,?,?,?)", [t, t, price, 1, pred, 0.6])
    for variant in ("improved", "baseline"):
        con.execute(f"COPY p TO '{d / f'predictions_{variant}.parquet'}' (FORMAT PARQUET)")
    con.close()

    cfg = {"ticker": ticker, "horizon": horizon, "seed": 42, "dataset_id": "DS_1",
           "model_version": "sklearn-LogReg", "feature_version_improved": "v1+vol",
           "train_period": ["2023-01-01", "2023-06-01"], "test_period": ["2024-01-01", "2024-03-01"]}
    (d / "config.json").write_text(json.dumps(cfg))
    if not sealed:
        return d
    curve = svc.bucket_returns(
        [{"event_time": (day + timedelta(days=i)).date(), "close": p, "y_pred": q_}
         for i, (p, q_) in enumerate(zip(prices, preds))], horizon)
    net = svc.compound(curve["net"])
    metrics = {
        "cumulative_return_strategy": net[-1] - 1 if net else 0.0,
        "cumulative_return_benchmark": svc.compound(curve["bench"])[-1] - 1,
        "sharpe_strategy": svc.sharpe(curve["net"], periods_per_year(ticker, horizon)),
        "max_drawdown_strategy": svc.max_drawdown(net),
        "n_return_buckets": curve["n_buckets"], "n_test": len(preds), "stat_pvalue": 0.4,
    }
    (d / "metrics.json").write_text(json.dumps({"baseline": metrics, "improved": metrics}))
    (d / "report.json").write_text(json.dumps({
        "experiment_id": exp_id, "hypothesis": "H1: test hypothesis",
        "research_question": "Q1?", "decision": decision, "config": cfg,
        "candidate_stat_pvalue": 0.4, "created_at": "2024-01-01T00:00:00+00:00",
    }))
    (d / "predictions.lock").write_text(json.dumps({"locked_at": "2024-01-01T00:00:00+00:00"}))
    if validation is not None:
        (d / "validation.json").write_text(json.dumps(validation))
    if holdout is not None:
        (d / "holdout_report.json").write_text(json.dumps(holdout))
    return d


def _hash_tree(root: Path) -> dict:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _sweep(paths=ENDPOINTS):
    return [api.route(p, {}) for p in paths]


# --- 1 & 2: the dashboard cannot mutate research state ----------------------
def test_connections_are_read_only(lab):
    con = q.connect()
    with pytest.raises(duckdb.Error):
        con.execute("CREATE TABLE dashboard_should_not_write (x INTEGER)")
    with pytest.raises(duckdb.Error):
        con.execute("INSERT INTO experiments VALUES (" + ",".join(["NULL"] * 18) + ")")
    con.close()


def test_no_source_file_is_modified_by_any_endpoint(lab):
    _seal_experiment(lab, "20240101_SPY_5d_aaaa", prices=_prices(60), preds=[1] * 60)
    (lab / "data" / "logs" / "heartbeat.json").write_text(json.dumps(
        {"timestamp": datetime.now(timezone.utc).isoformat(), "grid_remaining": 3}))
    before = _hash_tree(lab / "data")
    _sweep()
    api.route("/api/experiments/20240101_SPY_5d_aaaa", {})
    api.route("/api/performance/20240101_SPY_5d_aaaa", {})
    api.route("/api/risk/20240101_SPY_5d_aaaa", {})
    assert _hash_tree(lab / "data") == before


def test_dashboard_never_migrates_the_database(lab, monkeypatch):
    """migrate_db writes. No dashboard module may reach it."""
    import quant_loop_trader.data as data_mod

    def explode(*_a, **_k):
        raise AssertionError("dashboard called migrate_db")

    monkeypatch.setattr(data_mod, "migrate_db", explode)
    _seal_experiment(lab, "20240101_SPY_5d_bbbb", prices=_prices(60), preds=[1] * 60)
    _sweep()


def test_sealed_artifacts_are_not_touched_by_reading_predictions(lab):
    d = _seal_experiment(lab, "20240101_SPY_5d_cccc", prices=_prices(60), preds=[1, 0] * 30)
    before = _hash_tree(d)
    svc.curve("20240101_SPY_5d_cccc")
    svc.performance("20240101_SPY_5d_cccc")
    svc.risk("20240101_SPY_5d_cccc")
    assert _hash_tree(d) == before


# --- 3-7: the maths -------------------------------------------------------
def _prices(n, step=1.01):
    return [100 * (step ** i) for i in range(n)]


def test_compounded_return_is_multiplicative_not_additive():
    assert svc.compound([0.1, 0.1]) == pytest.approx([1.1, 1.21])
    assert svc.compound([0.5, -0.5])[-1] == pytest.approx(0.75)


def test_drawdown_is_equity_over_running_max():
    equity = [1.0, 1.2, 0.9, 1.5]
    assert svc.drawdown_series(equity) == pytest.approx([0.0, 0.0, 0.9 / 1.2 - 1, 0.0])
    # a first-period loss must register — the wealth path starts at 1.0
    assert svc.max_drawdown([0.8, 1.0]) == pytest.approx(-0.2)


def test_rolling_sharpe_matches_the_definition():
    rets = [0.01, -0.02, 0.03, 0.00, 0.02]
    ppy = 52.0
    got = svc.rolling(rets, 3, lambda w: svc.sharpe(w, ppy))
    assert got[0] is None and got[1] is None
    window = rets[:3]
    mean = sum(window) / 3
    sd = math.sqrt(sum((x - mean) ** 2 for x in window) / 3)
    assert got[2] == pytest.approx(math.sqrt(ppy) * mean / sd)


def test_bucket_returns_are_non_overlapping_and_cost_adjusted():
    prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    rows = [{"event_time": f"2024-01-{i + 1:02d}", "close": p, "y_pred": 1}
            for i, p in enumerate(prices)]
    b = svc.bucket_returns(rows, horizon=5)
    assert b["n_buckets"] == 2
    assert b["bench"] == pytest.approx([105 / 100 - 1, 110 / 105 - 1])
    # entry cost charged once on the first bucket, no switch afterwards
    assert b["net"][0] == pytest.approx(b["gross"][0] - 0.0005)
    assert b["net"][1] == pytest.approx(b["gross"][1])


def test_annualisation_uses_the_market_calendar_not_a_constant():
    assert periods_per_year("SPY", 1) == 252
    assert periods_per_year("SPY", 5) == 252 / 5
    assert periods_per_year("BTC", 1) == 365          # crypto trades every day
    assert periods_per_year("BTC-USD", 5) == 365 / 5


def test_risk_annualises_on_the_market_calendar(lab):
    _seal_experiment(lab, "20240101_BTC_5d_dddd", prices=_prices(60), preds=[1] * 60,
                     ticker="BTC")
    r = svc.risk("20240101_BTC_5d_dddd")
    assert r["calendar_days"] == 365
    assert r["periods_per_year"] == pytest.approx(365 / 5)


@pytest.mark.parametrize("experiment_id", sorted(
    p.parent.name for p in (REPO / "data" / "experiments").glob("*/metrics.json")
)[:12] or ["__none__"])
def test_curve_reproduces_the_sealed_metrics(experiment_id, monkeypatch):
    """The chart's series must reproduce the experiment's own sealed numbers."""
    if experiment_id == "__none__":
        pytest.skip("no local experiment artifacts")
    monkeypatch.setenv("QLT_ROOT", str(REPO))
    art = q.artifacts(experiment_id)
    sealed = (art["metrics"] or {}).get("improved") or {}
    curve = svc.curve(experiment_id, "improved")
    if not curve.get("available") or not sealed:
        pytest.skip("experiment has no reconstructable curve")
    last = curve["points"][-1]
    assert last["strategy"] - 1 == pytest.approx(sealed["cumulative_return_strategy"], rel=1e-9)
    assert last["benchmark"] - 1 == pytest.approx(sealed["cumulative_return_benchmark"], rel=1e-9)
    equity = [p["strategy"] for p in curve["points"]]
    assert svc.max_drawdown(equity) == pytest.approx(sealed["max_drawdown_strategy"], abs=1e-9)
    assert curve["reconciled"], f"no return convention reproduces {experiment_id}"
    # Older bundles were sealed with sqrt(252) on h-day buckets; the dashboard
    # annualizes on the bucket calendar. Accept either, but nothing else.
    got = svc.sharpe([p["return"] for p in curve["points"]], curve["periods_per_year"])
    legacy = got * math.sqrt(curve["bucket_days"])
    assert sealed["sharpe_strategy"] == pytest.approx(got, rel=1e-9) or \
        sealed["sharpe_strategy"] == pytest.approx(legacy, rel=1e-9)


# --- 8 & 9: cycle and counts match the real run state -----------------------
def _write_sessions(root: Path, sessions):
    (root / "data" / "logs" / "session.log").write_text("\n".join(json.dumps(s, indent=2)
                                                                  for s in sessions))


def test_cycle_progress_and_counts_come_from_session_state(lab):
    for i, eid in enumerate(["e1", "e2", "e3"]):
        _seal_experiment(lab, eid, prices=_prices(60), preds=[1] * 60,
                         decision="KEEP" if i == 0 else "REJECT")
    _write_sessions(lab, [
        {"session_started": "2024-01-01T00:00:00+00:00",
         "session_finished": "2024-01-01T00:05:00+00:00", "mode": "OBSERVATION",
         "budget": 2, "executed": 2, "grid_remaining": 7,
         "results": [{"experiment_id": "e1", "decision": "KEEP", "validation_status": "APPROVED"},
                     {"experiment_id": "e2", "decision": "REJECT", "validation_status": "REJECTED"}]},
        {"session_started": "2024-01-02T00:00:00+00:00",
         "session_finished": "2024-01-02T00:02:00+00:00", "mode": "OBSERVATION",
         "budget": 4, "executed": 1, "grid_remaining": 6,
         "results": [{"experiment_id": "e3", "decision": "REJECT", "validation_status": "REJECTED"}]},
    ])
    cycles = svc.cycles()
    assert [c["cycle_number"] for c in cycles] == [1, 2]
    assert cycles[0]["planned_experiments"] == 2 and cycles[0]["completed_experiments"] == 2
    assert cycles[0]["keeps"] == 1 and cycles[0]["rejects"] == 1
    assert cycles[0]["validation_passes"] == 1
    assert cycles[0]["elapsed_s"] == pytest.approx(300)
    assert cycles[1]["unused_budget"] == 3     # budget 4, one candidate available
    assert cycles[1]["failed_experiments"] == 0

    current = svc.current_cycle()
    assert current["cycle_number"] == 2 and current["status"] == "IDLE"
    assert current["progress"] == pytest.approx(0.25)
    assert svc.overview()["progress"]["experiments_completed"] == 3
    assert svc.overview()["progress"]["experiments_queued"] is None  # no heartbeat yet


def test_failed_run_in_a_cycle_is_counted(lab):
    _seal_experiment(lab, "ok1", prices=_prices(60), preds=[1] * 60)
    _seal_experiment(lab, "crashed1", prices=_prices(60), preds=[1] * 60, sealed=False)
    _write_sessions(lab, [{
        "session_started": "2024-01-01T00:00:00+00:00",
        "session_finished": "2024-01-01T00:05:00+00:00", "budget": 2, "executed": 2,
        "results": [{"experiment_id": "ok1", "decision": "KEEP"},
                    {"experiment_id": "crashed1", "decision": "REJECT"}]}])
    assert svc.cycles()[0]["failed_experiments"] == 1


# --- 10-13: honest behaviour in every real state ---------------------------
def test_missing_metrics_are_reported_as_unavailable_never_invented(lab):
    d = lab / "data" / "experiments" / "sparse"
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps({"experiment_id": "sparse", "decision": "REJECT"}))
    (d / "config.json").write_text(json.dumps({"ticker": "SPY", "horizon": 5}))
    perf = svc.performance("sparse")
    for field in ("net_return", "sharpe", "max_drawdown", "excess_return", "cagr",
                  "annualized_volatility", "p_value"):
        assert perf.get(field) is None, field
    assert svc.curve("sparse")["available"] is False
    assert svc.risk("sparse")["available"] is False
    row = [r for r in svc.experiment_index() if r["id"] == "sparse"][0]
    assert row["net_return"] is None and row["validation"] == "NOT_RUN"
    assert row["holdout"] == "NOT_RUN"


def test_empty_lab_serves_every_endpoint(lab):
    payloads = _sweep()
    overview = payloads[0]
    assert overview["progress"]["experiments_completed"] == 0
    assert overview["funnel"]["champions"] == 0
    assert svc.current_cycle()["status"] == "NO_DATA"
    assert api.route("/api/experiments", {})["experiments"] == []
    assert svc.champions()["champions"] == []
    assert svc.correlation_matrix([])["available"] is False


def test_missing_database_degrades_without_crashing(lab):
    (lab / "data" / "research.duckdb").unlink()
    sys = svc.system()
    assert sys["database"]["status"] == "UNAVAILABLE"
    assert any("database" in e for e in sys["errors"])
    assert svc.experiment_index() == []
    _sweep(["/api/overview", "/api/system", "/api/experiments"])


def test_autonomy_stopped_is_reported_as_stopped(lab, monkeypatch):
    monkeypatch.setenv("QLT_AUTONOMOUS_ENABLED", "false")
    _seal_experiment(lab, "e1", prices=_prices(60), preds=[1] * 60)
    sys = svc.system()
    assert sys["autonomy"] == "DISABLED"
    assert sys["workers"] == 0

    monkeypatch.setenv("QLT_AUTONOMOUS_ENABLED", "true")
    assert svc.system()["autonomy"] == "IDLE"      # enabled but nothing in flight


def test_running_experiment_is_visible_while_it_runs(lab, monkeypatch):
    monkeypatch.setenv("QLT_AUTONOMOUS_ENABLED", "true")
    _seal_experiment(lab, "done1", prices=_prices(60), preds=[1] * 60)
    _write_sessions(lab, [{
        "session_started": "2024-01-01T00:00:00+00:00",
        "session_finished": "2024-01-01T00:05:00+00:00", "budget": 1, "executed": 1,
        "results": [{"experiment_id": "done1", "decision": "KEEP"}]}])
    # an experiment directory with no sealed report is a run in flight
    _seal_experiment(lab, "20240102_SPY_5d_live", prices=_prices(60), preds=[1] * 60,
                     sealed=False)

    cycle = svc.current_cycle()
    assert cycle["status"] == "RUNNING"
    assert cycle["cycle_number"] == 2
    assert cycle["active_experiment"] == "20240102_SPY_5d_live"
    assert cycle["active_market"] == "SPY" and cycle["active_horizon"] == 5

    sys = svc.system()
    assert sys["autonomy"] == "RUNNING" and sys["workers"] == 1

    row = [r for r in svc.experiment_index() if r["id"] == "20240102_SPY_5d_live"][0]
    assert row["status"] == "RUNNING" and row["stage"] == "EXECUTION"
    assert row["net_return"] is None
    assert svc.overview()["progress"]["experiments_active"] == 1


def test_pipeline_stages_reflect_recorded_evidence(lab):
    _seal_experiment(
        lab, "staged", prices=_prices(60), preds=[1, 0] * 30, decision="KEEP",
        validation={
            "approval_status": "REJECTED",
            "issues_found": ["multiple_testing:fdr_not_significant:p0.4000"],
            "reviews": [
                {"reviewer": "statistical_reviewer", "approval_status": "APPROVED",
                 "tests_completed": ["majority_class_null_test"], "issues_found": []},
                {"reviewer": "adversarial_reviewer", "approval_status": "REJECTED",
                 "tests_completed": [], "issues_found": ["label_randomisation:acc"]},
                {"reviewer": "independent_replicator", "approval_status": "APPROVED",
                 "tests_completed": ["retrain_same_seed"], "issues_found": []},
            ],
            "hardening": {
                "walk_forward": {"stable_across_time": True, "mean_accuracy": 0.53, "folds": [1, 2, 3]},
                "multiple_testing": {"dsr": 0.55, "verdict": "LOW_CONFIDENCE", "n_trials": 4},
            },
        },
        holdout={"promoted": False, "holdout_accuracy": 0.5, "base_rate": 0.6, "n_holdout": 40},
    )
    stages = {s["key"]: s["status"] for s in svc.stages("staged")}
    assert stages["research_gate"] == "PASS"
    assert stages["replication"] == "PASS"
    assert stages["walk_forward"] == "PASS"
    assert stages["adversarial"] == "FAIL"
    assert stages["dsr"] == "FAIL"          # LOW_CONFIDENCE is not a pass
    assert stages["fdr"] == "FAIL"
    assert stages["validation"] == "FAIL"
    assert stages["holdout"] == "FAIL"
    assert stages["champion"] == "NOT_RUN"
    assert stages["data"] == "NOT_AVAILABLE"   # snapshot absent from disk
    assert svc.current_stage(svc.stages("staged")) == "ADVERSARIAL"

    rej = svc.rejections()
    reasons = {r["reason"]: r["count"] for r in rej["validation_issues"]}
    assert reasons["FDR failure"] == 1
    assert svc.funnel()["validation_pass"] == 0


def test_champion_lifecycle_and_correlation_gate(lab):
    for eid in ("a", "b"):
        _seal_experiment(lab, eid, prices=_prices(60), preds=[1, 0] * 30)
        _register(lab, eid, authoritative=True, status="champion")
    champs = svc.champions()
    assert {c["experiment_id"] for c in champs["champions"]} == {"a", "b"}
    corr = champs["correlation"]
    assert corr["available"] is True
    assert corr["matrix"][0][0] == pytest.approx(1.0)
    assert corr["highly_correlated"]      # identical series must be flagged


def test_experiment_filters_and_404(lab):
    _seal_experiment(lab, "20240101_SPY_5d_f1", prices=_prices(60), preds=[1] * 60)
    rows = api.route("/api/experiments", {"market": ["SPY"]})["experiments"]
    assert len(rows) == 1
    assert api.route("/api/experiments", {"market": ["BTC"]})["experiments"] == []
    assert api.route("/api/experiments", {"champion_only": ["true"]})["experiments"] == []
    with pytest.raises(KeyError):
        api.route("/api/experiments/does_not_exist", {})
    with pytest.raises(KeyError):
        api.route("/api/nope", {})


# --- authoritative population (proof 14) ------------------------------------


def _two_experiment_lab(lab):
    prices = [100 + i for i in range(40)]
    preds = [1, 0] * 20
    _seal_experiment(lab, "E_OLD", prices=prices, preds=preds, decision="KEEP")
    _seal_experiment(lab, "E_NEW", prices=prices, preds=preds, decision="KEEP")
    _register(lab, "E_OLD", authoritative=False, status="champion",
              hypothesis="H_OLD: quarantined idea")
    _register(lab, "E_NEW", authoritative=True, status="candidate",
              hypothesis="H_NEW: current idea")
    return lab


def test_quarantined_experiments_are_excluded_from_every_count(lab):
    """A run marked non-authoritative still has a directory on disk. It must not
    inflate experiment, hypothesis, funnel or rejection counts."""
    _two_experiment_lab(lab)
    assert [r["id"] for r in svc.authoritative()] == ["E_NEW"]
    assert svc.population() == {
        "basis": "AUTHORITATIVE", "on_disk": 2, "authoritative": 1, "quarantined": 1,
        "unrecorded": 0,
        "reason": "quarantined runs predate the current pipeline and are excluded",
    }
    f = svc.funnel()
    assert f["experiments"] == 1
    assert f["hypotheses"] == 1
    assert [h["hypothesis"] for h in svc.hypotheses()] == ["H1: test hypothesis"]
    assert svc.overview()["experiments_total"] == 1
    assert svc.default_experiment() == "E_NEW"


def test_lifecycle_counts_join_the_registry_to_authoritative_experiments(lab):
    """model_registry has no authoritative column of its own, so a quarantined
    experiment's champion row must not be counted as a champion."""
    _two_experiment_lab(lab)
    assert svc._registry_map()["E_OLD_improved"]["status"] == "champion"  # unfiltered
    assert "E_OLD_improved" not in svc._registry_map(authoritative_only=True)
    assert svc.champions()["counts"]["champion"] == 0
    assert svc.champions()["counts"]["candidate"] == 1
    assert svc.overview()["progress"]["champions"] == 0
    assert svc.funnel()["champions"] == 0


def test_baseline_records_never_reach_the_hypothesis_count(lab):
    """Each run writes a `baseline momentum` record alongside the real one."""
    _two_experiment_lab(lab)
    rows = q.query("SELECT hypothesis FROM experiments WHERE authoritative")
    assert sorted(r["hypothesis"] for r in rows) == ["H_NEW: current idea", "baseline momentum"]
    assert svc.funnel()["hypotheses"] == 1  # counted from report.json, one per directory


def test_population_is_unknown_not_empty_when_the_database_is_gone(lab):
    """Losing the database must not silently empty the dashboard."""
    _two_experiment_lab(lab)
    (lab / "data" / "research.duckdb").unlink()
    q.clear_caches()
    assert q.authoritative_ids() is None
    assert svc.population()["basis"] == "UNKNOWN"
    assert svc.population()["authoritative"] is None
    assert len(svc.authoritative()) == 2  # intact, flagged unknown
    assert all(r["authoritative"] is None for r in svc.experiment_index())


def test_experiments_endpoint_hides_quarantined_rows_by_default(lab):
    _two_experiment_lab(lab)
    default = api.route("/api/experiments", {})
    assert [r["id"] for r in default["experiments"]] == ["E_NEW"]
    assert default["population"]["quarantined"] == 1
    both = api.route("/api/experiments", {"include_quarantined": ["1"]})
    assert sorted(r["id"] for r in both["experiments"]) == ["E_NEW", "E_OLD"]


def test_champion_holdout_metrics_come_only_from_the_holdout_report(lab):
    """Research-window drawdown must never be presented as holdout drawdown."""
    prices = [100 + i for i in range(40)]
    _seal_experiment(lab, "E_H", prices=prices, preds=[1, 0] * 20, decision="KEEP",
                     holdout={"promoted": True, "holdout_accuracy": 0.61, "base_rate": 0.55,
                              "n_holdout": 145,
                              "economic_gate": {"compounded_net_return": 0.11,
                                                "sharpe_strategy": 1.5,
                                                "sharpe_benchmark": 1.4}})
    _register(lab, "E_H", authoritative=True, status="champion")
    h = svc._holdout_economics("E_H")
    assert h["status"] == "PASS"
    assert h["net_return"] == 0.11
    assert h["sharpe"] == 1.5
    assert h["accuracy"] == 0.61 and h["n"] == 145
    assert h["max_drawdown"] is None and h["annualized_volatility"] is None
    champ = svc.champions()["champions"][0]
    assert champ["metrics_basis"] == "research test window (sealed metrics.json)"
    assert champ["holdout"]["net_return"] == 0.11
    assert svc._holdout_economics("E_H")["n"] != champ["net_return"]


def test_cycles_separate_scheduler_progress_from_authoritative_evidence(lab):
    prices = [100 + i for i in range(40)]
    _seal_experiment(lab, "E_C", prices=prices, preds=[1, 0] * 20)
    _register(lab, "E_C", authoritative=False)
    (lab / "data" / "logs" / "session.log").write_text(json.dumps({
        "session_started": "2024-01-01T00:00:00+00:00",
        "session_finished": "2024-01-01T00:05:00+00:00",
        "budget": 1, "executed": 1, "grid_remaining": 4, "mode": "OBSERVATION",
        "results": [{"experiment_id": "E_C", "decision": "KEEP",
                     "validation_status": "REJECTED"}],
    }, indent=1))
    q.clear_caches()
    cyc = svc.cycles()[0]
    assert cyc["completed_experiments"] == 1        # the scheduler ran it
    assert cyc["grid_remaining_basis"] == "scheduler"
    assert cyc["authoritative_experiments"] == 0    # it is not current evidence


def test_an_unsealed_run_with_no_database_record_stays_visible(lab):
    """An in-flight experiment has no experiments row yet — it is unrecorded,
    not quarantined, and the control room must still show it."""
    _two_experiment_lab(lab)
    _seal_experiment(lab, "E_RUNNING", prices=_prices(40), preds=[1, 0] * 20, sealed=False)
    q.clear_caches()
    rows = {r["id"]: r["authoritative"] for r in svc.experiment_index()}
    assert rows == {"E_NEW": True, "E_OLD": False, "E_RUNNING": None}
    assert sorted(r["id"] for r in svc.authoritative()) == ["E_NEW", "E_RUNNING"]
    assert svc.population()["unrecorded"] == 1
