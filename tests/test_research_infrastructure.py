
from quant_loop_trader.research_memory import (
    search_memory, record_outcome, prior_confidence, duplicate_risk,
    register_features, register_model, _con,
)
from quant_loop_trader.experiment import run_experiment


def _fake_report(exp_id: str, decision: str) -> dict:
    return {
        "experiment_id": exp_id,
        "decision": decision,
        "hypothesis": "volatility regime classification improves momentum prediction",
        "economic_reasoning": "trend persistence differs across vol environments",
        "failure_condition": "no accuracy lift or Sharpe degrades",
        "error_analysis": {"overall_accuracy": 0.62, "regime_0_acc": 0.54, "false_positive_rate": 1.0},
        "improved_metrics": {"accuracy": 0.63},
        "improvement_delta_accuracy": 0.01 if decision != "REJECT" else 0.0,
        "config": {"ticker": "SPY", "horizon": 5},
        "research_branch": "research/mvrs",
        "creator_agent": "test",
    }


def test_migration_registries_exist():
    con = _con()
    tables = [r[0] for r in con.execute("show tables").fetchall()]
    for t in ["feature_registry", "model_registry", "research_memory"]:
        assert t in tables
    # provenance columns per constitution
    for t in ["feature_registry", "model_registry", "research_memory"]:
        cols = [r[0] for r in con.execute(f"describe {t}").fetchall()]
        assert "created_at" in cols and "version" in cols and "provenance_json" in cols
    con.close()


def test_register_features_idempotent():
    defs = [{"feature_id": "ret_1", "formula": "shift(close/close.shift(1)-1, 1)"}]
    register_features(defs)
    register_features(defs)
    con = _con()
    n = con.execute("select count(*) from feature_registry where feature_id='ret_1'").fetchone()[0]
    row = con.execute("select formula, validation_status from feature_registry where feature_id='ret_1'").fetchone()
    con.close()
    assert n == 1 and row[1] == "validated"


def test_register_model_and_status():
    register_model({"model_id": "m_test", "training_data_version": "ds_test", "feature_version": "v1"})
    register_model({"model_id": "m_test", "training_data_version": "ds_test", "feature_version": "v1", "status": "champion"})
    con = _con()
    rows = con.execute("select model_id, status from model_registry where model_id='m_test'").fetchall()
    con.close()
    assert len(rows) == 1 and rows[0][1] == "champion"


def test_record_outcome_failure_updates_belief():
    hyp = f"unique hypothesis {id(object())}: vol regime improves momentum"
    report = _fake_report(f"exp_fail_{id(object())}", "REJECT")
    report["hypothesis"] = hyp
    before = prior_confidence(hyp)
    assert before == 0.5  # unseen hypothesis starts neutral
    ids = record_outcome(report)
    assert len(ids) == 2
    after = search_memory(hyp)[0]
    assert after["memory_type"] == "failure"
    assert after["confidence"] < before  # belief decreased on failure


def test_duplicate_risk():
    # record several failures of the same hypothesis
    for i in range(4):
        record_outcome(_fake_report(f"exp_dup_{i}", "REJECT"))
    risk = duplicate_risk("volatility regime classification")
    assert risk["similar_failures"] >= 3 and risk["should_warn"] is True
    risk_none = duplicate_risk("quantum entanglement alpha")
    assert risk_none["should_warn"] is False


def test_e2e_registers_memory_and_models(tmp_path=None):
    report = run_experiment(ticker="SPY", horizon=5, start="2019-01-01", end="2024-12-31", seed=777)
    exp_id = report["experiment_id"]
    con = _con()
    models = con.execute("select model_id, status from model_registry where model_id like ?", [f"{exp_id}%"]).fetchall()
    feats = con.execute("select count(*) from feature_registry").fetchone()[0]
    mems = con.execute("select memory_type from research_memory where experiment_id like ?", [f"{exp_id}%"]).fetchall()
    con.close()
    assert len(models) == 2
    assert feats >= 7
    assert len(mems) >= 2  # outcome + knowledge


def test_quarantined_outcomes_excluded_from_decision_views(isolated_research):
    """Audit C1 remediation: pre-fix outcomes are non-authoritative and must be
    invisible to memory search, experiment listing, and DSR trial counts."""
    import duckdb
    import quant_loop_trader.data as dm
    from quant_loop_trader.research_memory import search_memory
    from quant_loop_trader.experiment import run_experiment, list_experiments
    from quant_loop_trader.agents import _experiment_count

    dm.migrate_db()
    def _raw_authoritative():
        con = duckdb.connect(str(dm.DB_PATH), read_only=True)
        n = con.execute("SELECT count(*) FROM experiments WHERE authoritative AND experiment_id NOT LIKE '%_baseline'").fetchone()[0]
        con.close()
        return n
    before = _raw_authoritative()
    report = run_experiment(ticker="SPY", horizon=5, start="2020-01-01", end="2022-12-31", seed=404)
    after = _raw_authoritative()
    assert after == before + 1  # new experiment is authoritative

    con = duckdb.connect(str(dm.DB_PATH))
    con.execute("UPDATE experiments SET authoritative = FALSE")
    con.execute("UPDATE research_memory SET authoritative = FALSE")
    con.close()

    assert list_experiments() == []                      # listing hides quarantined
    assert search_memory("volatility regime") == []      # memory reads hide quarantined
    assert _experiment_count() == 1  # DSR trials exclude quarantined (floor=1 is DSR-safe)
