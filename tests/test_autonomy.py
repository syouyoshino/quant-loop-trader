import json

import pytest

from quant_loop_trader.autonomy import run_session, select_candidates, review_memory, _already_run


@pytest.fixture(autouse=True)
def autonomous_enabled(monkeypatch):
    """Session tests exercise the gated path; the OFF path has its own test."""
    monkeypatch.setenv("QLT_AUTONOMOUS_ENABLED", "true")


def test_session_blocked_without_activation_key(monkeypatch):
    monkeypatch.delenv("QLT_AUTONOMOUS_ENABLED", raising=False)
    out = run_session(max_experiments=1)
    assert out["skipped"] == "autonomous_disabled" and out["executed"] == 0

    # The legacy tasks table may still exist in old databases/migrations, but the
    # simplified direct autonomy path must never enqueue work into it.
    import duckdb
    from quant_loop_trader.data import DB_PATH, migrate_db
    migrate_db()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    assert con.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0
    con.close()


def test_review_memory_returns_structure():
    m = review_memory()
    assert "total_memory_rows" in m and "recent_beliefs" in m


def test_duplicate_prevention():
    # fresh DB copy: first session executes; the executed config must not be re-selected
    s1 = run_session(max_experiments=1)
    assert s1["executed"] == 1
    key = s1["results"][0]["experiment_id"]
    candidates = select_candidates("SPY", 5, budget=10)
    cfgs = {f"{c['start']}_{c['end']}_{c['seed']}" for c in candidates}
    from quant_loop_trader.experiment import EXP_ROOT
    cfg = json.loads((EXP_ROOT / key / "config.json").read_text())
    executed_key = f"{cfg['start']}_{cfg['end']}_{cfg['seed']}"
    assert executed_key not in cfgs
    assert _already_run("SPY", 5, cfg["start"], cfg["end"], cfg["seed"])


def test_session_budget_respected_and_validates():
    summary = run_session(max_experiments=2, validate=True)
    assert summary["executed"] <= 2
    for r in summary["results"]:
        assert r["validation_status"] in ("APPROVED", "REJECTED")
        assert r["decision"] in ("KEEP", "IMPROVE", "REJECT")
        from quant_loop_trader.experiment import EXP_ROOT
        vpath = EXP_ROOT / r["experiment_id"] / "validation.json"
        assert vpath.exists()
    assert summary["mode"] == "OBSERVATION"


def test_quarantined_history_does_not_block_a_rerun():
    """Quarantined evidence is invalid, so it must not exhaust the frontier.

    Before this, the scheduler said "explored" for a config whose only evidence
    the science layer had already thrown away."""
    import duckdb

    from quant_loop_trader.data import DB_PATH, migrate_db
    from quant_loop_trader.experiment import EXP_ROOT

    s1 = run_session(max_experiments=1)
    assert s1["executed"] == 1
    key = s1["results"][0]["experiment_id"]
    cfg = json.loads((EXP_ROOT / key / "config.json").read_text())
    args = ("SPY", 5, cfg["start"], cfg["end"], cfg["seed"])
    assert _already_run(*args) is True

    migrate_db()
    con = duckdb.connect(str(DB_PATH))
    con.execute("UPDATE experiments SET authoritative=FALSE WHERE experiment_id LIKE ?",
                [f"{key}%"])
    con.close()
    assert _already_run(*args) is False
    assert any(c["start"] == cfg["start"] and c["end"] == cfg["end"]
               and c["seed"] == cfg["seed"] for c in select_candidates("SPY", 5, budget=50))


def test_baseline_records_alone_do_not_mark_a_config_explored():
    import duckdb

    from quant_loop_trader.data import DB_PATH, migrate_db
    from quant_loop_trader.experiment import EXP_ROOT

    s1 = run_session(max_experiments=1)
    key = s1["results"][0]["experiment_id"]
    cfg = json.loads((EXP_ROOT / key / "config.json").read_text())
    migrate_db()
    con = duckdb.connect(str(DB_PATH))
    con.execute("UPDATE experiments SET authoritative=FALSE "
                "WHERE experiment_id LIKE ? AND experiment_id NOT LIKE '%_baseline'",
                [f"{key}%"])
    con.close()
    assert _already_run("SPY", 5, cfg["start"], cfg["end"], cfg["seed"]) is False
