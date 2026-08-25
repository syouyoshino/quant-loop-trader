import json

import pytest
from pathlib import Path

from quant_loop_trader.autonomy import run_session, select_candidates, review_memory, _already_run


@pytest.fixture(autouse=True)
def autonomous_enabled(monkeypatch):
    """Session tests exercise the gated path; the OFF path has its own test."""
    monkeypatch.setenv("QLT_AUTONOMOUS_ENABLED", "true")


def test_session_blocked_without_activation_key(monkeypatch):
    monkeypatch.delenv("QLT_AUTONOMOUS_ENABLED", raising=False)
    out = run_session(max_experiments=1)
    assert out["skipped"] == "autonomous_disabled" and out["executed"] == 0
    from quant_loop_trader.data import migrate_db, DB_PATH as _real_unused
    import quant_loop_trader.data as _dm
    _dm.migrate_db()
    from quant_loop_trader.automation.queue import pending_count
    assert pending_count() == 0  # blocked session must not touch the queue


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
