import json
import shutil
from pathlib import Path

import pytest

from quant_loop_trader.autonomy import run_session, select_candidates, review_memory, _already_run


@pytest.fixture
def isolated_research(tmp_path, monkeypatch):
    """Copy the research DB so tests don't mutate or depend on real research state."""
    import quant_loop_trader.data as data_mod
    import quant_loop_trader.experiment as exp_mod
    import quant_loop_trader.research_memory as rm_mod
    import quant_loop_trader.autonomy as auto_mod
    import quant_loop_trader.agents as agents_mod

    db = tmp_path / "research.duckdb"  # fresh empty DB — migrations auto-apply
    for m in (data_mod, exp_mod, rm_mod, auto_mod):
        monkeypatch.setattr(m, "DB_PATH", db)
    exp_root = tmp_path / "experiments"
    exp_root.mkdir()
    for m in (exp_mod, agents_mod):
        monkeypatch.setattr(m, "EXP_ROOT", exp_root)
    return exp_root


def test_review_memory_returns_structure():
    m = review_memory()
    assert "total_memory_rows" in m and "recent_beliefs" in m


def test_duplicate_prevention(isolated_research):
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


def test_session_budget_respected_and_validates(isolated_research):
    summary = run_session(max_experiments=2, validate=True)
    assert summary["executed"] <= 2
    for r in summary["results"]:
        assert r["validation_status"] in ("APPROVED", "REJECTED")
        assert r["decision"] in ("KEEP", "IMPROVE", "REJECT")
        vpath = isolated_research / r["experiment_id"] / "validation.json"
        assert vpath.exists()
    assert summary["mode"] == "OBSERVATION"
