"""Every test runs against an isolated DB + experiments dir. The real research
DB must never be mutated by the test suite."""
import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_research(tmp_path, monkeypatch):
    import quant_loop_trader.data as data_mod
    import quant_loop_trader.experiment as exp_mod
    import quant_loop_trader.research_memory as rm_mod
    import quant_loop_trader.autonomy as auto_mod
    import quant_loop_trader.agents as agents_mod
    import quant_loop_trader.report as report_mod
    import quant_loop_trader.automation.queue as queue_mod

    db = tmp_path / "research.duckdb"
    for m in (data_mod, exp_mod, rm_mod, auto_mod, queue_mod):
        monkeypatch.setattr(m, "DB_PATH", db)
    exp_root = tmp_path / "experiments"
    exp_root.mkdir()
    for m in (exp_mod, agents_mod):
        monkeypatch.setattr(m, "EXP_ROOT", exp_root)
    yield exp_root
