"""Every test runs against isolated DB, experiments, cache, and snapshots."""
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
    import quant_loop_trader.automation.queue as queue_mod

    db = tmp_path / "research.duckdb"
    for m in (data_mod, exp_mod, rm_mod, auto_mod, queue_mod):
        if hasattr(m, "DB_PATH"):
            monkeypatch.setattr(m, "DB_PATH", db)

    proc = tmp_path / "processed"
    proc.mkdir()
    real_parquet = Path("data/processed/SPY.parquet")
    if real_parquet.exists():
        shutil.copy(real_parquet, proc / "SPY.parquet")
    monkeypatch.setattr(data_mod, "PROC_DIR", proc)
    monkeypatch.setattr(exp_mod, "PROC_DIR", proc)

    exp_root = tmp_path / "experiments"
    exp_root.mkdir()
    for m in (exp_mod, agents_mod):
        monkeypatch.setattr(m, "EXP_ROOT", exp_root)
    yield exp_root
