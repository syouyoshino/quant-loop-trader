"""Audit C1 regression: migration ledger must be persistent and one-shot.

A fresh process opening an existing DB must NOT re-run the quarantine backfill
and de-authorize post-fix experiments."""
import subprocess
import sys
import textwrap
from pathlib import Path

import duckdb


def _run_isolated(tmp: Path, code: str):
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                       capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "QLT_ROOT": str(tmp)},
                       cwd=str(tmp))
    assert r.returncode == 0, f"isolated run failed:\n{r.stderr[-800:]}"
    return r.stdout


def test_authoritative_row_survives_process_restart(tmp_path, monkeypatch):
    import quant_loop_trader.data as dm
    monkeypatch.setattr(dm, "DB_PATH", tmp_path / "research.duckdb")
    monkeypatch.setattr("quant_loop_trader.experiment.DB_PATH", tmp_path / "research.duckdb")
    dm.migrate_db()

    # insert a POST-fix authoritative experiment (simulating healthy new research)
    con = duckdb.connect(str(dm.DB_PATH))
    con.execute("""INSERT INTO datasets VALUES ('ds','SPY','2019-01-01','2024-12-31','test','v1',
                   'chk',1,'valid','snap','{}', current_timestamp)""")
    cols = [r[0] for r in con.execute("DESCRIBE experiments").fetchall()]
    assert "authoritative" in cols
    values = {c: None for c in cols}
    values.update({"experiment_id": "exp_post_fix", "dataset_id": "ds", "ticker": "SPY",
                   "horizon_days": 5, "version": "v1", "hypothesis": "h", "economic_reasoning": "r",
                   "research_question": "q", "model_version": "m", "feature_version": "f",
                   "seed": 1, "config_json": "{}", "metrics_json": "{}", "decision": "KEEP",
                   "parent_experiment_id": None, "provenance_json": "{}",
                   "created_at": "2025-08-25 12:00:00", "authoritative": True})
    con.execute(f"INSERT INTO experiments VALUES ({','.join('?' * len(cols))})",
                [values[c] for c in cols])
    con.close()

    # NEW PROCESS (separate interpreter): migrate again — must be a no-op on data
    code = """
        import sys
        sys.path.insert(0, %r)
        import quant_loop_trader.data as dm
        dm.DB_PATH = %r          # SAME database file as process 1
        dm._MIGRATED.clear()     # fresh-process semantics
        dm.migrate_db()
        import duckdb
        con = duckdb.connect(str(dm.DB_PATH), read_only=True)
        n_auth = con.execute("SELECT count(*) FROM experiments WHERE authoritative").fetchone()[0]
        applied = sorted(r[0] for r in con.execute("SELECT name FROM _schema_migrations").fetchall())
        print(n_auth, "|", applied)
    """ % (str(Path("src").resolve()), str(Path(dm.DB_PATH)))
    (tmp_path / "proc2").mkdir()
    out = _run_isolated(tmp_path / "proc2", code)
    n_auth, applied = out.strip().split("|")
    assert int(n_auth) == 1, f"restart re-quarantined authoritative rows! migrations ran: {applied}"
    assert "005_quarantine_backfill.sql" in applied  # recorded exactly once


def test_legacy_db_with_backfill_adopts_without_replaying(tmp_path, monkeypatch):
    """Legacy pre-tracker DB that already has the authoritative column adopts the
    ledger WITHOUT replaying the backfill UPDATE."""
    import duckdb
    import quant_loop_trader.data as dm
    db = tmp_path / "legacy.duckdb"
    monkeypatch.setattr(dm, "DB_PATH", db)

    # build a legacy DB using the REAL pre-004 schema, minus any migration ledger
    dm.migrate_db()
    con = duckdb.connect(str(db))
    con.execute("DROP TABLE _schema_migrations")
    con.execute("INSERT INTO datasets VALUES ('ds','SPY','2019-01-01','2024-12-31','test','v1','chk',1,'valid','snap','{}',current_timestamp)")
    con.execute("DELETE FROM experiments")  # legacy state: one authoritative row
    con.execute("""INSERT INTO experiments (experiment_id, dataset_id, ticker, horizon_days,
        version, hypothesis, economic_reasoning, research_question, model_version,
        feature_version, seed, config_json, metrics_json, decision, provenance_json,
        created_at, authoritative)
        VALUES ('exp_legacy','ds','SPY',5,'v1','h','r','q','m','f',1,'{}','{}','KEEP','{}',
        current_timestamp, TRUE)""")
    con.close()

    dm._MIGRATED.clear()  # simulate a brand-new process opening the same DB
    dm.migrate_db()       # adopt legacy DB
    con = duckdb.connect(str(db), read_only=True)
    # legacy row was already backfilled manually → stays authoritative (no replay)
    n_auth = con.execute("SELECT count(*) FROM experiments WHERE authoritative").fetchone()[0]
    applied = {r[0] for r in con.execute("SELECT name FROM _schema_migrations").fetchall()}
    con.close()
    assert n_auth == 1
    assert "005_quarantine_backfill.sql" in applied  # recorded once, not re-executed
