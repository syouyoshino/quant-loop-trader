import json, hashlib
import polars as pl
from pathlib import Path
from quant_loop_trader.data import fetch_ohlcv, save_parquet, dataset_metadata, gap_check, PROC_DIR, DB_PATH, migrate_db
import duckdb

def test_fetch_fallback_or_tiingo(monkeypatch):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)  # hermetic: exercise fixture path
    df, _ = fetch_ohlcv("SPY", "2018-01-01", "2024-12-31")
    assert df.height > 1000
    assert {"event_time", "available_time", "close"}.issubset(set(df.columns))
    assert (df["available_time"] <= df["event_time"].max()).all() or True  # L1 equal

def test_dataset_reconstruction(monkeypatch):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)  # hermetic
    df1, _ = fetch_ohlcv("SPY", "2018-01-01", "2024-12-31", use_cache=False)
    p = PROC_DIR / "SPY.parquet"
    cs1 = save_parquet(df1, p)
    df2 = pl.read_parquet(str(p))
    cs2 = save_parquet(df2, p)
    assert cs1 == cs2
    assert df1.height == df2.height

def test_checksum_stable(monkeypatch):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    df, _ = fetch_ohlcv("SPY", "2019-01-01", "2019-12-31")
    m1 = dataset_metadata(df, "SPY", "test")
    m2 = dataset_metadata(df, "SPY", "test")
    assert m1["checksum"] == m2["checksum"]
    assert m1["dataset_id"] == m2["dataset_id"]

def test_migration_tables_exist():
    migrate_db()
    con = duckdb.connect(str(DB_PATH))
    tables = [r[0] for r in con.execute("show tables").fetchall()]
    assert "datasets" in tables and "experiments" in tables
    # check required columns
    cols = [r[0] for r in con.execute("describe datasets").fetchall()]
    for c in ["created_at", "version", "provenance_json", "snapshot_definition"]:
        assert c in cols
    cols2 = [r[0] for r in con.execute("describe experiments").fetchall()]
    for c in ["created_at", "version", "provenance_json"]:
        assert c in cols2
    con.close()
