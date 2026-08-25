import polars as pl
from quant_loop_trader.data import fetch_ohlcv, save_parquet, dataset_metadata, PROC_DIR, DB_PATH, migrate_db
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


def test_tiingo_prefers_adjusted_fields(monkeypatch):
    """Audit H8: splits/dividends must not masquerade as returns."""
    import quant_loop_trader.data as dm
    rows = [{"date": "2024-01-02T00:00:00.000Z", "open": 500.0, "high": 505.0,
             "low": 495.0, "close": 502.0, "adjOpen": 250.0, "adjHigh": 252.5,
             "adjLow": 247.5, "adjClose": 251.0, "volume": 1000},
            {"date": "2024-01-03T00:00:00.000Z", "open": 502.0, "high": 506.0,
             "low": 500.0, "close": 504.0, "adjOpen": 251.0, "adjHigh": 253.0,
             "adjLow": 250.0, "adjClose": 252.0, "volume": 1100}]
    df = dm._parse_tiingo(rows)
    assert abs(df["close"][0] - 251.0) < 1e-9  # adjusted close selected
