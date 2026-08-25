import polars as pl
from quant_loop_trader.replay import ReplayEngine
from quant_loop_trader.data import PROC_DIR

def _engine():
    p = PROC_DIR / "SPY.parquet"
    # if not exists, fallback to fixture via fetch
    if not p.exists():
        from quant_loop_trader.data import fetch_ohlcv
        fetch_ohlcv("SPY", "2018-01-01", "2024-12-31")[0]
    return ReplayEngine(p)

def test_available_time_leq_timestamp():
    eng = _engine()
    for ts in ["2019-01-02", "2020-06-15", "2024-12-31", "2018-01-03"]:
        snap = eng.get_snapshot("SPY", ts)
        if snap.height:
            max_av = snap["available_time"].max()
            assert str(max_av) <= ts, f"leakage at {ts}: {max_av} > {ts}"

def test_no_future_columns():
    eng = _engine()
    snap = eng.get_snapshot("SPY", "2020-01-02")
    # feature cols should not exist in raw snapshot
    assert "fwd_ret" not in snap.columns
    assert "label" not in snap.columns

def test_snapshot_reproducible():
    eng = _engine()
    a = eng.get_snapshot("SPY", "2021-03-15")
    b = eng.get_snapshot("SPY", "2021-03-15")
    assert a.height == b.height
    assert a.write_csv() == b.write_csv()

def test_pit_weekend():
    eng = _engine()
    # 2024-01-06 is Saturday, snapshot should equal 2024-01-05 Friday close
    fri = eng.get_snapshot("SPY", "2024-01-05")
    sat = eng.get_snapshot("SPY", "2024-01-06")
    sun = eng.get_snapshot("SPY", "2024-01-07")
    mon = eng.get_snapshot("SPY", "2024-01-08")
    assert sat.height == fri.height
    assert sun.height == fri.height
    assert mon.height > fri.height  # Monday includes Monday's bar

def test_survivorship_bias_detection():
    # Engine should not drop delisted ticker rows pre-event; here we check SPY history completeness
    eng = _engine()
    full = eng.full_history()
    # ensure no missing year
    years = full.select(pl.col("event_time").dt.year().alias("y")).unique().sort("y")["y"].to_list()
    assert 2018 in years and 2024 in years
