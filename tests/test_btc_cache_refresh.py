from __future__ import annotations

from datetime import date

import polars as pl

import quant_loop_trader.data as dm


def _frame(days: list[str]) -> pl.DataFrame:
    return pl.DataFrame({
        "event_time": [date.fromisoformat(d) for d in days],
        "available_time": [date.fromisoformat(d) for d in days],
        "open": [100.0 + i for i in range(len(days))],
        "high": [101.0 + i for i in range(len(days))],
        "low": [99.0 + i for i in range(len(days))],
        "close": [100.5 + i for i in range(len(days))],
        "volume": [1000.0 + i for i in range(len(days))],
    }).with_columns(
        pl.col("event_time").cast(pl.Date),
        pl.col("available_time").cast(pl.Date),
    )


def test_invalid_crypto_cache_refreshes_from_tiingo(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "PROC_DIR", tmp_path)
    monkeypatch.setenv("TIINGO_API_KEY", "test-key")

    # Same requested endpoints but one missing interior UTC day: cache must fail
    # closed locally and then refresh rather than aborting before the network path.
    _frame(["2024-01-01", "2024-01-03"]).write_parquet(tmp_path / "BTCUSD.parquet")

    fresh_rows = [{
        "ticker": "btcusd",
        "priceData": [
            {"date": "2024-01-01T00:00:00.000Z", "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000},
            {"date": "2024-01-02T00:00:00.000Z", "open": 101, "high": 102, "low": 100, "close": 101.5, "volume": 1100},
            {"date": "2024-01-03T00:00:00.000Z", "open": 102, "high": 103, "low": 101, "close": 102.5, "volume": 1200},
        ],
    }]
    monkeypatch.setattr(dm, "_tiingo_fetch", lambda ticker, start, end, api_key: fresh_rows)

    df, source = dm.fetch_ohlcv("BTCUSD", "2024-01-01", "2024-01-03")
    assert source == "tiingo_crypto"
    assert df.height == 3
    assert str(df["event_time"].min()) == "2024-01-01"
    assert str(df["event_time"].max()) == "2024-01-03"
