"""Data acquisition + storage. Tiingo if key exists, else fixture. Stdlib-first."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ROOT = Path(os.environ.get("QLT_ROOT", Path.cwd()))
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"
PROC_DIR = ROOT / "data" / "processed"
MIGR_DIR = ROOT / "data" / "migrations"
DB_PATH = ROOT / "data" / "research.duckdb"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "SPY.csv"

TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"


def _checksum_df(df: pl.DataFrame) -> str:
    # ponytail: hash csv bytes, stable across polars versions; upgrade to parquet bytes hash if needed
    b = df.write_csv().encode()
    return hashlib.sha256(b).hexdigest()[:16]


def migrate_db(db_path: Path | None = None) -> None:
    import duckdb

    db_path = Path(db_path or DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    for sql_file in sorted(MIGR_DIR.glob("*.sql")):
        con.execute(sql_file.read_text())
    con.close()


def _tiingo_fetch(ticker: str, start: str, end: str, api_key: str) -> list[dict]:
    url = TIINGO_URL.format(ticker=ticker)
    params = {"startDate": start, "endDate": end, "format": "json", "resampleFreq": "daily"}
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}
    last_exc = None
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                wait = min(60, 2**attempt)
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = int(retry_after)
                logger.warning(json.dumps({"event": "tiingo_rate_limited", "wait": wait, "attempt": attempt}))
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            if attempt < 2:
                time.sleep(2**attempt)
            else:
                raise
    raise last_exc  # type: ignore[misc]


def _parse_tiingo(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema={"event_time": pl.Date, "available_time": pl.Date, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64})
    df = pl.DataFrame(rows)
    # tiingo returns "date" like "2024-01-02T00:00:00.000Z"
    df = df.select(
        pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.fZ", strict=False).alias("event_time"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Int64),
    )
    df = df.with_columns(
        pl.col("event_time").dt.date().alias("event_time"),
        pl.col("event_time").dt.date().alias("available_time"),  # L1: available at close; future datasets may lag
    )
    return df.sort("event_time")


def _load_fixture() -> pl.DataFrame | None:
    if FIXTURE_PATH.exists():
        df = pl.read_csv(str(FIXTURE_PATH), try_parse_dates=True)
        # ensure date cols are Date
        if "event_time" in df.columns:
            df = df.with_columns(
                pl.col("event_time").cast(pl.Date),
                pl.col("available_time").cast(pl.Date),
            )
        return df.sort("event_time")
    return None


def fetch_ohlcv(ticker: str = "SPY", start: str = "2018-01-01", end: str = "2024-12-31", use_cache: bool = True) -> pl.DataFrame:
    """Fetch OHLCV with PIT columns. Uses cache parquet if fresh, else Tiingo, else fixture."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = PROC_DIR / f"{ticker}.parquet"
    cache_meta = CACHE_DIR / f"{ticker}_{start}_{end}.json"

    # 1. parquet cache hit
    if use_cache and parquet_path.exists():
        try:
            df = pl.read_parquet(str(parquet_path))
            # validate date range covers request
            if df.height > 0:
                min_d = str(df["event_time"].min())
                max_d = str(df["event_time"].max())
                if min_d <= start and max_d >= end:
                    logger.info(json.dumps({"event": "cache_hit_parquet", "ticker": ticker, "rows": df.height}))
                    return df
        except Exception:
            pass

    # 2. Tiingo if key present
    api_key = os.getenv("TIINGO_API_KEY", "").strip()
    if api_key:
        try:
            rows = _tiingo_fetch(ticker, start, end, api_key)
            df = _parse_tiingo(rows)
            if df.height == 0:
                raise ValueError("Tiingo returned 0 rows")
            # missing data detection: gaps >3 trading days
            gap_check(df)
            save_parquet(df, parquet_path)
            # also raw json cache
            try:
                cache_meta.write_text(json.dumps(rows[:2]))  # minimal
            except Exception:
                pass
            logger.info(json.dumps({"event": "tiingo_fetch_ok", "ticker": ticker, "rows": df.height}))
            return df
        except Exception as e:
            logger.warning(json.dumps({"event": "tiingo_failed_fallback_fixture", "error": str(e)[:200]}))

    # 3. fixture fallback
    fixture = _load_fixture()
    if fixture is not None:
        logger.info(json.dumps({"event": "fixture_used", "ticker": ticker, "rows": fixture.height}))
        # filter to requested range
        fixture = fixture.filter((pl.col("event_time") >= pl.lit(start).str.strptime(pl.Date, "%Y-%m-%d")) & (pl.col("event_time") <= pl.lit(end).str.strptime(pl.Date, "%Y-%m-%d")))
        return fixture

    raise FileNotFoundError("No Tiingo key and no fixture at tests/fixtures/SPY.csv — cannot fetch data")


def gap_check(df: pl.DataFrame) -> None:
    if df.height < 2:
        return
    # trading days: expect gaps <=4 days (weekend+ holiday). >7 is suspicious
    diffs = df.select((pl.col("event_time").diff().dt.total_days()).alias("gap")).drop_nulls()
    max_gap = diffs["gap"].max()
    if max_gap is not None and max_gap > 7:
        logger.warning(json.dumps({"event": "large_gap_detected", "max_gap_days": int(max_gap)}))


def save_parquet(df: pl.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(path))
    cs = _checksum_df(df)
    logger.info(json.dumps({"event": "parquet_saved", "path": str(path), "rows": df.height, "checksum": cs}))
    return cs


def dataset_metadata(df: pl.DataFrame, ticker: str, source: str) -> dict:
    cs = _checksum_df(df)
    start = str(df["event_time"].min())
    end = str(df["event_time"].max())
    return {
        "dataset_id": f"{ticker}_{start}_{end}_{cs[:8]}",
        "ticker": ticker,
        "start_date": start,
        "end_date": end,
        "source": source,
        "version": "v1",
        "checksum": cs,
        "row_count": df.height,
        "validation_status": "valid",
        "snapshot_definition": "available_time <= prediction_timestamp, event_time daily close",
        "provenance_json": json.dumps({"source": source, "ticker": ticker, "rows": df.height}),
    }


def upsert_dataset(meta: dict, db_path: Path | None = None) -> None:
    import duckdb

    db_path = Path(db_path or DB_PATH)
    migrate_db(db_path)
    con = duckdb.connect(str(db_path))
    con.execute(
        "INSERT OR REPLACE INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)",
        [
            meta["dataset_id"],
            meta["ticker"],
            meta["start_date"],
            meta["end_date"],
            meta["source"],
            meta["version"],
            meta["checksum"],
            meta["row_count"],
            meta["validation_status"],
            meta["snapshot_definition"],
            meta["provenance_json"],
        ],
    )
    con.close()
