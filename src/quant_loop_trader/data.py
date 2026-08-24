"""Data acquisition + storage. Tiingo if key exists, else fixture. Stdlib-first."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ROOT = Path(os.environ.get("QLT_ROOT", Path.cwd()))
PROC_DIR = ROOT / "data" / "processed"
MIGR_DIR = ROOT / "data" / "migrations"
DB_PATH = ROOT / "data" / "research.duckdb"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "SPY.csv"

TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"


def _checksum_df(df: pl.DataFrame) -> str:
    # ponytail: hash csv bytes, stable across polars versions; upgrade to parquet bytes hash if needed
    b = df.write_csv().encode()
    return hashlib.sha256(b).hexdigest()[:16]


_MIGRATED: set[str] = set()


def migrate_db(db_path: Path | None = None) -> None:
    """Apply migrations once per (process, resolved db path). Idempotent SQL either way."""
    import duckdb

    db_path = Path(db_path or DB_PATH).resolve()
    if str(db_path) in _MIGRATED:
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    for sql_file in sorted(MIGR_DIR.glob("*.sql")):
        con.execute(sql_file.read_text())
    con.close()
    _MIGRATED.add(str(db_path))


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
    raise RuntimeError(f"Tiingo failed after retries: {last_exc}")


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


def fetch_ohlcv(ticker: str = "SPY", start: str = "2018-01-01", end: str = "2024-12-31", use_cache: bool = True) -> tuple[pl.DataFrame, str]:
    """Fetch OHLCV with PIT columns. Returns (df, actual_source)."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = PROC_DIR / f"{ticker}.parquet"

    # 1. parquet cache hit
    if use_cache and parquet_path.exists():
        try:
            df = pl.read_parquet(str(parquet_path))
            if df.height > 0:
                # parsed-date coverage check with a small grace window: a request
                # starting on a holiday/weekend is covered by the next trading day
                import datetime as _dt
                s = _dt.date.fromisoformat(start)
                e = _dt.date.fromisoformat(end)
                min_d, max_d = df["event_time"].min(), df["event_time"].max()
                if min_d <= s + _dt.timedelta(days=7) and max_d >= e:
                    df = df.filter(
                        (pl.col("event_time") >= pl.lit(s))
                        & (pl.col("event_time") <= pl.lit(e))
                    )
                    logger.info(json.dumps({"event": "cache_hit_parquet", "ticker": ticker, "rows": df.height}))
                    return df, "cache"
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
            logger.info(json.dumps({"event": "tiingo_fetch_ok", "ticker": ticker, "rows": df.height}))
            return df, "tiingo"
        except Exception as e:
            logger.warning(json.dumps({"event": "tiingo_failed_fallback_fixture", "error": str(e)[:200]}))

    # 3. fixture fallback — forbidden while a key is configured (never poison research silently)
    if api_key:
        raise RuntimeError("Tiingo failed and fixture fallback is forbidden while TIINGO_API_KEY is set")
    fixture = _load_fixture()
    if fixture is not None:
        logger.info(json.dumps({"event": "fixture_used", "ticker": ticker, "rows": fixture.height}))
        fixture = fixture.filter((pl.col("event_time") >= pl.lit(start).str.strptime(pl.Date, "%Y-%m-%d")) & (pl.col("event_time") <= pl.lit(end).str.strptime(pl.Date, "%Y-%m-%d")))
        save_parquet(fixture, parquet_path)  # single persistence site — downstream always finds the parquet
        return fixture, "fixture"

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


def dataset_metadata(df: pl.DataFrame, ticker: str, source: str, extra_provenance: dict | None = None) -> dict:
    cs = _checksum_df(df)
    start = str(df["event_time"].min())
    end = str(df["event_time"].max())
    prov = {"source": source, "ticker": ticker, "rows": df.height, **(extra_provenance or {})}
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
        "provenance_json": json.dumps(prov),
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
