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

from quant_loop_trader.market import is_crypto

load_dotenv()

logger = logging.getLogger(__name__)

ROOT = Path(os.environ.get("QLT_ROOT", Path.cwd()))
PROC_DIR = ROOT / "data" / "processed"
PKG_MIGR_DIR = Path(__file__).resolve().parent / "migrations"
MIGR_DIR = PKG_MIGR_DIR
DB_PATH = ROOT / "data" / "research.duckdb"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "SPY.csv"

TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
TIINGO_CRYPTO_URL = "https://api.tiingo.com/tiingo/crypto/prices"


def _checksum_df(df: pl.DataFrame) -> str:
    """Return the full SHA-256 of the canonical CSV representation."""
    b = df.write_csv().encode()
    return hashlib.sha256(b).hexdigest()


def seal_dataset_snapshot(df: pl.DataFrame, path: Path, expected_checksum: str) -> None:
    """Atomically create a content-addressed snapshot and fail on path collisions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pl.read_parquet(str(path))
        actual = _checksum_df(existing)
        if actual != expected_checksum:
            raise RuntimeError(
                "dataset_snapshot_collision:"
                f"path={path.name}:expected={expected_checksum}:actual={actual}"
            )
        return

    tmp = path.with_name(f".{path.name}.tmp")
    try:
        df.write_parquet(str(tmp))
        actual = _checksum_df(pl.read_parquet(str(tmp)))
        if actual != expected_checksum:
            raise RuntimeError(
                "dataset_snapshot_write_mismatch:"
                f"path={path.name}:expected={expected_checksum}:actual={actual}"
            )
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


_MIGRATED: set[str] = set()


def migrate_db(db_path: Path | None = None) -> None:
    """Apply unapplied migrations, tracked persistently in the database."""
    import duckdb

    db_path = Path(db_path or DB_PATH).resolve()
    if str(db_path) in _MIGRATED:
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    tracker_existed = True
    try:
        con.execute("SELECT 1 FROM _schema_migrations LIMIT 1")
    except duckdb.CatalogException:
        tracker_existed = False
    con.execute(
        "CREATE TABLE IF NOT EXISTS _schema_migrations ("
        "name TEXT PRIMARY KEY, applied_at TIMESTAMP DEFAULT current_timestamp)"
    )
    applied = {r[0] for r in con.execute("SELECT name FROM _schema_migrations").fetchall()}

    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    legacy_adopted = not tracker_existed and "experiments" in tables
    legacy_backfilled = False
    if legacy_adopted and "experiments" in tables:
        cols = [r[1] for r in con.execute("PRAGMA table_info('experiments')").fetchall()]
        legacy_backfilled = "authoritative" in cols

    for sql_file in sorted(MIGR_DIR.glob("*.sql")):
        name = sql_file.name
        if name in applied:
            continue
        if legacy_adopted and name == "005_quarantine_backfill.sql" and legacy_backfilled:
            con.execute("INSERT INTO _schema_migrations VALUES (?, current_timestamp)", [name])
            continue
        con.execute(sql_file.read_text())
        con.execute("INSERT INTO _schema_migrations VALUES (?, current_timestamp)", [name])
    con.close()
    _MIGRATED.add(str(db_path))


def _request_json(url: str, params: dict, api_key: str) -> list[dict]:
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
            payload = r.json()
            if not isinstance(payload, list):
                raise ValueError("Tiingo response was not a list")
            return payload
        except Exception as e:
            last_exc = e
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"Tiingo failed after retries: {last_exc}")


def _tiingo_fetch(ticker: str, start: str, end: str, api_key: str) -> list[dict]:
    if is_crypto(ticker):
        params = {
            "tickers": ticker.lower().replace("-", ""),
            "startDate": start,
            "endDate": end,
            "resampleFreq": "1day",
        }
        return _request_json(TIINGO_CRYPTO_URL, params, api_key)
    url = TIINGO_URL.format(ticker=ticker)
    params = {"startDate": start, "endDate": end, "format": "json", "resampleFreq": "daily"}
    return _request_json(url, params, api_key)


def _empty_ohlcv(volume_dtype=pl.Float64) -> pl.DataFrame:
    return pl.DataFrame(schema={
        "event_time": pl.Date,
        "available_time": pl.Date,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": volume_dtype,
    })


def _parse_tiingo(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return _empty_ohlcv(pl.Int64)
    df = pl.DataFrame(rows)
    has_adj = "adjClose" in df.columns
    prefix = "adj" if has_adj else ""
    df = df.select(
        pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.fZ", strict=False).alias("event_time"),
        pl.col(f"{prefix}Open").cast(pl.Float64).alias("open"),
        pl.col(f"{prefix}High").cast(pl.Float64).alias("high"),
        pl.col(f"{prefix}Low").cast(pl.Float64).alias("low"),
        pl.col(f"{prefix}Close").cast(pl.Float64).alias("close"),
        pl.col("volume").cast(pl.Int64),
    )
    if has_adj:
        logger.info(json.dumps({"event": "tiingo_adjusted_prices_used"}))
    df = df.with_columns(
        pl.col("event_time").dt.date().alias("event_time"),
        pl.col("event_time").dt.date().alias("available_time"),
    )
    return df.sort("event_time")


def _parse_tiingo_crypto(rows: list[dict], ticker: str) -> pl.DataFrame:
    """Flatten the exact requested Tiingo crypto pair into PIT daily OHLCV.

    Tiingo's crypto endpoint returns pair metadata with nested ``priceData``. A
    response containing a different pair must never be accepted for the requested
    ticker, even when that other pair has complete date coverage.
    """
    if not rows:
        return _empty_ohlcv()
    wanted = ticker.lower().replace("-", "")
    matches = [
        r for r in rows
        if str(r.get("ticker", "")).strip().lower().replace("-", "") == wanted
    ]
    if not matches:
        raise ValueError(f"requested_crypto_pair_missing:{ticker}")
    if len(matches) != 1:
        raise ValueError(f"requested_crypto_pair_ambiguous:{ticker}:matches={len(matches)}")
    pair = matches[0]
    price_data = pair.get("priceData") or []
    if not isinstance(price_data, list) or not price_data:
        return _empty_ohlcv()
    df = pl.DataFrame(price_data).select(
        pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.fZ", strict=False).alias("event_time"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    )
    df = df.with_columns(
        pl.col("event_time").dt.date().alias("event_time"),
        pl.col("event_time").dt.date().alias("available_time"),
    )
    return df.unique(subset=["event_time"], keep="last").sort("event_time")


def _load_fixture() -> pl.DataFrame | None:
    if FIXTURE_PATH.exists():
        df = pl.read_csv(str(FIXTURE_PATH), try_parse_dates=True)
        if "event_time" in df.columns:
            df = df.with_columns(
                pl.col("event_time").cast(pl.Date),
                pl.col("available_time").cast(pl.Date),
            )
        return df.sort("event_time")
    return None


def _slice_requested_window(df: pl.DataFrame, start: str, end: str) -> pl.DataFrame:
    """Return only observations the caller requested before validation/sealing."""
    import datetime as _dt

    s = _dt.date.fromisoformat(start)
    e = _dt.date.fromisoformat(end)
    if e < s:
        raise ValueError(f"invalid_date_range:{start}>{end}")
    return df.filter(
        (pl.col("event_time") >= pl.lit(s))
        & (pl.col("event_time") <= pl.lit(e))
    ).sort("event_time")


def coverage_check(df: pl.DataFrame, ticker: str, start: str, end: str) -> None:
    """Fail closed when a crypto response does not cover every requested UTC day.

    Internal gap detection alone cannot catch a response that is contiguous but
    truncated at the beginning or end. Crypto has a 24/7 calendar, so requested
    daily coverage is exact and can be verified deterministically.
    """
    if not is_crypto(ticker):
        return

    import datetime as _dt

    s = _dt.date.fromisoformat(start)
    e = _dt.date.fromisoformat(end)
    if e < s:
        raise ValueError(f"invalid_date_range:{start}>{end}")
    if df.height == 0:
        raise ValueError(f"crypto_coverage_empty:{ticker}:{start}:{end}")

    min_d = df["event_time"].min()
    max_d = df["event_time"].max()
    if min_d != s:
        raise ValueError(
            f"crypto_coverage_start:{ticker}:requested={s}:actual={min_d}"
        )
    if max_d != e:
        raise ValueError(
            f"crypto_coverage_end:{ticker}:requested={e}:actual={max_d}"
        )

    expected = (e - s).days + 1
    unique_days = df.select(pl.col("event_time").n_unique()).item()
    if unique_days != expected or df.height != expected:
        raise ValueError(
            f"crypto_coverage_count:{ticker}:expected={expected}:"
            f"rows={df.height}:unique_days={unique_days}"
        )


def fetch_ohlcv(ticker: str = "SPY", start: str = "2018-01-01", end: str = "2024-12-31",
                use_cache: bool = True) -> tuple[pl.DataFrame, str]:
    """Fetch OHLCV with PIT columns. Returns (df, actual_source).

    A malformed or incomplete crypto cache is never accepted. When a Tiingo key is
    available, cache validation failures fall through to a fresh network fetch
    instead of aborting the research session on stale local state.
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    ticker = ticker.upper()
    parquet_path = PROC_DIR / f"{ticker}.parquet"

    if use_cache and parquet_path.exists():
        try:
            df = pl.read_parquet(str(parquet_path))
            if df.height > 0:
                import datetime as _dt

                s = _dt.date.fromisoformat(start)
                e = _dt.date.fromisoformat(end)
                min_d, max_d = df["event_time"].min(), df["event_time"].max()
                if min_d <= s + _dt.timedelta(days=7) and max_d >= e:
                    df = _slice_requested_window(df, start, end)
                    gap_check(df, ticker=ticker)
                    coverage_check(df, ticker=ticker, start=start, end=end)
                    logger.info(json.dumps({"event": "cache_hit_parquet", "ticker": ticker, "rows": df.height}))
                    return df, "cache"
        except Exception as exc:
            logger.warning(json.dumps({
                "event": "cache_invalid_refresh",
                "ticker": ticker,
                "error": str(exc)[:200],
            }))

    api_key = os.getenv("TIINGO_API_KEY", "").strip()
    if api_key:
        try:
            rows = _tiingo_fetch(ticker, start, end, api_key)
            df = _parse_tiingo_crypto(rows, ticker) if is_crypto(ticker) else _parse_tiingo(rows)
            df = _slice_requested_window(df, start, end)
            if df.height == 0:
                raise ValueError("Tiingo returned 0 rows in requested window")
            gap_check(df, ticker=ticker)
            coverage_check(df, ticker=ticker, start=start, end=end)
            save_parquet(df, parquet_path)
            source = "tiingo_crypto" if is_crypto(ticker) else "tiingo_eod"
            logger.info(json.dumps({"event": "tiingo_fetch_ok", "ticker": ticker, "rows": df.height, "source": source}))
            return df, source
        except Exception as e:
            logger.warning(json.dumps({"event": "tiingo_failed_fallback_fixture", "error": str(e)[:200]}))

    if api_key:
        raise RuntimeError("Tiingo failed and fixture fallback is forbidden while TIINGO_API_KEY is set")
    if ticker != "SPY":
        raise ValueError(f"fixture fallback only covers SPY; no data source available for {ticker}")
    fixture = _load_fixture()
    if fixture is not None:
        logger.info(json.dumps({"event": "fixture_used", "ticker": ticker, "rows": fixture.height}))
        fixture = _slice_requested_window(fixture, start, end)
        save_parquet(fixture, parquet_path)
        return fixture, "fixture"

    raise FileNotFoundError("No Tiingo key and no fixture at tests/fixtures/SPY.csv — cannot fetch data")


def gap_check(df: pl.DataFrame, ticker: str = "SPY") -> None:
    if df.height < 2:
        return
    diffs = df.select((pl.col("event_time").diff().dt.total_days()).alias("gap")).drop_nulls()
    max_gap = diffs["gap"].max()
    if max_gap is None:
        return
    if is_crypto(ticker) and max_gap > 1:
        raise ValueError(f"crypto_calendar_gap:{ticker}:max_gap_days={int(max_gap)}")
    if not is_crypto(ticker) and max_gap > 7:
        logger.warning(json.dumps({"event": "large_gap_detected", "ticker": ticker, "max_gap_days": int(max_gap)}))


def save_parquet(df: pl.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(path))
    cs = _checksum_df(df)
    logger.info(json.dumps({"event": "parquet_saved", "path": str(path), "rows": df.height, "checksum": cs}))
    return cs


def dataset_metadata(df: pl.DataFrame, ticker: str, source: str,
                     extra_provenance: dict | None = None) -> dict:
    cs = _checksum_df(df)
    start = str(df["event_time"].min())
    end = str(df["event_time"].max())
    prov = {"source": source, "ticker": ticker, "rows": df.height, **(extra_provenance or {})}
    return {
        "dataset_id": f"{ticker}_{start}_{end}_{cs[:32]}",
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
    """Register a content-addressed dataset once; later reuse cannot rewrite provenance."""
    import duckdb

    db_path = Path(db_path or DB_PATH)
    migrate_db(db_path)
    con = duckdb.connect(str(db_path))
    con.execute(
        "INSERT OR IGNORE INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)",
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
