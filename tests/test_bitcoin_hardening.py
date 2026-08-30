from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import quant_loop_trader.data as dm
import quant_loop_trader.research_memory as rm
from quant_loop_trader.core import PIPELINE_VERSION
from quant_loop_trader.evaluation import evaluate
from quant_loop_trader.market import calendar_days, campaign_id, periods_per_year
from quant_loop_trader.validation.holdout import (
    _holdout_window_ready,
    holdout_boundary,
)
from quant_loop_trader.validation.walkforward import _market_hint


def test_tiingo_crypto_routes_to_crypto_endpoint(monkeypatch):
    seen = {}

    class Response:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return []

    def fake_get(url, params, headers, timeout):
        seen.update(url=url, params=params, headers=headers, timeout=timeout)
        return Response()

    monkeypatch.setattr(dm.requests, "get", fake_get)
    dm._tiingo_fetch("BTCUSD", "2024-01-01", "2024-01-31", "secret")
    assert seen["url"] == dm.TIINGO_CRYPTO_URL
    assert seen["params"]["tickers"] == "btcusd"
    assert seen["params"]["resampleFreq"] == "1day"


def test_tiingo_crypto_parser_flattens_nested_price_data():
    rows = [{
        "ticker": "btcusd",
        "baseCurrency": "btc",
        "quoteCurrency": "usd",
        "priceData": [
            {
                "date": "2024-01-01T00:00:00.000Z",
                "open": 42000.0,
                "high": 43000.0,
                "low": 41000.0,
                "close": 42500.0,
                "volume": 123.456,
            },
            {
                "date": "2024-01-02T00:00:00.000Z",
                "open": 42500.0,
                "high": 44000.0,
                "low": 42000.0,
                "close": 43500.0,
                "volume": 150.25,
            },
        ],
    }]
    df = dm._parse_tiingo_crypto(rows, "BTCUSD")
    assert df.height == 2
    assert df.columns == [
        "event_time", "open", "high", "low", "close", "volume", "available_time"
    ]
    assert df["event_time"][0].isoformat() == "2024-01-01"
    assert df["available_time"].to_list() == df["event_time"].to_list()
    assert abs(df["volume"][0] - 123.456) < 1e-12


def _crypto_frame(start: date, days: int) -> pl.DataFrame:
    dates = [start + timedelta(days=i) for i in range(days)]
    return pl.DataFrame({
        "event_time": dates,
        "available_time": dates,
        "open": [100.0 + i for i in range(days)],
        "high": [101.0 + i for i in range(days)],
        "low": [99.0 + i for i in range(days)],
        "close": [100.5 + i for i in range(days)],
        "volume": [10.0 + i for i in range(days)],
    })


def test_crypto_gap_check_is_fail_closed():
    df = pl.DataFrame({
        "event_time": ["2024-01-01", "2024-01-03"],
        "available_time": ["2024-01-01", "2024-01-03"],
        "close": [42000.0, 43000.0],
    }).with_columns(
        pl.col("event_time").str.strptime(pl.Date, "%Y-%m-%d"),
        pl.col("available_time").str.strptime(pl.Date, "%Y-%m-%d"),
    )
    with pytest.raises(ValueError, match="crypto_calendar_gap"):
        dm.gap_check(df, ticker="BTCUSD")


def test_crypto_coverage_check_requires_exact_requested_window():
    full = _crypto_frame(date(2024, 1, 1), 5)
    dm.coverage_check(full, "BTCUSD", "2024-01-01", "2024-01-05")

    with pytest.raises(ValueError, match="crypto_coverage_start"):
        dm.coverage_check(full.slice(1), "BTCUSD", "2024-01-01", "2024-01-05")
    with pytest.raises(ValueError, match="crypto_coverage_end"):
        dm.coverage_check(full.slice(0, 4), "BTCUSD", "2024-01-01", "2024-01-05")


def test_fetch_crypto_rejects_contiguous_but_truncated_tiingo_response(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "PROC_DIR", tmp_path)
    monkeypatch.setenv("TIINGO_API_KEY", "secret")

    truncated = _crypto_frame(date(2024, 1, 2), 4)
    payload = [{
        "ticker": "btcusd",
        "priceData": [
            {
                "date": f"{d.isoformat()}T00:00:00.000Z",
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            }
            for d, o, h, l, c, v in zip(
                truncated["event_time"].to_list(),
                truncated["open"].to_list(),
                truncated["high"].to_list(),
                truncated["low"].to_list(),
                truncated["close"].to_list(),
                truncated["volume"].to_list(),
            )
        ],
    }]
    monkeypatch.setattr(dm, "_tiingo_fetch", lambda *args, **kwargs: payload)

    with pytest.raises(RuntimeError, match="Tiingo failed"):
        dm.fetch_ohlcv("BTCUSD", "2024-01-01", "2024-01-05", use_cache=False)
    assert not (tmp_path / "BTCUSD.parquet").exists()


def test_crypto_calendar_is_canonical_365():
    assert calendar_days("BTCUSD") == 365
    assert periods_per_year("BTCUSD", 5) == 73
    assert calendar_days("SPY") == 252


def test_walkforward_legacy_caller_detects_24_7_calendar():
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(45)]
    assert _market_hint(dates, None) == "BTCUSD"
    assert _market_hint(dates, "SPY") == "SPY"


def test_evaluation_uses_crypto_calendar_and_cost_stress():
    prices = np.array([100, 102, 104, 106, 108, 111, 113, 116, 119, 121, 124], dtype=float)
    y_true = np.ones(10, dtype=int)
    y_pred = np.ones(10, dtype=int)
    y_prob = np.full(10, 0.8)

    btc = evaluate(y_true, y_pred, y_prob, prices, horizon=1, ticker="BTCUSD")
    spy = evaluate(y_true, y_pred, y_prob, prices, horizon=1, ticker="SPY")

    assert btc["calendar_days"] == 365
    assert btc["periods_per_year"] == 365
    assert spy["periods_per_year"] == 252
    assert btc["sharpe_strategy"] > spy["sharpe_strategy"]
    stress = btc["cost_sensitivity_compounded"]
    assert stress["5"] >= stress["10"] >= stress["25"] >= stress["50"]


def test_crypto_holdout_boundary_is_fixed_across_experiment_windows(monkeypatch):
    monkeypatch.delenv("QLT_CRYPTO_HOLDOUT_START", raising=False)
    monkeypatch.delenv("QLT_CRYPTO_CAMPAIGN_ID", raising=False)
    assert holdout_boundary("2018-01-01", "2024-12-31", ticker="BTCUSD") == "2024-01-01"
    assert holdout_boundary("2020-01-01", "2024-12-31", ticker="BTCUSD") == "2024-01-01"
    assert holdout_boundary("2022-01-01", "2023-06-30", ticker="BTCUSD") == "2024-01-01"


def test_custom_crypto_holdout_requires_new_campaign_id(monkeypatch):
    monkeypatch.setenv("QLT_CRYPTO_HOLDOUT_START", "2026-01-01")
    monkeypatch.delenv("QLT_CRYPTO_CAMPAIGN_ID", raising=False)
    with pytest.raises(ValueError, match="custom_crypto_holdout_requires_new_campaign_id"):
        campaign_id("BTCUSD")

    monkeypatch.setenv("QLT_CRYPTO_CAMPAIGN_ID", "btc_2026_v1")
    assert campaign_id("BTCUSD") == "btc_2026_v1"


def test_campaign_change_changes_experiment_fingerprint(monkeypatch):
    import quant_loop_trader.autonomy as autonomy

    monkeypatch.delenv("QLT_CRYPTO_HOLDOUT_START", raising=False)
    monkeypatch.delenv("QLT_CRYPTO_CAMPAIGN_ID", raising=False)
    old = autonomy._spec_fingerprint("BTCUSD", 5, "2020-01-01", "2026-08-29", 42)

    monkeypatch.setenv("QLT_CRYPTO_HOLDOUT_START", "2026-01-01")
    monkeypatch.setenv("QLT_CRYPTO_CAMPAIGN_ID", "btc_2026_v1")
    new = autonomy._spec_fingerprint("BTCUSD", 5, "2020-01-01", "2026-08-29", 42)
    assert new != old


def test_custom_crypto_grid_can_target_new_campaign(monkeypatch):
    import quant_loop_trader.autonomy as autonomy

    monkeypatch.setenv("QLT_CRYPTO_HOLDOUT_START", "2026-01-01")
    monkeypatch.setenv("QLT_CRYPTO_CAMPAIGN_ID", "btc_2026_v1")
    monkeypatch.setenv("QLT_CRYPTO_CAMPAIGN_STARTS", "2022-01-01")
    monkeypatch.setenv("QLT_CRYPTO_CAMPAIGN_ENDS", "2026-08-29")
    grid = autonomy._candidate_grid("BTCUSD")
    assert len(grid) == 3
    assert {row["start"] for row in grid} == {"2022-01-01"}
    assert {row["end"] for row in grid} == {"2026-08-29"}
    assert {row["seed"] for row in grid} == {42, 123, 777}


def test_early_crypto_window_cannot_consume_future_campaign_holdout(monkeypatch):
    monkeypatch.delenv("QLT_CRYPTO_HOLDOUT_START", raising=False)
    monkeypatch.delenv("QLT_CRYPTO_CAMPAIGN_ID", raising=False)
    ready, reason = _holdout_window_ready({
        "ticker": "BTCUSD",
        "start": "2020-01-01",
        "end": "2023-06-30",
        "horizon": 5,
    })
    assert ready is False
    assert reason == "holdout_not_in_experiment_window"


def test_pipeline_version_has_one_authoritative_value():
    import quant_loop_trader.autonomy as autonomy
    import quant_loop_trader.experiment as experiment

    assert experiment._pipeline_version() == PIPELINE_VERSION
    assert autonomy._spec_fingerprint("BTCUSD", 5, "2020-01-01", "2024-12-31", 42)


def test_research_memory_scopes_same_hypothesis_by_market(tmp_path, monkeypatch):
    db = tmp_path / "research.duckdb"
    monkeypatch.setattr(dm, "DB_PATH", db)
    monkeypatch.setattr(rm._data, "DB_PATH", db)
    dm._MIGRATED.clear()

    def report(ticker, exp_id, decision):
        return {
            "experiment_id": exp_id,
            "decision": decision,
            "hypothesis": "Adding volatility regime classification should improve momentum prediction",
            "failure_condition": "no lift",
            "economic_reasoning": "test",
            "config": {"ticker": ticker, "horizon": 5},
            "error_analysis": {},
            "improved_metrics": {"accuracy": 0.5},
            "improvement_delta_accuracy": 0.0,
            "research_branch": "test",
            "creator_agent": "test",
        }

    rm.record_outcome(report("SPY", "spy_exp", "REJECT"))
    rm.record_outcome(report("BTCUSD", "btc_exp", "KEEP"))

    btc = rm.search_memory("volatility regime", ticker="BTCUSD", horizon=5)
    spy = rm.search_memory("volatility regime", ticker="SPY", horizon=5)
    assert {r["experiment_id"] for r in btc} == {"btc_exp"}
    assert {r["experiment_id"] for r in spy} == {"spy_exp"}


def test_command_center_no_longer_hardcodes_market_poll():
    js = (Path(__file__).resolve().parents[1] / "dashboard/src/pages/terminal.js").read_text()
    assert "api.market('SPY')" not in js
    assert "state.marketTicker" in js