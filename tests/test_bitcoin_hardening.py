from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import quant_loop_trader.data as dm
import quant_loop_trader.research_memory as rm
from quant_loop_trader.core import PIPELINE_VERSION
from quant_loop_trader.evaluation import evaluate
from quant_loop_trader.market import calendar_days, periods_per_year
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
    assert holdout_boundary("2018-01-01", "2024-12-31", ticker="BTCUSD") == "2024-01-01"
    assert holdout_boundary("2020-01-01", "2024-12-31", ticker="BTCUSD") == "2024-01-01"
    assert holdout_boundary("2022-01-01", "2023-06-30", ticker="BTCUSD") == "2024-01-01"


def test_early_crypto_window_cannot_consume_future_campaign_holdout(monkeypatch):
    monkeypatch.delenv("QLT_CRYPTO_HOLDOUT_START", raising=False)
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
