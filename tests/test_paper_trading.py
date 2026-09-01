import pytest

from quant_loop_trader.experimental.paper_trading import (
    ExecutionSimulator,
    Order,
    PaperBroker,
    PortfolioTracker,
)


def test_broker_disabled_by_default(monkeypatch):
    monkeypatch.delenv("QLT_PAPER_ENABLED", raising=False)
    with pytest.raises(RuntimeError, match="disabled"):
        PaperBroker(allow=True)
    with pytest.raises(RuntimeError, match="disabled"):
        PaperBroker()


def test_broker_enables_with_double_key(monkeypatch):
    monkeypatch.setenv("QLT_PAPER_ENABLED", "true")
    b = PaperBroker(allow=True)
    o = Order(timestamp="2024-01-05", ticker="SPY", side="buy", quantity=10)
    fill = b.submit(o, bar_close=470.0)
    assert fill["filled"] and fill["price"] > 470.0
    assert b.tracker.cash < 100_000
    assert b.tracker.positions["SPY"].quantity == 10


def test_order_validation_and_risk_guards():
    sim = ExecutionSimulator()
    bad = Order("2024-01-05", "SPY", "hold", -5)
    with pytest.raises(ValueError, match="invalid order"):
        sim.fill(bad, 100.0)
    t = PortfolioTracker(1_000)
    with pytest.raises(ValueError, match="insufficient cash"):
        t.execute(Order("2024-01-05", "SPY", "buy", 100), 50.0)
    with pytest.raises(ValueError, match="insufficient position"):
        t.execute(Order("2024-01-05", "SPY", "sell", 10), 50.0)


def test_round_trip_equity_accounts_for_costs():
    t = PortfolioTracker(10_000)
    t.execute(Order("d1", "SPY", "buy", 10), 100.0)
    eq_mid = t.equity({"SPY": 100.0})
    assert eq_mid == pytest.approx(10_000, abs=0.01)
    t.execute(Order("d2", "SPY", "sell", 10), 110.0)
    assert t.equity({}) == pytest.approx(10_100)


def test_limit_orders_actually_bind():
    sim = ExecutionSimulator(slippage_bps=2)
    low = sim.fill(
        Order("t", "SPY", "buy", 10, order_type="limit", limit_price=99.0),
        100.0,
    )
    assert low["filled"] is False
    ok = sim.fill(
        Order("t", "SPY", "buy", 10, order_type="limit", limit_price=100.5),
        100.0,
    )
    assert ok["filled"] and ok["price"] <= 100.5
    hi = sim.fill(
        Order("t", "SPY", "sell", 10, order_type="limit", limit_price=101.0),
        100.0,
    )
    assert hi["filled"] is False
    ok2 = sim.fill(
        Order("t", "SPY", "sell", 10, order_type="limit", limit_price=99.9),
        100.0,
    )
    assert ok2["filled"] and ok2["price"] >= 99.9
    with pytest.raises(ValueError, match="unsupported order_type"):
        sim.fill(Order("t", "SPY", "buy", 1, order_type="iceberg"), 100.0)


def test_health_failed_tasks_escalate(monkeypatch, tmp_path):
    """Legacy failed task rows remain visible to health monitoring."""
    import duckdb
    import quant_loop_trader.data as dm
    from quant_loop_trader.monitoring.health import check_health
    from quant_loop_trader.monitoring.heartbeat import write_heartbeat

    write_heartbeat(tmp_path)
    dm.migrate_db()
    con = duckdb.connect(str(dm.DB_PATH))
    for i in range(5):
        con.execute(
            "INSERT INTO tasks VALUES (?, 'experiment', '{}', 'failed', 5, 'w', 3, NULL, '{}', 'v1', current_timestamp, current_timestamp)",
            [f"task_{i}"],
        )
    con.close()
    report = check_health(tmp_path, tmp_path / "experiments")
    assert report["status"] in ("degraded", "broken"), (
        f"failed tasks hidden by ordering: {report['status']}"
    )
