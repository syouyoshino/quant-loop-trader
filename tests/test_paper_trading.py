import pytest

from quant_loop_trader.paper_trading import Order, PaperBroker, ExecutionSimulator, PortfolioTracker


def test_broker_disabled_by_default(monkeypatch):
    monkeypatch.delenv("QLT_PAPER_ENABLED", raising=False)
    with pytest.raises(RuntimeError, match="disabled"):
        PaperBroker(allow=True)  # even explicit allow needs env
    with pytest.raises(RuntimeError, match="disabled"):
        PaperBroker()  # and env alone is not enough (allow=False default)


def test_broker_enables_with_double_key(monkeypatch):
    monkeypatch.setenv("QLT_PAPER_ENABLED", "true")
    b = PaperBroker(allow=True)
    o = Order(timestamp="2024-01-05", ticker="SPY", side="buy", quantity=10)
    fill = b.submit(o, bar_close=470.0)
    assert fill["filled"] and fill["price"] > 470.0  # slippage applied on buys
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
    assert eq_mid == pytest.approx(10_000, abs=0.01)  # marked to market, no PnL yet
    t.execute(Order("d2", "SPY", "sell", 10), 110.0)
    assert t.equity({}) == pytest.approx(10_100)


def test_limit_orders_actually_bind():
    from quant_loop_trader.paper_trading import ExecutionSimulator, Order
    sim = ExecutionSimulator(slippage_bps=2)  # 2bps of 100 = 0.02
    # buy limit below slipped market: NO fill (audit H10 — was silently a market order)
    low = sim.fill(Order("t", "SPY", "buy", 10, order_type="limit", limit_price=99.0), 100.0)
    assert low["filled"] is False
    # buy limit at/above slipped market: fills, never above the limit
    ok = sim.fill(Order("t", "SPY", "buy", 10, order_type="limit", limit_price=100.5), 100.0)
    assert ok["filled"] and ok["price"] <= 100.5
    # sell limit above slipped market: no fill; at/below: fills, never under the limit
    hi = sim.fill(Order("t", "SPY", "sell", 10, order_type="limit", limit_price=101.0), 100.0)
    assert hi["filled"] is False
    ok2 = sim.fill(Order("t", "SPY", "sell", 10, order_type="limit", limit_price=99.9), 100.0)
    assert ok2["filled"] and ok2["price"] >= 99.9
    # unknown order type rejected
    with pytest.raises(ValueError, match="unsupported order_type"):
        sim.fill(Order("t", "SPY", "buy", 1, order_type="iceberg"), 100.0)


def test_health_failed_tasks_escalate(monkeypatch, tmp_path):
    """Audit regression: lexical max('healthy','degraded') hid failures."""
    import duckdb
    from quant_loop_trader.monitoring.health import check_health
    import quant_loop_trader.data as dm
    from quant_loop_trader.monitoring.heartbeat import write_heartbeat
    write_heartbeat(tmp_path)
    dm.migrate_db()
    con = duckdb.connect(str(dm.DB_PATH))
    for i in range(5):
        con.execute(
            "INSERT INTO tasks VALUES (?, 'experiment', '{}', 'failed', 5, 'w', 3, NULL, '{}', 'v1', current_timestamp, current_timestamp)",
            [f"task_{i}"])
    con.close()
    report = check_health(tmp_path, tmp_path / "experiments")
    assert report["status"] in ("degraded", "broken"), f"failed tasks hidden by ordering: {report['status']}"
