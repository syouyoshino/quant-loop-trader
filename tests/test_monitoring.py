import datetime
import json

import pytest

from quant_loop_trader.monitoring.alerts import send_alert, alert_if
from quant_loop_trader.monitoring.heartbeat import write_heartbeat, read_heartbeat, is_stale
from quant_loop_trader.monitoring.health import check_health


def test_heartbeat_write_read_staleness(tmp_path):
    write_heartbeat(tmp_path, status="healthy", last_task="experiment_123", details={"executed": 4})
    hb = read_heartbeat(tmp_path)
    assert hb["status"] == "healthy" and hb["last_task"] == "experiment_123"
    assert not is_stale(tmp_path)
    # stale: heartbeat older than the threshold
    old = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=3)).isoformat()
    hb["timestamp"] = old
    (tmp_path / "heartbeat.json").write_text(json.dumps(hb))
    assert is_stale(tmp_path)
    assert is_stale(tmp_path / "nonexistent")  # never beat = stale


def test_alerts_noop_without_webhook(monkeypatch):
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    out = send_alert("session_failed", "critical", {"detail": 1})
    assert out["delivered"] is False
    with pytest.raises(ValueError):
        send_alert("x", severity="apocalyptic")


def test_alert_delivery_and_failure_isolation(monkeypatch):
    calls = []
    class R:
        status_code = 200
    def fake_post(url, json=None, timeout=None):
        calls.append((url, json))
        if "fail" in url:
            raise ConnectionError("boom")
        return R()
    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example/test")
    out = send_alert("grid_exhausted", "warning")
    assert out["delivered"] and calls[0][1]["event"] == "grid_exhausted"
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://fail.example/x")
    out = send_alert("worker_crash", "critical")   # must not raise
    assert out["delivered"] is False


def test_alert_if_conditional_helper():
    assert alert_if(False, "never") is None
    assert alert_if(True, "yes")["delivered"] is False  # noop without webhook


def test_health_check_aggregates(isolated_research, tmp_path):
    from quant_loop_trader.experiment import run_experiment
    write_heartbeat(tmp_path)
    run_experiment(ticker="SPY", horizon=5, start="2019-01-01", end="2024-12-31", seed=88)
    report = check_health(tmp_path, isolated_research)
    assert report["status"] in ("healthy", "degraded", "broken")
    assert report["checks"]["heartbeat"] == "ok"
    assert report["checks"]["database"] == "ok"
    assert report["last_experiment_artifact"]
    # stale heartbeat degrades but does not break
    report2 = check_health(tmp_path / "empty", isolated_research)
    assert report2["status"] == "degraded" and report2["checks"]["heartbeat"] == "stale"
