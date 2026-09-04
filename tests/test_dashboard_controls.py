from __future__ import annotations

import signal

import pytest

from quant_loop_trader.dashboard import api


def _payload():
    return {
        "ticker": "BTCUSD",
        "horizon": 5,
        "max_experiments": 100,
        "validate": True,
        "campaign_id": "btc_2026_v1",
        "holdout_start": "2026-01-01",
        "research_starts": ["2018-01-01", "2020-01-01", "2022-01-01"],
        "data_end": "2026-09-02",
    }


def _persisted_run(pid: int = 4321) -> dict:
    return {
        "pid": pid,
        "exit_code": None,
        "run": {
            **_payload(),
            "started_at": "2026-09-02T11:00:00+00:00",
            "log": "data/logs/dashboard-control.log",
        },
    }


def test_control_payload_preserves_campaign_boundary():
    cfg = api._normalise_control_payload(_payload())
    assert cfg["ticker"] == "BTCUSD"
    assert cfg["campaign_id"] == "btc_2026_v1"
    assert cfg["holdout_start"] == "2026-01-01"
    assert cfg["research_starts"][0] == "2018-01-01"
    assert cfg["data_end"] == "2026-09-02"


def test_control_payload_rejects_research_start_inside_holdout():
    payload = _payload()
    payload["research_starts"] = ["2026-02-01"]
    try:
        api._normalise_control_payload(payload)
    except ValueError as exc:
        assert "research_start_must_precede_holdout" in str(exc)
    else:
        raise AssertionError("invalid research start was accepted")


def test_control_launches_existing_autonomy_runner(tmp_path, monkeypatch):
    captured = {}

    class FakeProcess:
        pid = 4321
        returncode = None

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(api.q, "root", lambda: tmp_path)
    monkeypatch.setattr(api.subprocess, "Popen", fake_popen)
    api._CONTROL_PROCESS = None
    api._CONTROL_META = {}

    status = api._start_control_run(_payload())

    assert status["running"] is True
    assert status["recovered"] is False
    assert "quant_loop_trader.autonomy" in captured["cmd"]
    assert captured["env"]["QLT_AUTONOMOUS_ENABLED"] == "true"
    assert captured["env"]["QLT_CRYPTO_CAMPAIGN_ID"] == "btc_2026_v1"
    assert captured["env"]["QLT_CRYPTO_HOLDOUT_START"] == "2026-01-01"
    assert captured["env"]["QLT_CRYPTO_CAMPAIGN_ENDS"] == "2026-09-02"

    persisted = api._read_control_state()
    assert persisted["pid"] == 4321
    assert persisted["run"]["ticker"] == "BTCUSD"
    assert persisted["run"]["campaign_id"] == "btc_2026_v1"
    assert persisted["run"]["started_at"]

    api._CONTROL_PROCESS = None
    api._CONTROL_META = {}


def test_control_status_recovers_live_dashboard_run_after_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(api.q, "root", lambda: tmp_path)
    monkeypatch.setattr(api, "_pid_is_autonomy", lambda pid: int(pid) == 4321)
    api._CONTROL_PROCESS = None
    api._CONTROL_META = {}
    api._write_control_state(_persisted_run())

    status = api._control_status(True)

    assert status["running"] is True
    assert status["recovered"] is True
    assert status["pid"] == 4321
    assert status["run"]["ticker"] == "BTCUSD"
    assert status["run"]["campaign_id"] == "btc_2026_v1"


def test_control_recovered_run_blocks_duplicate_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(api.q, "root", lambda: tmp_path)
    monkeypatch.setattr(api, "_pid_is_autonomy", lambda pid: int(pid) == 4321)
    api._CONTROL_PROCESS = None
    api._CONTROL_META = {}
    api._write_control_state(_persisted_run())

    with pytest.raises(RuntimeError, match="research_run_already_active"):
        api._start_control_run(_payload())


def test_stop_control_can_terminate_recovered_process_group(tmp_path, monkeypatch):
    killed = []
    monkeypatch.setattr(api.q, "root", lambda: tmp_path)
    monkeypatch.setattr(api, "_pid_is_autonomy", lambda pid: int(pid) == 4321)
    monkeypatch.setattr(api.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    api._CONTROL_PROCESS = None
    api._CONTROL_META = {}
    api._write_control_state(_persisted_run())

    api._stop_control_run()

    assert killed == [(4321, signal.SIGTERM)]
    assert api._read_control_state()["stop_requested_at"]


def test_market_endpoint_defaults_to_btcusd(monkeypatch):
    monkeypatch.setattr(api.svc, "market", lambda ticker: {"ticker": ticker})

    assert api.route("/api/market", {}) == {"ticker": "BTCUSD"}
