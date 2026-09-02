from __future__ import annotations

from quant_loop_trader.dashboard import api


def _payload():
    return {
        "ticker": "BTCUSD",
        "horizon": 5,
        "max_experiments": 3,
        "validate": True,
        "campaign_id": "btc_2026_v1",
        "holdout_start": "2026-01-01",
        "research_starts": ["2018-01-01", "2020-01-01", "2022-01-01"],
        "data_end": "2026-09-02",
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
    assert "quant_loop_trader.autonomy" in captured["cmd"]
    assert captured["env"]["QLT_AUTONOMOUS_ENABLED"] == "true"
    assert captured["env"]["QLT_CRYPTO_CAMPAIGN_ID"] == "btc_2026_v1"
    assert captured["env"]["QLT_CRYPTO_HOLDOUT_START"] == "2026-01-01"
    assert captured["env"]["QLT_CRYPTO_CAMPAIGN_ENDS"] == "2026-09-02"

    api._CONTROL_PROCESS = None
    api._CONTROL_META = {}
