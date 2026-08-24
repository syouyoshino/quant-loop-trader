import pytest

from quant_loop_trader.automation import queue
from quant_loop_trader.automation.controller import ResearchController


def test_queue_enqueue_claim_complete(isolated_research):
    tid = queue.enqueue("experiment", {"ticker": "SPY", "horizon": 5}, priority=1)
    task = queue.claim_next()
    assert task["task_id"] == tid and task["payload"]["horizon"] == 5
    queue.complete(tid, {"ok": True})
    assert queue.pending_count() == 0
    # priority ordering: lower number first
    queue.enqueue("report", {}, priority=9)
    queue.enqueue("validate", {}, priority=2)
    assert queue.claim_next()["task_type"] == "validate"


def test_controller_is_inert_by_default(isolated_research, monkeypatch):
    monkeypatch.delenv("QLT_AUTONOMOUS_ENABLED", raising=False)
    c = ResearchController(enabled=True)  # even explicit True needs the env var
    assert c.active is False
    queue.enqueue("experiment", {})
    out = c.run_cycle()
    assert out["skipped"] == "controller_disabled" and out["ran"] == 0
    # nothing was claimed — the task is untouched for later activation
    assert queue.pending_count() == 1


def test_controller_runs_only_when_double_enabled(isolated_research, monkeypatch):
    ran = []
    monkeypatch.setattr("quant_loop_trader.automation.controller.WORKERS",
                        {"report": lambda **kw: ran.append(kw) or {"ok": 1}})
    monkeypatch.setenv("QLT_AUTONOMOUS_ENABLED", "true")
    c = ResearchController(enabled=True)
    assert c.active is True
    queue.enqueue("report", {"x": 1})
    out = c.run_cycle()
    assert out["ran"] == 1 and ran
