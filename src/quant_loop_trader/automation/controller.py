"""Research controller (Phase 10): orchestrates queue → worker → validation gate → memory.

SAFETY: the controller is INERT unless BOTH
    ResearchController(enabled=True)   # explicit in code
and env QLT_AUTONOMOUS_ENABLED=true  # explicit in environment
are present. Building the object, importing it, or running tests never triggers research.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

WORKERS = {
    "experiment": None,   # wired at activation: experiment.run_experiment
    "validate": None,     # wired at activation: agents.validate_experiment
    "ablation": None,
    "report": None,
}


class ResearchController:
    enabled: bool = False  # class-level default OFF

    def __init__(self, enabled: bool = False, max_tasks_per_cycle: int = 5):
        self.max_tasks_per_cycle = max_tasks_per_cycle
        self._enabled_flag = bool(enabled)
        self._active = self._enabled_flag and os.getenv("QLT_AUTONOMOUS_ENABLED", "").lower() == "true"
        if not self._active:
            logger.info(json.dumps({"event": "controller_inert",
                                    "reason": "requires enabled=True AND QLT_AUTONOMOUS_ENABLED=true"}))

    @property
    def active(self) -> bool:
        return self._active

    def run_cycle(self, db_path=None) -> dict:
        """Process up to max_tasks_per_cycle queued tasks. No-op when inert."""
        from quant_loop_trader.automation import queue as q

        if not self._active:
            return {"ran": 0, "skipped": "controller_disabled"}

        ran = []
        for _ in range(self.max_tasks_per_cycle):
            task = q.claim_next(db_path)
            if task is None:
                break
            worker = WORKERS.get(task["task_type"])
            if worker is None:
                q.complete(task["task_id"], {"error": "no worker registered"}, ok=False, db_path=db_path)
                continue
            try:
                result = worker(**task["payload"])
                q.complete(task["task_id"],
                           {"ok": True, "result": json.dumps(result, default=str)[:4000]},
                           ok=True, db_path=db_path)
                ran.append({"task_id": task["task_id"], "result": result})
            except Exception as e:
                q.complete(task["task_id"], {"error": str(e)[:300]}, ok=False, db_path=db_path)
        return {"ran": len(ran), "tasks": ran}
