"""Alerts (Task 5): provider-agnostic notification interface.

No provider required. Set ALERT_WEBHOOK_URL to any JSON-accepting endpoint
(Slack, Discord, generic webhook, email relay) — alerts become no-ops otherwise,
and a failing alert NEVER crashes the caller.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

SEVERITIES = ("info", "warning", "critical")


def send_alert(event: str, severity: str = "info", details: dict | None = None) -> dict:
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}")
    payload = {
        "event": event,
        "severity": severity,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details or {},
    }
    url = os.getenv("ALERT_WEBHOOK_URL", "").strip()
    if not url:
        logger.info(json.dumps({"event": "alert_noop", **payload}))
        return {"delivered": False, "reason": "no ALERT_WEBHOOK_URL configured"}

    try:
        import requests
        r = requests.post(url, json=payload, timeout=10)
        delivered = r.status_code < 300
        return {"delivered": delivered, "status_code": r.status_code}
    except Exception as e:
        # alerting must never take the lab down
        logger.warning(json.dumps({"event": "alert_failed", "error": str(e)[:120]}))
        return {"delivered": False, "reason": str(e)[:120]}


def alert_if(condition: bool, event: str, severity: str = "warning", details: dict | None = None) -> dict | None:
    return send_alert(event, severity, details) if condition else None
