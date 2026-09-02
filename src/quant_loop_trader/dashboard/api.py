"""HTTP API + static host for the Quant Loop research terminal.

GET endpoints remain read-only. Optional localhost-only control endpoints can be
enabled explicitly with ``--enable-controls`` to launch/stop the existing
autonomous research runner from the dashboard.

    python -m quant_loop_trader.dashboard.api --port 8787 --enable-controls
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import traceback
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from quant_loop_trader.dashboard import queries as q
from quant_loop_trader.dashboard import service as svc
from quant_loop_trader.dashboard.schemas import dumps

FRONTEND = q.root() / "dashboard"

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}

_ID = re.compile(r"^[A-Za-z0-9_.\-]+$")
_TICKER = re.compile(r"^[A-Za-z0-9.\-]+$")
_CONTROL_LOCK = threading.RLock()
_CONTROL_PROCESS: subprocess.Popen | None = None
_CONTROL_META: dict = {}


def _int(params: dict, key: str, default: int) -> int:
    try:
        return int(params.get(key, [default])[0])
    except (TypeError, ValueError):
        return default


def _iso_date(value: object, field: str) -> str:
    text = str(value or "").strip()
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid_{field}:{text}") from exc
    return text


def _control_status(enabled: bool) -> dict:
    global _CONTROL_PROCESS
    with _CONTROL_LOCK:
        proc = _CONTROL_PROCESS
        running = bool(proc and proc.poll() is None)
        exit_code = None if not proc or running else proc.returncode
        meta = dict(_CONTROL_META)
    return {
        "enabled": bool(enabled),
        "localhost_only": True,
        "running": running,
        "pid": proc.pid if proc else None,
        "exit_code": exit_code,
        "run": meta or None,
    }


def _normalise_control_payload(payload: dict) -> dict:
    ticker = str(payload.get("ticker", "BTCUSD")).strip().upper()
    if not _TICKER.match(ticker):
        raise ValueError("invalid_ticker")

    try:
        horizon = int(payload.get("horizon", 5))
        max_experiments = int(payload.get("max_experiments", 3))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_numeric_control") from exc
    if not 1 <= horizon <= 60:
        raise ValueError("horizon_out_of_range")
    if not 1 <= max_experiments <= 100:
        raise ValueError("max_experiments_out_of_range")

    validate = bool(payload.get("validate", True))
    campaign_id = str(payload.get("campaign_id", "btc_2026_v1")).strip()
    if not _ID.match(campaign_id):
        raise ValueError("invalid_campaign_id")

    holdout_start = _iso_date(payload.get("holdout_start", "2026-01-01"), "holdout_start")
    data_end = _iso_date(
        payload.get("data_end", datetime.now(timezone.utc).date().isoformat()),
        "data_end",
    )
    if data_end <= holdout_start:
        raise ValueError("data_end_must_be_after_holdout_start")

    raw_starts = payload.get("research_starts", ["2018-01-01", "2020-01-01", "2022-01-01"])
    if isinstance(raw_starts, str):
        raw_starts = [v.strip() for v in raw_starts.split(",") if v.strip()]
    if not isinstance(raw_starts, list) or not raw_starts:
        raise ValueError("research_starts_required")
    research_starts = [_iso_date(v, "research_start") for v in raw_starts]
    if any(v >= holdout_start for v in research_starts):
        raise ValueError("research_start_must_precede_holdout")

    return {
        "ticker": ticker,
        "horizon": horizon,
        "max_experiments": max_experiments,
        "validate": validate,
        "campaign_id": campaign_id,
        "holdout_start": holdout_start,
        "research_starts": research_starts,
        "data_end": data_end,
    }


def _start_control_run(payload: dict) -> dict:
    global _CONTROL_PROCESS, _CONTROL_META
    cfg = _normalise_control_payload(payload)

    with _CONTROL_LOCK:
        if _CONTROL_PROCESS and _CONTROL_PROCESS.poll() is None:
            raise RuntimeError("research_run_already_active")

        env = os.environ.copy()
        env["QLT_AUTONOMOUS_ENABLED"] = "true"
        env["QLT_CRYPTO_CAMPAIGN_ID"] = cfg["campaign_id"]
        env["QLT_CRYPTO_HOLDOUT_START"] = cfg["holdout_start"]
        env["QLT_CRYPTO_CAMPAIGN_STARTS"] = ",".join(cfg["research_starts"])
        env["QLT_CRYPTO_CAMPAIGN_ENDS"] = cfg["data_end"]

        cmd = [
            sys.executable,
            "-m",
            "quant_loop_trader.autonomy",
            "--ticker",
            cfg["ticker"],
            "--horizon",
            str(cfg["horizon"]),
            "--max-experiments",
            str(cfg["max_experiments"]),
        ]
        if not cfg["validate"]:
            cmd.append("--no-validate")

        log_dir = q.root() / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "dashboard-control.log"
        with log_path.open("ab") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=q.root(),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        _CONTROL_PROCESS = proc
        _CONTROL_META = {
            **cfg,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "log": str(log_path.relative_to(q.root())),
        }

    return _control_status(True)


def _stop_control_run() -> dict:
    global _CONTROL_PROCESS
    with _CONTROL_LOCK:
        proc = _CONTROL_PROCESS
        if not proc or proc.poll() is not None:
            return _control_status(True)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    return _control_status(True)


def filter_experiments(rows: list[dict], params: dict) -> list[dict]:
    """Server-side filters for the experiment table."""
    def one(key):
        v = params.get(key, [None])[0]
        return v or None

    market, status, stage = one("market"), one("status"), one("stage")
    cycle, hypothesis = one("cycle"), one("hypothesis")
    start, end = one("from"), one("to")
    champion_only = (one("champion_only") or "").lower() in ("1", "true", "yes")
    quarantined = (one("include_quarantined") or "").lower() in ("1", "true", "yes")
    out = []
    for r in rows:
        if not quarantined and r["authoritative"] is False:
            continue
        if market and r["market"] != market:
            continue
        if status and r["status"] != status:
            continue
        if stage and r["stage"] != stage:
            continue
        if cycle and str(r["cycle"]) != str(cycle):
            continue
        if hypothesis and hypothesis.lower() not in (r["hypothesis"] or "").lower():
            continue
        if start and (r["started"] or "") < start:
            continue
        if end and (r["started"] or "") > end:
            continue
        if champion_only and r["registry_status"] != "champion":
            continue
        out.append(r)
    return out


def route(path: str, params: dict):
    """Map a GET path to a JSON-serialisable payload. GET is read-only."""
    parts = [unquote(p) for p in path.strip("/").split("/") if p]
    if not parts or parts[0] != "api":
        raise KeyError(path)
    tail = parts[1:]

    def ident(value: str) -> str:
        if not _ID.match(value) or ".." in value:
            raise KeyError(value)
        return value

    match tail:
        case ["overview"]:
            return svc.overview()
        case ["cycles"]:
            return {"cycles": svc.cycles(), "current": svc.current_cycle()}
        case ["cycles", "current"]:
            return svc.current_cycle()
        case ["cycles", number]:
            found = [c for c in svc.cycles() if str(c["cycle_number"]) == number]
            if not found:
                raise KeyError(number)
            return found[0]
        case ["experiments"]:
            rows = filter_experiments(svc.experiment_index(), params)
            limit = _int(params, "limit", 500)
            return {
                "experiments": rows[:limit],
                "total": len(rows),
                "population": svc.population(),
                "filters": {
                    "markets": sorted({r["market"] for r in svc.experiment_index() if r["market"]}),
                    "statuses": sorted({r["status"] for r in svc.experiment_index() if r["status"]}),
                    "stages": sorted({r["stage"] for r in svc.experiment_index() if r["stage"]}),
                    "cycles": sorted({r["cycle"] for r in svc.experiment_index()
                                      if r["cycle"] is not None}),
                },
            }
        case ["experiments", eid]:
            return svc.experiment_detail(ident(eid))
        case ["hypotheses"]:
            return {"hypotheses": svc.hypotheses()}
        case ["champions"]:
            return svc.champions()
        case ["validation"]:
            rows = svc.authoritative()
            return {
                "experiments": [
                    {"id": r["id"], "cycle": r["cycle"], "status": r["status"],
                     "validation": r["validation"], "holdout": r["holdout"],
                     "p_value": r["p_value"], "dsr": r["dsr"],
                     "dsr_verdict": r["dsr_verdict"], "fdr": r["fdr"]}
                    for r in rows
                ],
                "rejections": svc.rejections(),
                "funnel": svc.funnel(),
            }
        case ["validation", eid]:
            return svc.validation_view(ident(eid))
        case ["performance", eid]:
            eid = ident(eid)
            variant = (params.get("variant", ["improved"])[0])
            variant = variant if variant in ("improved", "baseline") else "improved"
            return {
                "metrics": svc.performance(eid, variant),
                "baseline_metrics": svc.performance(eid, "baseline"),
                "curve": svc.curve(eid, variant),
            }
        case ["risk", eid]:
            eid = ident(eid)
            return {"risk": svc.risk(eid), "rolling": svc.rolling_performance(eid)}
        case ["market"]:
            ticker = params.get("ticker", ["SPY"])[0]
            return svc.market(ident(ticker))
        case ["system"]:
            return svc.system()
        case ["activity"]:
            return {"events": svc.activity(_int(params, "limit", 120))}
        case _:
            raise KeyError(path)


class Handler(BaseHTTPRequestHandler):
    server_version = "QuantLoopDashboard"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet by default
        if self.server.verbose:
            super().log_message(fmt, *args)

    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _local_control_allowed(self) -> bool:
        return bool(self.server.enable_controls and self.client_address[0] in ("127.0.0.1", "::1"))

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if parsed.path == "/api/control":
            self._send(200, dumps(_control_status(self._local_control_allowed())), "application/json")
            return
        if parsed.path.startswith("/api/"):
            try:
                self._send(200, dumps(route(parsed.path, params)), "application/json")
            except KeyError as exc:
                self._send(404, dumps({"error": "not_found", "detail": str(exc)}),
                           "application/json")
            except q.DataUnavailable as exc:
                self._send(503, dumps({"error": "data_unavailable", "detail": str(exc)}),
                           "application/json")
            except Exception as exc:
                if self.server.verbose:
                    traceback.print_exc()
                self._send(500, dumps({"error": "internal", "detail": str(exc)[:300]}),
                           "application/json")
            return
        self._static(parsed.path)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in ("/api/control/run", "/api/control/stop"):
            self._send(404, dumps({"error": "not_found"}), "application/json")
            return
        if not self._local_control_allowed():
            self._send(
                403,
                dumps({"error": "control_disabled", "detail": "launch with --enable-controls on localhost"}),
                "application/json",
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 16384:
                raise ValueError("request_too_large")
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("json_object_required")
            result = _start_control_run(payload) if parsed.path.endswith("/run") else _stop_control_run()
            self._send(202, dumps(result), "application/json")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, dumps({"error": "invalid_request", "detail": str(exc)}), "application/json")
        except RuntimeError as exc:
            self._send(409, dumps({"error": "conflict", "detail": str(exc)}), "application/json")
        except Exception as exc:
            if self.server.verbose:
                traceback.print_exc()
            self._send(500, dumps({"error": "internal", "detail": str(exc)[:300]}), "application/json")

    def _static(self, path: str):
        rel = path.lstrip("/") or "index.html"
        target = (FRONTEND / rel).resolve()
        try:
            target.relative_to(FRONTEND.resolve())
        except ValueError:
            self._send(403, b"forbidden", "text/plain")
            return
        if target.is_dir():
            target = target / "index.html"
        if not target.exists():
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, target.read_bytes(),
                   MIME.get(target.suffix, "application/octet-stream"))


def serve(
    host: str = "127.0.0.1",
    port: int = 8787,
    verbose: bool = False,
    enable_controls: bool = False,
):
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.verbose = verbose
    httpd.enable_controls = enable_controls
    return httpd


def main():
    p = argparse.ArgumentParser(description="Quant Loop research terminal")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--enable-controls",
        action="store_true",
        help="allow localhost POST controls that launch/stop autonomous research",
    )
    args = p.parse_args()
    httpd = serve(args.host, args.port, args.verbose, args.enable_controls)
    mode = "CONTROL" if args.enable_controls else "READ-ONLY"
    print(f"QUANT LOOP terminal → http://{args.host}:{args.port}  ({mode}; root: {q.root()})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
