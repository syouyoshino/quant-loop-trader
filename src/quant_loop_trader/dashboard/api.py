"""Read-only HTTP API + static host for the Quant Loop terminal.

Stdlib only: the research environment already carries duckdb; a web framework
would be one more thing to install and secure for a localhost observability
tool. Every handler delegates to service/queries — no SQL lives here.

    python -m quant_loop_trader.dashboard.api --port 8787
"""
from __future__ import annotations

import argparse
import re
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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


def _int(params: dict, key: str, default: int) -> int:
    try:
        return int(params.get(key, [default])[0])
    except (TypeError, ValueError):
        return default


def filter_experiments(rows: list[dict], params: dict) -> list[dict]:
    """Server-side filters for the experiment table."""
    def one(key):
        v = params.get(key, [None])[0]
        return v or None

    market, status, stage = one("market"), one("status"), one("stage")
    cycle, hypothesis = one("cycle"), one("hypothesis")
    start, end = one("from"), one("to")
    champion_only = (one("champion_only") or "").lower() in ("1", "true", "yes")
    out = []
    for r in rows:
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
    """Map a GET path to a JSON-serialisable payload. Read-only by construction."""
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
            rows = svc.experiment_index()
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

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
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


def serve(host: str = "127.0.0.1", port: int = 8787, verbose: bool = False):
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.verbose = verbose
    return httpd


def main():
    p = argparse.ArgumentParser(description="Quant Loop research terminal (read-only)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    httpd = serve(args.host, args.port, args.verbose)
    print(f"QUANT LOOP terminal → http://{args.host}:{args.port}  (root: {q.root()})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
