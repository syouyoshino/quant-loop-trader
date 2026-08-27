# Quant Loop Research Terminal

A read-only observability layer over the existing research system. It renders
what Quant Loop has actually recorded — nothing else. No metric on the screen is
generated, smoothed, or filled in: if the evidence is not on disk, the field
reads `N/A`.

## Launch

```bash
cd quant-loop-trader
PYTHONPATH=src .venv/bin/python -m quant_loop_trader.dashboard.api --port 8787
```

Then open <http://127.0.0.1:8787>. Add `-v` for request logging, `--host 0.0.0.0`
to expose it on the LAN (there is no authentication — keep it on localhost).

`PYTHONPATH=src` is needed because the package is installed non-editable (see
README); after `pip install .` the module resolves without it.

The terminal auto-refreshes in place — no page reloads:

| Stream | Endpoint | Interval |
|---|---|---|
| System / autonomy | `/api/system` | 2 s |
| Current cycle | `/api/cycles/current` | 2 s |
| Overview, funnel, rejections | `/api/overview` | 5 s |
| Experiment table | `/api/experiments` | 5 s |
| Activity feed | `/api/activity` | 5 s |
| Performance + equity curve | `/api/performance/{id}` | 12 s |
| Risk + rolling / edge decay | `/api/risk/{id}` | 12 s |
| Experiment detail | `/api/experiments/{id}` | 15 s |
| Champions | `/api/champions` | 15 s |
| Market | `/api/market` | 30 s |

## Layout

```
src/quant_loop_trader/dashboard/
    queries.py   every read of DuckDB / artifacts / logs (read-only by construction)
    service.py   equity, drawdown, rolling, cycles, funnel, rejections, assembly
    schemas.py   pipeline vocabulary, market calendars, JSON encoding
    api.py       stdlib HTTP routing + static host

dashboard/
    index.html   panel skeleton
    styles.css   terminal theme
    src/api      fetch client
    src/hooks    polling
    src/charts   ECharts wrappers (equity, drawdown, rolling)
    src/components  header, cycle, pipeline, system, funnel, metrics, risk,
                    validation, rejections, table, detail, champions, market, activity
    src/pages/terminal.js  selection state + poll cadences
    vendor/echarts.min.js  vendored so the terminal works offline
```

No build step, no node_modules: the frontend is native ES modules served
straight from disk.

Every panel collapses: click its header (the `▾` / `▸` marker) to fold it down
to a single line. Controls inside a header — variant, range, rolling window,
the table filters — are not collapse targets. The state is per browser, kept in
`localStorage` under `qlt.collapsed`, so a folded panel stays folded across
reloads. Collapsed panels keep polling; expanding one resizes its charts
immediately.

## Where every number comes from

| Panel | Source |
|---|---|
| Cycle / progress | `data/logs/session.log` session summaries + `data/logs/heartbeat.json` |
| Live pipeline | `report.json`, `validation.json`, `holdout_report.json`, `model_registry` |
| Equity / drawdown | `predictions_{variant}.parquet` recompounded with the engine's own bucket rules |
| Performance metrics | `metrics.json` (sealed), plus CAGR / annualised vol recomputed from the same series |
| Risk | equity curve reconstructed from the same predictions |
| Funnel / rejections | experiment artifacts + `validation.json` issue strings |
| Champions | `model_registry.status` lifecycle (`candidate → eligible → champion`) |
| Market regime | `data/processed/SPY.parquet` (research snapshot, not a live feed) |
| System | heartbeat, DuckDB, `deploy/*.plist` + `launchctl`, `git rev-parse` |

### Cycles

Quant Loop has no explicit "cycle" object. The dashboard reads one from what the
engine already writes: **a cycle is one `autonomy.run_session` session**, taken
from the session summaries appended to `data/logs/session.log`
(`session_started`, `session_finished`, `budget`, `executed`, `results[]`).
Nothing is persisted by the dashboard, and no experiment semantics change.

A cycle is `RUNNING` when an experiment directory exists without a sealed
`report.json` (the engine creates the directory first and seals at the end);
`STALLED` if that directory is older than 30 minutes; `IDLE` otherwise.
Experiments started outside a session (direct CLI runs) show `—` in the CYCLE
column rather than being invented into one.

### Return maths

The curve mirrors `evaluation.evaluate` exactly, so the chart and the sealed
metrics agree to floating-point tolerance (asserted in
`tests/test_dashboard.py::test_curve_reproduces_the_sealed_metrics`):

* non-overlapping `h`-day buckets, position taken from the bucket's first signal
* 5 bps charged per position change, plus entry into the first bucket
* wealth compounded multiplicatively; drawdown = `equity / running_max(equity) − 1`
* annualisation uses the market calendar — 252 d/yr for equities, 365 for crypto
  — divided by the horizon, never a blanket 252

Quant Loop's return convention changed over time (daily returns first, then
`h`-day buckets, then costs, then the first-bucket entry cost). Rather than
re-scoring old bundles under today's rule, the curve tries each convention and
keeps the one that reproduces that bundle's own sealed
`cumulative_return_strategy`; the chart footer names the convention it used. If
no convention reproduces the sealed value the footer says so in red and the
sealed metrics remain authoritative — the curve is never quietly reshaped to
agree. All 38 bundles in this repo reconcile today.

Sharpe is the one metric the dashboard recomputes differently from some older
bundles: those annualised `h`-day buckets with `sqrt(252)` instead of
`sqrt(252/h)`. The dashboard uses the bucket calendar; the sealed value is still
shown as recorded.

### Edge decay

Not a black-box score. The last 90 days of buckets are compared with everything
before them, and the terminal shows both Sharpes, both excess returns and the
rule it applied (`STABLE ≥ 0.70 × historical`, `WEAKENING ≥ 0.30`, else
`SEVERE_DECAY`). When the historical Sharpe is not positive there is no edge to
decay from, and the status reads `NOT AVAILABLE` with that reason.

## Guarantees

* Every DuckDB connection is opened `read_only=True`; `migrate_db` (which
  writes) is never imported.
* The dashboard imports neither `polars` nor `quant_loop_trader.data`; parquet is
  read through DuckDB.
* Tests hash the whole `data/` tree before and after hitting every endpoint and
  assert nothing changed.
* Anything the system has not produced renders as `N/A` / `NOT AVAILABLE` —
  including markets with no connector (funding, open interest, basis and
  liquidations are listed as not implemented rather than mocked).

## API

`GET /api/overview` · `/api/cycles` · `/api/cycles/current` · `/api/cycles/{n}` ·
`/api/experiments` · `/api/experiments/{id}` · `/api/hypotheses` ·
`/api/champions` · `/api/validation` · `/api/validation/{id}` ·
`/api/performance/{id}` · `/api/risk/{id}` · `/api/market` · `/api/system` ·
`/api/activity`

`/api/experiments` accepts `market`, `cycle`, `status`, `stage`, `hypothesis`,
`from`, `to`, `champion_only`, `limit`.

## Tests

```bash
.venv/bin/python -m pytest tests/test_dashboard.py -q
```
