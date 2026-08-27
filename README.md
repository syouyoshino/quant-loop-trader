# Quant Loop Trader

Historical-replay stock prediction engine. The intended flow is:

1. Fetch and cache market and macro data in `data/raw/` and `data/cache/`.
2. Produce replay-ready feature data in `data/processed/`.
3. Run predictions strictly one historical timestamp at a time, using only data available at that timestamp.
4. Record results for evaluation before any paper-trading integration.

## Layout

```text
src/quant_loop_trader/  Application code
data/                   Local data root (market data is ignored by Git)
tests/                  Automated checks
.env                    Your local API credentials (ignored by Git)
.env.example            Safe credential template
```

## Local setup

Create a virtual environment with Python 3.12 and install:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Copy the safe template if needed, then paste your own values into `.env`:

```bash
cp .env.example .env
```

Keep `ALPACA_PAPER=true` until the strategy has been tested through historical replay and paper trading.

## Minimum Viable Research Scientist (Level 1)

Single-command end-to-end loop — fetch → PIT snapshot → train baseline → predict → evaluate vs hidden future → autopsy → improved hypothesis → compare → store → reproduce:

```bash
# uses TIINGO_API_KEY if present, else fixture at tests/fixtures/SPY.csv (offline)
.venv/bin/python -m quant_loop_trader.experiment --ticker SPY --horizon 5 --start 2018-01-01 --end 2024-12-31

# outputs:
#   data/processed/SPY.parquet        — cached OHLCV with event_time/available_time
#   data/research.duckdb              — datasets + experiments tables (migrated)
#   data/experiments/{id}/report.json — full Experiment Framework record (20 fields)
#   data/experiments/{id}/predictions_*.parquet

# reproduce a prior experiment
.venv/bin/python -m quant_loop_trader.experiment --reproduce 20260822_SPY_5d_cca02216

# run the independent validation gate (statistical + adversarial + replication)
.venv/bin/python -m quant_loop_trader.experiment --validate <experiment_id>

# autonomous research session (observation mode, budgeted)
.venv/bin/python -m quant_loop_trader.autonomy --max-experiments 2

# run tests (no API required, uses fixture)
.venv/bin/python -m pytest -v
```

**Design:** SPY single market, 5-day horizon (param-driven for future 1/3/10/20), LogisticRegression + StandardScaler, 5 lagged features (ret_1, ret_5, ma10_gap, vol10, rsi14) + 2 vol-regime interactions for the improvement. `ReplayEngine.get_snapshot(ticker, timestamp)` enforces `available_time <= prediction_timestamp` — only `evaluation.py` sees future outcomes. Baseline vs improved compared on identical hidden test (time-split 70/30) with accuracy/precision/recall/Brier + Sharpe/vol/DD/turnover/cost; decision KEEP/IMPROVE/REJECT stored with lineage.

## Levels

- **L1 — MVP Research Scientist:** single-command loop (`experiment.py`), PIT replay, baseline vs improvement, DuckDB provenance, reproducible via `--reproduce`.
- **L2 — Research Infrastructure:** `research_memory.py` (institutional memory + belief updates + duplicate-risk detection), feature/model registries, migration `002_registries.sql`.
- **L3 — Multi-Agent Research Team:** `agents.py` — role/permission model (researcher cannot approve own work), validation gate with three independent reviewers: statistical (binomial significance vs coin-flip), adversarial (label-randomisation null test, regime concentration), and an independent replicator that rebuilds dataset→features→model from documented artifacts only. Champion promotion requires APPROVED from all three.
- **L4 — Autonomous Research Mode:** `autonomy.py` — budgeted observation-mode sessions: review memory → select unseen configs (duplicate prevention) → run experiments → validate → store knowledge. Scheduling = invoke from cron/launchd; sessions are crash-safe (each experiment commits independently). Champion promotion remains a human decision (REVIEW MODE).

## Platform phases (all built; autonomous mode OFF)

| Phase | Module(s) | Notes |
|---|---|---|
| 3 — Features | `features/` (technical, macro, fundamental, pit) | as-of availability joins; truncation-invariance leakage tests |
| 4 — Prediction | `models/prediction.py`, `experiment.run_horizons` | frozen Prediction objects (1/3/5/10/20d) |
| 5 — Models | `models/registry.py` | random/majority baselines, logistic, XGBoost (`pip install '.[advanced]'`) |
| 6 — Experiments | `experiment.py` registry API | list/get/compare, train+test period provenance |
| 7 — Validation | `validation/` | hidden holdout, ablation, bias dashboard, bootstrap CI |
| 8 — Portfolio | `portfolio/construction.py` | equal/vol/risk sizing, water-filled caps, drawdown stop |
| 9 — Strategies | `strategies/framework.py` | Strategy ABC + PIT-only contract; reference impl only |
| 10 — Automation | `automation/controller.py`, `queue.py` | task queue + controller, **inert** until `enabled=True` AND `QLT_AUTONOMOUS_ENABLED=true` |
| 11 — Paper prep | `paper_trading.py` | Order/simulator/tracker offline; broker **disabled** until `allow=True` AND `QLT_PAPER_ENABLED=true` |

## Research terminal (read-only dashboard)

Bloomberg-style observability over the running lab — cycle progress, live
pipeline, equity/drawdown/rolling charts, validation evidence, rejection
analytics and system health. It only reads: DuckDB is opened `read_only`, no
artifact is ever written, and anything Quant Loop has not produced renders as
`N/A`.

```bash
PYTHONPATH=src .venv/bin/python -m quant_loop_trader.dashboard.api --port 8787
# → http://127.0.0.1:8787
```

Backend in `src/quant_loop_trader/dashboard/` (queries / service / schemas /
api), frontend in `dashboard/` (native ES modules + vendored ECharts, no build
step). Full documentation: [docs/dashboard.md](docs/dashboard.md).

## Data connectors (`connectors/`)

Every connector returns `(pl.DataFrame, source)` where the frame satisfies the PIT contract: `event_time` + `available_time` Date columns, `available_time >= event_time`. Any such frame filters through `replay.pit_filter(df, ts)` — the same availability rule as `ReplayEngine.get_snapshot`.

| Connector | Source | event_time | available_time |
|---|---|---|---|
| `alpaca.fetch_bars` | Alpaca Market Data (IEX feed, historical only — no trading) | bar date (daily close) | same day |
| `fred.fetch_series` | FRED observations | observation period | period end **+ publication lag** (monthly ~15d, quarterly ~45d; `ponytail:` heuristic — ALFRED vintage_dates is the exact upgrade path) |
| `sec.fetch_company_facts` | SEC EDGAR XBRL companyfacts | fiscal period end | **exact filing date** — a Q2 report filed Aug 2 is unavailable until Aug 2 |

Unit tests mock all HTTP; integration tests hit live APIs and skip when credentials are absent. SEC raw payloads cached under `data/raw/sec/`.

## Continuous operation (7-day unattended run)

Installed via launchd (see `deploy/*.plist`, loaded into `~/Library/LaunchAgents`):

| Job | Schedule | Action |
|---|---|---|
| `com.quantloop.research-session` | daily 06:00 | `autonomy --max-experiments 4` — budgeted research session |
| `com.quantloop.weekly-report` | Sunday 18:00 | `report` + git-commit `data/reports/` |

Logs: `data/logs/session.log`. Reports: `data/reports/weekly_YYYY-Www.md`. Manage with:
```bash
launchctl kickstart gui/$(id -u)/com.quantloop.research-session  # run now
launchctl unload ~/Library/LaunchAgents/com.quantloop.research-session.plist  # stop
```
When the research grid is exhausted the loop idles by design (anti-mining governor, see report's "Research frontier").

Note: install non-editable (`pip install .`) — macOS security tooling on some machines flags editable-install `.pth` files as hidden, breaking imports. Reinstall after code changes, or run tests (which use `pythonpath = src`).
