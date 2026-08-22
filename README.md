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
#   data/experiments/experiments.jsonl — ledger

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

Note: install non-editable (`pip install .`) — macOS security tooling on some machines flags editable-install `.pth` files as hidden, breaking imports. Reinstall after code changes, or run tests (which use `pythonpath = src`).
