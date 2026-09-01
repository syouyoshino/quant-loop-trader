# Quant Loop Trader

A leakage-aware historical research engine for **BTCUSD** and SPY. The active runtime is deliberately small: acquire data → seal an immutable dataset → run an experiment → bind the exact strategy into `CandidateSpec` → validate/replay it → expose the final holdout once.

**This repository does not place real-money orders.** Broker execution is deliberately absent. Deferred portfolio, generalized strategy, hypothesis-generation, and paper-simulation code lives under `experimental/` and is not part of the BTC research runtime.

## Active research path

```text
Tiingo BTCUSD / SPY data
        ↓
exact PIT + calendar validation
        ↓
content-addressed immutable snapshot
        ↓
experiment.py
        ↓
CandidateSpec
(model + params + features + seed + dataset + campaign)
        ↓
validation ───── random replay
        ↓              ↓
independent replication + robustness evidence
        ↓
one-shot final holdout
```

The active core is intentionally limited to:

```text
candidate.py          canonical verified candidate identity
data.py               acquisition, BTC coverage checks, snapshots, DuckDB
market.py             market calendars + crypto campaign/holdout identity
replay.py             point-in-time snapshots
features/             causal feature construction
models/               research model registry + prediction objects
evaluation.py         predictive + economic metrics and cost stress
experiment.py         canonical experiment runner
random_replay.py      seeded historical random-start robustness replay
validation/           statistical, walk-forward, holdout and hardening checks
agents.py             adversarial + independent replication gate
research_memory.py    durable evidence/memory
 autonomy.py          the single autonomous research orchestrator
dashboard/            read-only observability
```

## Setup

Python 3.12 is the supported runtime:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

For BTCUSD, add a valid `TIINGO_API_KEY` to `.env`. BTC data fails closed unless the requested UTC daily window has exact 24/7 coverage.

## Run Bitcoin research

```bash
# One BTC experiment. Bars on/after the configured campaign holdout are not used
# for ordinary research training/testing.
.venv/bin/python -m quant_loop_trader.experiment \
  --ticker BTCUSD \
  --horizon 5 \
  --start 2018-01-01 \
  --end 2024-12-31

# Independent validation gate.
.venv/bin/python -m quant_loop_trader.experiment --validate <experiment_id>

# Candidate-bound random-start robustness replay.
.venv/bin/python -m quant_loop_trader.random_replay \
  --experiment <experiment_id> \
  --runs 100 \
  --trade-days 180

# Reproduce from the experiment's sealed dataset snapshot.
.venv/bin/python -m quant_loop_trader.experiment --reproduce <experiment_id>

# Budgeted autonomous BTC research. This is the ONE orchestration path.
QLT_AUTONOMOUS_ENABLED=true \
.venv/bin/python -m quant_loop_trader.autonomy \
  --ticker BTCUSD \
  --horizon 5 \
  --max-experiments 1
```

The default BTC campaign is `btc_pre2024_v1` with a permanent holdout beginning `2024-01-01`. Moving that boundary requires an explicitly new crypto campaign ID. See [docs/bitcoin.md](docs/bitcoin.md).

## What `CandidateSpec` prevents

Every trusted downstream check derives one strategy identity from a **verified experiment bundle**. It binds:

- ticker and horizon,
- model type, model parameters and seed,
- exact feature columns,
- immutable dataset snapshot and checksum,
- experiment fingerprint,
- crypto campaign and holdout boundary.

Random replay, validation and final holdout therefore cannot silently evaluate different versions of the candidate. Unsupported feature families or replication models fail closed instead of falling back to a different strategy.

Independent replication intentionally remains a separate implementation. That duplication is a scientific defense: a bug in the creator's model path should not automatically reproduce itself in the verifier.

## Bitcoin integrity rules

BTCUSD research retains the safeguards that materially affect evidence quality:

- exact 24/7 Tiingo daily coverage and gap checking,
- `event_time` / `available_time` point-in-time contract,
- causal lagged features plus truncation-invariance tests,
- horizon purge between training and test data,
- immutable content-addressed datasets,
- permanent campaign-level holdout,
- one-shot crash-safe holdout claims and sealed evidence,
- majority/significance checks and multiple-testing controls,
- adversarial feature/label tests,
- independent replication,
- random-start temporal robustness with overlap diagnostics,
- transaction costs and worst-entry-phase cost stress.

For multi-day models, replay reports overlapping windows explicitly and does **not** assume the repeated windows are statistically independent.

## Deferred capabilities

These capabilities are preserved but intentionally outside the active BTC core:

```text
experimental/
  hypothesis/          hypothesis generation/ranking
  portfolio/           multi-asset sizing and caps
  strategies/          generalized Strategy interface
  paper_trading.py     offline fill/portfolio simulator
```

They can be developed independently without increasing the complexity of the research-evidence path.

## Tests and Bitcoin smoke check

```bash
.venv/bin/python -m pytest -q -m "not integration"
```

GitHub Actions runs import checks, Ruff, and the offline suite on pushes and pull requests. A manual `workflow_dispatch` job runs the credentialed Tiingo BTCUSD coverage integration test and fails if `TIINGO_API_KEY` is unavailable.

## Research terminal

The dashboard remains read-only:

```bash
PYTHONPATH=src .venv/bin/python -m quant_loop_trader.dashboard.api --port 8787
# http://127.0.0.1:8787
```

It reads DuckDB and experiment artifacts but does not alter research state.

## Continuous operation

The bundled BTC launchd template invokes the direct autonomous research loop:

| Job | Schedule | Action |
|---|---|---|
| `com.quantloop.research-session` | daily 06:00 | `autonomy --ticker BTCUSD --horizon 5 --max-experiments 1` |
| `com.quantloop.weekly-report` | Sunday 18:00 | generate the weekly research report |

There is no second queue/controller orchestration layer. When the configured research frontier is exhausted, autonomy idles by design.

Logs: `data/logs/session.log`. Reports: `data/reports/weekly_YYYY-Www.md`.

```bash
launchctl kickstart gui/$(id -u)/com.quantloop.research-session
launchctl unload ~/Library/LaunchAgents/com.quantloop.research-session.plist
```

Editing a repository plist does not reload an already-installed LaunchAgent; redeploy/reload it explicitly.
