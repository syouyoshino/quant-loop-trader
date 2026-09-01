# Quant Loop Trader — Research Platform

Quant Loop Trader is an autonomous historical research laboratory for BTCUSD and SPY. It is **not** a live trading bot: nothing in the active core places real orders.

The platform is intentionally organized around one evidence path instead of multiple orchestration/framework layers.

## Active architecture

```text
connectors/          PIT data acquisition (Tiingo, Alpaca historical, FRED, SEC)
data.py              fetch/cache, exact BTC coverage, checksums, DuckDB migrations
market.py            market calendars + campaign/holdout identity
replay.py            ReplayEngine.get_snapshot (available_time <= T)
features/            causal technical/macro/fundamental feature construction
models/              prediction objects + model registry
experiment.py        acquire once → immutable snapshot → research experiment
candidate.py         canonical verified CandidateSpec
random_replay.py     candidate-bound random-start robustness testing
evaluation.py        prediction/economic metrics, costs and risk
validation/          walk-forward, multiple testing, holdout and diagnostics
agents.py            statistical/adversarial/independent-replication gate
research_memory.py   durable evidence and belief updates
autonomy.py          ONE budgeted autonomous orchestration path
monitoring/          heartbeat, health aggregation, alerts
dashboard/           read-only observability
```

Deferred capabilities live under `experimental/`:

```text
experimental/hypothesis/   hypothesis generation/ranking
experimental/portfolio/    multi-asset sizing
experimental/strategies/   generalized strategy interfaces
experimental/paper_trading.py  offline fill simulator
```

They remain tested but are not dependencies of BTC research execution.

## Canonical candidate identity

After an experiment is sealed, trusted downstream checks derive a `CandidateSpec` from the verified bundle. It binds:

- ticker and horizon,
- model type, parameters and seed,
- exact feature columns,
- immutable dataset snapshot/checksum,
- experiment fingerprint,
- market campaign and holdout boundary.

Validation, random replay and final holdout consume this same identity. They fail closed if the active campaign does not match the sealed candidate or if a candidate requires unsupported causal features.

Independent replication deliberately does **not** reuse the creator's model implementation. That separation remains because it protects against shared implementation bugs.

## Data flow

1. `fetch_ohlcv` acquires the requested market window.
2. BTCUSD must contain exactly one validated UTC daily observation for every requested calendar day.
3. The acquired dataframe is sealed as a content-addressed immutable dataset snapshot.
4. `ReplayEngine.get_snapshot(ticker, T)` exposes only observations with `available_time <= T`.
5. `build_train_test` creates labels/features and a horizon-purged ordered split from the sealed snapshot.
6. `experiment.py` trains/evaluates the research candidate and seals its artifacts.
7. `CandidateSpec` becomes the one downstream strategy definition.
8. Validation runs statistical/adversarial checks and an independent reconstruction.
9. `random_replay.py` tests the same candidate from seeded historical start points without loading campaign holdout observations.
10. Eligible candidates may consume the final holdout exactly once; evidence is sealed and committed atomically.
11. Outcomes update research memory.

## Bitcoin campaign boundary

BTCUSD defaults to the `btc_pre2024_v1` campaign with a permanent holdout beginning `2024-01-01`.

Changing the crypto holdout boundary requires a new campaign identity. This prevents observations that were once treated as hidden evidence from silently returning to the research set.

## Validation principles

The platform retains the safeguards that directly improve scientific reliability:

1. PIT availability and truncation-invariance leakage checks.
2. Horizon purge between training and testing.
3. Majority/base-rate significance checks.
4. Multiple-testing controls.
5. Adversarial label/feature destruction tests.
6. Walk-forward temporal checks.
7. Independent replication.
8. Candidate-bound random historical replay.
9. Transaction-cost and worst-entry-phase stress.
10. Permanent one-shot final holdout.
11. Immutable artifacts and checksums.

Ablation, regime breakdowns and other diagnostics explain a candidate; they do not justify silently testing a different candidate configuration.

## Autonomous operation

There is one autonomous research runner: `quant_loop_trader.autonomy`.

It remains disabled unless `QLT_AUTONOMOUS_ENABLED=true`.

```bash
QLT_AUTONOMOUS_ENABLED=true \
python -m quant_loop_trader.autonomy \
  --ticker BTCUSD \
  --horizon 5 \
  --max-experiments 1
```

The former task queue / `ResearchController` layer was removed because it duplicated orchestration already performed by `autonomy.py` and was not used by the scheduled BTC job.

## Safety boundaries

- No real-money broker execution exists in the active codebase.
- Autonomous research is off unless explicitly enabled with `QLT_AUTONOMOUS_ENABLED=true`.
- Experimental paper simulation remains separately double-keyed and does not connect to a live broker.
- Final holdout consumption is one-shot and crash-safe.
- Evaluation and campaign rules are not modifiable by research agents.
- Failed/rejected evidence remains part of the audit trail.

## Operations

```bash
python -m pytest -q -m "not integration"
python -m quant_loop_trader.monitoring.health
python -m quant_loop_trader.dashboard.api --port 8787
```

GitHub Actions also provides a manual credentialed Tiingo BTCUSD integration smoke test. It verifies that the real Tiingo crypto response satisfies the platform's exact daily-coverage contract.
