# Random-start replay

Random-start replay asks:

> If this exact strategy candidate were dropped into arbitrary points in historical BTC data, how often would it behave acceptably?

It is a robustness diagnostic. It does **not** replace walk-forward validation or the final untouched campaign holdout, and replay results are not an automatic promotion gate.

## Preferred mode: replay a verified candidate

Use the experiment id produced by `quant_loop_trader.experiment`:

```bash
python -m quant_loop_trader.random_replay \
  --experiment <experiment_id> \
  --runs 100 \
  --trade-days 180 \
  --min-training-days 730 \
  --seed 42
```

Candidate mode opens the experiment through the verified immutable bundle loader and binds the replay to its:

- ticker
- prediction horizon
- model seed
- model type and model parameters
- feature columns
- sealed dataset snapshot
- dataset identity/checksum
- experiment fingerprint and model/feature versions when available

Old experiment bundles created before explicit `model_type` / `feature_columns` fields are inferred as the current improved LogisticRegression strategy for backward compatibility. If a future candidate contains feature columns the replay engine cannot construct, it fails with `unsupported_candidate_features` instead of substituting another feature set.

Ticker, horizon, and source-snapshot overrides are forbidden in candidate mode so the command cannot accidentally test a different strategy while retaining the candidate label. `--seed` remains the replay-start sampling seed; the model uses the candidate's original model seed.

## Standalone mode

For an unbound diagnostic, the original interface remains available:

```bash
python -m quant_loop_trader.random_replay \
  --ticker BTCUSD \
  --horizon 5 \
  --runs 100 \
  --trade-days 180 \
  --min-training-days 730 \
  --data-start 2018-01-01 \
  --seed 42
```

Standalone mode uses the existing improved technical feature set and logistic model.

## Design

For each replay the engine:

1. Loads one immutable pre-holdout dataset snapshot.
2. Verifies the full PIT OHLCV schema (`event_time`, `available_time`, OHLC, volume), rejects null/duplicate timestamps, and rejects `available_time < event_time`.
3. Chooses a unique historical start using a seeded sampler that rotates roughly evenly across calendar years.
4. Trains on all usable history before that date.
5. Purges the final `horizon` training observations so their forward labels cannot cross into the simulated trading window.
6. Evaluates the next fixed number of observations using the normal Quant Loop financial/cost logic.
7. Records the result and repeats.

## Holdout safety

Random replay fails closed at the configured campaign holdout. It neither samples nor loads bars on or after `campaign_holdout_start`.

The default BTC campaign is still `btc_pre2024_v1`, whose permanent holdout begins `2024-01-01`; therefore the default replay universe ends on `2023-12-31`.

A deliberately new research campaign needs a new identity and a genuinely untouched later holdout, for example:

```bash
export QLT_CRYPTO_CAMPAIGN_ID=btc_2026_v1
export QLT_CRYPTO_HOLDOUT_START=2026-01-01
```

With that configuration, replay may use research data through `2025-12-31` but cannot load or trade 2026 holdout observations.

Candidate replay also rejects a candidate whose recorded campaign holdout boundary differs from the active campaign boundary.

## Overlapping windows

Hundreds of 180-day windows necessarily overlap in finite BTC history. The engine therefore does **not** claim replay outcomes are independent samples.

The summary records:

- `unique_start_dates`
- `median_start_gap_days`
- `overlapping_window_fraction`
- `statistical_independence: not_assumed`

Use `--min-start-gap-days` to reduce near-duplicate starts. If the requested run count cannot be satisfied at that spacing, replay fails instead of silently reducing the number of runs.

## Outputs and candidate binding

Each run creates:

```text
data/random_replays/<random_replay_id>/
  runs.csv
  summary.json
  config.json
```

Candidate-bound runs also create a separate diagnostic association outside the sealed experiment bundle:

```text
data/random_replays/by_experiment/<experiment_id>/<random_replay_id>.json
```

This preserves the experiment bundle's immutable hashes while making replay evidence discoverable by experiment id. The association includes a checksum of the replay `summary.json`; `latest_replay_for_experiment()` ignores a linked summary whose checksum no longer matches.

The aggregate includes profitable fraction, benchmark-beating fraction, 25 bps worst-phase profitability, median return/excess return/Sharpe/drawdown, 10th-percentile return, worst/best return, and starts per calendar year.

## Interpretation

Use the distribution, especially lower-tail return and drawdown. The intended sequence remains:

```text
research candidate
    -> walk-forward / hardening validation
    -> candidate-bound random-start robustness replay
    -> final untouched campaign holdout
```

Random replay remains diagnostic until enough real replay distributions exist to justify defensible promotion thresholds.
