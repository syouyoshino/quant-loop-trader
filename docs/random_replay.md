# Random-start replay

Random-start replay answers a different question from the fixed final holdout:

> If the current BTC strategy were dropped into an arbitrary point in research history, how often would it behave acceptably?

It is a robustness diagnostic. It does **not** replace walk-forward validation or the untouched campaign holdout.

## Design

For each replay the engine:

1. Loads one immutable pre-holdout dataset snapshot.
2. Chooses a unique historical start date using a seeded sampler that rotates roughly evenly across calendar years.
3. Trains the current improved strategy on all usable history before that date.
4. Purges the final `horizon` training observations so their forward labels cannot cross into the simulated trading window.
5. Trades/evaluates the next fixed number of observations using the existing Quant Loop evaluation and cost-stress logic.
6. Records the result and repeats.

The model, features, scaler and cost model are not reimplemented here. Random replay reuses the existing research stack.

## Holdout safety

Random replay fails closed at the configured campaign holdout. It neither samples nor loads bars on or after `campaign_holdout_start`.

For BTCUSD the default campaign is still `btc_pre2024_v1`, so without a deliberately new campaign the replay universe ends on 2023-12-31. To test a newer research universe, configure a new campaign only if the later holdout is genuinely untouched.

Example:

```bash
export QLT_CRYPTO_CAMPAIGN_ID=btc_2026_v1
export QLT_CRYPTO_HOLDOUT_START=2026-01-01
```

With that configuration, random replay can use data through 2025-12-31 but can never trade into 2026.

## Recommended first run

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

`--data-end` defaults to the day before a fixed campaign holdout. `--sample-start` and `--sample-end` can further restrict where replay starts may be selected.

For development, 100 runs is enough to expose obvious fragility. Increase the run count later for a denser robustness distribution. The sampler never duplicates exact start dates.

## Overlapping windows

Hundreds of 180-day windows necessarily overlap in a finite BTC history. The engine therefore does **not** claim the replay outcomes are independent statistical samples.

The summary explicitly reports:

- `unique_start_dates`
- `median_start_gap_days`
- `overlapping_window_fraction`
- `statistical_independence: not_assumed`

Use `--min-start-gap-days` if you want to reduce near-duplicate starts. A large spacing value can make the requested run count impossible, in which case the engine fails rather than silently lowering the number of runs.

## Outputs

Each run creates `data/random_replays/<random_replay_id>/` containing:

- `runs.csv` — one row per replay
- `summary.json` — configuration plus aggregate robustness metrics
- `config.json` — the reproducible replay configuration and dataset identity

The compact aggregate includes:

- profitable fraction
- fraction beating the same-window benchmark
- fraction remaining positive under the existing worst-phase 25 bps/side stress
- median strategy return
- median benchmark return
- median excess return
- median liquidated Sharpe
- median maximum drawdown
- 10th-percentile return
- worst and best replay returns
- number of starts sampled per calendar year

## Interpretation

Do not promote a strategy because one random-replay percentage looks attractive. Look at the distribution, especially the lower tail and drawdown.

The intended validation order remains:

```text
research candidate
    -> walk-forward / hardening validation
    -> random-start robustness replay
    -> final untouched campaign holdout
```

Random replay should initially remain diagnostic rather than an automatic champion gate. Promotion thresholds can be added later after enough real replay distributions have been observed to justify them.
