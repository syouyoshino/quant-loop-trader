# Weekly Research Report — 2026-W34
_Generated: 2026-08-23T08:00:29.846578+00:00 UTC. Mode: OBSERVATION. No champion promotion without human approval._

## Experiment activity
- Total experiment variants stored: 46
- Decision breakdown: KEEP: 1, REJECT: 21, IMPROVE: 1
- Last 7 days: 23 completed hypotheses
  - `20260822_SPY_5d_232bb059_improved` → IMPROVE (validation: REJECTED)
  - `20260822_SPY_5d_6f70080c_improved` → REJECT (validation: REJECTED)
  - `20260822_SPY_5d_19591254_improved` → REJECT (validation: REJECTED)
  - `20260822_SPY_5d_e9724bfc_improved` → REJECT (validation: REJECTED)
  - `20260822_SPY_5d_1ab4b9d8_improved` → REJECT (validation: n/a)
  - `20260822_SPY_5d_e232e392_improved` → REJECT (validation: REJECTED)
  - `20260822_SPY_5d_97eda584_improved` → REJECT (validation: n/a)
  - `20260822_SPY_5d_710970dd_improved` → KEEP (validation: n/a)
  - `20260822_SPY_5d_3e2360a4_improved` → REJECT (validation: n/a)
  - `20260822_SPY_5d_3766593f_improved` → REJECT (validation: REJECTED)

## Belief state (latest confidence per hypothesis)
- [model_knowledge] 0.60 — Baseline regime performance for SPY 5d
- [failure] 0.30 — unique hypothesis 4510293712: vol regime improves momentum
- [failure] 0.30 — unique hypothesis 4586822240: vol regime improves momentum
- [failure] 0.05 — volatility regime classification improves momentum prediction
- [partial] 0.05 — Adding volatility regime classification should improve momentum prediction becau

## Recent lessons
- (IMPROVE) Partial: accuracy improved but Sharpe degraded. Refine risk control before retest.
- (REJECT) Rejected: no accuracy lift or Sharpe degrades Vol-interaction features did not change predictions materially.
- (REJECT) Rejected: no accuracy lift or Sharpe degrades Vol-interaction features did not change predictions materially.
- (REJECT) Rejected: no accuracy lift or Sharpe degrades Vol-interaction features did not change predictions materially.
- (REJECT) Rejected: no accuracy lift or Sharpe degrades Vol-interaction features did not change predictions materially.
- (REJECT) Rejected: no accuracy lift or Sharpe degrades Vol-interaction features did not change predictions materially.
- (REJECT) Rejected: no accuracy lift or Sharpe degrades Vol-interaction features did not change predictions materially.
- (REJECT) Rejected: no accuracy lift or Sharpe degrades Vol-interaction features did not change predictions materially.

## Model registry
- rejected: 23, candidate: 22

## Data health
- SPY_2018-01-02_2024-12-31_dc70d173 rows=1761 status=valid checksum=dc70d173640bf0a2
- SPY_2018-01-02_2022-12-30_3843bdb8 rows=1259 status=valid checksum=3843bdb86fd66f5f
- SPY_2018-01-02_2023-06-30_85db0f49 rows=1383 status=valid checksum=85db0f49427098cb
- SPY_2019-01-02_2024-12-31_d8554beb rows=1510 status=valid checksum=d8554beb3a74216d
- SPY_2020-01-02_2022-12-30_73af7721 rows=756 status=valid checksum=73af7721cd66356b
- SPY_2020-01-02_2023-12-29_b3002e09 rows=1006 status=valid checksum=b3002e0938fbccb0
- SPY_2020-01-02_2023-06-30_bc487ba7 rows=880 status=valid checksum=bc487ba7970fabf5
- SPY_2020-01-02_2024-12-31_7d9d1aaf rows=1258 status=valid checksum=7d9d1aaff1ee7689

## Research frontier (anti-mining governor)
- Grid explored: 18/27 (remaining 9)
- When remaining=0 the loop idles by design: no duplicate mining.
