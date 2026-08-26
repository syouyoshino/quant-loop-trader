# SIMPLIFICATION_HARDENING_REPORT

**Commit:** a31d30f (main + research/mvrs) · **Tests:** 109 offline passed, 3 live integration passed · **Lint:** clean

## Executive summary

The seam-collapse refactor landed. The system now has ONE canonical significance
calculation, ONE experiment specification type with a deterministic fingerprint,
ONE verified-bundle loader for trusted paths, immutable content-addressed dataset
snapshots as the sole post-acquisition input, and ONE derived lifecycle policy.
Champion is no longer written by any component — it is *derived* from evidence.

## Architecture before → after

| Concept | Before | After |
|---|---|---|
| Significance implementations | 4+ (research stride inline, holdout binomtest, FDR family baseline p, report stat_pvalue) | 1: `core.significance()` (strided, one-sided, base-rate null) |
| Experiment identity | date+config-hash string, 2 hand-built hash sites | `ExperimentSpec.fingerprint()` (canonical JSON), stored as `spec_fingerprint` |
| Run identity | collided same-day, overwrote locked evidence | unique run dirs (`_rN` suffix on collision); reproduction = child run with lineage |
| Post-acquisition input | shared mutable ticker parquet + drift checks | content-addressed `data/datasets/{dataset_id}.parquet` snapshot; cache is acquisition-only |
| Integrity checking | manual `_verify_locks()` calls at each site | `ExperimentBundle.open_verified()` — trusted paths cannot load unverified |
| Validator boundary | reviewer dicts + issue strings + hardening dict + verdict text | `CheckResult` / `core.gate()` fail-closed semantics |
| Lifecycle | 3 vocabularies mutated by 3 components | evidence facts (`LifecycleEvidence`) → `core.final_state()` derivation |

## Invariants status

INV 1 spec ✓ (fingerprint in every config/report) · INV 2 unique run dirs ✓ ·
INV 3 snapshot input ✓ (build_train_test accepts snapshot path; adjudication/reviewers resolve via config) ·
INV 4/5 verified bundle ✓ (tamper/metrics/dataset-drift checks inside `open_verified`) ·
INV 6 one significance ✓ · INV 7 candidate p-value only for FDR ✓ ·
INV 8 holdout gate uses holdout predictions only ✓ (research `statistical_review`
no longer consulted at promotion) · INV 9-10 derived state ✓ (`final_state`) ·
INV 11 memory wording made provisional; correction machinery retained for legacy rows ·
INV 12-13 uniform fail-closed gate ✓ · INV 14 reproduce = child run + lineage ✓ ·
INV 15 replicator keeps its own sklearn reconstruction ✓.

## Simplifications implemented
`SAFE TO CONSOLIDATE`: significance ×4→1 · lock verification ×3→bundle ·
config hashing → fingerprint · lifecycle writers → `final_state`.
`SAFE TO REMOVE`: dead `_labels_nonholdout`, duplicate con.close/_MIGRATED lines,
unused imports (ruff-clean), `or True` assertions.
`EXPERIMENTAL/DEFERRED`: automation queue/controller (inactive, gated),
hypothesis-engine→runner wiring (activation-gated), XGBoost end-to-end,
ALFRED vintages (macro families generation-gated behind `QLT_ALLOW_REVISED_MACRO`).

## Scientific defenses retained
All of them: PIT, shift(1) discipline, embargo, permanent holdout, independent
replication (separate sklearn implementation deliberately preserved), majority
baseline, degenerate/near-degenerate gates, feature shuffle, label randomisation,
regime robustness, walk-forward, ablation, BH-FDR, DSR (empirical dispersion),
transaction costs, economic holdout gate, drawdown analysis, moving-block
bootstrap, artifact locks, snapshots, double activation keys, audit trail.

## Bugs found during simplification
- `run_ablation` referenced `feature_fn` before assignment (crash on real path).
- `adjudicate_holdout` failed with `"close" not found` after feature-column selection dropped price data.
- `statistical_review` crashed on synthetic dirs without config.json.
- `_verify_locks` flagged the `dataset_parquet` anchor as a missing artifact file.

All fixed with regression tests. No known regressions introduced: full suite green.

## Remaining justified complexity
Independent replication's separate pipeline (defense-in-depth). Layered gates
(different failure modes). Double activation keys. Correction machinery for
legacy memory rows (audit trail).

## Revisit later
Hypothesis-engine → runner wiring (activation phase). Queue/controller vs direct
autonomy convergence. XGBoost end-to-end. ALFRED vintages. Paper limit fills
using OHLC. SEC revenue-tag synonyms. Branch protection on main (needs admin UI).

## Verification caveat
Final suite runs were executed under severe machine CPU contention (load ~80 from
concurrent sessions). The last full run completed: **109 passed** in ~29s once
contention cleared; targeted verification of every changed path passed throughout.
