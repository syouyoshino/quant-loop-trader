# STREAMLINING_REPORT.md

**Mission:** maximum research capability, minimum unnecessary complexity.
**Process:** 3 independent read-only simplification reviewers → merged findings → 2 independent Capability Preservation Reviewers classified each KEEP/MODIFY/REMOVE → Lead Developer implemented only consensus-safe items.

## Pipeline executed

| Stage | Agents | Scope |
|---|---|---|
| Findings | Reviewer A (data/replay/features/evaluation/experiment), B (agents/autonomy/memory/report/deploy/tests), C (DB design/storage/artifacts) | full codebase |
| Classification | Capability Preservation Reviewer 1 & 2 | all 21 merged proposals |

## Implemented (consensus REMOVE / MODIFY)

| # | Component | Change | Benefit | Risk control |
|---|---|---|---|---|
| P1/P2 | `data.py` CACHE_DIR, RAW_DIR, dead datetime import | Deleted | Phantom dirs gone; −4 lines | grep-verified zero usage |
| P3 | `ReplayEngine.get_snapshot_range` | Deleted alias, zero callers | −3 lines | grep-verified |
| P4 | `experiments.jsonl` second ledger | Deleted — DuckDB `experiments` table is the single idempotent ledger; removes unbounded dedup scan + divergence-prone second truth | −7 lines, one source of truth | DuckDB rows carry identical fields |
| P5 | `migrate_db()` ~8×/run | Per-(process,path) memo; lazy ensure-migrated retained so fresh environments self-heal | ~8 file opens + SQL replays saved per run | keyed by resolved path so test-isolated fresh DBs still migrate (Reviewer 2 risk #3) |
| P9 | `dataset_metadata` provenance overwritten by caller | Caller passes `extra_provenance`; dead-on-arrival payload removed | clarity | report.py unaffected |
| P10 | `get_snapshot` redundant re-sort of sorted frame; unreachable `_to_date` fallback | Dropped sort, collapsed fallback; **leakage assert kept** (defense-in-depth) | faster snapshots ×2/run | truncation-invariance test added |
| P12 | autonomy: computed-then-popped summary field; unconsumed `duplicate_warning`; per-candidate duplicate scan | Removed; `grid_remaining` surfaced in summary instead | honest session summaries | none |
| P13 | agents.py imports stale post-refactor (`time_split`, `make_labels`, `fetch_ohlcv`, `ReplayEngine`) | Cleaned | hygiene | suite green |
| P14 | `validate_experiment` unused override args | Collapsed to config-only signature — reviewers *must* rebuild from documented config | smaller API, stronger reviewer independence | e2e gate test green |
| P15 | weekly plist `git commit` failing silently via TCC (`\|\| true`) | Git step removed from launchd; reports stay on disk, human commits | honest logs, no swallowed failure | README documents manual commit |
| P16 | tests writing `/tmp` fixtures | pytest `tmp_path_factory` module fixture | hermetic tests | — |
| — | `report.py` hardcoded relative path | Uses `EXP_ROOT` constant | cwd-independence | — |
| — | **Truncation-invariance test** (Reviewer 2's safeguard) | `test_features_truncation_invariant`: any feature reading rows > t fails on truncated recomputation | permanent fence on the highest-value invariant (lookahead) | runs in suite every commit |
| — | Fixture persistence | `fetch_ohlcv` persists fixture to parquet at its single save site; caller guards deleted | offline runs no longer crash downstream; one responsibility site | fixture only persisted when key absent |

## Rejected (consensus KEEP) — with reasons

| # | Proposal | Why kept |
|---|---|---|
| P6 | features.py direct-shift refactor | **Rejected unanimously**: failure mode is silent lookahead that every validation gate would then *confirm* (replication uses same features). Current churn is ugly but proven correct by truncation-invariance + shift tests. Complexity is the price of verified integrity. |
| P7 | Share ReplayEngine across pipeline calls | Divergence-regression risk on the just-repaired critical path for a ~2-parquet-read saving on 1.7k rows. Not worth touching. |
| P18 | Stop writing `model_knowledge` memory rows | Finder claim "never read" was wrong — weekly report's Belief State displays them; institutional journal per constitution. |
| P19 | Stop writing metrics.json | Unread by code but serves as human diff target; cost is one small write per experiment. |
| P20/P21 | Inline n_shuffles/tolerance; ROLES→docstring | Operational knobs for degraded-mode validation; ROLES is a machine-checked statement of the constitution (test asserts researcher cannot self-approve). |
| P17 | Drop review_memory() from session summary | Research-director traceability: what we believed entering each session. Cheap, load-bearing for audit trail. |
| A-rejected | Consolidate adversarial_review/independent_replication into creator's train path | The replicator rebuilding independently IS the validation strength. |

## Net effect

- ~60 lines removed; 2 phantom dirs, 1 double-ledger, 8 redundant DB open/migrate cycles per run, 1 silently-failing launchd step eliminated.
- Zero reduction in scientific validity: purge logic, PIT enforcement, three-reviewer gate, champion-promotion gating, heartbeat/backups/lockfile all untouched and now covered by 30 tests including a new lookahead tripwire.

## Verification

- Full suite: **30 passed** after changes.
- E2E re-run: experiment executes, validates, stores (see commit).
