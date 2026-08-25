"""Multi-agent research team — roles, permissions, structured messages, validation gates.

Agents are roles enforced by code paths, not processes (ponytail: single-module L3;
upgrade to task-queue workers when experiments run distributed at L4).
The creator (researcher role) cannot approve its own discovery: validation is a
separate pipeline owned by reviewer roles below.
"""
from __future__ import annotations

import hashlib
import math
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from quant_loop_trader.data import PROC_DIR
from quant_loop_trader.features import (
    add_improved_features,
    improved_feature_columns,
)
from quant_loop_trader.experiment import EXP_ROOT

logger = logging.getLogger(__name__)

# --- Agent permission model -------------------------------------------------
ROLES = {
    "research_director": {
        "can": ["approve_experiments", "reject_low_value", "allocate_resources"],
        "cannot": ["modify_evaluation_engine", "alter_scientific_standards"],
    },
    "quant_researcher": {
        "can": ["propose_experiments", "create_models", "analyse_results"],
        "cannot": ["approve_own_discoveries", "modify_evaluation_engine"],
    },
    "statistical_reviewer": {
        "can": ["check_significance", "check_robustness", "block_promotion"],
        "cannot": ["modify_results"],
    },
    "adversarial_reviewer": {
        "can": ["attack_discoveries", "run_hostile_tests", "block_promotion"],
        "cannot": ["modify_results"],
    },
    "independent_replicator": {
        "can": ["reproduce_from_documentation", "verify_artifacts"],
        "cannot": ["use_creator_privileges"],
    },
}


def _msg(reviewer: str, tests_completed: list[str], issues: list[str],
         required_changes: list[str] | None = None) -> dict:
    """Structured VALIDATION MESSAGE per communication protocol."""
    return {
        "reviewer": reviewer,
        "tests_completed": tests_completed,
        "issues_found": issues,
        "approval_status": "APPROVED" if not issues else "REJECTED",
        "required_changes": required_changes or [],
    }


def _load_predictions(exp_dir: Path) -> pl.DataFrame:
    return pl.read_parquet(str(exp_dir / "predictions_improved.parquet"))


def _verify_locks(exp_dir: Path) -> list[str]:
    """STEP 4 enforcement: predictions are immutable after creation. Any tampering
    with locked artifacts fails validation before any reviewer runs."""
    lock_path = exp_dir / "predictions.lock"
    if not lock_path.exists():
        return ["missing_predictions_lock"]
    locks = json.loads(lock_path.read_text())
    issues = []
    cfg_path = exp_dir / "config.json"
    # dataset drift check: parquet bytes now vs locked-at-experiment-time
    want = locks.get("dataset_parquet")
    ticker_name = json.loads(cfg_path.read_text()).get("ticker", "SPY") if cfg_path.exists() else "SPY"
    pq = PROC_DIR / f"{ticker_name}.parquet"
    if want and pq.exists():
        actual = hashlib.sha256(pq.read_bytes()).hexdigest()
        if actual != want:
            issues.append("dataset_drift:input_data_changed_since_experiment")
    for name, expected in locks.items():
        if name in ("locked_at", "dataset_parquet"):  # non-artifact anchors
            continue
        f = exp_dir / name
        if not f.exists():
            issues.append(f"locked_artifact_missing:{name}")
            continue
        actual = hashlib.sha256(f.read_bytes()).hexdigest()
        if actual != expected:
            issues.append(f"artifact_tampered:{name}")
    return issues


# --- Statistical Reviewer ---------------------------------------------------
def statistical_review(exp_dir: Path, alpha: float = 0.05) -> dict:
    """Significance vs the majority-class BASE RATE (not coin-flip), sample size,
    and a hard gate on degenerate constant predictions.
    ponytail: per-experiment binomial only; no Bonferroni correction across the
    overlapping-window grid family — add family-wise correction when grid >100 configs."""
    pred = _load_predictions(exp_dir)
    n = pred.height
    correct = int((pred["y_true"] == pred["y_pred"]).sum())
    issues = []
    # degenerate classifier: predicts a single class regardless of features
    if len(set(pred["y_pred"].to_list())) < 2:
        issues.append(f"degenerate_constant_predictions:{sorted(set(pred['y_pred'].to_list()))}")
    # near-degenerate: minority class of PREDICTIONS < 5% → majority collapse in disguise
    counts = pred["y_pred"].value_counts().sort("count")["count"]
    if n and counts[0] / n < 0.05:
        issues.append(f"near_degenerate_minority_rate:{counts[0] / n:.3f}")
    # null = majority-class accuracy on the same test labels
    p_base = float(max(pred["y_true"].mean(), 1 - pred["y_true"].mean()))
    from scipy.stats import binomtest
    pvalue = float(binomtest(correct, n, p_base).pvalue)
    if n < 100:
        issues.append(f"sample_size_too_small:{n}")
    if pvalue >= alpha:
        issues.append(f"not_significant_vs_base_rate{p_base:.3f}:p={pvalue:.4f}")
    logger.info(json.dumps({"event": "statistical_review", "n": n, "correct": correct,
                            "base_rate": p_base, "p": pvalue}))
    return _msg("statistical_reviewer",
                ["majority_class_null_test", "degenerate_prediction_gate", "sample_size_check"],
                issues)


# --- Adversarial Reviewer ---------------------------------------------------
def adversarial_review(exp_dir: Path, ticker: str, horizon: int,
                       start: str, end: str, seed: int, n_shuffles: int = 200) -> dict:
    """Hostile tests: label randomisation + regime concentration."""
    from quant_loop_trader.experiment import build_train_test
    train, test = build_train_test(ticker, start, end, horizon, add_improved_features, improved_feature_columns())

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    Xtr = train.select(improved_feature_columns()).to_numpy()
    ytr = train["label"].to_numpy()
    Xte = test.select(improved_feature_columns()).to_numpy()
    yte = test["label"].to_numpy()

    def _pipe():
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000, random_state=seed))])

    rng = np.random.default_rng(seed)
    pipe = _pipe()
    pipe.fit(Xtr, ytr)
    real_acc = float((pipe.predict(Xte) == yte).mean())

    # Test 0: feature shuffling — if destroying all feature information leaves
    # performance unchanged, the model is exploiting noise/majority class.
    Xte_shuf = rng.permuted(Xte, axis=0)  # independent permutation per column
    shuffled_acc = float((pipe.predict(Xte_shuf) == yte).mean())
    feature_dependent = real_acc - shuffled_acc > 0.01

    # Test 1: label randomisation — would shuffled targets give similar accuracy?
    null_accs = np.empty(n_shuffles)
    for i in range(n_shuffles):
        ytr_rand = rng.permutation(ytr)
        pipe_r = _pipe()
        pipe_r.fit(Xtr, ytr_rand)
        null_accs[i] = (pipe_r.predict(Xte) == yte).mean()
    null_p95 = float(np.quantile(null_accs, 0.95))

    # Test 2: regime concentration — is all edge from one vol bucket?
    pred_df = _load_predictions(exp_dir)
    acc_overall = float((pred_df["y_true"] == pred_df["y_pred"]).mean())
    vol = test["vol10"].to_numpy()[-pred_df.height:]
    qs = np.quantile(vol[np.isfinite(vol)], [0.5])
    regimes = np.digitize(vol, qs)
    reg_accs = [float((pred_df["y_true"].to_numpy()[regimes == r] ==
                       pred_df["y_pred"].to_numpy()[regimes == r]).mean())
                for r in np.unique(regimes)]
    # dead rule removed (max(reg)>2*acc unreachable); real check: any populated
    # regime below chance while overall above → edge concentrated elsewhere
    concentrated = len(reg_accs) > 1 and min(reg_accs) < 0.5 < acc_overall

    issues = []
    if not feature_dependent:
        issues.append(f"feature_shuffle:acc_unchanged_when_features_destroyed:{shuffled_acc:.3f}vs{real_acc:.3f}")
    if real_acc <= null_p95:
        issues.append(f"label_randomisation:acc{real_acc:.3f}<=null95{null_p95:.3f}")
    if concentrated:
        issues.append("regime_concentration:edge_driven_by_single_regime")
    logger.info(json.dumps({"event": "adversarial_review", "real_acc": real_acc,
                            "shuffled_acc": shuffled_acc, "null_p95": null_p95, "regime_accs": reg_accs}))
    return _msg("adversarial_reviewer",
                ["feature_shuffle_null_test", "label_randomisation_null_test", "regime_concentration_test"], issues)


# --- Independent Replicator -------------------------------------------------
def independent_replication(experiment_id: str, tolerance: float = 1e-9) -> dict:
    """Rebuild dataset→features→model→evaluation using ONLY documented artifacts;
    metrics must match the creator's claim."""
    exp_dir = EXP_ROOT / experiment_id
    cfg = json.loads((exp_dir / "config.json").read_text())
    report = json.loads((exp_dir / "report.json").read_text())

    from quant_loop_trader.experiment import build_train_test
    train, test = build_train_test(cfg["ticker"], cfg["start"], cfg["end"], cfg["horizon"],
                                   add_improved_features, improved_feature_columns())

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000, random_state=cfg["seed"]))])
    Xte = test.select(improved_feature_columns()).to_numpy()
    yte = test["label"].to_numpy()
    pipe.fit(train.select(improved_feature_columns()).to_numpy(), train["label"].to_numpy())
    ypred = pipe.predict(Xte)
    acc = float((ypred == yte).mean())
    try:
        prob = pipe.predict_proba(Xte)
    except Exception:
        prob = ypred.astype(float)
    from quant_loop_trader.evaluation import evaluate as _evaluate
    rep_metrics = _evaluate(yte, ypred, prob, test["close"].to_numpy(), horizon=cfg["horizon"])

    claimed = report["improved_metrics"]
    issues = []
    if abs(acc - claimed["accuracy"]) > tolerance:
        issues.append(f"replication_mismatch:got{acc:.6f}_claimed{claimed['accuracy']:.6f}")
    # audit H5: verify the ECONOMIC result too, not just classification accuracy —
    # a creator/reviewer-shared evaluate() bug must be caught by metric disagreement
    for key in ("sharpe_strategy", "cumulative_return_strategy",
                "transaction_cost_adj_return", "max_drawdown_strategy", "brier_score"):
        got_v, claim_v = rep_metrics[key], claimed[key]
        if abs(got_v - claim_v) > max(tolerance * 1000, 1e-9):
            issues.append(f"replication_metric_mismatch:{key}:got{got_v:.6f}_claimed{claim_v:.6f}")
    return _msg("independent_replicator",
                ["dataset_reconstruction", "feature_rebuild", "retrain_same_seed",
                 "metric_comparison"], issues)


# --- Orchestrator -----------------------------------------------------------
def validate_experiment(experiment_id: str) -> dict:
    """Full validation gate. Returns combined VALIDATION MESSAGE; stores validation.json.
    Reviewers always rebuild from the experiment's documented config — no overrides."""
    exp_dir = EXP_ROOT / experiment_id
    cfg = json.loads((exp_dir / "config.json").read_text())
    ticker, horizon = cfg["ticker"], cfg["horizon"]
    start, end, seed = cfg["start"], cfg["end"], cfg["seed"]

    # STEP 4: tamper check first — locked predictions must be untouched
    lock_issues = _verify_locks(exp_dir)

    reviews = [
        statistical_review(exp_dir),
        adversarial_review(exp_dir, ticker, horizon, start, end, seed),
        independent_replication(experiment_id),
    ]
    all_issues = [i for r in reviews for i in r["issues_found"]] + lock_issues

    # --- hardening layer (Task 1-3 wiring): walk-forward, ablation, DSR ---------
    hardening: dict = {}
    try:
        from quant_loop_trader.experiment import build_train_test
        from quant_loop_trader.validation.walkforward import WalkForwardValidator
        from quant_loop_trader.models.registry import LogisticModel
        train, test = build_train_test(ticker, start, end, horizon,
                                       add_improved_features, improved_feature_columns())
        full = pl.concat([train, test]).sort("event_time")
        wf = WalkForwardValidator(lambda: LogisticModel(seed=seed), n_folds=3)
        hardening["walk_forward"] = wf.run(full, improved_feature_columns(), horizon=horizon)
        if not hardening["walk_forward"]["stable_across_time"]:
            all_issues.append("walk_forward:not_stable_across_folds")

        from quant_loop_trader.validation.multiple_testing import deflated_sharpe_ratio
        sharpe = json.loads((exp_dir / "metrics.json").read_text())["improved"]["sharpe_strategy"]
        # DSR needs the EMPIRICAL dispersion of trial Sharpes (audit round-2) —
        # pulled from every authoritative improved experiment in the family
        m_improved = json.loads((exp_dir / "metrics.json").read_text())["improved"]
        ppy = 252 / horizon
        n_buckets = int(m_improved.get("n_return_buckets", test.height))
        # audit H3: PSR/DSR sampling variance must use the SAME units as the Sharpe —
        # per-bucket Sharpe over the actual number of h-day return observations.
        # audit H4: family scope = same ticker + horizon; LOW_CONFIDENCE blocks too.
        trial_sharpes_periodic = [s / np.sqrt(ppy)
                                  for s in _authoritative_trial_sharpes(ticker=ticker, horizon=horizon)]
        dsr = deflated_sharpe_ratio(m_improved["sharpe_strategy"] / np.sqrt(ppy),
                                    n_obs=n_buckets, n_trials=trial_sharpes_periodic)
        hardening["multiple_testing"] = {"n_trials": len(trial_sharpes_periodic),
                                         "n_return_buckets": n_buckets, **dsr}
        if dsr["verdict"] == "PROBABLY_LUCK":
            all_issues.append("multiple_testing:deflated_sharpe_probably_luck")
        elif dsr["verdict"] == "LOW_CONFIDENCE":
            all_issues.append(f"multiple_testing:dsr_low_confidence:{dsr['dsr']}")

        # FDR across the family's stored significance tests
        from quant_loop_trader.validation.multiple_testing import benjamini_hochberg
        family_pvals, current_p = _family_pvalues(exp_dir, ticker, horizon)
        if current_p is not None and len(family_pvals) >= 5:
            rejects = benjamini_hochberg(family_pvals, fdr=0.10)
            if not rejects[family_pvals.index(current_p)]:
                all_issues.append(f"multiple_testing:fdr_not_significant:p{current_p:.4f}")

        # ablation (audit H7): removing a kept component must never IMPROVE accuracy
        from quant_loop_trader.validation.ablation import run_ablation as _run_ablation
        abl = _run_ablation(ticker, start, end, horizon, seed, {
            "momentum": ["ret_1", "ret_5", "ma10_gap"],
            "volatility": ["vol10", "rsi14"],
        })
        hardening["ablation"] = abl.to_dicts()
        removable = abl.filter(pl.col("removed") != "(none)")
        if removable.height and removable["delta_vs_full"].max() > 0.02:
            all_issues.append(f"ablation:removal_improves_accuracy:{removable['delta_vs_full'].max():.3f}")
    except Exception as e:
        # FAIL CLOSED (audit C2): a broken validator must make approval impossible
        hardening["error"] = str(e)[:200]
        all_issues.append(f"hardening_error:{str(e)[:120]}")

    verdict = {
        "experiment_id": experiment_id,
        "reviews": reviews,
        "hardening": hardening,
        "approval_status": "APPROVED" if not all_issues else "REJECTED",
        "issues_found": all_issues,
    }
    (exp_dir / "validation.json").write_text(json.dumps(verdict, indent=2))
    # promotion policy: champion only survives every reviewer; enforced in DB, not prose
    import duckdb
    from quant_loop_trader.data import DB_PATH
    # APPROVED ≠ champion, and approval ≠ eligibility for every decision:
    # the experiment's PREDECLARED success criterion is authoritative.
    #   KEEP    → may become eligible → holdout may promote
    #   IMPROVE → research candidate (Sharpe degraded; never promotable)
    #   REJECT  → terminal rejected; validation can NEVER resurrect it
    decision = json.loads((exp_dir / "report.json").read_text())["decision"]
    if verdict["approval_status"] == "APPROVED" and decision == "KEEP":
        status = "eligible"
    elif decision == "REJECT":
        status = "rejected"
    else:
        status = "candidate"
    con = duckdb.connect(str(DB_PATH))
    con.execute("UPDATE model_registry SET status=? WHERE model_id=?", [status, f"{experiment_id}_improved"])
    con.close()

    # belief correction (audit AD1): a REJECTED verdict must supersede the premature
    # 'success' memory written at KEEP time — otherwise false beliefs persist forever
    if verdict["approval_status"] == "REJECTED":
        _correct_success_memories_for(experiment_id)

    logger.info(json.dumps({"event": "validation_complete", "experiment_id": experiment_id,
                            "status": verdict["approval_status"], "registry_status": status}))
    return verdict


def _authoritative_trial_sharpes(ticker: str | None = None,
                                 horizon: int | None = None) -> list[float]:
    """Empirical Sharpe dispersion across the authoritative experiment family
    (audit H4: family scope = same ticker + horizon)."""
    import duckdb
    from quant_loop_trader.data import DB_PATH, migrate_db
    migrate_db()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute(
        "SELECT metrics_json FROM experiments WHERE authoritative "
        "AND experiment_id NOT LIKE '%_baseline' "
        "AND (? IS NULL OR ticker = ?) AND (? IS NULL OR horizon_days = ?)",
        [ticker, ticker, horizon, horizon],
    ).fetchall()
    con.close()
    out = []
    for (mj,) in rows:
        try:
            s_val = json.loads(mj).get("sharpe_strategy")
            if isinstance(s_val, (int, float)) and math.isfinite(s_val):
                out.append(float(s_val))
        except Exception:
            continue
    return out


def _experiment_count() -> int:
    import duckdb
    from quant_loop_trader.data import DB_PATH, migrate_db
    migrate_db()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    n = con.execute("SELECT count(*) FROM experiments WHERE authoritative AND experiment_id NOT LIKE '%_baseline'").fetchone()[0]
    con.close()
    return max(1, n)


def _correct_success_memories_for(experiment_id: str) -> None:
    """Find premature success memories by their experiment_id column (audit cycle-2:
    free-text search could never match) and supersede them."""
    import duckdb
    from quant_loop_trader.data import DB_PATH, migrate_db
    migrate_db()
    con = duckdb.connect(str(DB_PATH))
    ids = [r[0] for r in con.execute(
        "SELECT memory_id FROM research_memory WHERE experiment_id = ? AND memory_type = 'success'",
        [experiment_id],
    ).fetchall()]
    con.close()
    for mid in ids:
        _correct_success_memory(mid)


def _correct_success_memory(memory_id: str) -> None:
    """Supersede (never delete) a premature success memory — audit trail preserved."""
    import duckdb
    from quant_loop_trader.data import DB_PATH, migrate_db
    migrate_db()
    con = duckdb.connect(str(DB_PATH))
    row = con.execute("SELECT confidence FROM research_memory WHERE memory_id=?", [memory_id]).fetchone()
    if row is None:
        con.close()
        return
    new_conf = max(0.05, float(row[0]) - 0.2)
    con.execute(
        "UPDATE research_memory SET memory_type='failure', lesson=?, confidence=?, "
        "provenance_json=? WHERE memory_id=?",
        ["CORRECTED: validation gate rejected this KEEP after promotion-time optimism.",
         new_conf,
         json.dumps({"corrected_by": "validate_experiment"}), memory_id],
    )
    con.close()
    logger.info(json.dumps({"event": "memory_corrected", "memory_id": memory_id, "confidence": new_conf}))


def _family_pvalues(current_exp_dir, ticker: str, horizon: int):
    """Stored binomial p-values across the authoritative same-ticker/same-horizon
    family, plus the current experiment's own p-value (audit H4 FDR wiring)."""
    pvals, current_p = [], None
    import duckdb
    from quant_loop_trader.data import DB_PATH, migrate_db
    migrate_db()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    ids = [r[0].replace("_improved", "") for r in con.execute(
        "SELECT experiment_id FROM experiments WHERE authoritative AND ticker=? "
        "AND horizon_days=? AND experiment_id NOT LIKE '%_baseline'",
        [ticker, horizon]).fetchall()]
    con.close()
    for eid in ids:
        try:
            rep_file = EXP_ROOT / eid / "report.json"
            if not rep_file.exists():
                continue
            pv = json.loads(rep_file.read_text()).get("stat_pvalue")
            if pv is None:
                continue
            if eid == current_exp_dir.name:
                current_p = float(pv)
            pvals.append(float(pv))
        except Exception:
            continue
    return pvals, current_p
