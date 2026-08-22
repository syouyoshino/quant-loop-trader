"""Multi-agent research team — roles, permissions, structured messages, validation gates.

Agents are roles enforced by code paths, not processes (ponytail: single-module L3;
upgrade to task-queue workers when experiments run distributed at L4).
The creator (researcher role) cannot approve its own discovery: validation is a
separate pipeline owned by reviewer roles below.
"""
from __future__ import annotations

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
    # null = majority-class accuracy on the same test labels
    p_base = float(max(pred["y_true"].mean(), 1 - pred["y_true"].mean()))
    from scipy.stats import binomtest
    pvalue = float(binomtest(correct, n, p_base).pvalue)
    if n < 100:
        issues.append(f"sample_size_too_small:{n}")
    if pvalue >= alpha:
        issues.append(f"not_significant_vs_base_rate{p_base:.3f}:p={pvalue:.4f}")
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

    pipe = _pipe()
    pipe.fit(Xtr, ytr)
    real_acc = float((pipe.predict(Xte) == yte).mean())

    # Test 1: label randomisation — would shuffled targets give similar accuracy?
    rng = np.random.default_rng(seed)
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
    if real_acc <= null_p95:
        issues.append(f"label_randomisation:acc{real_acc:.3f}<=null95{null_p95:.3f}")
    if concentrated:
        issues.append("regime_concentration:edge_driven_by_single_regime")
    logger.info(json.dumps({"event": "adversarial_review", "real_acc": real_acc,
                            "null_p95": null_p95, "regime_accs": reg_accs}))
    return _msg("adversarial_reviewer",
                ["label_randomisation_null_test", "regime_concentration_test"], issues)


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
    pipe.fit(train.select(improved_feature_columns()).to_numpy(), train["label"].to_numpy())
    yte = test["label"].to_numpy()
    acc = float((pipe.predict(test.select(improved_feature_columns()).to_numpy()) == yte).mean())

    claimed = report["improved_metrics"]["accuracy"]
    issues = []
    if abs(acc - claimed) > tolerance:
        issues.append(f"replication_mismatch:got{acc:.6f}_claimed{claimed:.6f}")
    logger.info(json.dumps({"event": "independent_replication", "got": acc, "claimed": claimed}))
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

    reviews = [
        statistical_review(exp_dir),
        adversarial_review(exp_dir, ticker, horizon, start, end, seed),
        independent_replication(experiment_id),
    ]
    all_issues = [i for r in reviews for i in r["issues_found"]]
    verdict = {
        "experiment_id": experiment_id,
        "reviews": reviews,
        "approval_status": "APPROVED" if not all_issues else "REJECTED",
        "issues_found": all_issues,
    }
    (exp_dir / "validation.json").write_text(json.dumps(verdict, indent=2))
    # promotion policy: champion only survives every reviewer; enforced in DB, not prose
    import duckdb
    from quant_loop_trader.data import DB_PATH
    status = "champion" if verdict["approval_status"] == "APPROVED" else "rejected"
    con = duckdb.connect(str(DB_PATH))
    con.execute("UPDATE model_registry SET status=? WHERE model_id=?", [status, f"{experiment_id}_improved"])
    con.close()
    logger.info(json.dumps({"event": "validation_complete", "experiment_id": experiment_id,
                            "status": verdict["approval_status"], "registry_status": status}))
    return verdict
