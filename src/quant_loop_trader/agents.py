"""Research reviewer roles and the fail-closed validation gate.

Reviewer separation is retained deliberately: independent reconstruction and
hostile tests are scientific defenses, not orchestration duplication.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import polars as pl

from quant_loop_trader.candidate import CandidateSpec
from quant_loop_trader.experiment import EXP_ROOT
from quant_loop_trader.features import add_improved_features, improved_feature_columns
from quant_loop_trader.market import (
    DEFAULT_CRYPTO_CAMPAIGN_ID,
    DEFAULT_CRYPTO_HOLDOUT_START,
    is_crypto,
    periods_per_year,
)

logger = logging.getLogger(__name__)

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
    """Compatibility wrapper around the one authoritative bundle verifier."""
    from quant_loop_trader.bundle import BundleIntegrityError, ExperimentBundle

    try:
        ExperimentBundle.open_verified(exp_dir.name, exp_dir.parent)
        return []
    except BundleIntegrityError as exc:
        text = str(exc)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return [text]


# --- Statistical Reviewer ---------------------------------------------------
def statistical_review(exp_dir: Path, alpha: float = 0.05) -> dict:
    pred = _load_predictions(exp_dir)
    cfg_file = exp_dir / "config.json"
    horizon = json.loads(cfg_file.read_text())["horizon"] if cfg_file.exists() else 1
    from quant_loop_trader.core import significance
    sig = significance(
        pred["y_true"].to_numpy(),
        pred["y_pred"].to_numpy(),
        horizon=horizon,
        alpha=alpha,
    )

    issues = []
    if len(set(pred["y_pred"].to_list())) < 2:
        issues.append(
            f"degenerate_constant_predictions:{sorted(set(pred['y_pred'].to_list()))}"
        )
    counts = pred["y_pred"].value_counts().sort("count")["count"]
    if sig.n_effective and counts[0] / pred.height < 0.05:
        issues.append(f"near_degenerate_minority_rate:{counts[0] / pred.height:.3f}")
    if sig.n_effective < max(20, 100 // max(horizon, 1)):
        issues.append(f"sample_size_too_small:{sig.n_effective}")
    if not sig.passed:
        issues.append(
            f"not_significant_vs_base_rate{sig.base_rate:.3f}:p={sig.pvalue:.4f}"
        )
    logger.info(json.dumps({"event": "statistical_review", **sig.to_dict()}))
    return _msg(
        "statistical_reviewer",
        [
            "majority_class_null_test",
            "degenerate_prediction_gate",
            "near_degenerate_gate",
            "sample_size_check",
        ],
        issues,
    )


# --- Adversarial Reviewer ---------------------------------------------------
def adversarial_review(exp_dir: Path, ticker: str, horizon: int,
                       start: str, end: str, seed: int, n_shuffles: int = 200,
                       parquet_path: Path | None = None,
                       feature_cols: list[str] | tuple[str, ...] | None = None,
                       model_type: str = "logistic",
                       model_params: dict | None = None) -> dict:
    """Hostile tests rebuilt from the exact candidate and immutable snapshot."""
    from quant_loop_trader.experiment import build_train_test
    from quant_loop_trader.models.registry import build_model

    cols = list(feature_cols or improved_feature_columns())
    train, test = build_train_test(
        ticker, start, end, horizon, add_improved_features,
        cols, parquet_path=parquet_path,
    )

    Xtr = train.select(cols).to_numpy()
    ytr = train["label"].to_numpy()
    Xte = test.select(cols).to_numpy()
    yte = test["label"].to_numpy()

    def _model():
        return build_model(model_type, seed=seed, **dict(model_params or {}))

    rng = np.random.default_rng(seed)
    model = _model()
    model.fit(Xtr, ytr)
    real_acc = float((model.predict(Xte) == yte).mean())

    Xte_shuf = rng.permuted(Xte, axis=0)
    shuffled_acc = float((model.predict(Xte_shuf) == yte).mean())
    feature_dependent = real_acc - shuffled_acc > 0.01

    null_accs = np.empty(n_shuffles)
    for i in range(n_shuffles):
        ytr_rand = rng.permutation(ytr)
        model_r = _model()
        model_r.fit(Xtr, ytr_rand)
        null_accs[i] = (model_r.predict(Xte) == yte).mean()
    null_p95 = float(np.quantile(null_accs, 0.95))

    pred_df = _load_predictions(exp_dir)
    acc_overall = float((pred_df["y_true"] == pred_df["y_pred"]).mean())
    vol = test["vol10"].to_numpy()[-pred_df.height:]
    qs = np.quantile(vol[np.isfinite(vol)], [0.5])
    regimes = np.digitize(vol, qs)
    reg_accs = [
        float((
            pred_df["y_true"].to_numpy()[regimes == r]
            == pred_df["y_pred"].to_numpy()[regimes == r]
        ).mean())
        for r in np.unique(regimes)
    ]
    concentrated = len(reg_accs) > 1 and min(reg_accs) < 0.5 < acc_overall

    issues = []
    if not feature_dependent:
        issues.append(
            f"feature_shuffle:acc_unchanged_when_features_destroyed:"
            f"{shuffled_acc:.3f}vs{real_acc:.3f}"
        )
    if real_acc <= null_p95:
        issues.append(f"label_randomisation:acc{real_acc:.3f}<=null95{null_p95:.3f}")
    if concentrated:
        issues.append("regime_concentration:edge_driven_by_single_regime")
    logger.info(json.dumps({
        "event": "adversarial_review",
        "real_acc": real_acc,
        "shuffled_acc": shuffled_acc,
        "null_p95": null_p95,
        "regime_accs": reg_accs,
        "model_type": model_type,
        "feature_count": len(cols),
    }))
    return _msg(
        "adversarial_reviewer",
        [
            "feature_shuffle_null_test",
            "label_randomisation_null_test",
            "regime_concentration_test",
        ],
        issues,
    )


# --- Independent Replicator -------------------------------------------------
def independent_replication(experiment_id: str, tolerance: float = 1e-9) -> dict:
    """Rebuild independently from the sealed snapshot and canonical candidate spec."""
    from quant_loop_trader.bundle import ExperimentBundle
    from quant_loop_trader.experiment import build_train_test

    bundle = ExperimentBundle.open_verified(experiment_id, EXP_ROOT)
    candidate = CandidateSpec.from_bundle(bundle)
    report = bundle.report
    cols = list(candidate.feature_columns)
    train, test = build_train_test(
        candidate.ticker,
        candidate.research_start,
        candidate.research_end,
        candidate.horizon,
        add_improved_features,
        cols,
        parquet_path=candidate.dataset_snapshot,
    )

    # Deliberately independent implementation. Do not call registry.build_model
    # here: a shared model bug would otherwise make creator and replicator agree.
    if candidate.model_type != "logistic":
        return _msg(
            "independent_replicator",
            ["dataset_reconstruction", "feature_rebuild"],
            [f"replication_unsupported_model:{candidate.model_type}"],
        )

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=candidate.model_seed)),
    ])
    Xte = test.select(cols).to_numpy()
    yte = test["label"].to_numpy()
    pipe.fit(
        train.select(cols).to_numpy(),
        train["label"].to_numpy(),
    )
    ypred = pipe.predict(Xte)
    acc = float((ypred == yte).mean())
    try:
        prob = pipe.predict_proba(Xte)[:, 1]
    except Exception:
        prob = ypred.astype(float)

    from quant_loop_trader.evaluation import evaluate as _evaluate
    rep_metrics = _evaluate(
        yte,
        ypred,
        prob,
        test["close"].to_numpy(),
        horizon=candidate.horizon,
        ticker=candidate.ticker,
    )

    claimed = report["improved_metrics"]
    issues = []
    if abs(acc - claimed["accuracy"]) > tolerance:
        issues.append(
            f"replication_mismatch:got{acc:.6f}_claimed{claimed['accuracy']:.6f}"
        )
    for key in (
        "sharpe_strategy",
        "sharpe_strategy_liquidated",
        "cumulative_return_strategy",
        "transaction_cost_adj_return",
        "transaction_cost_adj_return_compounded",
        "max_drawdown_strategy",
        "brier_score",
    ):
        if key not in claimed:
            continue
        got_v, claim_v = rep_metrics[key], claimed[key]
        if abs(got_v - claim_v) > max(tolerance * 1000, 1e-9):
            issues.append(
                f"replication_metric_mismatch:{key}:got{got_v:.6f}_claimed{claim_v:.6f}"
            )
    return _msg(
        "independent_replicator",
        [
            "dataset_reconstruction",
            "feature_rebuild",
            "retrain_same_seed",
            "metric_comparison",
        ],
        issues,
    )


# --- Orchestrator -----------------------------------------------------------
def validate_experiment(experiment_id: str) -> dict:
    """Full validation gate over one verified immutable experiment bundle."""
    from quant_loop_trader.bundle import BundleIntegrityError, ExperimentBundle
    from quant_loop_trader.core import CheckResult, LifecycleEvidence, final_state, gate

    exp_dir = EXP_ROOT / experiment_id
    try:
        bundle = ExperimentBundle.open_verified(experiment_id, EXP_ROOT)
        candidate = CandidateSpec.from_bundle(bundle)
    except (BundleIntegrityError, ValueError, FileNotFoundError) as exc:
        issues = _verify_locks(exp_dir) if exp_dir.exists() else []
        verdict = {
            "experiment_id": experiment_id,
            "reviews": [],
            "hardening": {},
            "approval_status": "REJECTED",
            "issues_found": issues or [f"candidate_integrity:{exc}"],
        }
        if exp_dir.exists():
            (exp_dir / "validation.json").write_text(json.dumps(verdict, indent=2))
        return verdict

    ticker, horizon = candidate.ticker, candidate.horizon
    start, end, seed = candidate.research_start, candidate.research_end, candidate.model_seed
    cols = list(candidate.feature_columns)

    reviews = [
        statistical_review(bundle.exp_dir),
        adversarial_review(
            bundle.exp_dir,
            ticker,
            horizon,
            start,
            end,
            seed,
            parquet_path=candidate.dataset_snapshot,
            feature_cols=cols,
            model_type=candidate.model_type,
            model_params=candidate.model_params,
        ),
        independent_replication(experiment_id),
    ]

    checks: list[CheckResult] = []
    for review in reviews:
        review_issues = tuple(review["issues_found"])
        checks.append(CheckResult(
            name=review["reviewer"],
            passed=not review_issues,
            evidence={"tests_completed": review["tests_completed"]},
            issues=review_issues,
        ))

    hardening: dict = {}
    hardening_issues: list[str] = []
    try:
        from quant_loop_trader.experiment import build_train_test
        from quant_loop_trader.models.registry import build_model
        from quant_loop_trader.validation.walkforward import WalkForwardValidator

        train, test = build_train_test(
            ticker,
            start,
            end,
            horizon,
            add_improved_features,
            cols,
            parquet_path=candidate.dataset_snapshot,
        )
        full = pl.concat([train, test]).sort("event_time")
        wf = WalkForwardValidator(
            lambda: build_model(
                candidate.model_type,
                seed=seed,
                **dict(candidate.model_params),
            ),
            n_folds=3,
        )
        hardening["walk_forward"] = wf.run(
            full, cols, horizon=horizon, ticker=ticker
        )
        if not hardening["walk_forward"]["stable_across_time"]:
            hardening_issues.append("walk_forward:not_stable_across_folds")

        from quant_loop_trader.validation.multiple_testing import deflated_sharpe_ratio
        m_improved = bundle.metrics["improved"]
        ppy = periods_per_year(ticker, horizon)
        n_buckets = int(m_improved.get("n_return_buckets", test.height))
        trial_sharpes_periodic = [
            s / np.sqrt(ppy)
            for s in _authoritative_trial_sharpes(
                ticker=ticker,
                horizon=horizon,
                campaign=candidate.campaign,
            )
        ]
        current_sharpe = m_improved.get(
            "sharpe_strategy_liquidated", m_improved["sharpe_strategy"]
        )
        dsr = deflated_sharpe_ratio(
            current_sharpe / np.sqrt(ppy),
            n_obs=n_buckets,
            n_trials=trial_sharpes_periodic,
        )
        hardening["multiple_testing"] = {
            "campaign_id": candidate.campaign,
            "n_trials": len(trial_sharpes_periodic),
            "n_return_buckets": n_buckets,
            "periods_per_year": ppy,
            **dsr,
        }
        if dsr["verdict"] == "PROBABLY_LUCK":
            hardening_issues.append("multiple_testing:deflated_sharpe_probably_luck")
        elif dsr["verdict"] == "LOW_CONFIDENCE":
            hardening_issues.append(
                f"multiple_testing:dsr_low_confidence:{dsr['dsr']}"
            )

        from quant_loop_trader.validation.multiple_testing import benjamini_hochberg
        family_pvals, current_p = _family_pvalues(
            bundle.exp_dir,
            ticker,
            horizon,
            campaign=candidate.campaign,
        )
        if current_p is not None and len(family_pvals) >= 5:
            rejects = benjamini_hochberg(family_pvals, fdr=0.10)
            if not rejects[family_pvals.index(current_p)]:
                hardening_issues.append(
                    f"multiple_testing:fdr_not_significant:p{current_p:.4f}"
                )

        # Ablation is diagnostic for the current canonical improved feature family.
        # Do not silently apply a different feature grouping to future candidates.
        if tuple(cols) == tuple(improved_feature_columns()):
            from quant_loop_trader.validation.ablation import run_ablation
            abl = run_ablation(
                ticker,
                start,
                end,
                horizon,
                seed,
                {
                    "momentum": ["ret_1", "ret_5", "ma10_gap"],
                    "volatility": ["vol10", "rsi14"],
                },
                parquet_path=candidate.dataset_snapshot,
            )
            hardening["ablation"] = abl.to_dicts()
            removable = abl.filter(pl.col("removed") != "(none)")
            if removable.height and removable["delta_vs_full"].max() > 0.02:
                hardening_issues.append(
                    f"ablation:removal_improves_accuracy:"
                    f"{removable['delta_vs_full'].max():.3f}"
                )
        else:
            hardening["ablation"] = {
                "status": "not_applicable",
                "reason": "candidate_feature_set_differs_from_canonical_ablation_groups",
            }
    except Exception as exc:
        hardening["error"] = str(exc)[:200]
        hardening_issues.append(f"hardening_error:{str(exc)[:120]}")

    checks.append(CheckResult(
        name="hardening",
        passed=not hardening_issues,
        evidence=hardening,
        issues=tuple(hardening_issues),
    ))
    approved, all_issues = gate(checks)

    verdict = {
        "experiment_id": experiment_id,
        "candidate": {
            "ticker": candidate.ticker,
            "horizon": candidate.horizon,
            "campaign_id": candidate.campaign,
            "model_type": candidate.model_type,
            "model_params": candidate.model_params,
            "feature_columns": cols,
            "spec_fingerprint": candidate.spec_fingerprint,
        },
        "reviews": reviews,
        "hardening": hardening,
        "approval_status": "APPROVED" if approved else "REJECTED",
        "issues_found": all_issues,
    }
    (bundle.exp_dir / "validation.json").write_text(json.dumps(verdict, indent=2))

    decision = bundle.report["decision"]
    status = final_state(LifecycleEvidence(
        research_screen=decision,
        validation="PASS" if approved else "FAIL",
        holdout="NOT_RUN",
    ))

    import duckdb
    from quant_loop_trader.data import DB_PATH

    con = duckdb.connect(str(DB_PATH))
    con.execute(
        "UPDATE model_registry SET status=? WHERE model_id=?",
        [status, f"{experiment_id}_improved"],
    )
    con.close()

    if not approved:
        _correct_success_memories_for(experiment_id)

    logger.info(json.dumps({
        "event": "validation_complete",
        "experiment_id": experiment_id,
        "status": verdict["approval_status"],
        "registry_status": status,
    }))
    return verdict


def _campaign_from_config(raw_config, ticker: str) -> str | None:
    """Resolve stored campaign identity, including pre-v4 BTC evidence."""
    try:
        cfg = json.loads(raw_config) if isinstance(raw_config, str) else dict(raw_config or {})
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    cid = cfg.get("campaign_id")
    if cid:
        return str(cid)
    if is_crypto(ticker):
        if cfg.get("campaign_holdout_start") == DEFAULT_CRYPTO_HOLDOUT_START:
            return DEFAULT_CRYPTO_CAMPAIGN_ID
        return None
    return "default"


def _authoritative_trial_sharpes(ticker: str | None = None,
                                 horizon: int | None = None,
                                 campaign: str | None = None) -> list[float]:
    import duckdb
    from quant_loop_trader.data import DB_PATH, migrate_db

    migrate_db()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute(
        "SELECT metrics_json, config_json, ticker FROM experiments WHERE authoritative "
        "AND experiment_id NOT LIKE '%_baseline' "
        "AND (? IS NULL OR ticker = ?) AND (? IS NULL OR horizon_days = ?)",
        [ticker, ticker, horizon, horizon],
    ).fetchall()
    con.close()
    out = []
    for mj, cfg_json, row_ticker in rows:
        try:
            if campaign is not None and _campaign_from_config(cfg_json, row_ticker) != campaign:
                continue
            metrics = json.loads(mj)
            s_val = metrics.get("sharpe_strategy_liquidated", metrics.get("sharpe_strategy"))
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
    n = con.execute(
        "SELECT count(*) FROM experiments WHERE authoritative "
        "AND experiment_id NOT LIKE '%_baseline'"
    ).fetchone()[0]
    con.close()
    return max(1, n)


def _correct_success_memories_for(experiment_id: str) -> None:
    import duckdb
    from quant_loop_trader.data import DB_PATH, migrate_db

    migrate_db()
    con = duckdb.connect(str(DB_PATH))
    ids = [r[0] for r in con.execute(
        "SELECT memory_id FROM research_memory "
        "WHERE experiment_id = ? AND memory_type = 'success'",
        [experiment_id],
    ).fetchall()]
    con.close()
    for mid in ids:
        _correct_success_memory(mid)


def _correct_success_memory(memory_id: str) -> None:
    import duckdb
    from quant_loop_trader.data import DB_PATH, migrate_db

    migrate_db()
    con = duckdb.connect(str(DB_PATH))
    row = con.execute(
        "SELECT confidence, provenance_json FROM research_memory WHERE memory_id=?",
        [memory_id],
    ).fetchone()
    if row is None:
        con.close()
        return
    new_conf = max(0.05, float(row[0]) - 0.2)
    try:
        provenance = json.loads(row[1]) if row[1] else {}
    except (json.JSONDecodeError, TypeError):
        provenance = {}
    provenance["corrected_by"] = "validate_experiment"
    con.execute(
        "UPDATE research_memory SET memory_type='failure', lesson=?, confidence=?, "
        "provenance_json=? WHERE memory_id=?",
        [
            "CORRECTED: validation gate rejected this KEEP after promotion-time optimism.",
            new_conf,
            json.dumps(provenance),
            memory_id,
        ],
    )
    con.close()
    logger.info(json.dumps({
        "event": "memory_corrected",
        "memory_id": memory_id,
        "confidence": new_conf,
    }))


def _family_pvalues(current_exp_dir, ticker: str, horizon: int,
                    campaign: str | None = None):
    """Verified candidate p-values for the authoritative same-campaign family."""
    pvals, current_p = [], None
    import duckdb
    from quant_loop_trader.bundle import BundleIntegrityError, ExperimentBundle
    from quant_loop_trader.data import DB_PATH, migrate_db

    migrate_db()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute(
        "SELECT experiment_id, config_json FROM experiments WHERE authoritative AND ticker=? "
        "AND horizon_days=? AND experiment_id NOT LIKE '%_baseline'",
        [ticker, horizon],
    ).fetchall()
    con.close()
    ids = [
        experiment_id.replace("_improved", "")
        for experiment_id, config_json in rows
        if campaign is None or _campaign_from_config(config_json, ticker) == campaign
    ]
    for eid in ids:
        try:
            bundle = ExperimentBundle.open_verified(eid, EXP_ROOT)
            pv = bundle.report.get("candidate_stat_pvalue")
            if pv is None:
                continue
            if eid == current_exp_dir.name:
                current_p = float(pv)
            pvals.append(float(pv))
        except (BundleIntegrityError, FileNotFoundError, KeyError, ValueError, TypeError):
            continue
    return pvals, current_p