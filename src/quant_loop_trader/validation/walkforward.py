"""Walk-forward validation (Task 2): expanding-window folds, PIT-safe.

"If this strategy was discovered in the past, would it have continued working?"
"""
from __future__ import annotations

import polars as pl


def make_folds(event_dates: list, n_folds: int = 4, min_train_frac: float = 0.5) -> list[dict]:
    """Expanding-window folds: train on everything before fold_end, validate the
    following segment. First fold trains on >= min_train_frac of history."""
    n = len(event_dates)
    first_cut = int(n * min_train_frac)
    seg = (n - first_cut) // n_folds
    cuts = [first_cut + i * seg for i in range(n_folds)] + [n]
    folds = []
    for i in range(n_folds):
        folds.append({
            "window_id": f"wf_{i:02d}",
            "train_idx": (0, cuts[i]),
            "validation_idx": (cuts[i], cuts[i + 1]),
        })
    return folds


class WalkForwardValidator:
    """Rolling validation over any PIT-clean frame with event_time/label/features."""

    def __init__(self, model_builder, n_folds: int = 4):
        self.model_builder = model_builder
        self.n_folds = n_folds

    def run(self, df: pl.DataFrame, feature_cols: list[str], horizon: int = 1) -> dict:
        from quant_loop_trader.evaluation import evaluate
        import numpy as np

        df = df.sort("event_time")
        dates = df["event_time"].to_list()
        folds_spec = make_folds(dates, self.n_folds)
        results = []
        all_accs, all_bases = [], []
        for spec in folds_spec:
            a, b = spec["train_idx"]
            c, d = spec["validation_idx"]
            # purge h boundary rows: their labels read validation-window prices (audit QA2)
            train = df.slice(a, max(1, (b - a) - horizon))
            test = df.slice(c, d - c)
            m = self.model_builder()
            m.fit(train.select(feature_cols).to_numpy(), train["label"].to_numpy(),
                  train_period=(str(train["event_time"].min()), str(train["event_time"].max())))
            yte = test["label"].to_numpy()
            ypred = m.predict(test.select(feature_cols).to_numpy())
            try:
                prob = m.predict_proba(test.select(feature_cols).to_numpy())
            except Exception:
                prob = ypred.astype(float)
            prices = test["close"].to_numpy() if "close" in test.columns else np.array([])
            metrics = evaluate(yte, ypred, prob, prices, horizon=horizon)  # real label horizon (audit QA2)
            acc = metrics["accuracy"]
            # per-fold majority-class baseline (audit M2: 0.5 is not the bar)
            base_i = float(max(np.mean(yte), 1 - np.mean(yte)))
            rec = {
                "window_id": spec["window_id"],
                "training_period": [str(train["event_time"].min()), str(train["event_time"].max())],
                "validation_period": [str(test["event_time"].min()), str(test["event_time"].max())],
                "performance": metrics["cumulative_return_strategy"],
                "benchmark_performance": metrics["cumulative_return_benchmark"],
                "risk_metrics": {"max_drawdown_strategy": metrics["max_drawdown_strategy"],
                                 "sharpe_strategy": metrics["sharpe_strategy"]},
                "prediction_accuracy": acc,
                "fold_base_rate": base_i,
                "beats_fold_baseline": bool(acc > base_i),
                "n_validation": test.height,
            }
            results.append(rec)
            all_accs.append(acc)
            all_bases.append(base_i)

        # stability answer: persisted accuracy AND beat each fold's own baseline
        stable = len(all_accs) >= 2 and (max(all_accs) - min(all_accs)) <= 0.15 \
                 and float(np.mean(all_accs)) > 0.5
        return {
            "folds": results,
            "mean_accuracy": round(float(np.mean(all_accs)), 4),
            "accuracy_dispersion": round(float(np.std(all_accs)), 4),
            "stable_across_time": bool(stable and all(b < a for a, b in zip(all_accs, all_bases))),
            "folds_beat_baseline": int(sum(1 for a, b in zip(all_accs, all_bases) if a > b)),
        }
