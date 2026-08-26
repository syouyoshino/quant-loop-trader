"""Ablation testing: which feature GROUPS actually create improvement?
A strategy dependent on one fragile group dies here."""
from __future__ import annotations

import polars as pl


def run_ablation(ticker: str, start: str, end: str, horizon: int, seed: int,
                 feature_groups: dict[str, list[str]],
                 model_builder=None) -> pl.DataFrame:
    """Train on all features, then with each single group removed.
    Returns per-group delta vs full-model accuracy (negative = group was helping)."""
    from quant_loop_trader.experiment import build_train_test
    from quant_loop_trader.models.registry import LogisticModel
    from sklearn.metrics import accuracy_score

    model_builder = model_builder or (lambda: LogisticModel(seed=seed))
    feature_fn = feature_fn or add_improved_features
    feat_cols = feat_cols or improved_feature_columns()
    all_cols = sorted({c for cols in feature_groups.values() for c in cols} | set(feat_cols))
    missing = [c for c in all_cols if c not in feat_cols]
    if missing:
        raise ValueError(f"ablation groups reference unknown features: {missing}")

    train, test = build_train_test(ticker, start, end, horizon, feature_fn, feat_cols)
    yte = test["label"].to_numpy()

    def _acc(cols: list[str]) -> float:
        m = model_builder().fit(train.select(cols).to_numpy(), train["label"].to_numpy())
        return float(accuracy_score(yte, m.predict(test.select(cols).to_numpy())))

    base_acc = _acc(all_cols)
    rows = [{"removed": "(none)", "accuracy": base_acc, "delta_vs_full": 0.0}]
    for gname, gcols in feature_groups.items():
        keep = [c for c in all_cols if c not in set(gcols)]
        if not keep:
            continue
        a = _acc(keep)
        rows.append({"removed": gname, "accuracy": a, "delta_vs_full": a - base_acc})
    return pl.DataFrame(rows)
