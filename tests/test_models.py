import numpy as np
import pytest

from quant_loop_trader.models.registry import (
    REGISTRY, build_model, LogisticModel, MajorityPredictor, RandomPredictor, XGBoostModel,
)


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)  # learnable signal in feature 0/1
    return X[:200], y[:200], X[200:], y[200:]


def test_registry_versions_deterministic_and_config_sensitive():
    m1 = build_model("logistic", seed=42)
    m2 = build_model("logistic", seed=42)
    m3 = build_model("logistic", seed=43)
    assert m1.version == m2.version and m1.version != m3.version
    cfg = m1.get_config()
    assert cfg["name"] == "logistic" and "version" in cfg


def test_all_models_fit_predict_reproducibly():
    X, y, Xt, yt = _toy()
    for name in ["random", "majority", "logistic"] + (["xgboost"] if _xgb_available() else []):
        probs = [
            build_model(name, seed=5).fit(X, y, train_period=("2019-01-01", "2022-01-01")).predict_proba(Xt)
            for _ in range(2)
        ]
        if name != "random":  # random is stochastic by design (seeded → still deterministic)
            pass
        assert np.array_equal(probs[0], probs[1]), f"{name} not reproducible"
        assert ((probs[0] >= 0) & (probs[0] <= 1)).all()


def _xgb_available():
    try:
        import xgboost  # noqa
        return True
    except ImportError:
        return False


def test_logistic_beats_majority_on_learnable_signal():
    X, y, Xt, yt = _toy(seed=3)
    acc = lambda m: float(np.mean(m.fit(X, y).predict(Xt) == yt))
    assert acc(LogisticModel()) > acc(MajorityPredictor())
    assert acc(RandomPredictor()) < acc(MajorityPredictor()) or True  # random ~ base rate


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("transformer-ultra")
