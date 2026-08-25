import numpy as np
import pytest

from quant_loop_trader.models.registry import (
    build_model, LogisticModel, MajorityPredictor, RandomPredictor,
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
    majority_acc = acc(MajorityPredictor())
    random_acc = acc(RandomPredictor(seed=7))
    # sentinel always predicts the TRAIN majority class; accuracy equals that
    # class's share of the test set (whatever the test base rate turns out to be)
    train_majority = max(np.mean(y), 1 - np.mean(y))
    expected_share = (yt == (np.mean(y) >= 0.5)).mean()
    assert majority_acc == pytest.approx(expected_share)
    assert abs(random_acc - max(yt.mean(), 1 - yt.mean())) < 0.15  # random ≈ test base rate


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("transformer-ultra")


def test_majority_predictor_respects_class_zero_majority():
    """Audit H3 regression: sentinel must predict DOWN when class 0 dominates."""
    X = np.zeros((100, 2))
    y = np.array([0] * 70 + [1] * 30)
    m = MajorityPredictor().fit(X, y)
    preds = m.predict(X)
    assert (preds == 0).all(), "majority-class sentinel predicted the minority class"
    assert m.predict_proba(X)[0] < 0.5  # P(class=1) below threshold
