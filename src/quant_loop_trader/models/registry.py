"""Model interface (Phase 5). Every model versions itself, stores its config,
and reproduces results from (seed, config, training period)."""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    name = "base"

    def __init__(self, seed: int = 42, **params):
        self.seed = int(seed)
        self.params = params
        self._train_period: tuple[str, str] | None = None

    @property
    def version(self) -> str:
        cfg = json.dumps({"name": self.name, "seed": self.seed, "params": self.params}, sort_keys=True)
        return f"{self.name}-v1-{hashlib.sha256(cfg.encode()).hexdigest()[:8]}"

    def get_config(self) -> dict:
        return {"name": self.name, "seed": self.seed, "params": self.params,
                "version": self.version, "train_period": self._train_period}

    def fit(self, X: np.ndarray, y: np.ndarray, train_period: tuple[str, str] | None = None) -> "BaseModel":
        self._train_period = train_period
        return self._fit(X, y)

    # deterministic helper for stochastic models
    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    @abstractmethod
    def _fit(self, X, y) -> "BaseModel": ...

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """P(class=1). The ONLY prediction surface downstream code may use."""

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class RandomPredictor(BaseModel):
    """Random baseline — the luck floor. Every real model must beat it AND the
    majority-class baseline; this exists so the comparison is executable."""
    name = "random"

    def _fit(self, X, y):
        self._rate = float(np.mean(y)) if len(y) else 0.5
        return self

    def predict_proba(self, X):
        return self._rng().random(len(X))


class MajorityPredictor(BaseModel):
    """Always predicts the training majority class. The degenerate-model sentinel:
    any candidate that cannot beat THIS is noise."""
    name = "majority"

    def _fit(self, X, y):
        self._p = float(np.mean(y)) if len(y) else 0.5
        return self

    def predict_proba(self, X):
        return np.full(len(X), max(self._p, 1 - self._p))


class LogisticModel(BaseModel):
    name = "logistic"

    def _fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        self._pipe = Pipeline([("scaler", StandardScaler()),
                               ("clf", LogisticRegression(max_iter=1000, random_state=self.seed))])
        self._pipe.fit(X, y)
        return self

    def predict_proba(self, X):
        return self._pipe.predict_proba(X)[:, 1]


class XGBoostModel(BaseModel):
    name = "xgboost"

    def __init__(self, seed: int = 42, n_estimators: int = 200, max_depth: int = 3,
                 learning_rate: float = 0.05, **kw):
        super().__init__(seed=seed, n_estimators=n_estimators, max_depth=max_depth,
                         learning_rate=learning_rate, **kw)

    def _fit(self, X, y):
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise RuntimeError("xgboost not installed — pip install '.[advanced]'") from e
        self._m = XGBClassifier(random_state=self.seed, n_jobs=1,
                                **{k: v for k, v in self.params.items()})
        self._m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self._m.predict_proba(X)[:, 1]


REGISTRY = {
    cls.name: cls for cls in (RandomPredictor, MajorityPredictor, LogisticModel, XGBoostModel)
}


def build_model(name: str, seed: int = 42, **params) -> BaseModel:
    if name not in REGISTRY:
        raise ValueError(f"unknown model '{name}'; available: {sorted(REGISTRY)}")
    return REGISTRY[name](seed=seed, **params)
