"""Canonical research primitives (simplification refactor).

ONE definition each for: experiment specification + fingerprint, significance,
uniform gate results, dataset snapshot identity, and derived lifecycle state.

If you are tempted to recompute any of these elsewhere — don't. Import from here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Experiment specification — the ONE immutable identity of requested research
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentSpec:
    ticker: str
    start: str
    end: str
    horizon: int
    seed: int
    hypothesis_id: str = "baseline_vol_regime"
    hypothesis: str = ""
    economic_reasoning: str = ""
    pipeline_version: int = 2

    def fingerprint(self) -> str:
        """Deterministic identity of the REQUESTED experiment (not the run)."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Significance — the ONE statistical significance calculation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignificanceResult:
    n_effective: int      # non-overlapping observations actually tested
    correct: int
    base_rate: float      # majority-class accuracy of the same rows
    pvalue: float         # one-sided: P(>= correct | base rate)
    alpha: float
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def significance(y_true: np.ndarray, y_pred: np.ndarray,
                 horizon: int = 1, alpha: float = 0.05) -> SignificanceResult:
    """Binomial significance of directional accuracy vs the majority-class base
    rate, computed on NON-OVERLAPPING h-day observations (audit H-round3):
    adjacent h-day labels share price paths, so every row is not an independent
    Bernoulli trial. One-sided 'greater' — the research question is whether the
    model is BETTER than the trivial baseline, not merely different."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    h = max(1, int(horizon))
    n_pairs = min(len(yt), len(yp))
    # non-overlapping stride: keep every h-th aligned pair
    idx = np.arange(0, n_pairs - h + 1, h) if n_pairs >= h else np.arange(0)
    if len(idx) == 0:
        return SignificanceResult(0, 0, 0.5, 1.0, alpha, False)
    t = yt[idx]
    p = yp[idx]
    n_eff = len(t)
    correct = int((t == p).sum())
    base_rate = float(max(t.mean(), 1 - t.mean()))
    from scipy.stats import binomtest
    pvalue = float(binomtest(correct, n_eff, base_rate, alternative="greater").pvalue)
    return SignificanceResult(n_eff, correct, base_rate, pvalue, alpha, pvalue < alpha)


# ---------------------------------------------------------------------------
# Uniform gate results — every required check resolves to exactly one of these
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    evidence: dict = field(default_factory=dict)
    issues: tuple[str, ...] = ()

    @staticmethod
    def fail(name: str, issues: list[str] | tuple[str, ...], evidence: dict | None = None) -> "CheckResult":
        return CheckResult(name, False, evidence or {}, tuple(issues))

    @staticmethod
    def ok(name: str, evidence: dict | None = None) -> CheckResult:
        return CheckResult(name, True, evidence or {}, ())

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed,
                "issues_found": list(self.issues), "evidence": self.evidence}


def gate(checks: list[CheckResult]) -> tuple[bool, list[str]]:
    """Generic fail-closed gate semantics, defined ONCE.
    Any check that did not resolve to passed=True blocks approval."""
    issues = []
    for c in checks:
        for i in c.issues:
            issues.append(f"{c.name}:{i}" if ":" not in i else i)
        if not c.passed and not c.issues:
            issues.append(f"{c.name}:failed")
    return (len(issues) == 0), issues


# ---------------------------------------------------------------------------
# Derived lifecycle state — evidence facts in, one status out
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LifecycleEvidence:
    research_screen: str          # KEEP | IMPROVE | REJECT
    validation: str = "NOT_RUN"   # NOT_RUN | PASS | FAIL
    holdout: str = "NOT_RUN"      # NOT_RUN | PASS | FAIL


def final_state(e: LifecycleEvidence) -> str:
    """THE lifecycle policy. Components record evidence; this derives status.
    No component may independently declare eligible/champion."""
    if e.research_screen == "REJECT":
        return "rejected"
    if e.validation == "FAIL" or e.holdout == "FAIL":
        return "rejected"
    if e.research_screen == "IMPROVE":
        return "candidate"                      # Sharpe degraded → never promotable
    # research_screen == KEEP from here
    if e.validation == "NOT_RUN":
        return "candidate"
    if e.holdout == "NOT_RUN":
        return "eligible"
    return "champion"                           # holdout PASS
