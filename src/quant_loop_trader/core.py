"""Canonical research primitives.

One definition each for experiment identity, significance, gate semantics, and
lifecycle state. Authoritative paths import these primitives rather than rebuild
their own policy.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

PIPELINE_VERSION = 4


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
    campaign_id: str = "default"
    holdout_start: str | None = None
    pipeline_version: int = PIPELINE_VERSION

    def fingerprint(self) -> str:
        """Deterministic identity of the requested scientific experiment.

        Campaign identity and holdout boundary are resolved centrally rather than
        trusting every caller to remember them. Moving a crypto holdout therefore
        necessarily creates a different experiment identity.
        """
        from quant_loop_trader.market import campaign_holdout_start, campaign_id

        payload_dict = asdict(self)
        payload_dict["campaign_id"] = campaign_id(self.ticker)
        payload_dict["holdout_start"] = campaign_holdout_start(self.ticker)
        payload = json.dumps(payload_dict, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class SignificanceResult:
    n_effective: int
    correct: int
    base_rate: float
    pvalue: float
    alpha: float
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def significance(y_true: np.ndarray, y_pred: np.ndarray,
                 horizon: int = 1, alpha: float = 0.05) -> SignificanceResult:
    """One-sided binomial significance vs the same-sample majority base rate.

    h-day directional labels overlap in price paths, so only every h-th aligned
    observation is treated as an independent Bernoulli trial.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    h = max(1, int(horizon))
    n_pairs = min(len(yt), len(yp))
    idx = np.arange(0, n_pairs - h + 1, h) if n_pairs >= h else np.arange(0)
    if len(idx) == 0:
        return SignificanceResult(0, 0, 0.5, 1.0, alpha, False)
    t = yt[idx]
    p = yp[idx]
    n_eff = len(t)
    correct = int((t == p).sum())
    base_rate = float(max(t.mean(), 1 - t.mean()))
    from scipy.stats import binomtest

    pvalue = float(
        binomtest(correct, n_eff, base_rate, alternative="greater").pvalue
    )
    return SignificanceResult(
        n_eff, correct, base_rate, pvalue, alpha, pvalue < alpha
    )


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    evidence: dict = field(default_factory=dict)
    issues: tuple[str, ...] = ()

    @staticmethod
    def fail(name: str, issues: list[str] | tuple[str, ...],
             evidence: dict | None = None) -> "CheckResult":
        return CheckResult(name, False, evidence or {}, tuple(issues))

    @staticmethod
    def ok(name: str, evidence: dict | None = None) -> "CheckResult":
        return CheckResult(name, True, evidence or {}, ())

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "issues_found": list(self.issues),
            "evidence": self.evidence,
        }


def gate(checks: list[CheckResult]) -> tuple[bool, list[str]]:
    """Generic fail-closed gate semantics."""
    issues = []
    for check in checks:
        for issue in check.issues:
            issues.append(
                f"{check.name}:{issue}" if ":" not in issue else issue
            )
        if not check.passed and not check.issues:
            issues.append(f"{check.name}:failed")
    return len(issues) == 0, issues


@dataclass(frozen=True)
class LifecycleEvidence:
    research_screen: str
    validation: str = "NOT_RUN"
    holdout: str = "NOT_RUN"


def final_state(e: LifecycleEvidence) -> str:
    """Derive the only model lifecycle state from recorded evidence.

    Invalid or causally impossible evidence is rejected loudly rather than being
    allowed to fall through to a promotable state.
    """
    if e.research_screen not in {"KEEP", "IMPROVE", "REJECT"}:
        raise ValueError(f"invalid research_screen:{e.research_screen}")
    if e.validation not in {"NOT_RUN", "PASS", "FAIL"}:
        raise ValueError(f"invalid validation:{e.validation}")
    if e.holdout not in {"NOT_RUN", "PASS", "FAIL"}:
        raise ValueError(f"invalid holdout:{e.holdout}")
    if e.holdout != "NOT_RUN" and e.validation != "PASS":
        raise ValueError("holdout evidence requires validation=PASS")

    if e.research_screen == "REJECT":
        return "rejected"
    if e.validation == "FAIL" or e.holdout == "FAIL":
        return "rejected"
    if e.research_screen == "IMPROVE":
        return "candidate"
    if e.validation == "NOT_RUN":
        return "candidate"
    if e.holdout == "NOT_RUN":
        return "eligible"
    return "champion"