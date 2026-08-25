"""Multiple-testing correction (Task 3): FDR control + deflated Sharpe.

"The more things tested, the easier it is to find fake winners."

DSR follows Bailey & López de Prado: the expected maximum Sharpe under H0 is
driven by the EMPIRICAL dispersion of the trial Sharpes actually observed —
trial count alone is not a substitute (audit round-2).
"""
from __future__ import annotations

import math

import numpy as np


def benjamini_hochberg(pvals: list[float], fdr: float = 0.10) -> list[bool]:
    """BH procedure: returns reject/not-reject per hypothesis (input order preserved)."""
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    rejects = [False] * n
    max_k = -1
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= fdr * rank / n:
            max_k = rank
    if max_k > 0:
        for idx in order[:max_k]:
            rejects[idx] = True
    return rejects


def family_wise_stats(db_counts: dict) -> dict:
    """Experiment-family accounting from the registry."""
    total = db_counts.get("total", 0)
    return {
        "total_experiments": total,
        "variations_per_family": db_counts.get("variations", total),
        "successful_candidates": db_counts.get("successes", 0),
        "failed_candidates": db_counts.get("failures", 0),
        "families": db_counts.get("families", 1),
    }


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_cdf_inv(p: float) -> float:
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def expected_max_sharpe_h0(trial_sharpes: list[float]) -> float:
    """E[max Sharpe] under H0 given the empirical dispersion of tried strategies."""
    n = len(trial_sharpes)
    if n < 2:
        return 0.0
    emc = 0.5772156649
    z1 = _norm_cdf_inv(1 - 1 / n)
    ze = _norm_cdf_inv(1 - 1 / (n * math.e))
    var = float(np.var(trial_sharpes, ddof=1))
    return math.sqrt(max(var, 1e-12)) * ((1 - emc) * z1 + emc * ze)


def deflated_sharpe_ratio(sharpe: float, n_obs: int, n_trials: int | list[float],
                          skew: float = 0.0, kurtosis: float = 3.0) -> dict:
    """Probabilistic/Deflated Sharpe Ratio.

    n_trials may be an int (count only — variance then cannot be estimated and the
    deflation threshold falls back to 0, i.e. plain PSR with an explicit warning)
    or a list of the trial Sharpes themselves (preferred: real empirical dispersion).
    Finite-sample skew/kurtosis enter via the standard PSR variance term.
    """
    if isinstance(n_trials, (list, tuple)):
        trial_sharpes = [float(s) for s in n_trials]
        n_trials_n = len(trial_sharpes)
        emax = expected_max_sharpe_h0(trial_sharpes)
        dispersion_note = "empirical"
    else:
        n_trials_n = int(n_trials)
        # count-only fallback: no dispersion information exists. Threshold 0 =
        # plain PSR; we refuse to fabricate variance from the count (audit round-2).
        emax = 0.0
        dispersion_note = "count_only_no_dispersion"

    if n_obs <= 1 or n_trials_n <= 0:
        return {"dsr": 0.0, "expected_max_sharpe_h0": 0.0, "verdict": "INSUFFICIENT_DATA",
                "dispersion_note": dispersion_note}

    sr_var = (1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe ** 2) / max(n_obs - 1, 1)
    if sr_var <= 0:
        return {"dsr": 0.0, "expected_max_sharpe_h0": emax, "verdict": "INSUFFICIENT_DATA",
                "dispersion_note": dispersion_note}
    psr = _norm_cdf((sharpe - emax) / math.sqrt(sr_var))
    verdict = "GENUINE" if psr >= 0.95 else "LOW_CONFIDENCE" if psr >= 0.5 else "PROBABLY_LUCK"
    return {"dsr": round(psr, 4), "expected_max_sharpe_h0": round(emax, 3),
            "dispersion_note": dispersion_note, "n_trials": n_trials_n, "verdict": verdict}
