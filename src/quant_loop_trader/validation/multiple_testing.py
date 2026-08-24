"""Multiple-testing correction (Task 3): FDR control + deflated Sharpe.

"The more things tested, the easier it is to find fake winners."
"""
from __future__ import annotations

import math

import numpy as np


def benjamini_hochberg(pvals: list[float], fdr: float = 0.10) -> list[bool]:
    """BH procedure: returns reject/not-reject per hypothesis (sorted-input order preserved)."""
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


def deflated_sharpe_ratio(sharpe: float, n_obs: int, n_trials: int,
                          skew: float = 0.0, kurtosis: float = 3.0) -> dict:
    """Bailey & López de Prado DSR: probability that the BEST of n_trials Sharpes
    is genuinely > 0, given selection bias under multiple testing.

    Expected maximum Sharpe under H0 grows ~ sqrt(2 ln n_trials); we deflate by it.
    """
    if n_obs <= 1 or n_trials <= 0:
        return {"dsr": 0.0, "expected_max_sharpe_h0": 0.0, "verdict": "INSUFFICIENT_DATA"}
    # variance of trial Sharpes across the search — estimated conservatively from trial count
    var_sharpe_trials = max(math.log(max(n_trials, 2)), 1.0)
    emc = 0.5772156649  # Euler-Mascheroni
    expected_max_h0 = math.sqrt(var_sharpe_trials) * ((1 - emc) * _norm_cdf_inv(1 - 1 / n_trials)
                                                      + emc * _norm_cdf_inv(1 - 1 / (n_trials * math.e)))
    sr_std = math.sqrt((1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2) / max(n_obs - 1, 1))
    if sr_std <= 0:
        return {"dsr": 0.0, "expected_max_sharpe_h0": expected_max_h0, "verdict": "INSUFFICIENT_DATA"}
    psr = _norm_cdf((sharpe - expected_max_h0) / sr_std)
    verdict = "GENUINE" if psr >= 0.95 else "LOW_CONFIDENCE" if psr >= 0.5 else "PROBABLY_LUCK"
    return {"dsr": round(psr, 4), "expected_max_sharpe_h0": round(expected_max_h0, 3),
            "deflated_sharpe_threshold": round(expected_max_h0, 3), "verdict": verdict}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_cdf_inv(p: float) -> float:
    """Inverse normal CDF via bisection on erf — no scipy dependency here."""
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
