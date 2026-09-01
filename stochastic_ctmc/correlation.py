"""Adjacent-dwell correlation: the statistic that detects the joint's rank.

The 2-D joint density factorises iff adjacent dwells are independent (see
dwell.joint_density_rank). So the practical detector of topology is the
correlation between a filled dwell and the empty dwell that follows it.

All moments are closed form. With
    f(t_f, t_e) = phi exp(Q_FF t_f) Q_FE exp(Q_EE t_e) Q_EF 1
and the standard integrals (for A with strictly negative spectrum)
    int_0^inf exp(At) dt = -A^-1,   int_0^inf t exp(At) dt = A^-2,
    int_0^inf t^2 exp(At) dt = -2 A^-3
every quantity below is a matrix expression -- no quadrature, no fitting. This
gives an exact prediction to test the simulator against, and lets you size an
experiment before running it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .gillespie import SSAResult
from .generator import validate_generator
from .dwell import _blocks, entry_distribution, dwell_runs

__all__ = [
    "adjacent_pairs",
    "AdjacentMoments",
    "analytic_adjacent_moments",
    "empirical_correlation",
    "CorrelationEstimate",
    "pairs_needed",
]


def adjacent_pairs(ssa_result: SSAResult, level_map, first_level: int = 1):
    """(t_first, t_next) for every adjacent dwell pair in a trajectory.

    Returns durations of each ``first_level`` run paired with the run that
    immediately follows it. The first run is dropped (its entry distribution is
    the fixed initial state, not the equilibrium entry law) and the last is
    dropped (right-censored at t_max).

    Returns
    -------
    (t_first, t_next) : ndarray, ndarray
    """
    run_levels, run_times = dwell_runs(ssa_result, level_map)
    if run_times.size < 3:
        return np.empty(0), np.empty(0)

    run_levels = run_levels[1:-1]
    run_times = run_times[1:-1]
    if run_times.size < 2:
        return np.empty(0), np.empty(0)

    idx = np.flatnonzero(run_levels[:-1] == first_level)
    return run_times[idx], run_times[idx + 1]


@dataclass(frozen=True)
class AdjacentMoments:
    """Exact moments of an adjacent dwell pair (t_f at `level`, t_e after it)."""

    mean_f: float
    mean_e: float
    var_f: float
    var_e: float
    cov: float
    rho: float


def analytic_adjacent_moments(Q, level_map, level: int = 1) -> AdjacentMoments:
    """Closed-form moments and Pearson rho for adjacent dwells.

    Let phi = entry law into F, and note that
        psi = phi (-Q_FF)^-1 Q_FE
    is the entry law into E: it is non-negative and psi 1_E = phi 1_F = 1,
    because Q_FF 1_F + Q_FE 1_E = 0 (rows of a generator sum to zero).

        E[t_f]     = -phi Q_FF^-1 1
        E[t_f^2]   =  2 phi Q_FF^-2 1
        E[t_e]     = -psi Q_EE^-1 1
        E[t_e^2]   =  2 psi Q_EE^-2 1
        E[t_f t_e] =  phi Q_FF^-2 Q_FE Q_EE^-2 Q_EF 1

    rho = 0 exactly iff the joint factorises. Note rho = 0 does NOT prove
    independence in general (it is one moment, not the full law) -- but for a
    rank-1 joint the factorisation is exact, so rho = 0 is implied, and rho != 0
    is sufficient to prove rank > 1.
    """
    Q = validate_generator(Q)
    phi = entry_distribution(Q, level_map, level)
    F, E, Q_FF, Q_FE, Q_EE, Q_EF = _blocks(Q, level_map, level)

    one_F = np.ones(F.size)
    one_E = np.ones(E.size)

    iFF = np.linalg.inv(Q_FF)
    iEE = np.linalg.inv(Q_EE)
    iFF2 = iFF @ iFF
    iEE2 = iEE @ iEE

    psi = phi @ (-iFF) @ Q_FE

    mean_f = float(-phi @ iFF @ one_F)
    mean_e = float(-psi @ iEE @ one_E)
    var_f = float(2.0 * phi @ iFF2 @ one_F) - mean_f ** 2
    var_e = float(2.0 * psi @ iEE2 @ one_E) - mean_e ** 2

    e_fe = float(phi @ iFF2 @ Q_FE @ iEE2 @ Q_EF @ one_F)
    cov = e_fe - mean_f * mean_e
    rho = cov / np.sqrt(var_f * var_e)

    return AdjacentMoments(mean_f, mean_e, var_f, var_e, cov, float(rho))


@dataclass(frozen=True)
class CorrelationEstimate:
    """Sample Pearson correlation with a Fisher-z confidence interval."""

    rho: float
    ci_low: float
    ci_high: float
    n_pairs: int
    z_score: float          # (arctanh rho) * sqrt(n-3): sigmas from rho = 0

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


def empirical_correlation(t_first, t_next, confidence: float = 0.95) -> CorrelationEstimate:
    """Sample correlation of adjacent dwells, with a Fisher-z interval.

    Fisher's z = arctanh(r) is approximately normal with standard error
    1/sqrt(n-3) regardless of the true rho, which the raw sampling distribution
    of r is not. The interval is built in z and mapped back with tanh.
    """
    t_first = np.asarray(t_first, dtype=float)
    t_next = np.asarray(t_next, dtype=float)
    n = t_first.size
    if n < 5:
        raise ValueError(f"need at least 5 pairs, got {n}")

    r = float(np.corrcoef(t_first, t_next)[0, 1])
    z = np.arctanh(np.clip(r, -1 + 1e-15, 1 - 1e-15))
    se = 1.0 / np.sqrt(n - 3)
    crit = stats.norm.ppf(0.5 * (1 + confidence))

    return CorrelationEstimate(
        rho=r,
        ci_low=float(np.tanh(z - crit * se)),
        ci_high=float(np.tanh(z + crit * se)),
        n_pairs=int(n),
        z_score=float(z * np.sqrt(n - 3)),
    )


def pairs_needed(rho: float, n_sigma: float = 5.0) -> int:
    """Adjacent dwell pairs required to reject rho = 0 at `n_sigma`.

    From the Fisher-z standard error 1/sqrt(n-3):
        n = 3 + (n_sigma / arctanh(rho))^2
    Sizes the experiment before running it -- and says plainly when a topology
    is undetectable in practice.
    """
    if rho == 0.0:
        return int(np.iinfo(np.int64).max)
    return int(np.ceil(3 + (n_sigma / np.arctanh(abs(rho))) ** 2))