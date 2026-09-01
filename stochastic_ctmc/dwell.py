"""Aggregated-Markov dwell times: the observable is a level, not a state.

Sub-states within an observable level are invisible (a trap is 'filled' or
'empty'; the drain current cannot resolve which filled sub-state). A dwell
therefore runs until the LEVEL changes, merging hidden intra-level transitions.
This aggregation is what makes topology inference nontrivial: 1-D dwell
histograms lose state connectivity, 2-D joint histograms partially recover it.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .gillespie import SSAResult
from .generator import validate_generator, stationary_distribution

__all__ = [
    "aggregate_dwells",
    "dwell_runs",
    "entry_distribution",
    "dwell_pdf_1d",
    "dwell_pdf_2d",
    "gateway_block_rank",
    "joint_density_rank",
]


def dwell_runs(ssa_result: SSAResult, level_map) -> tuple[np.ndarray, np.ndarray]:
    """Merge an SSA trajectory into observable-level runs.

    Parameters
    ----------
    ssa_result : SSAResult
        Trajectory from ``gillespie_ssa``.
    level_map : array-like, shape (n_states,)
        ``level_map[s]`` is the observable level of state ``s`` (e.g. 0 = empty,
        1 = filled). States sharing a level are indistinguishable.

    Returns
    -------
    (run_levels, run_times) : ndarray, ndarray
        Level of each run and its duration, in trajectory order. Consecutive
        runs always differ in level. Censoring is NOT applied here.
    """
    lm = np.asarray(level_map, dtype=int)
    states = np.asarray(ssa_result.states, dtype=int)
    if states.max(initial=0) >= lm.shape[0]:
        raise ValueError("level_map too short for the states in this trajectory")

    levels = lm[states]

    edges = np.concatenate([ssa_result.times, [ssa_result.t_max]])
    durations = np.diff(edges)
    if np.any(durations < 0):
        raise ValueError("trajectory times are not monotonic")

    # Run starts: index 0, plus every index where the level differs from the
    # previous one. Hidden (same-level) transitions are absorbed into the run.
    starts = np.concatenate([[0], np.flatnonzero(np.diff(levels)) + 1])
    run_times = np.add.reduceat(durations, starts)
    run_levels = levels[starts]
    return run_levels, run_times


def aggregate_dwells(
    ssa_result: SSAResult,
    level_map,
    *,
    drop_first: bool = True,
    drop_last: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Dwell-time samples for observable levels 0 and 1.

    The first run is dropped by default: the chain starts at a fixed state, but
    later runs are entered with the equilibrium entry distribution over that
    level's sub-states, so run #1 is drawn from a different law. The last run is
    dropped because it is right-censored at ``t_max`` (the trajectory ended, the
    dwell did not).

    Returns
    -------
    (dwells_0, dwells_1) : ndarray, ndarray
        Durations of level-0 and level-1 dwells.
    """
    run_levels, run_times = dwell_runs(ssa_result, level_map)

    lo = 1 if drop_first else 0
    hi = run_times.shape[0] - (1 if drop_last else 0)
    if hi <= lo:
        return np.empty(0), np.empty(0)

    run_levels = run_levels[lo:hi]
    run_times = run_times[lo:hi]

    return run_times[run_levels == 0], run_times[run_levels == 1]


# --- Colquhoun-Hawkes analytic densities ------------------------------------

def _blocks(Q, level_map, level: int = 1):
    """Partition Q into F (states at `level`) and E (all others) blocks."""
    lm = np.asarray(level_map, dtype=int)
    F = np.flatnonzero(lm == level)
    E = np.flatnonzero(lm != level)
    if F.size == 0 or E.size == 0:
        raise ValueError("both observable levels must be occupied by some state")
    return (F, E,
            Q[np.ix_(F, F)], Q[np.ix_(F, E)],
            Q[np.ix_(E, E)], Q[np.ix_(E, F)])


def _krylov_basis(v: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Rows v, vA, vA^2, ..., vA^(n-1).

    span{v exp(At) : t >= 0} equals the row space of this matrix: the matrix
    exponential is a power series in A, so it cannot reach outside the Krylov
    space of v. Its rank is the number of distinct A-modes that v actually
    excites -- which is what decides whether v exp(At) traces a ray (rank 1) or
    a genuine curve (rank > 1).
    """
    n = A.shape[0]
    rows = [np.asarray(v, dtype=float).ravel()]
    for _ in range(n - 1):
        rows.append(rows[-1] @ A)
    return np.array(rows)


def entry_distribution(Q, level_map, level: int = 1) -> np.ndarray:
    """Equilibrium entry probabilities into each sub-state of `level`.

    A dwell in the aggregate begins wherever the chain crossed into it. At
    stationarity the flux into F sub-state j is (pi_E Q_EF)_j; normalising gives
    phi_F. This is what makes dwell #1 of a trajectory started at a fixed state
    unrepresentative -- see aggregate_dwells(drop_first=True).
    """
    Q = validate_generator(Q)
    pi = stationary_distribution(Q)
    F, E, _, _, _, Q_EF = _blocks(Q, level_map, level)
    flux = pi[E] @ Q_EF
    total = flux.sum()
    if total <= 0:
        raise ValueError("no probability flux enters this level")
    return flux / total


def dwell_pdf_1d(Q, level_map, t, level: int = 1) -> np.ndarray:
    """Colquhoun-Hawkes density of an aggregated dwell at `level`.

        f(t) = phi_F exp(Q_FF t) (-Q_FF) 1

    Reads as: enter F distributed as phi_F, survive within F for time t
    (exp(Q_FF t) -- hidden intra-F transitions allowed and invisible), then
    leave F at total rate (-Q_FF)1. For a single-state F this collapses to
    Exp(exit rate); for |F| > 1 it is a mixture of exponentials.
    """
    Q = validate_generator(Q)
    phi = entry_distribution(Q, level_map, level)
    F, _, Q_FF, _, _, _ = _blocks(Q, level_map, level)
    exit_rates = (-Q_FF) @ np.ones(F.size)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return np.array([phi @ expm(Q_FF * ti) @ exit_rates for ti in t])


def dwell_pdf_2d(Q, level_map, t_f, t_e, level: int = 1) -> np.ndarray:
    """Joint density of a `level` dwell of length t_f followed by an adjacent
    dwell of length t_e at the other level.

        f(t_f, t_e) = phi_F exp(Q_FF t_f) Q_FE exp(Q_EE t_e) Q_EF 1

    Writing u(t_f) = phi exp(Q_FF t_f) Q_FE and v(t_e) = exp(Q_EE t_e) Q_EF 1,
    the joint is the inner product u(t_f) . v(t_e). It factorises into
    g(t_f) h(t_e) -- i.e. adjacent dwells are INDEPENDENT and the 2-D histogram
    says nothing the 1-D marginals do not -- exactly when u or v traces a single
    ray as its argument varies. See joint_density_rank: a rank-2 gateway is NOT
    enough on its own; the aggregate must also have internal structure.

    Returns shape (len(t_f), len(t_e)).
    """
    Q = validate_generator(Q)
    phi = entry_distribution(Q, level_map, level)
    F, E, Q_FF, Q_FE, Q_EE, Q_EF = _blocks(Q, level_map, level)
    tail = Q_EF @ np.ones(F.size)

    t_f = np.atleast_1d(np.asarray(t_f, dtype=float))
    t_e = np.atleast_1d(np.asarray(t_e, dtype=float))

    left = np.array([phi @ expm(Q_FF * x) @ Q_FE for x in t_f])      # (nf, |E|)
    right = np.array([expm(Q_EE * y) @ tail for y in t_e])           # (ne, |E|)
    return left @ right.T


def gateway_block_rank(Q, level_map, level: int = 1, tol: float = 1e-9) -> int:
    """rank(Q_FE) for the ONE crossing block out of `level`.

    Named for what it computes. It was called gateway_rank, which collided with
    topology.min_gateway_rank -- a different function, over both crossing blocks
    rather than one, with a different tolerance convention. Same spelling, and
    which one a caller got depended on their import line.

    NECESSARY but NOT SUFFICIENT for adjacent-dwell correlation -- if every
    sub-state of F has the same total exit rate and F has no internal
    transitions, exp(Q_FF t) is a scalar multiple of the identity and the joint
    density factorises however many gateways exist. Use joint_density_rank for
    the actual invariant.

    Also NOT the same as 'chain vs branched': a branched scheme whose sub-states
    all exit through one gateway is still rank 1.

    ``tol`` is a ratio to the largest singular value, not an absolute cutoff, so
    the verdict cannot move when every rate is rescaled. Q_FE scales uniformly
    under Q -> kQ, so the bare ratio suffices here; the Krylov bases inside
    joint_density_rank do not, and need _scale_free_rank instead.
    """
    Q = validate_generator(Q)
    _, _, _, Q_FE, _, _ = _blocks(Q, level_map, level)
    return _ratio_rank(Q_FE, tol)


def _ratio_rank(M, tol: float) -> int:
    """Numerical rank by sigma_i / sigma_1 > tol.

    sigma_i carries units of rate, so an absolute cutoff is not admissible:
    quoting the same physical system in ms^-1 instead of s^-1 scales the whole
    spectrum and can drag it across a fixed threshold. Only the ratio is
    dimensionless. Correct for any matrix that scales UNIFORMLY under
    Q -> kQ; see _scale_free_rank for the ones that do not.
    """
    s = np.linalg.svd(np.asarray(M, dtype=float), compute_uv=False)
    if s.size == 0 or s[0] <= 0.0:
        return 0
    return int(np.count_nonzero(s / s[0] > tol))


def _scale_free_rank(M, tol: float) -> int:
    """_ratio_rank after normalising each row to unit length.

    For the Krylov bases in joint_density_rank the bare ratio is not enough:
    row i is v A^i, so Q -> kQ multiplies row i by k^i and the matrix does not
    scale uniformly. Row-normalising absorbs the k^i exactly, and cannot change
    the rank -- scaling rows by nonzero constants is an invertible row
    operation.
    """
    M = np.asarray(M, dtype=float)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return _ratio_rank(M / np.where(norms > 0.0, norms, 1.0), tol)


def joint_density_rank(Q, level_map, level: int = 1, tol: float = 1e-9) -> int:
    """Rank of the 2-D joint dwell density -- the quantity the 2-D histogram sees.

    The joint is u(t_f) . v(t_e) with
        u(t_f) = phi exp(Q_FF t_f) Q_FE      spans  K(phi, Q_FF) @ Q_FE
        v(t_e) = exp(Q_EE t_e) Q_EF 1        spans  K(Q_EF 1, Q_EE^T)
    so its rank is bounded by the smaller of those two dimensions (generically
    equal to it). Rank 1 <=> adjacent dwells independent <=> the 2-D histogram
    is redundant with the 1-D marginals.

    Three things must all hold for rank > 1: phi must excite >= 2 modes of
    Q_FF (internal structure or unequal exit rates), Q_FE must have rank >= 2
    (multiple gateways), and Q_EF 1 must excite >= 2 modes of Q_EE.

    ``tol`` is a RATIO to the largest singular value of the row-normalised
    basis, not an absolute singular-value cutoff -- see _scale_free_rank. This
    makes the verdict invariant under rescaling every rate by a common factor.
    """
    Q = validate_generator(Q)
    phi = entry_distribution(Q, level_map, level)
    F, _, Q_FF, Q_FE, Q_EE, Q_EF = _blocks(Q, level_map, level)

    left = _krylov_basis(phi, Q_FF) @ Q_FE
    right = _krylov_basis(Q_EF @ np.ones(F.size), Q_EE.T)

    return int(min(_scale_free_rank(left, tol), _scale_free_rank(right, tol)))