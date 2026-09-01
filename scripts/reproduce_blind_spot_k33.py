"""Reproduce the K(3,3) correlation blind spot and the chi2 experiment that
detects it.

Recap. topology.py proves that at TWO sub-states per level (K(2,2)) rho = 0
forces the joint to rank 1 -- correlation is a complete topology test there,
no blind spot exists. At THREE sub-states (K(3,3)) the marginals only fix pi
and both margins of the flux matrix W, leaving a (n-1)^2 = 4 dimensional
family of degenerate partners on which Cov is LINEAR in W. rho = 0 is then a
hyperplane through the product coupling rather than an isolated point, and
zero_correlation_partner constructs a point on it with a genuinely higher-rank
joint -- the blind spot correlation cannot see but chi2 can.

This script reproduces, end to end:
  1. the degenerate pair (analytic, exact)          -- topology.py
  2. the chi-squared test's exact power              -- independence.py
  3. a simulation confirming both                    -- gillespie.py

Run:  python -u scripts/reproduce_blind_spot_k33.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats, optimize, linalg

from stochastic_ctmc.topology import (
    bipartite_generator_from_coupling, independence_partner, min_gateway_rank,
    mixture_params,
)
from stochastic_ctmc.dwell import joint_density_rank
from stochastic_ctmc.correlation import (
    analytic_adjacent_moments, adjacent_pairs, empirical_correlation,
)
from stochastic_ctmc.independence import (
    independence_test, chi2_noncentrality, pairs_needed_chi2,
)
from stochastic_ctmc.gillespie import gillespie_ssa

LEV6 = [0, 0, 0, 1, 1, 1]
N_BINS = 6
DOF = (N_BINS - 1) ** 2
N_SIGMA = 5.0

# Centring basis for R^3: columns e_0 - e_2, e_1 - e_2. Any zero-sum vector in
# R^3 is U @ (its first two coordinates); see zero_correlation_partner in
# topology.py for the derivation this mirrors.
_U = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])


def _rho_zero_slice_basis(a, b):
    """Basis of {T (2x2) : a^T T b = 0} -- the rho = 0 slice.

    a = x[:2] - x[2], b = y[:2] - y[2] are the centred sub-state mean dwells;
    see topology.zero_correlation_partner's docstring for why this reduces the
    linear covariance condition to this one equation.
    """
    G = np.outer(a, b).ravel()
    return [v.reshape(2, 2) for v in linalg.null_space(G[None, :]).T]


def _build(pi_E, pi_F, W0, deviation_bases, theta, margin=0.999):
    """W0 perturbed along theta (in the rho = 0 slice), scaled to stay
    strictly inside the transportation polytope {W >= 0, W1 = D, W^T1 = K}."""
    M = sum(t * m for t, m in zip(theta, deviation_bases))
    if not np.any(M < 0):
        return None
    step = margin * np.min(-W0[M < 0] / M[M < 0])
    W = W0 + step * M
    if W.min() <= 0:
        return None
    return bipartite_generator_from_coupling(pi_E, pi_F, W)


def strongest_blind_spot(rng, n_targets=25, n_starts=6):
    """Search the rho = 0 slice for the generator with the largest exact
    chi-squared noncentrality (chi2_noncentrality), starting from n_targets
    random marginal signatures with n_starts restarts each.

    This is a search for a FAVOURABLE EXAMPLE, not part of the theorem: the
    theorem is that a rank > 1, rho = 0 partner exists at all, and
    topology.zero_correlation_partner constructs one directly with no search.
    Detectability varies over orders of magnitude across the family (by
    ~10^3 in ad hoc sampling), driven by how well separated the sub-state
    exit rates are -- reported below alongside the winning example.
    """
    best_lam, best_Q, best_info = 0.0, None, None
    for _ in range(n_targets):
        pi = rng.uniform(0.2, 2.0, 6)
        pi = pi / pi.sum()
        D = rng.uniform(0.3, 3.0, 3)
        K = rng.uniform(0.3, 3.0, 3)
        K = K * (D.sum() / K.sum())            # both margins must total C
        C = D.sum()
        x, y = pi[:3] / D, pi[3:] / K
        a, b = x[:2] - x[2], y[:2] - y[2]
        W0 = np.outer(D, K) / C
        bases = [_U @ T @ _U.T for T in _rho_zero_slice_basis(a, b)]

        def neg_lambda(theta):
            n = np.linalg.norm(theta)
            if n < 1e-9:
                return 0.0
            Q = _build(pi[:3], pi[3:], W0, bases, theta / n)
            if Q is None:
                return 0.0
            try:
                return -chi2_noncentrality(Q, LEV6, N_BINS)
            except ValueError:
                return 0.0

        for _ in range(n_starts):
            theta0 = rng.normal(size=len(bases))
            result = optimize.minimize(
                neg_lambda, theta0 / np.linalg.norm(theta0),
                method="Nelder-Mead",
                options=dict(maxiter=400, xatol=1e-4, fatol=1e-12))
            if -result.fun > best_lam:
                theta_hat = result.x / np.linalg.norm(result.x)
                Q = _build(pi[:3], pi[3:], W0, bases, theta_hat)
                rates = -np.diag(Q)
                best_lam = -result.fun
                best_Q = Q
                best_info = dict(
                    e_rate_spread=rates[:3].max() / rates[:3].min(),
                    f_rate_spread=rates[3:].max() / rates[3:].min(),
                )
    return best_lam, best_Q, best_info


def _signature(Q, level_map):
    return np.concatenate([np.concatenate(mixture_params(Q, level_map, lev))
                           for lev in (0, 1)])


def main():
    rng = np.random.default_rng(2024)
    lam, blind, info = strongest_blind_spot(rng)
    flat = independence_partner(blind, LEV6)

    print("=== 1. the degenerate pair (analytic, exact) ===")
    print(f"{'generator':<22}{'rho':>13}{'gw_rank':>9}{'joint_rank':>12}")
    for tag, Q in (("rank-1 partner", flat), ("blind spot", blind)):
        m = analytic_adjacent_moments(Q, LEV6, 1)
        print(f"{tag:<22}{m.rho:+13.2e}{min_gateway_rank(Q, LEV6):9d}"
              f"{joint_density_rank(Q, LEV6):12d}")
    print(f"marginal signature difference : "
          f"{np.max(np.abs(_signature(blind, LEV6) - _signature(flat, LEV6))):.2e}")
    print(f"E/F sub-state exit rate spread of the blind spot: "
          f"{info['e_rate_spread']:.1f}x / {info['f_rate_spread']:.1f}x "
          f"(detectability tracks this; near 1x collapses toward a K(2,2) "
          f"dwell degeneracy and the effect vanishes)")

    m_blind = analytic_adjacent_moments(blind, LEV6, 1)
    cycle = m_blind.mean_f + m_blind.mean_e

    print("\n=== 2. exact chi-squared power (no Monte Carlo) ===")
    lam_flat = chi2_noncentrality(flat, LEV6, N_BINS)
    crit = stats.chi2.isf(2.0 * stats.norm.sf(N_SIGMA), DOF)
    print(f"lambda/pair  blind spot     = {lam:.4e}   V_true = "
          f"{np.sqrt(lam / (N_BINS - 1)):.5f}")
    print(f"lambda/pair  rank-1 partner = {lam_flat:.4e}   (zero, as it must be)")
    print(f"dof = {DOF}, chi2 critical for {N_SIGMA:.0f} sigma = {crit:.2f}")
    for power in (0.5, 0.9, 0.95):
        n = pairs_needed_chi2(blind, LEV6, N_BINS, n_sigma=N_SIGMA, power=power)
        print(f"  power {power:>4.0%} -> n = {n:,} pairs")
    print("correlation: rho = 0 exactly, so no n suffices -- an identity, "
          "not a power limit")

    print("\n=== 3. simulation ===")
    n_head = 50_000
    print(f"(a) headline, ~{n_head:,} pairs each")
    for tag, Q, seed in (("rank-1 partner", flat, 41), ("blind spot", blind, 42)):
        tf, te = adjacent_pairs(
            gillespie_ssa(Q, 0, n_head * cycle * 1.15, rng=seed), LEV6, 1)
        c = empirical_correlation(tf, te)
        i = independence_test(tf, te, n_bins=N_BINS)
        predicted = DOF + chi2_noncentrality(Q, LEV6, N_BINS) * i.n_pairs
        print(f"  {tag:<16}: rho={c.rho:+.5f} z={c.z_score:+6.2f} | "
              f"chi2={i.statistic:9.1f} (predicted {predicted:7.1f}) "
              f"p={i.p_value:.2e} V={i.cramers_v:.5f} n={i.n_pairs:,} "
              f"n_sigma={i.n_sigma:.1f}")

    n5 = pairs_needed_chi2(blind, LEV6, N_BINS, n_sigma=N_SIGMA, power=0.9)
    n_reps = 15
    print(f"\n(b) sizing check: {n_reps} replicates at the 90%-power n = {n5:,}")
    sigmas, chis = [], []
    for rep in range(n_reps):
        tf, te = adjacent_pairs(
            gillespie_ssa(blind, 0, n5 * cycle * 1.15, rng=1000 + rep), LEV6, 1)
        i = independence_test(tf[:n5], te[:n5], n_bins=N_BINS)
        sigmas.append(i.n_sigma)
        chis.append(i.statistic)
    predicted_chi2 = DOF + lam * n5
    print(f"  observed chi2  mean {np.mean(chis):7.1f}  "
          f"(predicted {predicted_chi2:.1f})")
    print(f"  observed sigma mean {np.mean(sigmas):5.2f}  "
          f"median {np.median(sigmas):5.2f}  "
          f"min {np.min(sigmas):5.2f}  max {np.max(sigmas):5.2f}")
    print(f"  fraction reaching 5 sigma: "
          f"{np.mean(np.array(sigmas) >= N_SIGMA):.0%}  (target power 90%)")


if __name__ == "__main__":
    main()
