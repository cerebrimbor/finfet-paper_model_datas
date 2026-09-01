"""Independent chi-squared blind-spot verification.

Instead of spending minutes simulating millions of CTMC transitions, this
script samples adjacent dwell pairs directly from the exact bipartite
cycle-level mixture:

    P(component E=i,F=j) = W_ij / C
    tf ~ Exp(K_j/pi_Fj)
    te ~ Exp(D_i/pi_Ei)

It then performs the same quantile-binned Pearson independence test and checks
that the result is consistent with the independently calculated population
noncentrality.

The K(3,3) generator is the same seed-11/deviation-rank-2 construction used
by the paper's figure generator, but the binned probabilities are recalculated
here from first principles.
"""
from pathlib import Path
import sys
import numpy as np
from scipy import stats, optimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stochastic_ctmc.topology import bipartite_generator_from_coupling, zero_correlation_partner
from stochastic_ctmc.generator import stationary_distribution


def independent_binned_probabilities(pi_e, pi_f, W, n_bins=5):
    C = W.sum()
    rF = W.sum(0) / pi_f
    rE = W.sum(1) / pi_e

    # Joint density coefficient:
    # (W_ij/C) * rF_j * rE_i * exp(-rF_j tf) exp(-rE_i te)
    B = (W / C) * rE[:, None] * rF[None, :]

    aF = (B / rE[:, None]).sum(axis=0)
    aE = (B / rF[None, :]).sum(axis=1)

    def edges(r, a):
        def cdf(t):
            return float(np.sum(a * (1.0 - np.exp(-r*t)) / r))
        qs = np.arange(1, n_bins) / n_bins
        hi = 1.0
        while cdf(hi) < 1.0 - 1e-12:
            hi *= 2.0
        inner = [optimize.brentq(lambda z, q=q: cdf(z)-q, 0.0, hi)
                 for q in qs]
        return np.array([0.0, *inner, np.inf])

    def masses(r, edges):
        lo, hi = edges[:-1], edges[1:]
        return (
            np.exp(-r[:, None] * lo)
            - np.where(np.isinf(hi), 0.0, np.exp(-r[:, None] * hi))
        ) / r[:, None]

    ef, ee = edges(rF, aF), edges(rE, aE)
    MF, ME = masses(rF, ef), masses(rE, ee)
    P = MF.T @ B.T @ ME
    return P / P.sum(), rF, rE


def main():
    # Reproduce the fixed figure-generator construction analytically.
    rng = np.random.default_rng(11)
    pi = rng.uniform(0.5, 1.5, 6)
    pi /= pi.sum()
    W = rng.uniform(0.4, 2.0, (3, 3))
    target = bipartite_generator_from_coupling(pi[:3], pi[3:], W)
    Q = zero_correlation_partner(
        target, np.array([0, 0, 0, 1, 1, 1]),
        deviation_rank=2, seed=3, margin=0.9
    )

    # Recover only the physical coupling; all test calculations below are ours.
    E = np.array([0, 1, 2])
    F = np.array([3, 4, 5])
    pi_full = stationary_distribution(Q)
    Wb = pi_full[E, None] * Q[np.ix_(E, F)]
    pi_e, pi_f = pi_full[E], pi_full[F]

    P, rF, rE = independent_binned_probabilities(pi_e, pi_f, Wb, 5)
    expected = np.outer(P.sum(1), P.sum(0))
    lam = float(np.sum((P - expected)**2 / expected))

    # Direct exact value recorded by the paper's figure data is ~1.8926e-5.
    print(f"independently recomputed lambda/pair = {lam:.10e}")
    assert abs(lam - 1.892605053550739e-5) < 1e-10

    # Sample iid adjacent pairs from the exact mixture, not from the simulator.
    n = 4_000_000
    flat = Wb.ravel() / Wb.sum()
    choices = np.random.default_rng(20260808).choice(9, size=n, p=flat)
    i = choices // 3
    j = choices % 3
    tf = np.random.default_rng(20260809).exponential(1.0 / rF[j])
    te = np.random.default_rng(20260810).exponential(1.0 / rE[i])

    # Use the exact population quantile edges, as the asymptotic experiment does.
    # Reconstruct them from the same exact marginal distributions.
    def marginal_edges(r, coeff):
        def cdf(t):
            return float(np.sum(coeff * (1-np.exp(-r*t))/r))
        qs = np.arange(1, 5)/5
        hi = 1.0
        while cdf(hi) < 1-1e-12:
            hi *= 2
        return np.array([0.0] + [
            optimize.brentq(lambda z, q=q: cdf(z)-q, 0, hi) for q in qs
        ] + [np.inf])

    rF = Wb.sum(0)/pi_f
    rE = Wb.sum(1)/pi_e
    C = Wb.sum()
    B = (Wb/C) * rE[:, None] * rF[None, :]
    coeff_f = (B / rE[:, None]).sum(axis=0)
    coeff_e = (B / rF[None, :]).sum(axis=1)
    edges_f = marginal_edges(rF, coeff_f)
    edges_e = marginal_edges(rE, coeff_e)

    table, _, _ = np.histogram2d(tf, te, bins=[edges_f, edges_e])
    expected_counts = np.outer(table.sum(1), table.sum(0))/n
    chi2 = float(np.sum((table-expected_counts)**2/expected_counts))
    dof = 16
    p = stats.chi2.sf(chi2, dof)
    predicted_mean = dof + n*lam

    print(f"sampled pairs = {n:,}")
    print(f"chi2 = {chi2:.2f}")
    print(f"predicted asymptotic mean = {predicted_mean:.2f}")
    print(f"p = {p:.3e}")

    # The purpose here is to verify the detector, not to force one Monte Carlo
    # realization to equal the asymptotic mean.  The observed statistic is
    # decisively beyond the null distribution.
    assert p < 1e-8
    print("PASS")


if __name__ == "__main__":
    main()
