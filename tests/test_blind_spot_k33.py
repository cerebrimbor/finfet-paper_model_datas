"""The blind spot exists from three sub-states per level, and not before.

test_independence.py proves the negative half at K(2,2): there
Cov = det(W)(mE0-mE1)(mF0-mF1)/C^2 factorises, so rho = 0 only when the joint
genuinely has rank 1 and correlation is a complete topology test.

This file proves the positive half. In the coupling chart the 1-D marginals fix
pi and both margins of the flux matrix W, so a generator's degenerate partners
are exactly the transportation polytope {W >= 0, W 1 = D, W^T 1 = K}, of
dimension (n_E - 1)(n_F - 1). On it

    Cov(t_f, t_e) = (1/C) x^T (W - D K^T / C) y

is LINEAR in W, so rho = 0 is a hyperplane through the product coupling rather
than an isolated point. Writing the deviation as W~ = U_E T U_F^T reduces the
condition to a^T T b = 0. At two sub-states T is a scalar and this forces
T = 0; from three up, T can have any rank up to n-1 while rho stays exactly
zero. That is the blind spot, and it is reached by CANCELLATION -- all
sub-state dwell rates stay distinct and W~ stays large.
"""

import numpy as np
import pytest
from scipy import stats

from stochastic_ctmc.generator import n_state_generator, is_detailed_balance
from stochastic_ctmc.dwell import joint_density_rank
from stochastic_ctmc.correlation import analytic_adjacent_moments
from stochastic_ctmc.topology import (
    bipartite_generator_from_coupling, coupling_from_generator,
    independence_partner, zero_correlation_partner, min_gateway_rank,
    gateway_singular_ratios, mixture_params, bipartite_reversible_generator,
    LEVELS_4,
)
from stochastic_ctmc.independence import (
    joint_mixture_form, chi2_noncentrality, pairs_needed_chi2,
)

LEV6 = [0, 0, 0, 1, 1, 1]


def _signature(Q, level_map):
    """The 1-D-observable content: mixture rates and coefficients, both levels."""
    return np.concatenate([np.concatenate(mixture_params(Q, level_map, lev))
                           for lev in (0, 1)])


def _target(rng, n=3):
    pi = rng.uniform(0.5, 1.5, 2 * n)
    pi = pi / pi.sum()
    W = rng.uniform(0.4, 2.0, (n, n))
    return bipartite_generator_from_coupling(pi[:n], pi[n:], W)


def _rank_k_coupling(rank, rng, n=3):
    """Strictly positive n x n flux matrix of exactly `rank`."""
    W = np.zeros((n, n))
    for _ in range(rank):
        W += np.outer(rng.uniform(0.3, 2.0, n), rng.uniform(0.3, 2.0, n))
    return W


# --- the chart --------------------------------------------------------------

def test_coupling_chart_round_trips():
    rng = np.random.default_rng(0)
    pi = rng.uniform(0.5, 1.5, 6)
    pi = pi / pi.sum()
    W = rng.uniform(0.4, 2.0, (3, 3))
    Q = bipartite_generator_from_coupling(pi[:3], pi[3:], W)

    pi_E, pi_F, W_back = coupling_from_generator(Q, LEV6)
    assert np.allclose(pi_E, pi[:3]) and np.allclose(pi_F, pi[3:])
    assert np.allclose(W_back, W)
    assert is_detailed_balance(Q, atol=1e-12)


def test_coupling_chart_rejects_intra_level_edges():
    """CHAIN has 0-1 and 2-3 edges inside the levels, so it is not bipartite and
    its dwells are not single exponentials. The chart must refuse it."""
    chain = n_state_generator(
        {(0, 1): 5.0, (1, 0): 5.0, (1, 2): 1.0, (2, 1): 1.0,
         (2, 3): 4.0, (3, 2): 4.0}, n_states=4)
    with pytest.raises(ValueError, match="intra-level"):
        coupling_from_generator(chain, LEVELS_4)


# --- instrument: the detector must separate all three ranks -----------------

def test_gateway_rank_separates_rank_one_two_and_three():
    """A rank-2 3x3 gateway has sigma_2/sigma_1 healthy and sigma_3/sigma_1 ~ 0,
    a signature distinct from both rank 1 and rank 3. Checked before any null
    result is read from this detector."""
    rng = np.random.default_rng(20)
    for want in (1, 2, 3):
        W = _rank_k_coupling(want, rng)
        pi = rng.uniform(0.3, 1.0, 6)
        pi = pi / pi.sum()
        Q = bipartite_generator_from_coupling(pi[:3], pi[3:], W)

        assert np.linalg.matrix_rank(W) == want
        assert min_gateway_rank(Q, LEV6) == want
        ratios = gateway_singular_ratios(Q, LEV6)[0]
        assert np.sum(ratios > 1e-6) == want
        assert np.all(ratios[want:] < 1e-10)


def test_joint_rank_follows_gateway_rank_only_when_rates_are_distinct():
    """Joint rank = gateway rank when all sub-state exit rates differ, because
    the joint is sum_jk M_jk exp(-S_j t_f) exp(-R_k t_e) with M a diagonal
    rescaling of the gateway. Collide two rates and the exponentials stop being
    independent, so the joint drops while the gateway does not."""
    rng = np.random.default_rng(21)
    W = _rank_k_coupling(3, rng)
    D = W.sum(axis=1)

    pi = rng.uniform(0.3, 1.0, 6)
    pi = pi / pi.sum()
    Q = bipartite_generator_from_coupling(pi[:3], pi[3:], W)
    assert min_gateway_rank(Q, LEV6) == 3 and joint_density_rank(Q, LEV6) == 3

    pi[1] = pi[0] * D[1] / D[0]          # forces exit rates R_0 == R_1
    pi = pi / pi.sum()
    Q_deg = bipartite_generator_from_coupling(pi[:3], pi[3:], W)
    rates = -np.diag(Q_deg)[:3]
    assert abs(rates[0] - rates[1]) < 1e-9 * rates[0]
    assert min_gateway_rank(Q_deg, LEV6) == 3      # gateway untouched
    assert joint_density_rank(Q_deg, LEV6) == 2   # joint collapses


# --- the two extremes of the polytope ---------------------------------------

def test_independence_partner_is_rank_one_with_zero_rho():
    rng = np.random.default_rng(1)
    target = _target(rng)
    Q0 = independence_partner(target, LEV6)

    assert abs(analytic_adjacent_moments(Q0, LEV6, 1).rho) < 1e-14
    assert min_gateway_rank(Q0, LEV6) == 1
    assert joint_density_rank(Q0, LEV6) == 1
    assert np.max(np.abs(_signature(Q0, LEV6) - _signature(target, LEV6))) < 1e-9


@pytest.mark.parametrize("deviation_rank, want_joint_rank", [(1, 2), (2, 3)])
def test_blind_spot_exists_at_three_substates(deviation_rank, want_joint_rank):
    """rho = 0 exactly, marginals identical, yet the joint has rank > 1.

    Paired with independence_partner this is a genuine degenerate pair: same
    1-D dwell densities, same (zero) adjacent-dwell correlation, different 2-D
    joints. Correlation cannot separate them; a rank test can.
    """
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        target = _target(rng)
        blind = zero_correlation_partner(target, LEV6,
                                         deviation_rank=deviation_rank,
                                         seed=seed)

        assert abs(analytic_adjacent_moments(blind, LEV6, 1).rho) < 1e-12
        assert joint_density_rank(blind, LEV6) == want_joint_rank
        assert min_gateway_rank(blind, LEV6) == want_joint_rank
        assert is_detailed_balance(blind, atol=1e-10)
        assert np.max(np.abs(_signature(blind, LEV6)
                             - _signature(target, LEV6))) < 1e-9

        # Indistinguishable from the rank-1 partner by marginals AND by rho.
        Q0 = independence_partner(target, LEV6)
        assert joint_density_rank(Q0, LEV6) == 1
        assert np.max(np.abs(_signature(blind, LEV6)
                             - _signature(Q0, LEV6))) < 1e-9


def test_blind_spot_is_cancellation_not_degeneracy():
    """GUARDRAIL: classify the branch rather than merely reporting rho = 0.

    Degeneracy would mean some sub-state dwell rates coincide or the deviation
    from the product coupling vanishes -- the K(2,2) story repeating, and worth
    nothing. Cancellation means all rates stay distinct and W~ stays large
    while x^T W~ y happens to vanish. Only the latter is the blind spot.
    """
    for seed in range(8):
        rng = np.random.default_rng(200 + seed)
        target = _target(rng)
        blind = zero_correlation_partner(target, LEV6, deviation_rank=2,
                                         seed=seed)

        rates = -np.diag(blind)
        for level_rates in (rates[:3], rates[3:]):
            gaps = np.diff(np.sort(level_rates))
            assert np.min(gaps) > 1e-6 * np.max(level_rates), "rates collided"

        pi_E, pi_F, W = coupling_from_generator(blind, LEV6)
        W0 = np.outer(W.sum(axis=1), W.sum(axis=0)) / W.sum()
        deviation = W - W0
        assert np.linalg.norm(deviation) > 1e-3 * np.linalg.norm(W0)
        s = np.linalg.svd(deviation, compute_uv=False)
	assert s[1] / s[0] > 1e-8
	assert s[2] / s[0] < 1e-12


def test_no_blind_spot_at_two_substates_by_construction():
    """The same constructor, run at K(2,2), must refuse -- and say why.

    At two sub-states the perturbation space is one dimensional, so the single
    linear condition a^T T b = 0 annihilates it. This is the constructive form
    of the K(2,2) theorem: not that the search failed, but that the solution
    set is a point and that point is the product coupling.
    """
    q22 = bipartite_reversible_generator(np.log([0.3, 0.2, 0.25, 0.25]),
                                         np.log([1.0, 0.4, 0.7, 2.0]))
    with pytest.raises(ValueError, match="two sub-states per level"):
        zero_correlation_partner(q22, LEVELS_4, deviation_rank=1)


# --- sizing the chi-squared experiment, exactly, before running it ----------
#
# joint_mixture_form is the 2-D analogue of dwell.mixture_params: diagonalise
# both blocks and keep the joint instead of marginalising it away. Checked
# against mixture_params itself before anything is built on it, the same
# instrument-first discipline as gateway_rank_ratio above.

def test_joint_mixture_form_marginals_match_mixture_params():
    rng = np.random.default_rng(40)
    for _ in range(20):
        pi = rng.uniform(0.5, 1.5, 6)
        pi = pi / pi.sum()
        W = rng.uniform(0.3, 3.0, (3, 3))
        Q = bipartite_generator_from_coupling(pi[:3], pi[3:], W)

        rF, rE, B = joint_mixture_form(Q, LEV6, level=1)
        aF = (B / rE[None, :]).sum(axis=1)
        r_mix, c_mix = mixture_params(Q, LEV6, level=1)

        oa, ob = np.argsort(rF), np.argsort(r_mix)
        assert np.allclose(np.sort(rF), np.sort(r_mix), atol=1e-9)
        assert np.allclose(aF[oa], c_mix[ob], atol=1e-9)


def test_chi2_noncentrality_zero_on_rank_one_positive_from_three():
    """lambda tracks the rank test exactly: ~0 on the independence partner,
    strictly positive on both the target and a genuine blind spot."""
    rng = np.random.default_rng(41)
    pi = rng.uniform(0.5, 1.5, 6)
    pi = pi / pi.sum()
    W = rng.uniform(0.3, 3.0, (3, 3))
    target = bipartite_generator_from_coupling(pi[:3], pi[3:], W)
    flat = independence_partner(target, LEV6)
    blind = zero_correlation_partner(target, LEV6, deviation_rank=2, seed=0)

    assert chi2_noncentrality(flat, LEV6, n_bins=6) < 1e-12
    assert chi2_noncentrality(target, LEV6, n_bins=6) > 1e-8
    assert chi2_noncentrality(blind, LEV6, n_bins=6) > 1e-12
    assert abs(analytic_adjacent_moments(blind, LEV6, 1).rho) < 1e-12


def test_pairs_needed_chi2_is_monotone_in_power_and_matches_ncx2():
    """Higher power demands more pairs, and the returned n actually clears the
    noncentral-chi2 power equation it was solved from -- not merely plausible
    numbers, checked against scipy's own noncentral chi2 survival function."""
    rng = np.random.default_rng(42)
    pi = rng.uniform(0.5, 1.5, 6)
    pi = pi / pi.sum()
    W = rng.uniform(0.3, 3.0, (3, 3))
    target = bipartite_generator_from_coupling(pi[:3], pi[3:], W)
    blind = zero_correlation_partner(target, LEV6, deviation_rank=2, seed=1)

    n50 = pairs_needed_chi2(blind, LEV6, n_bins=6, power=0.5)
    n90 = pairs_needed_chi2(blind, LEV6, n_bins=6, power=0.9)
    n95 = pairs_needed_chi2(blind, LEV6, n_bins=6, power=0.95)
    assert n50 < n90 < n95

    dof = 25
    lam = chi2_noncentrality(blind, LEV6, n_bins=6)
    crit = stats.chi2.isf(2.0 * stats.norm.sf(5.0), dof)
    for n, power in ((n50, 0.5), (n90, 0.9), (n95, 0.95)):
        assert stats.ncx2.sf(crit, dof, lam * n) == pytest.approx(power, abs=1e-4)


def test_pairs_needed_chi2_is_effectively_infinite_on_rank_one():
    rng = np.random.default_rng(43)
    pi = rng.uniform(0.5, 1.5, 6)
    pi = pi / pi.sum()
    W = rng.uniform(0.3, 3.0, (3, 3))
    flat = independence_partner(bipartite_generator_from_coupling(
        pi[:3], pi[3:], W), LEV6)
    assert pairs_needed_chi2(flat, LEV6, n_bins=6) > 10 ** 15
