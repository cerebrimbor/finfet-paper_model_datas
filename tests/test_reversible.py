"""Does the degenerate pair survive the reversibility constraint?

Kienker (1989): reversible aggregated Markov models admit equivalence classes.
If distinguishability requires irreversibility, the claim is 'we detect net
cyclic flux', not 'we detect topology'. This file decides which.
"""

import numpy as np
import pytest

from stochastic_ctmc.generator import (
    n_state_generator, is_detailed_balance, stationary_distribution,
)
from stochastic_ctmc.gillespie import gillespie_ssa
from stochastic_ctmc.dwell import gateway_block_rank, joint_density_rank, dwell_pdf_1d
from stochastic_ctmc.topology import (
    bipartite_reversible_generator,
    match_marginals_reversible,
    explore_reversible_manifold,
    max_correlation_reversible,
    LEVELS_4,
)
from stochastic_ctmc.correlation import (
    analytic_adjacent_moments, adjacent_pairs, empirical_correlation, pairs_needed,
)

CHAIN = n_state_generator(
    {(0, 1): 5.0, (1, 0): 5.0,
     (1, 2): 1.0, (2, 1): 1.0,
     (2, 3): 4.0, (3, 2): 4.0},
    n_states=4,
)
LEVELS = list(LEVELS_4)


# --- The parameterisation is reversible by construction ---------------------

def test_reversible_parameterisation_always_satisfies_detailed_balance():
    rng = np.random.default_rng(0)
    for _ in range(20):
        Q = bipartite_reversible_generator(rng.normal(0, 1, 4),
                                           np.log(rng.uniform(0.1, 5, 4)))
        assert is_detailed_balance(Q, atol=1e-8)


def test_reversible_parameterisation_recovers_its_own_pi():
    log_pi = np.array([0.3, -0.5, 0.8, -0.1])
    pi_expected = np.exp(log_pi) / np.exp(log_pi).sum()
    Q = bipartite_reversible_generator(log_pi, np.log([1.0, 2.0, 0.5, 1.5]))
    assert np.allclose(stationary_distribution(Q), pi_expected)


def test_log_pi_gauge_shift_leaves_Q_unchanged():
    # pi is normalised after exp, so log_pi -> log_pi + const is a flat
    # direction. This is exactly why optimisers must fix the gauge.
    lc = np.log([1.0, 2.0, 0.5, 1.5])
    a = bipartite_reversible_generator(np.array([0.3, -0.5, 0.8, -0.1]), lc)
    b = bipartite_reversible_generator(np.array([0.3, -0.5, 0.8, -0.1]) + 7.0, lc)
    assert np.allclose(a, b)


def test_reversible_parameterisation_rejects_bad_input():
    with pytest.raises(ValueError):
        bipartite_reversible_generator([0.0, 0.0], np.log([1.0, 2.0, 0.5, 1.5]))
    with pytest.raises(ValueError):
        bipartite_reversible_generator([0.0, np.inf, 0.0, 0.0], np.zeros(4))
    with pytest.raises(ValueError):
        bipartite_reversible_generator([0.0, 500.0, 0.0, 0.0], np.zeros(4))


def test_reversible_bipartite_still_has_two_gateways():
    Q = bipartite_reversible_generator([0.3, -0.5, 0.8, -0.1],
                                       np.log([1.0, 2.0, 0.5, 1.5]))
    assert gateway_block_rank(Q, LEVELS) == 2


def test_a_reversible_bipartite_can_be_correlated():
    # Existence, independent of any marginal matching: reversibility alone does
    # NOT force rho = 0.
    Q = bipartite_reversible_generator([0.3, -0.9, 1.2, -0.4],
                                       np.log([1.0, 3.0, 0.4, 2.0]))
    assert is_detailed_balance(Q, atol=1e-8)
    assert abs(analytic_adjacent_moments(Q, LEVELS, 1).rho) > 1e-3


# --- THE QUESTION -----------------------------------------------------------

@pytest.mark.slow
def test_reversible_degenerate_pair_exists():
    res = match_marginals_reversible(CHAIN, LEVELS)
    assert res.success, f"no reversible match: {res!r}"
    assert is_detailed_balance(res.Q, atol=1e-8)
    assert gateway_block_rank(res.Q, LEVELS) == 2

    t = np.linspace(0.0, 12.0, 400)
    for lev in (0, 1):
        assert np.max(np.abs(dwell_pdf_1d(CHAIN, LEVELS, t, lev)
                             - dwell_pdf_1d(res.Q, LEVELS, t, lev))) < 1e-7


@pytest.mark.slow
def test_manifold_is_one_dimensional_and_rho_varies_along_it():
    # 7 free parameters, 6 independent constraints => a curve of solutions.
    # If rho were constant along it, 'the' correlation would be well defined;
    # it is not, so the paper must report a range or an optimum.
    sols = explore_reversible_manifold(CHAIN, LEVELS, n_samples=120)
    assert len(sols) >= 10, f"only {len(sols)} feasible points found"
    rhos = np.array([s.rho for s in sols])
    print(f"\nmanifold: {len(sols)} points, "
          f"rho in [{rhos.min():+.5f}, {rhos.max():+.5f}], "
          f"|rho| max = {np.abs(rhos).max():.5f}")
    for s in sols:
        assert is_detailed_balance(s.Q, atol=1e-8)
        assert s.signature_error < 1e-7


@pytest.mark.slow
def test_best_reversible_partner_is_correlated_and_measurable():
    """The headline number: how strong is the 2-D signal when irreversibility
    is ruled out as an explanation?"""
    res = max_correlation_reversible(CHAIN, LEVELS)
    assert res.success, f"no feasible partner found: {res!r}"
    assert is_detailed_balance(res.Q, atol=1e-8)
    assert joint_density_rank(res.Q, LEVELS) == 2
    assert joint_density_rank(CHAIN, LEVELS) == 1

    n_req = pairs_needed(res.rho, n_sigma=5.0)
    print(f"\n--- reversible degenerate pair ---")
    print(f"rho              = {res.rho:.6f}")
    print(f"pairs @ 5 sigma  = {n_req:,}")
    print(f"marginal error   = {res.signature_error:.2e}")
    print(f"Q_matched =\n{np.array2string(res.Q, precision=4)}")

    assert abs(res.rho) > 1e-3, (
        f"best reversible partner has rho={res.rho:.2e}: the correlation test "
        "cannot separate them. The 2-D joint still differs in rank, so a rank "
        "test may work -- but the simple correlation statistic does not."
    )


@pytest.mark.slow
def test_reversible_pair_separates_in_simulation():
    res = max_correlation_reversible(CHAIN, LEVELS)
    assert res.success and abs(res.rho) > 1e-3

    n_req = pairs_needed(res.rho, n_sigma=5.0)
    m = analytic_adjacent_moments(res.Q, LEVELS, 1)
    t_max = float(np.clip(4.0 * n_req * (m.mean_f + m.mean_e), 5000.0, 600_000.0))

    est_c = empirical_correlation(
        *adjacent_pairs(gillespie_ssa(CHAIN, 1, t_max, rng=31), LEVELS, 1))
    est_m = empirical_correlation(
        *adjacent_pairs(gillespie_ssa(res.Q, 2, t_max, rng=32), LEVELS, 1))

    print(f"\nchain   : rho={est_c.rho:+.5f}  z={est_c.z_score:+.2f}  n={est_c.n_pairs:,}")
    print(f"matched : rho={est_m.rho:+.5f}  z={est_m.z_score:+.2f}  n={est_m.n_pairs:,}")

    assert not est_c.excludes_zero
    assert est_m.excludes_zero