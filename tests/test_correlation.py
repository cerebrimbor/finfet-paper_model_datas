"""Adjacent-dwell correlation: analytic moments, simulator agreement, and the
detectability of the degenerate pair."""

import numpy as np
import pytest

from stochastic_ctmc.generator import (
    two_state_generator, n_state_generator, is_detailed_balance,
)
from stochastic_ctmc.gillespie import gillespie_ssa
from stochastic_ctmc.dwell import aggregate_dwells, joint_density_rank
from stochastic_ctmc.topology import match_marginals, bipartite_generator, LEVELS_4
from stochastic_ctmc.correlation import (
    adjacent_pairs,
    analytic_adjacent_moments,
    empirical_correlation,
    pairs_needed,
)

CHAIN = n_state_generator(
    {(0, 1): 5.0, (1, 0): 5.0,
     (1, 2): 1.0, (2, 1): 1.0,
     (2, 3): 4.0, (3, 2): 4.0},
    n_states=4,
)
LEVELS = list(LEVELS_4)


# --- Pairing bookkeeping (deterministic) ------------------------------------

def test_adjacent_pairs_alternate_and_are_censored():
    from stochastic_ctmc.gillespie import SSAResult
    # Runs: [0(1s), 1(2s), 0(3s), 1(4s), 0(5s)] -- drop first and last.
    ssa = SSAResult(
        times=np.array([0.0, 1.0, 3.0, 6.0, 10.0]),
        states=np.array([0, 1, 0, 1, 0]),
        t_max=15.0,
        absorbed=False,
    )
    t_f, t_e = adjacent_pairs(ssa, [0, 1], first_level=1)
    # Surviving runs: 1(2s), 0(3s), 1(4s). Filled runs with a successor: 2s->3s.
    assert np.allclose(t_f, [2.0])
    assert np.allclose(t_e, [3.0])


def test_adjacent_pairs_short_trajectory_is_empty():
    from stochastic_ctmc.gillespie import SSAResult
    ssa = SSAResult(times=np.array([0.0]), states=np.array([0]),
                    t_max=1.0, absorbed=True)
    t_f, t_e = adjacent_pairs(ssa, [0, 1])
    assert t_f.size == 0 and t_e.size == 0


# --- Analytic moments against elementary cases ------------------------------

def test_two_state_moments_are_exponential_and_uncorrelated():
    k_c, k_e = 1.0, 2.0
    m = analytic_adjacent_moments(two_state_generator(k_c, k_e), [0, 1], level=1)
    assert np.isclose(m.mean_f, 1 / k_e)      # filled exits by emission
    assert np.isclose(m.mean_e, 1 / k_c)      # empty exits by capture
    assert np.isclose(m.var_f, 1 / k_e ** 2)  # Exp: var = mean^2
    assert np.isclose(m.var_e, 1 / k_c ** 2)
    assert np.isclose(m.rho, 0.0, atol=1e-12)


def test_chain_is_uncorrelated_because_joint_is_rank_one():
    assert joint_density_rank(CHAIN, LEVELS) == 1
    m = analytic_adjacent_moments(CHAIN, LEVELS, level=1)
    assert np.isclose(m.rho, 0.0, atol=1e-12)


def test_moments_are_consistent_with_pdf_quadrature():
    from stochastic_ctmc.dwell import dwell_pdf_1d
    Q = bipartite_generator([1.0, 0.5, 1.5, 2.5, 2.0, 3.0, 4.0, 2.0])
    t = np.linspace(0, 40, 40001)
    for lev, attr in ((1, "mean_f"), (0, "mean_e")):
        pdf = dwell_pdf_1d(Q, LEVELS, t, lev)
        quad = np.trapezoid(t * pdf, t)
        m = analytic_adjacent_moments(Q, LEVELS, level=1)
        assert np.isclose(quad, getattr(m, attr), rtol=1e-4)


def test_entry_law_into_E_is_consistent():
    # psi = phi (-Q_FF)^-1 Q_FE must equal the stationary entry law into E.
    from stochastic_ctmc.dwell import entry_distribution, _blocks
    Q = bipartite_generator([1.0, 0.5, 1.5, 2.5, 2.0, 3.0, 4.0, 2.0])
    phi = entry_distribution(Q, LEVELS, 1)
    _, _, Q_FF, Q_FE, _, _ = _blocks(Q, LEVELS, 1)
    psi = phi @ np.linalg.inv(-Q_FF) @ Q_FE
    assert np.allclose(psi, entry_distribution(Q, LEVELS, 0))


# --- Simulator must reproduce the analytic correlation ----------------------

@pytest.mark.slow
def test_simulated_correlation_matches_analytic_for_chain():
    m = analytic_adjacent_moments(CHAIN, LEVELS, level=1)
    ssa = gillespie_ssa(CHAIN, initial_state=1, t_max=30000.0, rng=11)
    t_f, t_e = adjacent_pairs(ssa, LEVELS, first_level=1)
    est = empirical_correlation(t_f, t_e)
    # True rho is exactly 0; the estimate must be within its own CI of it.
    assert est.ci_low < m.rho < est.ci_high
    assert abs(est.z_score) < 4.0


@pytest.mark.slow
def test_simulated_correlation_matches_analytic_for_bipartite():
    Q = bipartite_generator([1.0, 0.5, 1.5, 2.5, 2.0, 3.0, 4.0, 2.0])
    m = analytic_adjacent_moments(Q, LEVELS, level=1)
    assert abs(m.rho) > 1e-3, "test fixture must actually be correlated"

    ssa = gillespie_ssa(Q, initial_state=2, t_max=30000.0, rng=12)
    t_f, t_e = adjacent_pairs(ssa, LEVELS, first_level=1)
    est = empirical_correlation(t_f, t_e)
    assert est.ci_low < m.rho < est.ci_high


# --- THE EXPERIMENT ---------------------------------------------------------

@pytest.mark.slow
def test_degenerate_pair_is_detectable_from_finite_data():
    """Two topologies, identical 1-D marginals. One correlated, one not.
    Simulate both and separate them statistically."""
    res = match_marginals(CHAIN, LEVELS)
    assert res.success

    m_chain = analytic_adjacent_moments(CHAIN, LEVELS, level=1)
    m_match = analytic_adjacent_moments(res.Q, LEVELS, level=1)

    assert np.isclose(m_chain.rho, 0.0, atol=1e-12)
    assert abs(m_match.rho) > 1e-3, (
        f"matched generator has rho={m_match.rho:.2e} -- the 2-D signal is real "
        "but too weak for a correlation test; a rank test on the full joint "
        "would be needed instead"
    )

    n_req = pairs_needed(m_match.rho, n_sigma=5.0)
    assert n_req < 2_000_000, f"needs {n_req} pairs -- impractical"

    # Simulate ~4x the required pairs from each generator.
    t_max = 4.0 * n_req * (m_match.mean_f + m_match.mean_e)
    t_max = float(np.clip(t_max, 5000.0, 400_000.0))

    ssa_c = gillespie_ssa(CHAIN, initial_state=1, t_max=t_max, rng=21)
    est_c = empirical_correlation(*adjacent_pairs(ssa_c, LEVELS, first_level=1))

    ssa_m = gillespie_ssa(res.Q, initial_state=2, t_max=t_max, rng=22)
    est_m = empirical_correlation(*adjacent_pairs(ssa_m, LEVELS, first_level=1))

    # The chain's correlation is consistent with zero; the matched one is not.
    assert not est_c.excludes_zero, f"chain spuriously correlated: {est_c}"
    assert est_m.excludes_zero, f"matched pair undetected: {est_m}"

    # And the two estimates are themselves separated.
    assert abs(est_m.z_score) > 3.0


def test_pairs_needed_scales_as_rho_squared():
    # n - 3 = (n_sigma / arctanh rho)^2, so halving rho roughly quadruples n.
    assert pairs_needed(0.1) > pairs_needed(0.2) > pairs_needed(0.5)
    assert pairs_needed(0.0) > 10 ** 12
    r = 0.05
    assert np.isclose(pairs_needed(r, 5.0), 3 + (5.0 / np.arctanh(r)) ** 2, rtol=0.01)


# --- Reversibility: framing check, not a pass/fail ---------------------------

@pytest.mark.slow
def test_report_reversibility_of_the_degenerate_pair():
    """Kienker (1989): reversible aggregated Markov models admit equivalence
    classes under similarity transform. If CHAIN is reversible and the matched
    generator is not, the distinguishability may rest on irreversibility (net
    cyclic flux) rather than topology per se. Physically that is legitimate for
    RTN -- multi-state NMP trap models carry net flux under bias -- but the
    claim must be framed accordingly. This test records the fact; it does not
    fail on it.
    """
    res = match_marginals(CHAIN, LEVELS)
    assert res.success
    print(f"\nCHAIN reversible:   {is_detailed_balance(CHAIN)}")
    print(f"matched reversible: {is_detailed_balance(res.Q)}")
    print(f"rho(matched) = {analytic_adjacent_moments(res.Q, LEVELS, 1).rho:.6f}")
    print(f"pairs needed at 5 sigma: "
          f"{pairs_needed(analytic_adjacent_moments(res.Q, LEVELS, 1).rho):,}")