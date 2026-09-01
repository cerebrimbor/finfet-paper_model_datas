"""Colquhoun-Hawkes densities: analytic oracle, normalisation, and the
factorisation condition the whole topology argument rests on."""

import numpy as np
import pytest

from stochastic_ctmc.generator import two_state_generator, n_state_generator
from stochastic_ctmc.gillespie import gillespie_ssa
from stochastic_ctmc.dwell import (
    aggregate_dwells,
    entry_distribution,
    dwell_pdf_1d,
    dwell_pdf_2d,
    gateway_block_rank,
    joint_density_rank,
)

# --- Three 4-state topologies. States 0,1 = empty (level 0); 2,3 = filled. ---

# CHAIN: E2 <-> E1 <-> F1 <-> F2.
# Only F1 <-> E1 crosses the boundary => ONE gateway => rank(Q_FE) = 1.
CHAIN = n_state_generator(
    {(0, 1): 5.0, (1, 0): 5.0,
     (1, 2): 1.0, (2, 1): 1.0,
     (2, 3): 4.0, (3, 2): 4.0},
    n_states=4,
)

# TWO_GATEWAY: every filled state connects to every empty state, AND the two
# filled states have different total exit rates (5 vs 6), so Q_FF is not a
# multiple of the identity. Both conditions => correlated adjacent dwells.
TWO_GATEWAY = n_state_generator(
    {(0, 2): 1.0, (0, 3): 0.5,      # E1 -> F1, F2   (total 1.5)
     (1, 2): 1.5, (1, 3): 2.5,      # E2 -> F1, F2   (total 4.0)
     (2, 0): 2.0, (2, 1): 3.0,      # F1 -> E1, E2   (total 5.0)
     (3, 0): 4.0, (3, 1): 2.0},     # F2 -> E1, E2   (total 6.0)
    n_states=4,
)

# EQUAL_EXIT: two gateways, but both filled states have total exit rate 5 and
# there are no F1 <-> F2 transitions. Q_FF = -5 I, so exp(Q_FF t) = e^{-5t} I
# and the F-side factor is a scalar times a fixed vector. The counterexample
# proving rank(Q_FE) >= 2 is NOT sufficient for correlation.
EQUAL_EXIT = n_state_generator(
    {(0, 2): 1.0, (0, 3): 0.5,
     (1, 2): 1.5, (1, 3): 2.5,
     (2, 0): 2.0, (2, 1): 3.0,      # total 5.0
     (3, 0): 4.0, (3, 1): 1.0},     # total 5.0  <-- equal
    n_states=4,
)

LEVELS = [0, 0, 1, 1]


# --- 2-state: must collapse to the elementary exponential -------------------

def test_two_state_pdf_is_exponential():
    k_c, k_e = 1.0, 2.0
    Q = two_state_generator(k_c, k_e)
    t = np.linspace(0, 3, 50)
    assert np.allclose(dwell_pdf_1d(Q, [0, 1], t, level=1), k_e * np.exp(-k_e * t))
    assert np.allclose(dwell_pdf_1d(Q, [0, 1], t, level=0), k_c * np.exp(-k_c * t))


def test_two_state_entry_distribution_is_trivial():
    assert np.allclose(entry_distribution(two_state_generator(1.0, 2.0), [0, 1]), [1.0])


def test_two_state_ranks_are_one():
    Q = two_state_generator(1.0, 2.0)
    assert gateway_block_rank(Q, [0, 1]) == 1
    assert joint_density_rank(Q, [0, 1]) == 1


# --- Normalisation and consistency ------------------------------------------

def test_pdf_1d_integrates_to_one():
    t = np.linspace(0, 60, 60001)
    for Q in (CHAIN, TWO_GATEWAY, EQUAL_EXIT):
        for lev in (0, 1):
            assert np.isclose(np.trapezoid(dwell_pdf_1d(Q, LEVELS, t, lev), t),
                              1.0, atol=1e-4)


def test_pdf_2d_integrates_to_one():
    # Tolerance is set by trapezoid quadrature error at this dt, not by the
    # code: the density is sharply peaked at t=0. Truncation at t=40 is
    # negligible (slowest mode decays with rate ~0.5 => e^-20).
    t = np.linspace(0, 40, 2001)
    for Q in (CHAIN, TWO_GATEWAY, EQUAL_EXIT):
        joint = dwell_pdf_2d(Q, LEVELS, t, t)
        inner = np.trapezoid(joint, t, axis=1)
        assert np.isclose(np.trapezoid(inner, t), 1.0, atol=2e-3)


def test_pdf_2d_marginalises_to_pdf_1d():
    # Integrating out the adjacent dwell must return the 1-D density exactly:
    #   int f(t_f, t_e) dt_e = phi exp(Q_FF t_f) Q_FE (-Q_EE)^-1 Q_EF 1
    #                        = phi exp(Q_FF t_f) (-Q_FF) 1
    t_f = np.linspace(0.05, 3.0, 25)
    t_e = np.linspace(0, 60, 24001)
    for Q in (CHAIN, TWO_GATEWAY, EQUAL_EXIT):
        marg = np.trapezoid(dwell_pdf_2d(Q, LEVELS, t_f, t_e), t_e, axis=1)
        assert np.allclose(marg, dwell_pdf_1d(Q, LEVELS, t_f), rtol=1e-3)


# --- The central claim: what the 2-D histogram can and cannot see ------------

def _numeric_joint_rank(Q):
    t_f = np.linspace(0.05, 2.0, 20)
    t_e = np.linspace(0.05, 2.0, 20)
    joint = dwell_pdf_2d(Q, LEVELS, t_f, t_e)
    return joint, np.linalg.matrix_rank(joint, tol=1e-10 * joint.max())


def test_chain_one_gateway_factorises():
    assert gateway_block_rank(CHAIN, LEVELS) == 1
    assert joint_density_rank(CHAIN, LEVELS) == 1

    joint, rank = _numeric_joint_rank(CHAIN)
    assert rank == 1

    t = np.linspace(0.05, 2.0, 20)
    f_f = dwell_pdf_1d(CHAIN, LEVELS, t, level=1)
    f_e = dwell_pdf_1d(CHAIN, LEVELS, t, level=0)
    assert np.allclose(joint, np.outer(f_f, f_e), rtol=1e-8)


def test_two_gateway_does_not_factorise():
    assert gateway_block_rank(TWO_GATEWAY, LEVELS) == 2
    assert joint_density_rank(TWO_GATEWAY, LEVELS) == 2

    joint, rank = _numeric_joint_rank(TWO_GATEWAY)
    assert rank == 2

    t = np.linspace(0.05, 2.0, 20)
    f_f = dwell_pdf_1d(TWO_GATEWAY, LEVELS, t, level=1)
    f_e = dwell_pdf_1d(TWO_GATEWAY, LEVELS, t, level=0)
    assert not np.allclose(joint, np.outer(f_f, f_e), rtol=1e-3)


def test_equal_exit_rates_kill_correlation_despite_two_gateways():
    # The counterexample. rank(Q_FE) = 2, yet the joint is rank 1: with
    # Q_FF = -5 I the filled-side factor collapses to e^{-5 t_f} times a fixed
    # vector. Multiple gateways alone do NOT imply correlated dwells.
    assert gateway_block_rank(EQUAL_EXIT, LEVELS) == 2
    assert joint_density_rank(EQUAL_EXIT, LEVELS) == 1

    joint, rank = _numeric_joint_rank(EQUAL_EXIT)
    assert rank == 1

    t = np.linspace(0.05, 2.0, 20)
    f_f = dwell_pdf_1d(EQUAL_EXIT, LEVELS, t, level=1)
    f_e = dwell_pdf_1d(EQUAL_EXIT, LEVELS, t, level=0)
    assert np.allclose(joint, np.outer(f_f, f_e), rtol=1e-8)


def test_equal_exit_filled_dwell_is_a_single_exponential():
    # Corollary: with Q_FF = -5 I the aggregated filled dwell is Exp(5) exactly,
    # even though the filled level has two sub-states.
    t = np.linspace(0, 2, 200)
    assert np.allclose(dwell_pdf_1d(EQUAL_EXIT, LEVELS, t, level=1),
                       5.0 * np.exp(-5.0 * t))


# --- The rank verdict may not depend on the time unit -----------------------

@pytest.mark.parametrize("k", [1e-6, 1e-3, 1e3, 1e6])
def test_joint_density_rank_is_invariant_under_a_change_of_time_units(k):
    """Q -> kQ is the SAME physical system quoted in different time units, so no
    rank verdict may move. This was a real bug: joint_density_rank thresholded
    singular values against an ABSOLUTE tolerance, and because the Krylov basis
    has row i equal to v A^i, rescaling multiplies row i by k^i and drags the
    spectrum across a fixed cut. Note a bare sigma_i/sigma_1 ratio does not fix
    it either -- the matrix does not scale uniformly -- so the rows are
    normalised before the ratio is taken.
    """
    for Q in (CHAIN, TWO_GATEWAY, EQUAL_EXIT):
        assert joint_density_rank(k * Q, LEVELS) == joint_density_rank(Q, LEVELS)


def test_joint_density_rank_scale_invariant_on_random_generators():
    """The fixed topologies above have rank verdicts that are structurally
    forced; this sweeps generic rate values, where the verdict is decided
    numerically and a units-dependent cut would actually bite.
    """
    rng = np.random.default_rng(4242)
    for _ in range(40):
        edges = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 0), (2, 1), (3, 0), (3, 1)]
        Q = n_state_generator({e: float(r) for e, r in
                               zip(edges, np.exp(rng.uniform(-2, 2, 8)))},
                              n_states=4)
        base = joint_density_rank(Q, LEVELS)
        for k in (1e-6, 1e-3, 1e3, 1e6):
            assert joint_density_rank(k * Q, LEVELS) == base


# --- The analytic density must match the simulator --------------------------

@pytest.mark.slow
def test_pdf_1d_matches_gillespie_histogram():
    ssa = gillespie_ssa(CHAIN, initial_state=1, t_max=40000.0, rng=3)
    _, d_filled = aggregate_dwells(ssa, LEVELS)
    assert d_filled.size > 5000

    edges = np.linspace(0, 3.0, 31)
    widths = np.diff(edges)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # NOT density=True: that normalises over the bin RANGE, so mass beyond
    # t=3 would inflate every bin by 1/P(dwell <= 3). Normalise by the full
    # sample count instead.
    raw, _ = np.histogram(d_filled, bins=edges)
    density = raw / (d_filled.size * widths)
    sigma = np.sqrt(np.maximum(raw, 1)) / (d_filled.size * widths)

    pred = dwell_pdf_1d(CHAIN, LEVELS, centres, level=1)

    # Agreement within Poisson counting statistics -- a principled band that
    # scales with sample size, rather than a hand-picked rtol.
    mask = pred > 0.05 * pred.max()
    assert np.all(np.abs(density[mask] - pred[mask]) < 4.0 * sigma[mask])


@pytest.mark.slow
def test_gillespie_mean_dwell_matches_analytic_mean():
    t = np.linspace(0, 80, 80001)
    for init, Q in ((1, CHAIN), (0, TWO_GATEWAY), (0, EQUAL_EXIT)):
        pdf = dwell_pdf_1d(Q, LEVELS, t, level=1)
        mean_analytic = np.trapezoid(t * pdf, t)
        ssa = gillespie_ssa(Q, initial_state=init, t_max=30000.0, rng=5)
        _, d_filled = aggregate_dwells(ssa, LEVELS)
        assert np.isclose(d_filled.mean(), mean_analytic, rtol=0.05)