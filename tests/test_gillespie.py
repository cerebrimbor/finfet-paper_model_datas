"""Phase 1 tests for the Gillespie SSA.

The statistical tests use fixed seeds and generous tolerances so they are
deterministic and do not flake, while still being tight enough to catch a
genuinely wrong sampler.
"""

import numpy as np
import pytest

from stochastic_ctmc.generator import (
    two_state_generator,
    n_state_generator,
    stationary_distribution,
)
from stochastic_ctmc.gillespie import gillespie_ssa


# --- Stationary distribution match (ergodic average) ------------------------

def test_two_state_occupancy_matches_stationary():
    a, b = 1.0, 2.0
    Q = two_state_generator(a, b)
    pi = stationary_distribution(Q)

    res = gillespie_ssa(Q, initial_state=0, t_max=20000.0, rng=42)
    occ = res.occupancy_fractions(n_states=2)

    # One long trajectory should reproduce pi within ~1%.
    assert np.allclose(occ, pi, atol=0.01)


def test_three_state_occupancy_matches_stationary():
    Q = n_state_generator(
        {(0, 1): 1.0, (1, 0): 1.0, (1, 2): 2.0, (2, 1): 2.0, (0, 2): 0.5, (2, 0): 0.5},
        n_states=3,
    )
    pi = stationary_distribution(Q)

    res = gillespie_ssa(Q, initial_state=0, t_max=50000.0, rng=7)
    occ = res.occupancy_fractions(n_states=3)
    assert np.allclose(occ, pi, atol=0.015)


# --- Waiting-time distribution ---------------------------------------------

def test_exit_times_are_exponential_with_correct_rate():
    # From a symmetric 2-state chain, dwell times in state 0 are Exp(rate a0=k01).
    k = 3.0
    Q = two_state_generator(k, k)
    res = gillespie_ssa(Q, initial_state=0, t_max=30000.0, rng=1)

    edges = np.concatenate([res.times, [res.t_max]])
    durations = np.diff(edges)
    # The final interval is right-censored at t_max whatever state it is in:
    # drop it, then select. (Selecting then dropping discards a complete dwell
    # when the run ends in state 1.)
    dwell_in_0 = durations[:-1][res.states[:-1] == 0]

    # Mean of Exp(k) is 1/k.
    assert np.isclose(dwell_in_0.mean(), 1.0 / k, rtol=0.05)


# --- Ensemble distribution at fixed time ------------------------------------

def test_ensemble_relaxes_to_stationary():
    a, b = 1.0, 2.0
    Q = two_state_generator(a, b)
    pi = stationary_distribution(Q)

    rng = np.random.default_rng(123)
    t_obs = 20.0  # >> relaxation time 1/(a+b)
    n = 4000
    final_states = np.array(
        [gillespie_ssa(Q, 0, t_obs, rng=rng, validate=False).state_at(t_obs) for _ in range(n)]
    )
    frac1 = final_states.mean()
    assert abs(frac1 - pi[1]) < 0.03


# --- Reproducibility --------------------------------------------------------

def test_same_seed_gives_identical_trajectory():
    Q = two_state_generator(1.5, 0.5)
    r1 = gillespie_ssa(Q, 0, 100.0, rng=99)
    r2 = gillespie_ssa(Q, 0, 100.0, rng=99)
    assert np.array_equal(r1.times, r2.times)
    assert np.array_equal(r1.states, r2.states)


# --- Edge cases -------------------------------------------------------------

def test_absorbing_state_halts():
    Q = n_state_generator({(0, 1): 5.0}, n_states=2)  # state 1 absorbing
    res = gillespie_ssa(Q, initial_state=0, t_max=100.0, rng=3)
    assert res.absorbed
    assert res.states[-1] == 1  # ends trapped in the absorbing state


def test_start_in_absorbing_state_never_moves():
    Q = n_state_generator({(1, 0): 5.0}, n_states=2)  # state 0 absorbing
    res = gillespie_ssa(Q, initial_state=0, t_max=100.0, rng=3)
    assert res.absorbed
    assert len(res.states) == 1 and res.states[0] == 0


def test_zero_propensity_everywhere_is_absorbing():
    Q = n_state_generator({}, n_states=3)  # no transitions at all
    res = gillespie_ssa(Q, initial_state=2, t_max=50.0, rng=0)
    assert res.absorbed and res.states[-1] == 2


def test_rejects_bad_initial_state():
    Q = two_state_generator(1.0, 1.0)
    with pytest.raises(ValueError):
        gillespie_ssa(Q, initial_state=5, t_max=1.0)


def test_invalid_generator_is_rejected():
    bad = np.array([[-1.0, 1.0], [2.0, -1.0]])  # row 1 sums to +1
    with pytest.raises(Exception):
        gillespie_ssa(bad, initial_state=0, t_max=1.0)