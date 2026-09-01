"""Dwell-time aggregation: 2-state oracle, merge logic, censoring."""

import numpy as np
import pytest

from stochastic_ctmc.generator import two_state_generator, n_state_generator
from stochastic_ctmc.gillespie import gillespie_ssa, SSAResult
from stochastic_ctmc.dwell import aggregate_dwells, dwell_runs


# --- Merge logic, deterministic (no RNG) ------------------------------------

def test_merge_absorbs_hidden_transitions():
    # States 0,1 -> level 0; states 2,3 -> level 1.
    # Path: 0 -(1)-> 1 -(1)-> 2 -(1)-> 3 -(1)-> 0, unit dwell each.
    ssa = SSAResult(
        times=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        states=np.array([0, 1, 2, 3, 0]),
        t_max=5.0,
        absorbed=False,
    )
    levels, times = dwell_runs(ssa, [0, 0, 1, 1])
    # 0->1 is hidden (both level 0), 2->3 is hidden (both level 1).
    assert np.array_equal(levels, [0, 1, 0])
    assert np.allclose(times, [2.0, 2.0, 1.0])  # last run truncated at t_max


def test_merge_is_noop_when_every_transition_flips_level():
    ssa = SSAResult(
        times=np.array([0.0, 1.0, 2.0]),
        states=np.array([0, 1, 0]),
        t_max=3.0,
        absorbed=False,
    )
    levels, times = dwell_runs(ssa, [0, 1])
    assert np.array_equal(levels, [0, 1, 0])
    assert np.allclose(times, [1.0, 1.0, 1.0])


def test_censoring_drops_first_and_last():
    ssa = SSAResult(
        times=np.array([0.0, 1.0, 2.0, 3.0]),
        states=np.array([0, 1, 0, 1]),
        t_max=4.0,
        absorbed=False,
    )
    d0, d1 = aggregate_dwells(ssa, [0, 1])
    # Runs are [0,1,0,1]; drop run 0 and run 3 -> one level-1 and one level-0.
    assert np.allclose(d0, [1.0])
    assert np.allclose(d1, [1.0])

    d0f, d1f = aggregate_dwells(ssa, [0, 1], drop_first=False, drop_last=False)
    assert d0f.shape == (2,) and d1f.shape == (2,)


def test_short_trajectory_returns_empty():
    ssa = SSAResult(times=np.array([0.0]), states=np.array([0]),
                    t_max=1.0, absorbed=True)
    d0, d1 = aggregate_dwells(ssa, [0, 1])
    assert d0.size == 0 and d1.size == 0


def test_level_map_too_short_is_rejected():
    ssa = SSAResult(times=np.array([0.0, 1.0]), states=np.array([0, 2]),
                    t_max=2.0, absorbed=False)
    with pytest.raises(ValueError):
        dwell_runs(ssa, [0, 1])


# --- Statistical oracle: 2-state, exit rate sets the dwell ------------------

@pytest.mark.slow
def test_two_state_dwells_are_exponential_with_exit_rates():
    k_c, k_e = 1.0e3, 2.0e3          # empty->filled, filled->empty
    Q = two_state_generator(k_c, k_e)
    # Mean cycle = 1/k_c + 1/k_e = 1.5 ms; 15 s => ~10^4 cycles.
    ssa = gillespie_ssa(Q, initial_state=0, t_max=15.0, rng=0)
    d_empty, d_filled = aggregate_dwells(ssa, [0, 1])

    assert d_empty.size > 5000 and d_filled.size > 5000

    # Dwell in a state is Exp(exit rate). Empty exits by capture, filled by
    # emission -- NOT the rate that entered the state.
    assert np.isclose(d_empty.mean(), 1.0 / k_c, rtol=0.05)
    assert np.isclose(d_filled.mean(), 1.0 / k_e, rtol=0.05)

    # Exponential: std == mean.
    assert np.isclose(d_empty.std(), 1.0 / k_c, rtol=0.05)
    assert np.isclose(d_filled.std(), 1.0 / k_e, rtol=0.05)


@pytest.mark.slow
def test_two_state_matches_unmerged_oracle():
    # For a 2-state chain the merge must be a no-op: reproduce the raw
    # inter-transition durations exactly.
    k = 3.0
    Q = two_state_generator(k, k)
    ssa = gillespie_ssa(Q, initial_state=0, t_max=30000.0, rng=1)

    edges = np.concatenate([ssa.times, [ssa.t_max]])
    durations = np.diff(edges)
    # Drop the censored final interval FIRST, then filter by state. Filtering
    # first and then dropping discards a complete dwell whenever the trajectory
    # happens to end in the other state.
    raw_in_0 = durations[:-1][ssa.states[:-1] == 0]

    d0, _ = aggregate_dwells(ssa, [0, 1], drop_first=False, drop_last=True)
    assert np.array_equal(d0, raw_in_0)


@pytest.mark.slow
def test_aggregated_four_state_dwell_is_not_exponential():
    # Chain E2 <-> E1 <-> F1 <-> F2, levels [0,0,1,1].
    # Escaping the filled level requires reaching F1 first, so the aggregated
    # filled dwell is a sum/mixture of exponentials -- std/mean != 1.
    Q = n_state_generator(
        {(0, 1): 5.0, (1, 0): 5.0,      # E2 <-> E1
         (1, 2): 1.0, (2, 1): 1.0,      # E1 <-> F1  (the single gateway)
         (2, 3): 4.0, (3, 2): 4.0},     # F1 <-> F2
        n_states=4,
    )
    ssa = gillespie_ssa(Q, initial_state=1, t_max=20000.0, rng=2)
    _, d_filled = aggregate_dwells(ssa, [0, 0, 1, 1])
    assert d_filled.size > 1000
    cv = d_filled.std() / d_filled.mean()
    assert cv > 1.05  # hyperexponential, not a single Exp