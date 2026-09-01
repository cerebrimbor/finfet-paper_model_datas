"""Phase 4 tests: the RTN <-> CTMC structural identity (item 15).

There is no SPICE leg to test: DVTSHIFT is an exact gate-voltage offset, so the
device is a memoryless nonlinearity captured by one DC sweep (see
``device_provenance/``), and the subprocess wrapper it would have needed was
removed rather than left as dead scaffolding.
"""

import numpy as np

from stochastic_ctmc.config import RTNConfig
from stochastic_ctmc.generator import two_state_generator
from stochastic_ctmc.hh import gate_rates
from stochastic_ctmc.circuit import (
    rtn_generator,
    simulate_rtn,
    rtn_stationary_filled,
    STRUCTURAL_IDENTITY,
)


# --- Item 15: the structural identity ---------------------------------------

def test_rtn_generator_is_literally_the_two_state_generator():
    kc, ke = 1.0e3, 2.0e3
    assert np.array_equal(rtn_generator(kc, ke), two_state_generator(kc, ke))


def test_rtn_trap_shares_structure_with_ion_channel_gate():
    # If we pick trap rates equal to an HH gate's rates at some voltage, the two
    # generators are identical -- an ion channel and a transistor trap become the
    # same CTMC.
    r = gate_rates(-40.0)
    trap = rtn_generator(r.alpha_m, r.beta_m)
    gate = two_state_generator(r.alpha_m, r.beta_m)
    assert np.array_equal(trap, gate)


def test_structural_identity_covers_all_four_domains():
    domains = {d.domain for d in STRUCTURAL_IDENTITY}
    assert len(domains) == 4
    # Every domain maps state1 to the "active" level that modulates its observable.
    for d in STRUCTURAL_IDENTITY:
        assert d.a_meaning and d.b_meaning and d.observable


def test_rtn_occupancy_matches_ctmc_stationary():
    cfg = RTNConfig(capture_rate=1.0e3, emission_rate=3.0e3, delta_vth=5e-3)
    trace = simulate_rtn(cfg, t_max=2.0, rng=0)  # 2 s is many trap lifetimes
    p_filled = rtn_stationary_filled(cfg)         # = kc/(kc+ke) = 0.25
    assert np.isclose(p_filled, 0.25)
    assert np.isclose(trace.stationary_filled_fraction(), p_filled, atol=0.02)


def test_vth_signal_is_two_level_telegraph():
    cfg = RTNConfig(capture_rate=2.0e3, emission_rate=2.0e3, delta_vth=5e-3)
    trace = simulate_rtn(cfg, t_max=0.05, rng=1)
    t = np.linspace(0, 0.05, 500)
    sig = trace.vth_signal(t)
    levels = np.unique(np.round(sig, 12))
    assert set(levels).issubset({0.0, cfg.delta_vth})
    assert len(levels) == 2  # both levels actually visited on this window
