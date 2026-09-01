"""Phase 3 tests: deterministic-limit recovery (item 11) and cross-validation of
a voltage-clamped HH gate against the Phase 1/2 CTMC (item 12)."""

import dataclasses

import numpy as np
import pytest

from stochastic_ctmc.config import HHConfig
from stochastic_ctmc.generator import two_state_generator, stationary_distribution
from stochastic_ctmc.hh import (
    gate_rates,
    steady_state_gate,
    simulate_hh,
    FOX_LU_STATUS,
)


def test_item9_is_flagged():
    # The blocker is surfaced in code, not silently ignored.
    assert "BLOCKED" in FOX_LU_STATUS


# --- Item 11: deterministic HH recovered at the zero-noise limit -------------

def test_deterministic_hh_fires_action_potential():
    cfg = HHConfig()
    res = simulate_hh(cfg, t_max=50.0, dt=0.01, stochastic=False)
    # Standard params with I_ext=10 uA/cm^2 produce spiking: V crosses well above 0.
    assert res.V.max() > 20.0
    # Gating variables stay in [0, 1] throughout.
    for g in (res.m, res.h, res.n):
        assert g.min() >= -1e-9 and g.max() <= 1 + 1e-9


def test_large_N_stochastic_approaches_deterministic():
    # Zero-noise limit is N -> inf. A huge channel count should track the
    # deterministic trajectory closely.
    det = simulate_hh(HHConfig(), t_max=20.0, dt=0.01, stochastic=False)
    big = dataclasses.replace(HHConfig(), n_na_channels=10_000_000, n_k_channels=3_000_000)
    sto = simulate_hh(big, t_max=20.0, dt=0.01, stochastic=True, rng=0)
    rmsd = np.sqrt(np.mean((det.V - sto.V) ** 2))
    assert rmsd < 2.0  # mV; small relative to the ~100 mV spike amplitude


def test_zero_noise_flag_matches_deterministic_exactly():
    cfg = HHConfig()
    a = simulate_hh(cfg, t_max=10.0, dt=0.01, stochastic=False, rng=0)
    b = simulate_hh(cfg, t_max=10.0, dt=0.01, stochastic=False, rng=999)
    # With noise off, the RNG is irrelevant: trajectories are identical.
    assert np.array_equal(a.V, b.V) and np.array_equal(a.m, b.m)


# --- Item 12: voltage-clamped gate == Phase 1/2 two-state CTMC ---------------

@pytest.mark.slow
def test_clamped_gate_stationary_matches_ctmc():
    V_clamp = -40.0
    r = gate_rates(V_clamp)
    a, b = r.alpha_m, r.beta_m          # m-gate rates at the clamp voltage (1/ms)

    # Phase 1 prediction: stationary open fraction = pi[1] of the 2-state CTMC.
    Q = two_state_generator(a, b)
    pi = stationary_distribution(Q)
    p_star = pi[1]
    assert np.isclose(p_star, steady_state_gate(a, b))

    # Small channel count so the stationary variance is measurable.
    N = 200
    cfg = dataclasses.replace(HHConfig(), n_na_channels=N)

    t_max, dt, n_seeds = 40.0, 0.01, 300
    finals = np.empty(n_seeds)
    for i in range(n_seeds):
        res = simulate_hh(cfg, t_max=t_max, dt=dt, stochastic=True, rng=i, clamp_V=V_clamp)
        finals[i] = res.m[-1]           # stationary m after many relaxation times

    # Mean matches the CTMC stationary occupancy.
    assert np.isclose(finals.mean(), p_star, atol=0.02)

    # Variance matches the binomial channel law p(1-p)/N (the Phase 2 identity).
    expected_var = p_star * (1 - p_star) / N
    assert np.isclose(finals.var(), expected_var, rtol=0.35)


def test_clamped_gate_relaxes_from_zero():
    # A gate started away from steady state relaxes to alpha/(alpha+beta).
    V_clamp = -50.0
    r = gate_rates(V_clamp)
    a, b = r.alpha_n, r.beta_n
    p_star = steady_state_gate(a, b)

    cfg = dataclasses.replace(HHConfig(), n_k_channels=5000)
    # Average a handful of seeds to suppress noise, then check the endpoint mean.
    vals = [simulate_hh(cfg, 30.0, 0.01, stochastic=True, rng=s, clamp_V=V_clamp).n[-1]
            for s in range(40)]
    assert np.isclose(np.mean(vals), p_star, atol=0.03)
