"""Phase 5 tests: plotting (17), export (18), and the cross-phase consistency
capstone (19)."""

import dataclasses
import os

import numpy as np
import pytest

from stochastic_ctmc.config import HHConfig, RTNConfig
from stochastic_ctmc.generator import two_state_generator, stationary_distribution
from stochastic_ctmc.gillespie import gillespie_ssa
from stochastic_ctmc.sde import channel_langevin_terms, euler_maruyama, deterministic_relaxation
from stochastic_ctmc.ensemble import ensemble_em
from stochastic_ctmc.hh import gate_rates, simulate_hh
from stochastic_ctmc.circuit import simulate_rtn, rtn_stationary_filled
from stochastic_ctmc.plotting import (
    plot_trajectory_comparison, plot_error_band, plot_convergence, save_figure,
)
from stochastic_ctmc.export import (
    export_columns_csv, hdf5_available, export_bundle_hdf5, read_bundle_hdf5,
)


# --- Item 17: plotting ------------------------------------------------------

def test_plots_render_and_save(tmp_path):
    Q = two_state_generator(2.0, 1.0)
    t = np.linspace(0, 5, 200)
    ref = deterministic_relaxation(Q, 0.0, t)

    fig1 = plot_trajectory_comparison([("A", t, ref), ("B", t, ref * 0.9)])
    p1 = save_figure(fig1, os.path.join(tmp_path, "traj.png"))
    assert os.path.getsize(p1) > 0

    summ = ensemble_em(*channel_langevin_terms(Q, 300), x0=0.0, t_max=5.0, dt=2e-3,
                       n_seeds=20, grid=t, clip=(0, 1))
    fig2 = plot_error_band(summ, reference=ref)
    p2 = save_figure(fig2, os.path.join(tmp_path, "band.png"))
    assert os.path.getsize(p2) > 0

    fig3 = plot_convergence(np.array([1e-1, 1e-2, 1e-3]), np.array([3e-1, 1e-1, 3e-2]))
    p3 = save_figure(fig3, os.path.join(tmp_path, "conv.png"))
    assert os.path.getsize(p3) > 0


# --- Item 18: export --------------------------------------------------------

def test_csv_export_roundtrip(tmp_path):
    path = os.path.join(tmp_path, "traj.csv")
    export_columns_csv(path, {"t": np.linspace(0, 1, 5), "x": np.arange(5.0)})
    lines = open(path).read().splitlines()
    assert lines[0] == "t,x" and len(lines) == 6


def test_csv_export_rejects_ragged(tmp_path):
    with pytest.raises(ValueError):
        export_columns_csv(os.path.join(tmp_path, "bad.csv"),
                           {"a": np.zeros(3), "b": np.zeros(4)})


def test_hdf5_bundle_roundtrip(tmp_path):
    if not hdf5_available():
        pytest.skip("h5py not installed")
    path = os.path.join(tmp_path, "bundle.h5")
    data = {"gillespie": np.arange(10.0), "em": np.linspace(0, 1, 10)}
    export_bundle_hdf5(path, data, attrs={"generator": "two_state(2,1)"})
    back = read_bundle_hdf5(path)
    assert np.allclose(back["gillespie"], data["gillespie"])
    assert np.allclose(back["em"], data["em"])


# --- Item 19: cross-phase consistency capstone ------------------------------

@pytest.mark.slow
def test_one_generator_consistent_across_all_phases():
    """The same 2-state generator, read four different ways, must give the same
    stationary occupancy p* = a/(a+b)."""
    # Anchor the shared rates to an HH gate at a fixed voltage, so the generator
    # is simultaneously an HH-gate, a CTMC, an EM-Langevin, and an RTN trap.
    V = -45.0
    r = gate_rates(V)
    a, b = r.alpha_m, r.beta_m
    Q = two_state_generator(a, b)
    p_star = stationary_distribution(Q)[1]
    assert np.isclose(p_star, a / (a + b))

    N = 300
    tol = 0.02

    # (1) Gillespie SSA -- one long trajectory's time-average occupancy.
    ss = gillespie_ssa(Q, initial_state=0, t_max=8000.0, rng=0)
    p_gillespie = ss.occupancy_fractions(2)[1]

    # (2) Euler-Maruyama channel Langevin -- ensemble stationary mean.
    drift, diffusion = channel_langevin_terms(Q, N)
    finals_em = np.array([
        euler_maruyama(drift, diffusion, p_star, t_max=50.0, dt=5e-3, rng=1000 + i,
                       clip=(0, 1)).X[-1]
        for i in range(200)
    ])
    p_em = finals_em.mean()

    # (3) Stochastic HH gate, voltage-clamped at V -- m-gate stationary mean.
    cfg = dataclasses.replace(HHConfig(), n_na_channels=N)
    finals_hh = np.array([
        simulate_hh(cfg, t_max=50.0, dt=5e-3, stochastic=True, rng=i, clamp_V=V).m[-1]
        for i in range(200)
    ])
    p_hh = finals_hh.mean()

    # (4) Circuit RTN trap with capture=a, emission=b -- filled fraction.
    rtn_cfg = RTNConfig(capture_rate=a, emission_rate=b, delta_vth=5e-3)
    p_rtn = simulate_rtn(rtn_cfg, t_max=8000.0, rng=0).stationary_filled_fraction()
    assert np.isclose(rtn_stationary_filled(rtn_cfg), p_star)

    # All four independent realisations agree with p* and with each other.
    estimates = {"gillespie": p_gillespie, "em": p_em, "hh": p_hh, "rtn": p_rtn}
    for name, val in estimates.items():
        assert abs(val - p_star) < tol, f"{name}={val:.4f} vs p*={p_star:.4f}"
    spread = max(estimates.values()) - min(estimates.values())
    assert spread < 2 * tol, f"cross-phase spread too large: {estimates}"
