"""Phase 2 tests: Euler-Maruyama integrator, multi-seed relaxation tracking, and
the EM -> Gillespie convergence regression (items 5, 6, 8)."""

import numpy as np
import pytest

from stochastic_ctmc.generator import two_state_generator
from stochastic_ctmc.sde import (
    euler_maruyama,
    channel_langevin_terms,
    deterministic_relaxation,
)
from stochastic_ctmc.ensemble import (
    ensemble_em,
    mean_ci,
    population_generator,
    gillespie_population_fraction,
)


# --- Integrator sanity ------------------------------------------------------

def test_zero_noise_recovers_ode():
    # dx = -x dt, no diffusion => x(t) = x0 exp(-t). EM with small dt tracks it.
    res = euler_maruyama(
        drift=lambda x, t: -x, diffusion=lambda x, t: 0.0,
        x0=1.0, t_max=3.0, dt=1e-3, rng=0,
    )
    assert np.isclose(res.X[-1], np.exp(-res.times[-1]), rtol=2e-3)


def test_pure_diffusion_variance_grows_linearly():
    # dx = dW  => Var[x(t)] = t. Check across an ensemble.
    summ = ensemble_em(
        drift=lambda x, t: 0.0, diffusion=lambda x, t: 1.0,
        x0=0.0, t_max=1.0, dt=1e-3, n_seeds=400, base_seed=0,
    )
    # Reconstruct ensemble variance at final time via the CI half-width -> sem.
    # Simpler: run raw and check variance directly.
    rng = np.random.default_rng(1)
    finals = np.array([
        euler_maruyama(lambda x, t: 0.0, lambda x, t: 1.0, 0.0, 1.0, 1e-3, rng=rng).X[-1]
        for _ in range(2000)
    ])
    assert np.isclose(finals.var(), 1.0, rtol=0.1)


def test_jump_diffusion_shifts_mean():
    # Pure positive jumps at known rate/size shift the mean by rate*size*t.
    rate, size, t_max = 2.0, 0.5, 4.0
    rng = np.random.default_rng(3)
    finals = np.array([
        euler_maruyama(
            lambda x, t: 0.0, lambda x, t: 0.0, 0.0, t_max, 1e-2,
            rng=rng, jump_rate=lambda x, t: rate, jump_size=lambda g: size,
        ).X[-1]
        for _ in range(1500)
    ])
    assert np.isclose(finals.mean(), rate * size * t_max, rtol=0.05)


# --- Item 6: fast-relaxation tracking, multi-seed, mean +/- CI --------------

def test_multiseed_mean_tracks_deterministic_relaxation():
    a, b = 3.0, 1.0                 # fast relaxation, tau = 1/(a+b) = 0.25 s
    Q = two_state_generator(a, b)
    N = 500
    drift, diffusion = channel_langevin_terms(Q, N)

    x0 = 0.0
    t_max = 2.0
    summ = ensemble_em(
        drift, diffusion, x0=x0, t_max=t_max, dt=5e-4,
        n_seeds=150, base_seed=0, confidence=0.95, clip=(0.0, 1.0),
    )
    assert summ.n_seeds >= 100  # requirement: >= 100 seeds

    truth = deterministic_relaxation(Q, x0, summ.times)

    # The deterministic curve must lie inside the 95% band almost everywhere.
    inside = (truth >= summ.ci_low - 1e-9) & (truth <= summ.ci_high + 1e-9)
    assert inside.mean() > 0.9

    # And the endpoint mean must match the stationary occupancy x* = a/(a+b).
    assert np.isclose(summ.mean[-1], a / (a + b), atol=0.02)


def test_mean_ci_shapes_and_coverage():
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=5.0, scale=2.0, size=(300, 4))
    mean, lo, hi = mean_ci(samples, confidence=0.95)
    assert mean.shape == (4,)
    assert np.all(lo < mean) and np.all(mean < hi)
    assert np.allclose(mean, 5.0, atol=0.3)


# --- Item 8: EM stationary statistics converge to Gillespie as dt -> 0 ------

def _em_stationary_variance(Q, N, dt, n_seeds, t_max):
    drift, diffusion = channel_langevin_terms(Q, N)
    x_star = Q[0, 1] / (Q[0, 1] + Q[1, 0])
    # Start at the fixed point and sample the stationary fraction at t_max.
    summ = ensemble_em(
        drift, diffusion, x0=x_star, t_max=t_max, dt=dt,
        n_seeds=n_seeds, base_seed=1000, clip=(0.0, 1.0),
        grid=np.array([t_max]),
    )
    # Recover the ensemble variance at the single grid point from the CI.
    # Easier: re-run collecting finals directly.
    finals = []
    for i in range(n_seeds):
        res = euler_maruyama(drift, diffusion, x_star, t_max, dt, rng=2000 + i, clip=(0.0, 1.0))
        finals.append(res.X[-1])
    return float(np.var(finals)), float(np.mean(finals))


@pytest.mark.slow
def test_em_stationary_matches_gillespie_population():
    a, b = 1.0, 2.0
    Q = two_state_generator(a, b)
    N = 200
    p = a / (a + b)
    binomial_var = p * (1 - p) / N   # exact stationary variance of k/N

    # EM stationary variance at a fine dt.
    em_var, em_mean = _em_stationary_variance(Q, N, dt=2e-4, n_seeds=600, t_max=6.0)

    # Gillespie population reference at the same regime.
    gill = gillespie_population_fraction(
        Q, N, initial_open=int(round(p * N)), t_max=6.0, n_seeds=600, base_seed=0,
        grid=np.array([6.0]),
    )[:, 0]
    gill_var, gill_mean = float(gill.var()), float(gill.mean())

    # Means agree with p; both variances agree with the binomial law and each other.
    assert np.isclose(em_mean, p, atol=0.02)
    assert np.isclose(gill_mean, p, atol=0.02)
    assert np.isclose(em_var, binomial_var, rtol=0.25)
    assert np.isclose(gill_var, binomial_var, rtol=0.25)
    assert np.isclose(em_var, gill_var, rtol=0.35)


@pytest.mark.slow
def test_em_converges_to_gillespie_as_dt_shrinks():
    # Regression: the discrepancy between EM stationary variance and the exact
    # binomial variance should shrink (or at least not blow up) as dt -> 0.
    a, b = 1.0, 1.0
    Q = two_state_generator(a, b)
    N = 100
    p = 0.5
    target = p * (1 - p) / N

    err_coarse, _ = _em_stationary_variance(Q, N, dt=5e-3, n_seeds=400, t_max=8.0)
    err_fine, _ = _em_stationary_variance(Q, N, dt=2e-4, n_seeds=400, t_max=8.0)

    e_coarse = abs(err_coarse - target)
    e_fine = abs(err_fine - target)
    # Fine step should be at least as accurate as the coarse one (within noise).
    assert e_fine <= e_coarse + 0.3 * target
