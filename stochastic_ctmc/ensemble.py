"""Phase 2 support: multi-seed ensembles, confidence intervals, and the exact
population reference used to benchmark the Euler-Maruyama diffusion approximation.

Two things live here:

* ``ensemble_em`` / ``mean_ci`` -- run an SDE over many seeds and summarise the
  ensemble as mean +/- confidence interval on a common time grid (item 6).
* ``population_generator`` -- the exact CTMC of ``N`` independent 2-state
  channels, expressed as an ``(N+1)``-state birth-death generator built with the
  *same* ``n_state_generator`` from Phase 1. Simulating this with the Gillespie
  SSA gives the exact reference the EM approximation must converge to (item 8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats

from .generator import n_state_generator, validate_generator
from .gillespie import gillespie_ssa
from .sde import euler_maruyama, EMResult

__all__ = [
    "EnsembleSummary",
    "ensemble_em",
    "mean_ci",
    "population_generator",
    "gillespie_population_fraction",
]


@dataclass
class EnsembleSummary:
    """Ensemble mean and confidence band on a shared time grid."""

    times: np.ndarray      # (G,)
    mean: np.ndarray       # (G,)
    ci_low: np.ndarray     # (G,)
    ci_high: np.ndarray    # (G,)
    n_seeds: int
    confidence: float

    @property
    def half_width(self) -> np.ndarray:
        """Half the CI width at each grid point."""
        return 0.5 * (self.ci_high - self.ci_low)


def mean_ci(samples: np.ndarray, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and two-sided ``confidence`` CI across axis 0 (the seed axis).

    Uses the Student-t standard-error-of-the-mean interval, appropriate for a
    finite number of seeds.
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.shape[0]
    mean = samples.mean(axis=0)
    if n < 2:
        return mean, mean.copy(), mean.copy()
    sem = samples.std(axis=0, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.5 * (1 + confidence), df=n - 1)
    half = t_crit * sem
    return mean, mean - half, mean + half


def ensemble_em(
    drift: Callable[[float, float], float],
    diffusion: Callable[[float, float], float],
    x0: float,
    t_max: float,
    dt: float,
    n_seeds: int,
    *,
    base_seed: int = 0,
    confidence: float = 0.95,
    grid: np.ndarray | None = None,
    **em_kwargs,
) -> EnsembleSummary:
    """Run ``n_seeds`` Euler-Maruyama trajectories and summarise as mean +/- CI.

    Each seed uses ``base_seed + i`` so the ensemble is fully reproducible. All
    trajectories share the same ``dt`` grid; if ``grid`` is given the results are
    sampled onto it (nearest step) instead.
    """
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1")

    first: EMResult = euler_maruyama(drift, diffusion, x0, t_max, dt, rng=base_seed, **em_kwargs)
    ref_times = first.times if grid is None else np.asarray(grid, dtype=float)
    n_grid = ref_times.shape[0]

    all_x = np.empty((n_seeds, n_grid), dtype=float)

    def sample(res: EMResult) -> np.ndarray:
        if grid is None:
            return res.X
        idx = np.searchsorted(res.times, ref_times, side="right") - 1
        idx = np.clip(idx, 0, res.X.shape[0] - 1)
        return res.X[idx]

    all_x[0] = sample(first)
    for i in range(1, n_seeds):
        res = euler_maruyama(drift, diffusion, x0, t_max, dt, rng=base_seed + i, **em_kwargs)
        all_x[i] = sample(res)

    mean, lo, hi = mean_ci(all_x, confidence=confidence)
    return EnsembleSummary(
        times=ref_times, mean=mean, ci_low=lo, ci_high=hi,
        n_seeds=n_seeds, confidence=confidence,
    )


def population_generator(Q2: np.ndarray, n_channels: int) -> np.ndarray:
    """Exact ``(N+1)``-state birth-death generator for ``N`` independent 2-state channels.

    State ``k`` = number of channels currently in state 1. With single-channel
    rates ``a`` (0->1) and ``b`` (1->0):

        k -> k+1 at rate  a * (N - k)   (an empty channel opens)
        k -> k-1 at rate  b * k         (an open channel closes)

    Built with :func:`n_state_generator`, i.e. the *same* Phase 1 constructor --
    this is the concrete sense in which the population reference shares the CTMC
    core with the single-trap model.
    """
    Q2 = validate_generator(Q2)
    if Q2.shape != (2, 2):
        raise ValueError("population_generator expects a 2-state single-channel generator")
    a = Q2[0, 1]
    b = Q2[1, 0]
    N = int(n_channels)

    rates: dict[tuple[int, int], float] = {}
    for k in range(N + 1):
        if k < N:
            rates[(k, k + 1)] = a * (N - k)
        if k > 0:
            rates[(k, k - 1)] = b * k
    return n_state_generator(rates, n_states=N + 1)


def gillespie_population_fraction(
    Q2: np.ndarray,
    n_channels: int,
    initial_open: int,
    t_max: float,
    n_seeds: int,
    *,
    base_seed: int = 0,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Ensemble of open-fraction trajectories from the exact population CTMC.

    Returns an array of shape ``(n_seeds, len(grid))`` (or per-seed final samples
    if ``grid`` is None) giving ``k/N``, the exact Gillespie counterpart of the
    Langevin ``x``.
    """
    Qpop = population_generator(Q2, n_channels)
    N = int(n_channels)

    if grid is not None:
        grid = np.asarray(grid, dtype=float)
        out = np.empty((n_seeds, grid.shape[0]), dtype=float)
    else:
        out = np.empty((n_seeds, 1), dtype=float)

    for i in range(n_seeds):
        res = gillespie_ssa(Qpop, initial_open, t_max, rng=base_seed + i, validate=False)
        if grid is None:
            out[i, 0] = res.state_at(t_max) / N
        else:
            for j, tg in enumerate(grid):
                out[i, j] = res.state_at(tg) / N
    return out
