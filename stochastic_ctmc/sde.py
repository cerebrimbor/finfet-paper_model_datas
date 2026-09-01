"""Phase 2: Euler-Maruyama integrator for jump-diffusion SDEs.

A jump-diffusion SDE has the general form

    dX = f(X, t) dt  +  g(X, t) dW  +  h(X, t) dN

where ``dW`` is a Wiener increment (Gaussian, variance ``dt``) and ``dN`` is a
compound-Poisson jump increment (rate ``lambda(X, t)``, mark size from
``jump_size``). The Euler-Maruyama scheme discretises this as

    X_{n+1} = X_n + f dt + g sqrt(dt) Z_n + (jumps that fired in [t, t+dt))

with ``Z_n ~ N(0, 1)``.

The link to Phase 1: the CTMC of an ion channel / RTN trap has a *diffusion
approximation* (the chemical Langevin equation of Fox & Lu, 1994). Its drift and
diffusion are read straight off the same generator matrix, so a Gillespie
ensemble and an Euler-Maruyama ensemble are two views of one object. As
``dt -> 0`` the EM statistics converge to the Gillespie statistics (item 8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .generator import validate_generator

__all__ = [
    "EMResult",
    "euler_maruyama",
    "channel_langevin_terms",
    "deterministic_relaxation",
]


@dataclass
class EMResult:
    """One Euler-Maruyama trajectory."""

    times: np.ndarray   # shape (M+1,)
    X: np.ndarray       # shape (M+1,) for scalar state, or (M+1, d)
    dt: float


def euler_maruyama(
    drift: Callable[[float, float], float],
    diffusion: Callable[[float, float], float],
    x0: float,
    t_max: float,
    dt: float,
    *,
    rng: np.random.Generator | int | None = None,
    jump_rate: Callable[[float, float], float] | None = None,
    jump_size: Callable[[np.random.Generator], float] | None = None,
    clip: tuple[float, float] | None = None,
) -> EMResult:
    """Integrate a scalar jump-diffusion SDE with the Euler-Maruyama scheme.

    Parameters
    ----------
    drift, diffusion : callable(x, t) -> float
        The ``f`` and ``g`` coefficients.
    x0 : float
        Initial condition.
    t_max, dt : float
        Horizon and fixed time step. The number of steps is ``ceil(t_max/dt)``.
    rng : np.random.Generator | int | None
        Random source (int seed for reproducibility).
    jump_rate : callable(x, t) -> float, optional
        Poisson intensity ``lambda``. If ``None`` the SDE is a pure diffusion.
    jump_size : callable(rng) -> float, optional
        Draws one jump mark. Required if ``jump_rate`` is given.
    clip : (lo, hi), optional
        Reflect/clamp the state into ``[lo, hi]`` after each step. Channel/trap
        occupancy fractions live in ``[0, 1]``; the diffusion approximation can
        otherwise wander slightly outside on coarse steps.

    Returns
    -------
    EMResult
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")
    if jump_rate is not None and jump_size is None:
        raise ValueError("jump_size must be provided when jump_rate is set")

    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    n_steps = int(np.ceil(t_max / dt))
    times = np.empty(n_steps + 1, dtype=float)
    X = np.empty(n_steps + 1, dtype=float)
    times[0] = 0.0
    X[0] = float(x0)

    sqrt_dt = np.sqrt(dt)
    normals = generator.standard_normal(n_steps)

    x = float(x0)
    t = 0.0
    for n in range(n_steps):
        f = drift(x, t)
        g = diffusion(x, t)
        x = x + f * dt + g * sqrt_dt * normals[n]

        if jump_rate is not None:
            # Number of jumps in this step ~ Poisson(lambda * dt); usually 0 or 1.
            n_jumps = generator.poisson(max(0.0, jump_rate(x, t)) * dt)
            for _ in range(int(n_jumps)):
                x = x + jump_size(generator)

        if clip is not None:
            lo, hi = clip
            # Reflect at the boundaries (keeps the diffusion measure-preserving,
            # unlike a hard clamp which piles probability on the wall).
            if x < lo:
                x = lo + (lo - x)
            if x > hi:
                x = hi - (x - hi)
            x = min(max(x, lo), hi)

        t += dt
        X[n + 1] = x
        times[n + 1] = t

    return EMResult(times=times, X=X, dt=dt)


def channel_langevin_terms(Q: np.ndarray, n_channels: int):
    """Drift and diffusion of the diffusion approximation for a 2-state channel.

    For a population of ``N`` independent 2-state channels/traps with generator
    ``Q = [[-a, a], [b, -b]]`` (``a`` = 0->1 rate, ``b`` = 1->0 rate), let ``x``
    be the *fraction* currently in state 1. The chemical Langevin equation is

        dx = [a (1 - x) - b x] dt  +  sqrt( (a (1 - x) + b x) / N ) dW.

    The drift's fixed point is the stationary occupancy ``x* = a/(a+b)`` and,
    linearised there, the process is Ornstein-Uhlenbeck with stationary variance
    ``x*(1 - x*)/N`` -- exactly the binomial channel-number variance. That
    identity is what the Phase 1 <-> Phase 2 regression test checks.

    Returns
    -------
    (drift, diffusion) : callables ``(x, t) -> float``.
    """
    Q = validate_generator(Q)
    if Q.shape != (2, 2):
        raise ValueError("channel_langevin_terms is defined for 2-state generators")
    if n_channels <= 0:
        raise ValueError("n_channels must be positive")

    a = Q[0, 1]  # 0 -> 1 rate
    b = Q[1, 0]  # 1 -> 0 rate

    def drift(x: float, t: float) -> float:
        return a * (1.0 - x) - b * x

    def diffusion(x: float, t: float) -> float:
        # Guard the sqrt against tiny negative arguments from over-stepping.
        var = (a * (1.0 - x) + b * x) / n_channels
        return np.sqrt(max(var, 0.0))

    return drift, diffusion


def deterministic_relaxation(Q: np.ndarray, x0: float, t: np.ndarray) -> np.ndarray:
    """Closed-form mean relaxation ``x(t)`` of the 2-state occupancy fraction.

    ``x(t) = x* + (x0 - x*) exp(-(a + b) t)`` with ``x* = a / (a + b)``. Used as
    the ground truth for the fast-relaxation tracking test (item 6): the
    multi-seed EM mean must follow this curve within its confidence band.
    """
    Q = validate_generator(Q)
    a = Q[0, 1]
    b = Q[1, 0]
    x_star = a / (a + b)
    return x_star + (x0 - x_star) * np.exp(-(a + b) * np.asarray(t, dtype=float))
