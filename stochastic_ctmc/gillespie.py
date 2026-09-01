"""Gillespie stochastic simulation algorithm (SSA) for a CTMC.

This is the *exact* simulation of a continuous-time Markov chain defined by a
generator matrix ``Q`` (see :mod:`stochastic_ctmc.generator`). Every other phase
is benchmarked against the trajectories this module produces.

The direct method (Gillespie 1977):

1. In state ``i`` the total exit rate is ``a0 = -Q[i, i]`` (sum of the row's
   off-diagonal propensities).
2. The waiting time until the next transition is ``Exp(a0)``: ``tau = -ln(u)/a0``.
3. The next state ``j != i`` is chosen with probability ``Q[i, j] / a0``.

Absorbing states (``a0 == 0``) end the trajectory: no further transition can
occur, so the walker stays put until ``t_max``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .generator import validate_generator

__all__ = ["SSAResult", "gillespie_ssa"]


@dataclass
class SSAResult:
    """Result of a single Gillespie SSA run.

    Attributes
    ----------
    times : ndarray, shape (K,)
        Transition times, starting at 0.0. ``times[k]`` is the time at which the
        chain *entered* ``states[k]``.
    states : ndarray, shape (K,)
        State occupied on each inter-transition interval. The chain is in
        ``states[k]`` on ``[times[k], times[k+1])`` (and past ``times[-1]`` up to
        ``t_max``).
    t_max : float
        Simulation horizon.
    absorbed : bool
        True if the run halted early because it reached an absorbing state.
    """

    times: np.ndarray
    states: np.ndarray
    t_max: float
    absorbed: bool

    def state_at(self, t: float) -> int:
        """State occupied at continuous time ``t`` (right-continuous step function)."""
        idx = int(np.searchsorted(self.times, t, side="right") - 1)
        idx = max(0, idx)
        return int(self.states[idx])

    def occupancy_fractions(self, n_states: int) -> np.ndarray:
        """Time-weighted fraction of ``[0, t_max]`` spent in each state.

        This is the empirical estimate of the stationary distribution from one
        long trajectory (ergodic average).
        """
        dwell = np.zeros(n_states, dtype=float)
        edges = np.concatenate([self.times, [self.t_max]])
        durations = np.diff(edges)
        for s, d in zip(self.states, durations):
            dwell[int(s)] += d
        total = dwell.sum()
        if total <= 0:
            raise ValueError("trajectory has zero total duration")
        return dwell / total


def gillespie_ssa(
    Q: np.ndarray,
    initial_state: int,
    t_max: float,
    *,
    rng: np.random.Generator | int | None = None,
    max_steps: int | None = None,
    validate: bool = True,
) -> SSAResult:
    """Simulate one CTMC trajectory with Gillespie's direct method.

    Parameters
    ----------
    Q : (N, N) ndarray
        CTMC generator matrix. Validated on entry unless ``validate=False``.
    initial_state : int
        Starting state index in ``0 .. N-1``.
    t_max : float
        Simulate until this time is reached or exceeded.
    rng : np.random.Generator | int | None
        Random source. An int is used as a seed; ``None`` draws a fresh default
        generator. Pass an explicit seed for reproducibility across phases.
    max_steps : int | None
        Safety cap on the number of transitions. ``None`` means unlimited (the
        loop still terminates because ``t`` advances past ``t_max``).
    validate : bool
        Run :func:`validate_generator` on ``Q`` first. Kept as an option only for
        hot loops that have already validated the matrix.

    Returns
    -------
    SSAResult
    """
    if validate:
        Q = validate_generator(Q)
    else:
        Q = np.asarray(Q, dtype=float)

    n_states = Q.shape[0]
    if not (0 <= initial_state < n_states):
        raise ValueError(f"initial_state {initial_state} out of range for {n_states} states")
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")

    if isinstance(rng, np.random.Generator):
        generator = rng
    else:
        generator = np.random.default_rng(rng)

    # Precompute per-row exit rates and normalised jump distributions.
    exit_rates = -np.diag(Q)  # a0 for each state
    off = Q.copy()
    np.fill_diagonal(off, 0.0)

    times = [0.0]
    states = [int(initial_state)]
    t = 0.0
    state = int(initial_state)
    absorbed = False
    step = 0

    while t < t_max:
        a0 = exit_rates[state]
        if a0 <= 0.0:
            # Absorbing state: no transition will ever fire.
            absorbed = True
            break

        # Exponential waiting time.
        tau = generator.exponential(1.0 / a0)
        t += tau
        if t >= t_max:
            break

        # Choose the destination state with probability Q[state, j] / a0.
        probs = off[state] / a0
        state = int(generator.choice(n_states, p=probs))

        times.append(t)
        states.append(state)

        step += 1
        if max_steps is not None and step >= max_steps:
            break

    return SSAResult(
        times=np.asarray(times, dtype=float),
        states=np.asarray(states, dtype=int),
        t_max=float(t_max),
        absorbed=absorbed,
    )
