"""Phase 4: circuit-level mapping -- GAAFET random-telegraph-noise (RTN).

Central claim (item 15, the paper's "structural identity")
----------------------------------------------------------
A single oxide trap in a gate-all-around FET is a 2-state system: **empty** and
**filled**. Carriers are captured (empty -> filled) at rate ``k_c`` and emitted
(filled -> empty) at rate ``k_e``. Its generator is

    Q_RTN = [[-k_c,  k_c],
             [ k_e, -k_e]]

which is *literally* :func:`two_state_generator(k_c, k_e)` -- the same object a
Phase 1 CTMC, a Phase 2 channel-Langevin gate, and a Phase 3 Hodgkin-Huxley gate
are built from. When the trap is filled it shifts the transistor threshold
voltage by ``delta_Vth``, producing the familiar two-level telegraph signal in
the drain current. So an ion channel flickering open/closed and a trap
capturing/emitting a carrier are the *same* stochastic process wearing different
physical clothes -- that is the structural identity this module makes concrete
and testable.

No SPICE in the loop
--------------------
This module deliberately contains no circuit-simulator subprocess. The trap's
effect on the device is a threshold-voltage offset, and ``DVTSHIFT`` in the
BSIM-CMG card was verified to be an *exact* gate-voltage offset: re-extracting
the shift between the baseline and trap Id-Vg sweeps at matched drain current
gives 5.000000 mV across 200 of the 201 swept points, spread 2 nV -- zero to
within the interpolation error. The single excluded point is ``Vg = 0``: the
trap curve is displaced towards higher gate voltage, so its current there lies
below every current in the baseline sweep and can only be matched by
extrapolating off the start of that curve, not by interpolating within it.
(See ``device_provenance/``, reproduced against ngspice 46 + OSDI BSIMCMG.)
The transistor is therefore a memoryless nonlinearity, fully captured by one DC
sweep -- the CTMC decides *when* the trap is filled, and the sweep says what
that does to the current. Running a simulator per timestep would recompute a
function already known in closed form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RTNConfig
from .generator import two_state_generator, stationary_distribution
from .gillespie import gillespie_ssa, SSAResult

__all__ = [
    "rtn_generator",
    "DomainMap",
    "STRUCTURAL_IDENTITY",
    "simulate_rtn",
    "RTNTrace",
    "rtn_stationary_filled",
]


def rtn_generator(capture_rate: float, emission_rate: float) -> np.ndarray:
    """Generator of a single RTN trap = ``two_state_generator(k_c, k_e)``.

    State 0 = empty, state 1 = filled. This is deliberately a thin alias over the
    Phase 1 constructor: the point of the paper is that there is nothing
    circuit-specific about the *structure* -- only the interpretation of the two
    rates changes.
    """
    return two_state_generator(capture_rate, emission_rate)


@dataclass(frozen=True)
class DomainMap:
    """How one physical system maps onto the shared 2-state CTMC ``(a, b, N)``.

    ``a`` is the 0->1 rate, ``b`` the 1->0 rate, ``N`` the population size that
    sets the diffusion-approximation noise amplitude (1 for a single unit).
    """

    domain: str
    state0: str
    state1: str
    a_meaning: str   # what the 0->1 rate is called in this domain
    b_meaning: str   # what the 1->0 rate is called in this domain
    observable: str  # what the state modulates in the measured signal


# The four domains, all mapped onto the identical generator structure.
STRUCTURAL_IDENTITY: tuple[DomainMap, ...] = (
    DomainMap(
        domain="Generic CTMC (Phase 1)",
        state0="state 0", state1="state 1",
        a_meaning="rate k01", b_meaning="rate k10",
        observable="occupancy",
    ),
    DomainMap(
        domain="Ion channel gate (Phase 3, HH)",
        state0="closed", state1="open",
        a_meaning="alpha(V)", b_meaning="beta(V)",
        observable="gating variable / conductance",
    ),
    DomainMap(
        domain="Channel population (Phase 2)",
        state0="closed fraction", state1="open fraction",
        a_meaning="opening rate a", b_meaning="closing rate b",
        observable="open fraction x",
    ),
    DomainMap(
        domain="GAAFET oxide trap (Phase 4, RTN)",
        state0="empty", state1="filled",
        a_meaning="capture rate k_c", b_meaning="emission rate k_e",
        observable="threshold-voltage shift delta_Vth",
    ),
)


@dataclass
class RTNTrace:
    """A random-telegraph-noise trace derived from an RTN CTMC trajectory."""

    ssa: SSAResult
    delta_vth: float

    def vth_signal(self, t: np.ndarray) -> np.ndarray:
        """Threshold-voltage offset at each time in ``t`` (0 when empty, delta when filled)."""
        return np.array([self.ssa.state_at(float(ti)) * self.delta_vth for ti in t])

    def stationary_filled_fraction(self) -> float:
        """Time-averaged fraction of the trace spent in the filled state."""
        return float(self.ssa.occupancy_fractions(2)[1])


def simulate_rtn(
    cfg: RTNConfig,
    t_max: float,
    *,
    rng: np.random.Generator | int | None = None,
    initial_state: int = 0,
) -> RTNTrace:
    """Simulate a single-trap RTN telegraph signal with the Gillespie SSA.

    Reuses the Phase 1 SSA verbatim -- the trap trajectory is produced by exactly
    the same engine that produces ion-channel gating trajectories.
    """
    Q = rtn_generator(cfg.capture_rate, cfg.emission_rate)
    ssa = gillespie_ssa(Q, initial_state=initial_state, t_max=t_max, rng=rng)
    return RTNTrace(ssa=ssa, delta_vth=cfg.delta_vth)


def rtn_stationary_filled(cfg: RTNConfig) -> float:
    """Analytic stationary filled probability ``k_c / (k_c + k_e)`` via the CTMC."""
    Q = rtn_generator(cfg.capture_rate, cfg.emission_rate)
    return float(stationary_distribution(Q)[1])
