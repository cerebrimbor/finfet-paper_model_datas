"""Central configuration: the single source of truth for parameters (Phase 5, #16).

Every phase imports its parameters from here so that the "same generator matrix
across all models" claim is enforced by construction rather than by copy-paste.
Values are grouped into frozen dataclasses; instantiate the defaults or override
fields explicitly.

Nothing in this module imports heavy machinery, so it is safe to import from any
phase (including plotting and data-export utilities) without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CTMCConfig:
    """A named CTMC defined by directed transition rates (units: 1/s).

    ``rates`` maps ``(i, j) -> rate`` for the 0->1 style transitions; it feeds
    directly into ``n_state_generator``. This is the canonical description shared
    by every phase.
    """

    name: str
    n_states: int
    rates: dict[tuple[int, int], float]
    initial_state: int = 0


@dataclass(frozen=True)
class SSAConfig:
    """Gillespie SSA run parameters."""

    t_max: float = 10.0
    seed: int = 0


@dataclass(frozen=True)
class EMConfig:
    """Euler-Maruyama jump-diffusion parameters (Phase 2)."""

    dt: float = 1e-4
    t_max: float = 10.0
    n_seeds: int = 100
    confidence: float = 0.95


@dataclass(frozen=True)
class HHConfig:
    """Stochastic Hodgkin-Huxley parameters (Phase 3, Fox & Lu 1994).

    Standard squid-axon values (mS/cm^2, mV, uF/cm^2). ``n_channels`` sets the
    channel-noise amplitude: noise ~ 1/sqrt(N), so the deterministic HH model is
    recovered as ``n_channels -> inf`` (see Phase 3, item 11).
    """

    c_m: float = 1.0
    g_na: float = 120.0
    g_k: float = 36.0
    g_leak: float = 0.3
    e_na: float = 50.0
    e_k: float = -77.0
    e_leak: float = -54.4
    n_na_channels: int = 6000
    n_k_channels: int = 1800
    i_ext: float = 10.0


@dataclass(frozen=True)
class RTNConfig:
    """GAAFET random-telegraph-noise trap parameters (Phase 4).

    A single trap is a 2-state CTMC (empty <-> filled) whose capture/emission
    rates map onto the same generator structure as an ion-channel gate — the
    paper's structural-identity claim (item 15).
    """

    capture_rate: float = 1.0e3   # empty -> filled  (1/s)
    emission_rate: float = 2.0e3  # filled -> empty  (1/s)
    delta_vth: float = 5.0e-3     # threshold-voltage shift when trap filled (V)


# --- Canonical shared CTMCs -------------------------------------------------
# These are the exact generators reused across Gillespie / EM / HH / RTN so that
# cross-phase consistency tests (item 19) compare like with like.

TWO_STATE = CTMCConfig(
    name="two_state_channel",
    n_states=2,
    rates={(0, 1): 1.0, (1, 0): 2.0},
    initial_state=0,
)

THREE_STATE = CTMCConfig(
    name="three_state_reversible",
    n_states=3,
    # A reversible 3-cycle: chosen so detailed balance holds (for the DB test)
    # while still exercising N>2 assembly.
    rates={
        (0, 1): 1.0, (1, 0): 1.0,
        (1, 2): 2.0, (2, 1): 2.0,
        (0, 2): 0.5, (2, 0): 0.5,
    },
    initial_state=0,
)


@dataclass(frozen=True)
class Config:
    """Top-level bundle passed around between phases."""

    ctmc: CTMCConfig = TWO_STATE
    ssa: SSAConfig = field(default_factory=SSAConfig)
    em: EMConfig = field(default_factory=EMConfig)
    hh: HHConfig = field(default_factory=HHConfig)
    rtn: RTNConfig = field(default_factory=RTNConfig)


DEFAULT = Config()
