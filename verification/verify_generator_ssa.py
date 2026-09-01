"""Independent verification of the CTMC generator and Gillespie SSA.

This script deliberately does NOT call the project's stationary_distribution()
or Gillespie helper to compute the reference values.  It uses the defining
equations directly, then compares long-run SSA statistics against them.

Run from the repository root:
    python verification/verify_generator_ssa.py
"""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stochastic_ctmc.generator import two_state_generator
from stochastic_ctmc.gillespie import gillespie_ssa


def main():
    k01, k10 = 1.0, 2.0
    Q = two_state_generator(k01, k10)

    # Independent reference values from the two-state CTMC equations.
    p1 = k01 / (k01 + k10)
    p0 = k10 / (k01 + k10)

    assert np.allclose(Q, [[-k01, k01], [k10, -k10]])

    # Long trajectory: time occupancy should converge to stationary mass.
    ss = gillespie_ssa(Q, 0, t_max=2_000_000.0, rng=12345)
    occ = ss.occupancy_fractions(2)

    # Number of transitions should be of the expected order.
    expected_rate = 2 * k01 * k10 / (k01 + k10)
    transition_rate = (len(ss.times) - 1) / ss.t_max

    print("CTMC / Gillespie independent verification")
    print(f"stationary p0={p0:.8f}, p1={p1:.8f}")
    print(f"empirical  p0={occ[0]:.8f}, p1={occ[1]:.8f}")
    print(f"expected transition rate={expected_rate:.6f}/s")
    print(f"observed transition rate={transition_rate:.6f}/s")

    assert abs(occ[0] - p0) < 0.003
    assert abs(occ[1] - p1) < 0.003
    assert abs(transition_rate - expected_rate) / expected_rate < 0.02

    print("PASS")


if __name__ == "__main__":
    main()
