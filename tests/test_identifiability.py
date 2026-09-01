"""Is the chain UNIQUELY identified, or is some degenerate partner invisible?

max |rho| says how little data you need in the best case. min |rho| decides the
strength of the claim:
  inf|rho| > 0  ->  every reversible degenerate partner is detectably
                    correlated; the chain is uniquely identified by the 2-D
                    statistic.
  inf|rho| = 0  ->  some partner is invisible to the correlation test; the
                    claim must narrow to a rank test on the binned joint.
"""

import numpy as np
import pytest

from stochastic_ctmc.generator import n_state_generator, is_detailed_balance
from stochastic_ctmc.dwell import joint_density_rank, gateway_block_rank
from stochastic_ctmc.topology import (
    extremal_correlation_reversible, explore_reversible_manifold, LEVELS_4,
)
from stochastic_ctmc.correlation import pairs_needed

CHAIN = n_state_generator(
    {(0, 1): 5.0, (1, 0): 5.0,
     (1, 2): 1.0, (2, 1): 1.0,
     (2, 3): 4.0, (3, 2): 4.0},
    n_states=4,
)
LEVELS = list(LEVELS_4)


def test_sense_is_validated():
    with pytest.raises(ValueError):
        extremal_correlation_reversible(CHAIN, LEVELS, sense="sideways")


@pytest.mark.slow
def test_matched_exit_rates_equal_the_chains_eigenvalues():
    """The bipartite partner has diagonal Q_FF, so its mixture rates ARE its
    bare exit rates -- and they must equal the eigenvalues of the chain's
    coupled Q_FF block. Same observable time constants, different mechanism:
    aggregation degeneracy made concrete."""
    res = extremal_correlation_reversible(CHAIN, LEVELS, sense="max", n_samples=80)
    assert res.success

    for idx, label in (([2, 3], "filled"), ([0, 1], "empty")):
        blk = CHAIN[np.ix_(idx, idx)]
        eig = np.sort(-np.linalg.eigvals(blk).real)
        exits = np.sort(-np.diag(res.Q)[idx])
        assert np.allclose(eig, exits, rtol=1e-6), f"{label}: {eig} vs {exits}"


@pytest.mark.slow
def test_report_correlation_range_over_the_manifold():
    hi = extremal_correlation_reversible(CHAIN, LEVELS, sense="max", n_samples=200)
    lo = extremal_correlation_reversible(CHAIN, LEVELS, sense="min", n_samples=200)
    assert hi.success and lo.success

    print(f"\n--- reversible degenerate-partner manifold ---")
    print(f"max |rho| = {abs(hi.rho):.6f}  -> {pairs_needed(hi.rho):>14,} pairs @ 5 sigma")
    print(f"min |rho| = {abs(lo.rho):.6f}  -> {pairs_needed(lo.rho):>14,} pairs @ 5 sigma")

    for r in (hi, lo):
        assert is_detailed_balance(r.Q, atol=1e-8)
        assert r.signature_error < 1e-7
        assert gateway_block_rank(r.Q, LEVELS) == 2

    assert abs(lo.rho) <= abs(hi.rho) + 1e-9


@pytest.mark.slow
def test_worst_case_partner_is_still_detectable():
    """THE CLAIM. If this passes, every reversible degenerate partner is
    correlated and the chain is uniquely identified by adjacent-dwell
    correlation. If it fails, report the rank test instead and narrow the
    abstract accordingly."""
    lo = extremal_correlation_reversible(CHAIN, LEVELS, sense="min", n_samples=200)
    assert lo.success

    # The rank difference survives even at the correlation minimum.
    assert joint_density_rank(CHAIN, LEVELS) == 1
    assert joint_density_rank(lo.Q, LEVELS) == 2

    n_req = pairs_needed(lo.rho, n_sigma=5.0)
    print(f"\nworst-case rho = {lo.rho:+.6f}, needs {n_req:,} pairs @ 5 sigma")
    print(f"Q_worst =\n{np.array2string(lo.Q, precision=4)}")

    assert abs(lo.rho) > 1e-4, (
        f"inf|rho| ~= {lo.rho:.2e}: some reversible degenerate partner is "
        "effectively invisible to the correlation test. The joint still differs "
        "in RANK -- switch the detector to a rank test on the binned joint and "
        "narrow the claim."
    )
    assert n_req < 50_000_000, f"{n_req:,} pairs is not a feasible experiment"

