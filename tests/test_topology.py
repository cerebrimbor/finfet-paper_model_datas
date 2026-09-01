"""The degenerate pair: identical 1-D marginals, different 2-D joints.

If these tests fail, the paper has no contribution.
"""

import numpy as np
import pytest

from stochastic_ctmc.generator import n_state_generator
from stochastic_ctmc.dwell import (
    dwell_pdf_1d, dwell_pdf_2d, gateway_block_rank, joint_density_rank,
)
from stochastic_ctmc.topology import (
    bipartite_generator, mixture_params, match_marginals, LEVELS_4,
)

# E2 <-> E1 <-> F1 <-> F2: one gateway (F1 <-> E1), rank(Q_FE) = 1.
CHAIN = n_state_generator(
    {(0, 1): 5.0, (1, 0): 5.0,
     (1, 2): 1.0, (2, 1): 1.0,
     (2, 3): 4.0, (3, 2): 4.0},
    n_states=4,
)
LEVELS = list(LEVELS_4)


# --- Constructor ------------------------------------------------------------

def test_bipartite_generator_is_valid_and_rank_two():
    Q = bipartite_generator([1.0, 0.5, 1.5, 2.5, 2.0, 3.0, 4.0, 2.0])
    assert np.allclose(Q.sum(axis=1), 0.0)
    assert gateway_block_rank(Q, LEVELS) == 2


def test_bipartite_has_no_intra_level_transitions():
    Q = bipartite_generator([1.0, 0.5, 1.5, 2.5, 2.0, 3.0, 4.0, 2.0])
    assert Q[0, 1] == 0.0 and Q[1, 0] == 0.0   # E1 <-> E2
    assert Q[2, 3] == 0.0 and Q[3, 2] == 0.0   # F1 <-> F2


def test_bipartite_rejects_wrong_length():
    with pytest.raises(ValueError):
        bipartite_generator([1.0, 2.0, 3.0])


# --- Mixture parameterisation ----------------------------------------------

def test_mixture_params_reconstruct_the_pdf():
    for Q in (CHAIN, bipartite_generator([1.0, 0.5, 1.5, 2.5, 2.0, 3.0, 4.0, 2.0])):
        for lev in (0, 1):
            r, c = mixture_params(Q, LEVELS, lev)
            t = np.linspace(0, 5, 120)
            recon = (c[None, :] * np.exp(-r[None, :] * t[:, None])).sum(axis=1)
            assert np.allclose(recon, dwell_pdf_1d(Q, LEVELS, t, lev), atol=1e-12)


def test_mixture_params_normalise():
    # int f dt = sum_k c_k / r_k = 1.
    for lev in (0, 1):
        r, c = mixture_params(CHAIN, LEVELS, lev)
        assert np.isclose((c / r).sum(), 1.0)


def test_bipartite_mixture_rates_are_the_exit_rates():
    # Q_FF is diagonal for a bipartite topology, so the filled-dwell rates are
    # just the two total exit rates -- no matrix algebra needed to predict them.
    Q = bipartite_generator([1.0, 0.5, 1.5, 2.5, 2.0, 3.0, 4.0, 2.0])
    r, _ = mixture_params(Q, LEVELS, level=1)
    assert np.allclose(np.sort(r), np.sort([2.0 + 3.0, 4.0 + 2.0]))


# --- THE RESULT -------------------------------------------------------------

@pytest.mark.slow
def test_degenerate_pair_exists():
    """A rank-2 generator with the chain's exact 1-D marginals."""
    res = match_marginals(CHAIN, LEVELS)
    assert res.success, f"no match found: {res!r}"
    assert res.signature_error < 1e-7


@pytest.mark.slow
def test_degenerate_pair_is_invisible_to_1d_analysis():
    res = match_marginals(CHAIN, LEVELS)
    assert res.success

    t = np.linspace(0.0, 12.0, 400)
    for lev in (0, 1):
        f_chain = dwell_pdf_1d(CHAIN, LEVELS, t, lev)
        f_match = dwell_pdf_1d(res.Q, LEVELS, t, lev)
        # Agreement to ~1e-9 absolute: no finite dwell-time histogram can ever
        # separate these two topologies from marginals alone.
        assert np.max(np.abs(f_chain - f_match)) < 1e-7


@pytest.mark.slow
def test_degenerate_pair_is_visible_to_2d_analysis():
    res = match_marginals(CHAIN, LEVELS)
    assert res.success

    # The topologies genuinely differ.
    assert gateway_block_rank(CHAIN, LEVELS) == 1
    assert gateway_block_rank(res.Q, LEVELS) == 2
    assert joint_density_rank(CHAIN, LEVELS) == 1
    assert joint_density_rank(res.Q, LEVELS) == 2

    t = np.linspace(0.02, 4.0, 40)
    j_chain = dwell_pdf_2d(CHAIN, LEVELS, t, t)
    j_match = dwell_pdf_2d(res.Q, LEVELS, t, t)

    assert np.linalg.matrix_rank(j_chain, tol=1e-10 * j_chain.max()) == 1
    assert np.linalg.matrix_rank(j_match, tol=1e-10 * j_match.max()) == 2

    # And the joints differ by an amount no experiment would call negligible.
    rel = np.max(np.abs(j_chain - j_match)) / j_chain.max()
    assert rel > 0.01, f"joints differ by only {rel:.2%} -- too weak to detect"


@pytest.mark.slow
def test_matched_generator_is_not_a_relabelling_of_the_chain():
    # Guard against the trivial escape: the optimiser must not have found the
    # chain itself under a permutation. Bipartite has zero intra-level rates by
    # construction, so it CANNOT be the chain -- assert the topologies differ
    # where it counts.
    res = match_marginals(CHAIN, LEVELS)
    assert res.success
    assert res.Q[2, 3] == 0.0 and CHAIN[2, 3] == 4.0
    assert not np.allclose(np.sort(res.Q.ravel()), np.sort(CHAIN.ravel()))