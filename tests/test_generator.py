"""Phase 1 tests for the CTMC generator matrix and its validation."""

import numpy as np
import pytest

from stochastic_ctmc.generator import (
    GeneratorError,
    two_state_generator,
    n_state_generator,
    validate_generator,
    stationary_distribution,
    is_detailed_balance,
)


# --- Construction -----------------------------------------------------------

def test_two_state_generator_rows_sum_to_zero():
    Q = two_state_generator(1.0, 2.0)
    assert np.allclose(Q.sum(axis=1), 0.0)
    assert Q[0, 1] == 1.0 and Q[1, 0] == 2.0
    assert Q[0, 0] == -1.0 and Q[1, 1] == -2.0


def test_two_state_generator_rejects_negative_rate():
    with pytest.raises(GeneratorError):
        two_state_generator(-1.0, 2.0)


def test_n_state_generator_assembles_diagonal():
    Q = n_state_generator({(0, 1): 1.0, (1, 0): 1.0, (1, 2): 3.0, (2, 1): 0.5}, n_states=3)
    assert np.allclose(Q.sum(axis=1), 0.0)
    assert Q[1, 1] == -(1.0 + 3.0)  # row 1 leaves to 0 (rate 1) and 2 (rate 3)


def test_n_state_generator_rejects_diagonal_input():
    with pytest.raises(GeneratorError):
        n_state_generator({(1, 1): 2.0}, n_states=3)


def test_n_state_generator_rejects_out_of_bounds():
    with pytest.raises(GeneratorError):
        n_state_generator({(0, 5): 1.0}, n_states=3)


# --- Validation -------------------------------------------------------------

def test_validate_accepts_well_formed():
    Q = two_state_generator(0.7, 1.3)
    out = validate_generator(Q)
    assert out.dtype == float


def test_validate_rejects_non_square():
    with pytest.raises(GeneratorError):
        validate_generator(np.zeros((2, 3)))


def test_validate_rejects_negative_offdiagonal():
    Q = np.array([[1.0, -1.0], [1.0, -1.0]])
    with pytest.raises(GeneratorError):
        validate_generator(Q)


def test_validate_rejects_nonzero_row_sum():
    Q = np.array([[-1.0, 1.0], [2.0, -1.0]])  # row 1 sums to +1
    with pytest.raises(GeneratorError):
        validate_generator(Q)


def test_validate_rejects_nonfinite():
    Q = np.array([[-np.inf, np.inf], [1.0, -1.0]])
    with pytest.raises(GeneratorError):
        validate_generator(Q)


def test_validate_is_scale_invariant():
    """Q -> kQ is the same chain quoted in a different unit of time, so validity
    cannot depend on k. atol used to be an absolute cutoff, which broke this in
    both directions: a numerically assembled generator carries row-sum residuals
    of order eps*max|Q|, so rescaling to fast rates pushed a perfectly good
    generator over a fixed threshold, while rescaling to slow rates would drop a
    genuine error under it. Both directions are asserted here.
    """
    rng = np.random.default_rng(31337)
    edges = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)]
    Q = n_state_generator({e: float(r) for e, r in
                           zip(edges, np.exp(rng.uniform(-2, 2, 6)))}, n_states=3)

    for k in (1e-12, 1e-6, 1e3, 1e9, 1e12):
        validate_generator(k * Q)                     # must not raise

    bad = np.array([[-1.0, 1.0], [2.0, -1.0]])        # row 1 sums to +1
    for k in (1e-12, 1e-6, 1e3, 1e9, 1e12):
        with pytest.raises(GeneratorError):
            validate_generator(k * bad)


# --- Stationary distribution ------------------------------------------------

def test_two_state_stationary_matches_analytic():
    # For rates a (0->1) and b (1->0): pi = [b, a] / (a + b).
    a, b = 1.0, 3.0
    Q = two_state_generator(a, b)
    pi = stationary_distribution(Q)
    assert np.allclose(pi, np.array([b, a]) / (a + b))


def test_stationary_is_left_null_vector():
    Q = n_state_generator(
        {(0, 1): 1.0, (1, 0): 1.0, (1, 2): 2.0, (2, 1): 2.0, (0, 2): 0.5, (2, 0): 0.5},
        n_states=3,
    )
    pi = stationary_distribution(Q)
    assert np.allclose(pi @ Q, 0.0, atol=1e-9)
    assert np.isclose(pi.sum(), 1.0)


# --- Detailed balance -------------------------------------------------------

def test_detailed_balance_holds_for_reversible_chain():
    # Any 2-state chain is reversible.
    assert is_detailed_balance(two_state_generator(1.0, 4.0))
    # Symmetric 3-cycle is reversible.
    Q = n_state_generator(
        {(0, 1): 1.0, (1, 0): 1.0, (1, 2): 2.0, (2, 1): 2.0, (0, 2): 0.5, (2, 0): 0.5},
        n_states=3,
    )
    assert is_detailed_balance(Q)


def test_detailed_balance_fails_for_cyclic_chain():
    # A pure one-directional 3-cycle (0->1->2->0) has net probability flux and is
    # NOT reversible, even though it has a valid stationary distribution.
    Q = n_state_generator({(0, 1): 1.0, (1, 2): 1.0, (2, 0): 1.0}, n_states=3)
    pi = stationary_distribution(Q)
    assert np.allclose(pi, 1 / 3)  # symmetric => uniform stationary
    assert not is_detailed_balance(Q, pi)


# --- Edge cases -------------------------------------------------------------

def test_absorbing_state_generator_is_valid():
    # State 1 is absorbing: no outgoing rate. Still a valid generator.
    Q = n_state_generator({(0, 1): 1.0}, n_states=2)
    validate_generator(Q)
    assert Q[1, 1] == 0.0


def test_zero_generator_is_valid():
    Q = n_state_generator({}, n_states=3)
    assert np.allclose(Q, 0.0)
    validate_generator(Q)
