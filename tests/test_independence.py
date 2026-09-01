"""The chi-squared independence test, and the boundary where it is not needed.

chi2 on the quantile-binned joint is a rank-1 test, and in general it catches
dependence that correlation misses -- the y = |x| case below is the miniature.
On K(2,2), however -- two sub-states per level, reversible -- it buys nothing
over rho, and that boundary is proved here rather than assumed:

    Cov(t_f, t_e) = det(W) (mE0 - mE1) (mF0 - mF1) / C^2

so rho vanishes only when the joint genuinely has rank 1. There is no rho = 0
blind spot to close at two sub-states. A blind spot needs >= 3 sub-states per
level, where the gateway's rank profile stops being a single scalar and can
therefore decouple from the single scalar rho.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from stochastic_ctmc.generator import n_state_generator, is_detailed_balance
from stochastic_ctmc.dwell import joint_density_rank, dwell_pdf_1d
from stochastic_ctmc.topology import (
    partner_with_rho, bipartite_reversible_generator, gateway_rank_ratio,
    mixture_params, LEVELS_4,
)
from stochastic_ctmc.correlation import analytic_adjacent_moments
from stochastic_ctmc.independence import (
    quantile_bin_edges, binned_joint, independence_test,
)

CHAIN = n_state_generator(
    {(0, 1): 5.0, (1, 0): 5.0,
     (1, 2): 1.0, (2, 1): 1.0,
     (2, 3): 4.0, (3, 2): 4.0},
    n_states=4,
)
LEVELS = list(LEVELS_4)


# --- Binning mechanics ------------------------------------------------------

def test_quantile_bins_are_equal_frequency():
    rng = np.random.default_rng(0)
    t = rng.exponential(2.0, 10000)
    counts, _ = np.histogram(t, bins=quantile_bin_edges(t, 5))
    assert np.all(np.abs(counts - 2000) < 100)


def test_quantile_bins_capture_every_sample():
    rng = np.random.default_rng(1)
    t = rng.exponential(1.0, 5000)
    assert np.histogram(t, bins=quantile_bin_edges(t, 4))[0].sum() == 5000


def test_quantile_bins_reject_too_few():
    with pytest.raises(ValueError):
        quantile_bin_edges(np.arange(100.0), 1)


def test_binned_joint_totals_match():
    rng = np.random.default_rng(2)
    a, b = rng.exponential(1.0, 3000), rng.exponential(2.0, 3000)
    table = binned_joint(a, b, quantile_bin_edges(a, 4), quantile_bin_edges(b, 4))
    assert table.sum() == 3000


# --- Calibration: the test must not cry wolf --------------------------------

def test_independence_test_accepts_independent_data():
    rng = np.random.default_rng(3)
    n = 20000
    res = independence_test(rng.exponential(1.0, n), rng.exponential(2.0, n))
    assert not res.rejects_independence
    assert res.cramers_v < 0.05
    assert res.min_expected > 100


def test_independence_test_detects_obvious_dependence():
    rng = np.random.default_rng(4)
    n = 20000
    x = rng.exponential(1.0, n)
    res = independence_test(x, x + 0.3 * rng.exponential(1.0, n))
    assert res.rejects_independence
    assert res.n_sigma > 10


def test_independence_test_detects_uncorrelated_dependence():
    # THE POINT, in miniature: y = |x| with x symmetric has rho = 0 exactly,
    # yet x and y are maximally dependent. Correlation is blind; chi2 is not.
    rng = np.random.default_rng(5)
    n = 20000
    x = rng.normal(0, 1, n)
    y = np.abs(x) + 0.05 * rng.normal(0, 1, n)
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.05
    assert independence_test(x, y).rejects_independence


def test_independence_test_rejects_unpaired_or_tiny_input():
    with pytest.raises(ValueError):
        independence_test(np.zeros(500), np.zeros(400))
    with pytest.raises(ValueError):
        independence_test(np.zeros(50), np.zeros(50))


# --- NO BLIND SPOT AT TWO SUB-STATES ----------------------------------------

def _reversible(pi, c):
    """Reversible K(2,2) generator from stationary weights and conductances."""
    pi = np.asarray(pi, dtype=float)
    return bipartite_reversible_generator(np.log(pi / pi.sum()),
                                          np.log(np.asarray(c, dtype=float)))


def _cov_closed_form(pi, c):
    """det(W) (mE0 - mE1) (mF0 - mF1) / C^2."""
    pi = np.asarray(pi, dtype=float)
    pi = pi / pi.sum()
    c = np.asarray(c, dtype=float)
    d0, d1 = c[0] + c[1], c[2] + c[3]      # W row sums: E sub-state exit rates
    k0, k1 = c[0] + c[2], c[1] + c[3]      # W col sums: F sub-state exit rates
    det_w = c[0] * c[3] - c[1] * c[2]
    return (det_w * (pi[0] / d0 - pi[1] / d1) * (pi[2] / k0 - pi[3] / k1)
            / c.sum() ** 2)


def test_adjacent_covariance_factorises_into_three_mechanisms():
    """Cov(t_f, t_e) = det(W) (mE0 - mE1) (mF0 - mF1) / C^2 on reversible K(2,2).

    W is the conductance matrix; its (i,j) entry IS the stationary flux across
    edge i-j, because detailed balance gives pi_i q_ij = c_ij. A bipartite level
    has no internal transitions, so a dwell is a single exponential fixed by the
    sub-state entered, and the adjacent sub-state joint is exactly W / C. The
    covariance therefore has three independent ways to vanish, not one.

    Tolerance is 1e-8 relative rather than machine epsilon because the library
    forms cov as E[t_f t_e] - E[t_f]E[t_e], a difference of near-equal terms
    that sheds roughly -log10|rho| significant digits to cancellation.
    """
    rng = np.random.default_rng(11)
    for _ in range(200):
        pi = rng.uniform(0.1, 1.0, 4)
        c = rng.uniform(0.05, 5.0, 4)
        cov = analytic_adjacent_moments(_reversible(pi, c), LEVELS, 1).cov
        assert cov == pytest.approx(_cov_closed_form(pi, c), rel=1e-8, abs=1e-18)


def test_rho_zero_forces_a_rank_one_joint_at_two_substates():
    """Every way of making rho vanish also collapses the 2-D joint to rank 1.

    det(W) = 0 degenerates the gateway. mE0 = mE1 makes Q_EE a multiple of the
    identity, so exp(Q_EE t_e) = exp(-lam t_e) I and the t_e dependence leaves
    f(t_f, t_e) as a scalar factor -- the joint separates whatever the gateway
    does. mF0 = mF1 is the same statement on the other level. Hence on K(2,2)
    rho = 0 iff the joint factorises, and correlation has NO blind spot.
    """
    rng = np.random.default_rng(12)
    for _ in range(30):
        pi = rng.uniform(0.2, 1.0, 4)
        c = rng.uniform(0.2, 3.0, 4)

        generic = _reversible(pi, c)
        assert joint_density_rank(generic, LEVELS) == 2
        assert abs(analytic_adjacent_moments(generic, LEVELS, 1).rho) > 1e-12

        d0, d1 = c[0] + c[1], c[2] + c[3]
        k0, k1 = c[0] + c[2], c[1] + c[3]
        branches = {
            "det W = 0": (pi, [c[0], c[1], c[2], c[1] * c[2] / c[0]]),
            "mE0 = mE1": ([pi[0], pi[0] * d1 / d0, pi[2], pi[3]], c),
            "mF0 = mF1": ([pi[0], pi[1], pi[2], pi[2] * k1 / k0], c),
        }
        for name, (pi_d, c_d) in branches.items():
            Q = _reversible(pi_d, c_d)
            assert abs(analytic_adjacent_moments(Q, LEVELS, 1).rho) < 1e-9, name
            assert joint_density_rank(Q, LEVELS) == 1, name

        # The two dwell-degeneracy branches keep a FULL-RANK gateway: rho = 0
        # does not imply the gateway collapsed, only that the joint did.
        for name in ("mE0 = mE1", "mF0 = mF1"):
            pi_d, c_d = branches[name]
            assert gateway_rank_ratio(_reversible(pi_d, c_d), LEVELS) > 1e-3, name


@pytest.mark.slow
def test_no_blind_spot_for_two_substates():
    """Searching for the blind spot on CHAIN's manifold correctly comes up empty.

    CHAIN's pinned marginals are genuine two-exponential mixtures with distinct
    rates, so mE0 = mE1 and mF0 = mF1 are both excluded on this manifold. That
    leaves det(W) = 0 as the only route to rho = 0 -- and det(W) = 0 is exactly
    a rank-1 gateway. So rho and the rank obstruction vanish together and no
    rank-2 partner with rho = 0 exists to be found. The solver still returns a
    perfectly marginal-matched generator; it is simply rank 1.
    """
    res = partner_with_rho(CHAIN, 0.0, LEVELS, n_restarts=40)

    assert not res.success, (
        "a rank-2 partner with rho = 0 was found, contradicting "
        f"Cov = det(W)(mE0-mE1)(mF0-mF1)/C^2: {res!r}")
    assert res.best_rank_ratio < 1e-6, (
        f"search reached gateway ratio {res.best_rank_ratio:.2e}, which is "
        "genuinely non-degenerate -- theorem or filter is wrong")
    assert joint_density_rank(res.Q, LEVELS) == 1
    assert is_detailed_balance(res.Q, atol=1e-8)
    assert res.signature_error < 1e-7          # still 1-D indistinguishable

    t = np.linspace(0.0, 12.0, 400)
    for lev in (0, 1):
        assert np.max(np.abs(dwell_pdf_1d(CHAIN, LEVELS, t, lev)
                             - dwell_pdf_1d(res.Q, LEVELS, t, lev))) < 1e-7

    print("\n--- no blind spot on K(2,2) ---")
    print(f"rho reached        = {res.rho:+.3e}")
    print(f"best gateway ratio = {res.best_rank_ratio:.3e} (threshold 1e-6)")
    print(f"joint rank         = {joint_density_rank(res.Q, LEVELS)}")


@pytest.mark.slow
def test_rho_is_a_fixed_multiple_of_the_gateway_determinant():
    """rho = k det(gateway), with k fixed by the pinned marginals alone.

        k = psi0 psi1 mE0 mE1 (mE0 - mE1)(mF0 - mF1) / sqrt(var_f var_e)

    using pi0 = mE0 psi0 C to clear the stationary weights. Every factor is a
    1-D marginal quantity, which is what makes k a constant of the manifold and
    rho an exact proxy for the gateway determinant. Compared on magnitude: the
    sign tracks sub-state labelling, and sub-states are unobservable.
    """
    def k_from_marginals(Q):
        rE, aE = mixture_params(Q, LEVELS, level=0)   # f = sum a_k exp(-r_k t)
        rF, aF = mixture_params(Q, LEVELS, level=1)
        psiE, psiF = aE / rE, aF / rF                 # mixture weights
        mE, mF = 1.0 / rE, 1.0 / rF                   # sub-state mean dwells
        var = lambda psi, r: 2.0 * np.sum(psi / r**2) - np.sum(psi / r) ** 2
        return (psiE[0] * psiE[1] * mE[0] * mE[1]
                * (mE[0] - mE[1]) * (mF[0] - mF[1])
                / np.sqrt(var(psiF, rF) * var(psiE, rE)))

    predicted = abs(k_from_marginals(CHAIN))
    measured = []
    for target in (0.02, 0.01, 1e-3, -1e-3):
        r = partner_with_rho(CHAIN, target, LEVELS, n_restarts=25)
        assert r.success, f"target {target}: {r!r}"
        measured.append(r.rho / np.linalg.det(r.Q[np.ix_([0, 1], [2, 3])]))

    print(f"\nk predicted from CHAIN's marginals alone: {predicted:.8f}")
    print("k measured on the manifold: "
          + ", ".join(f"{k:+.8f}" for k in measured))

    assert np.allclose(np.abs(measured), predicted, rtol=1e-6)
    assert np.ptp(measured) < 1e-6 * abs(np.mean(measured))


# --- THE THEOREM SURVIVES INTRA-LEVEL CONNECTIVITY --------------------------
#
# Everything above assumes a strictly bipartite K(2,2): no e0--e1 or f0--f1
# edge, so a level's dwell is a single exponential fixed by the sub-state
# entered, and "equal mean dwells" reduces to "equal exit rates". That
# hypothesis is not innocent. The canonical device-physics trap is Grasser's
# four-state NMP model, whose two charge-exchange gateways are accompanied by
# intra-level structural relaxation at fixed charge state -- exactly the edges
# the bipartite argument excludes. A theorem that needed them absent would not
# cover the system the paper is about.
#
# It does not need them absent. Generator rows sum to zero, so
# Q_EE 1 = -Q_EF 1 identically, and therefore
#
#     m_E prop 1  <=>  1 is an eigenvector of Q_EE  <=>  Q_EF 1 prop 1
#
# which collapses the Krylov space K(Q_EF 1, Q_EE^T) to span{1} and forces the
# joint to rank 1 -- with no assumption on Q_EE's off-diagonal entries. The
# covariance identity survives verbatim once m_E and m_F are read as the
# mean-dwell VECTORS (-Q_EE)^-1 1 and (-Q_FF)^-1 1 rather than reciprocals of
# exit rates, which they equal only when the level is internally unconnected.

def _reversible_intra(pi, c, a_E=0.0, a_F=0.0):
    """Reversible 2+2 generator, optionally with intra-level edges.

    ``pi`` = (p0, p1, q0, q1) stationary weights, normalised here; ``c`` the 2x2
    gateway conductance matrix, c[i, j] being the stationary flux on edge
    e_i -- f_j; ``a_E``, ``a_F`` the conductances of e0--e1 and f0--f1. Detailed
    balance is structural: rate(x->y) = c_xy / pi_x gives pi_x q_xy = c_xy =
    pi_y q_yx on every edge, intra-level ones included.

    topology.bipartite_reversible_generator is deliberately not reused -- it
    parameterises the strictly bipartite manifold, and intra-level edges are the
    whole point here. A zero entry of ``c`` omits that gateway, which is how the
    NMP perfect-matching topology is expressed.
    """
    pi = np.asarray(pi, dtype=float)
    pi = pi / pi.sum()
    c = np.asarray(c, dtype=float)
    rates = {}
    for i in (0, 1):
        for j in (0, 1):
            if c[i, j] > 0.0:
                rates[(i, 2 + j)] = c[i, j] / pi[i]
                rates[(2 + j, i)] = c[i, j] / pi[2 + j]
    if a_E > 0.0:
        rates[(0, 1)], rates[(1, 0)] = a_E / pi[0], a_E / pi[1]
    if a_F > 0.0:
        rates[(2, 3)], rates[(3, 2)] = a_F / pi[2], a_F / pi[3]
    return n_state_generator(rates, n_states=4)


def _mean_dwell_vectors(Q):
    """((-Q_EE)^-1 1, (-Q_FF)^-1 1): mean dwell from each sub-state of each level.

    Equal to 1/exit-rate only when the level has no internal edges; with them it
    is the phase-type mean, and it is this vector the extended identity needs.
    """
    Q = np.asarray(Q, dtype=float)
    return (np.linalg.solve(-Q[np.ix_([0, 1], [0, 1])], np.ones(2)),
            np.linalg.solve(-Q[np.ix_([2, 3], [2, 3])], np.ones(2)))


def _extended_cov(Q, c):
    """det(W)(mE0 - mE1)(mF0 - mF1) / C^2, with the vector-valued means."""
    mE, mF = _mean_dwell_vectors(Q)
    c = np.asarray(c, dtype=float)
    return np.linalg.det(c) * (mE[0] - mE[1]) * (mF[0] - mF[1]) / c.sum() ** 2


def _zero_cov_branches(pi, c):
    """The three loci where the identity makes Cov vanish, as values of c11.

    The dwell branches are written as equal TOTAL GATEWAY EXIT RATES, which the
    row-sum identity makes equivalent to equal mean dwells whatever the
    intra-level edges are -- so these loci do not move when a_E or a_F changes.
    That independence is itself asserted below.
    """
    pi = np.asarray(pi, dtype=float)
    p0, p1, q0, q1 = pi / pi.sum()
    out = {}
    if c[0, 0] > 0.0:
        out["det W = 0"] = c[0, 1] * c[1, 0] / c[0, 0]
    out["mE0 = mE1"] = p1 * (c[0, 0] + c[0, 1]) / p0 - c[1, 0]
    out["mF0 = mF1"] = q1 * (c[0, 0] + c[1, 0]) / q0 - c[0, 1]
    return out


_INTRA_MODES = [("E only", True, False), ("F only", False, True),
                ("both", True, True)]


@pytest.mark.parametrize("label,on_E,on_F", _INTRA_MODES)
def test_covariance_identity_survives_intra_level_edges(label, on_E, on_F):
    """Cov = det(W)(mE0-mE1)(mF0-mF1)/C^2 with m = (-Q)^-1 1, intra-level edges on.

    Same tolerance rationale as the bipartite version above: cov is formed as a
    difference of near-equal terms and sheds digits to cancellation.
    """
    rng = np.random.default_rng(707)
    worst = 0.0
    for _ in range(120):
        pi = rng.uniform(0.1, 0.4, 4)
        c = np.exp(rng.uniform(-1.5, 1.5, size=(2, 2)))
        a_E = float(rng.uniform(0.05, 2.0)) if on_E else 0.0
        a_F = float(rng.uniform(0.05, 2.0)) if on_F else 0.0
        Q = _reversible_intra(pi, c, a_E, a_F)
        assert is_detailed_balance(Q, atol=1e-10)
        cov = analytic_adjacent_moments(Q, LEVELS, 1).cov
        worst = max(worst, abs(cov - _extended_cov(Q, c)) / max(abs(cov), 1e-30))
    assert worst < 1e-8, f"{label}: max relative error {worst:.3e}"


@pytest.mark.parametrize("label,on_E,on_F", _INTRA_MODES)
def test_zero_covariance_still_forces_rank_one_with_intra_level_edges(label, on_E, on_F):
    """rho = 0 => joint rank 1, on every branch, with intra-level connectivity.

    Also pins the mechanism: on the mE0 = mE1 branch the mean dwells really do
    coincide (the row-sum identity, not an accident of the bipartite case) while
    the gateway stays non-degenerate -- so the rank collapsed for the stated
    structural reason and not because W quietly went singular.
    """
    rng = np.random.default_rng(808)
    checked = 0
    for _ in range(40):
        pi = rng.uniform(0.15, 0.4, 4)
        c = np.exp(rng.uniform(-1.0, 1.0, size=(2, 2)))
        a_E = float(rng.uniform(0.05, 2.0)) if on_E else 0.0
        a_F = float(rng.uniform(0.05, 2.0)) if on_F else 0.0

        assert joint_density_rank(_reversible_intra(pi, c, a_E, a_F), LEVELS) == 2

        for name, c11 in _zero_cov_branches(pi, c).items():
            if c11 <= 1e-6:
                continue
            cd = np.array([[c[0, 0], c[0, 1]], [c[1, 0], c11]])
            Q = _reversible_intra(pi, cd, a_E, a_F)
            assert abs(analytic_adjacent_moments(Q, LEVELS, 1).rho) < 1e-9, (label, name)
            assert joint_density_rank(Q, LEVELS) == 1, (label, name)
            if name == "mE0 = mE1":
                mE, _ = _mean_dwell_vectors(Q)
                assert abs(mE[0] - mE[1]) < 1e-10 * mE[0], (label, name)
                assert gateway_rank_ratio(Q, LEVELS) > 1e-3, (label, name)
            checked += 1
    assert checked > 50


def test_nmp_perfect_matching_topology_has_no_blind_spot():
    """Grasser's four-state trap: gateways form a PERFECT MATCHING (c01 = c10 = 0)
    plus intra-level relaxation on both levels.

    This is the physically important topology and it sits on the boundary of the
    conductance orthant, which random sampling of the interior never visits. Its
    gateway determinant c00*c11 cannot vanish while both gateways exist, so the
    dwell branches are the only routes to rho = 0 -- and they still collapse the
    joint to rank 1.
    """
    rng = np.random.default_rng(909)
    worst, checked = 0.0, 0
    for _ in range(80):
        pi = rng.uniform(0.15, 0.4, 4)
        c = np.diag(np.exp(rng.uniform(-1.5, 1.5, 2)))
        a_E, a_F = float(rng.uniform(0.05, 2.0)), float(rng.uniform(0.05, 2.0))

        Q = _reversible_intra(pi, c, a_E, a_F)
        cov = analytic_adjacent_moments(Q, LEVELS, 1).cov
        worst = max(worst, abs(cov - _extended_cov(Q, c)) / max(abs(cov), 1e-30))

        for name, c11 in _zero_cov_branches(pi, c).items():
            if name == "det W = 0" or c11 <= 1e-6:
                continue
            cd = np.array([[c[0, 0], 0.0], [0.0, c11]])
            Qz = _reversible_intra(pi, cd, a_E, a_F)
            assert abs(analytic_adjacent_moments(Qz, LEVELS, 1).rho) < 1e-9, name
            assert joint_density_rank(Qz, LEVELS) == 1, name
            checked += 1
    assert worst < 1e-8, f"NMP identity max relative error {worst:.3e}"
    assert checked > 100


# The draw that exposed the absolute-tolerance bug in joint_density_rank: a
# root-find left Delta m_E at -7.4e-12, the E-side Krylov ratio sigma_2/sigma_1
# came out 4.99e-10, and the old ABSOLUTE tol=1e-9 read that as rank 2. Recorded
# verbatim so the regression is pinned to the exact numbers, not to an RNG
# stream that a refactor could silently renumber.
_NMP_BUG = dict(
    pi=[0.3722289219375432, 0.17509949068631228,
        0.30813808305308166, 0.14453350432306292],
    c=[[0.49484827286976985, 0.0], [0.0, 0.2327806235019816]],
    a_E=2.1372454335791686,
    a_F=1.4108538419456327,
)


def test_nmp_draw_that_broke_the_absolute_tolerance_reads_rank_one():
    Q = _reversible_intra(_NMP_BUG["pi"], _NMP_BUG["c"],
                          _NMP_BUG["a_E"], _NMP_BUG["a_F"])
    mE, _ = _mean_dwell_vectors(Q)
    assert abs(mE[0] - mE[1]) < 1e-10                      # sits on the locus
    assert abs(analytic_adjacent_moments(Q, LEVELS, 1).rho) < 1e-12
    assert joint_density_rank(Q, LEVELS) == 1
    for k in (1e-3, 1e3, 1e6):
        assert joint_density_rank(k * Q, LEVELS) == 1, f"scale {k:g}"


def test_joint_density_rank_scale_invariant_with_intra_level_edges():
    rng = np.random.default_rng(1010)
    for _ in range(30):
        pi = rng.uniform(0.15, 0.4, 4)
        c = np.exp(rng.uniform(-1.5, 1.5, size=(2, 2)))
        Q = _reversible_intra(pi, c, float(rng.uniform(0.0, 2.0)),
                              float(rng.uniform(0.0, 2.0)))
        base = joint_density_rank(Q, LEVELS)
        for k in (1e-6, 1e-3, 1e3, 1e6):
            assert joint_density_rank(k * Q, LEVELS) == base, f"scale {k:g}"


@pytest.mark.slow
def test_searched_not_constructed_no_rank_two_zero_correlation_point():
    """Search rather than construction: sweep a gateway conductance, root-find
    every Cov = 0 crossing, and read the rank there.

    The constructed-loci test above assumes the identity in order to know where
    the zeros are. This one does not -- it finds them numerically, so it would
    catch a zero the identity does not predict.
    """
    rng = np.random.default_rng(1111)
    roots, offenders = 0, []
    for _ in range(60):
        pi = rng.uniform(0.15, 0.4, 4)
        c = np.exp(rng.uniform(-1.0, 1.0, size=(2, 2)))
        a_E, a_F = float(rng.uniform(0.02, 2.0)), float(rng.uniform(0.02, 2.0))

        def cov_at(x):
            cd = np.array([[c[0, 0], c[0, 1]], [c[1, 0], x]])
            return analytic_adjacent_moments(
                _reversible_intra(pi, cd, a_E, a_F), LEVELS, 1).cov

        grid = np.geomspace(1e-2, 20.0, 60)
        v = np.array([cov_at(x) for x in grid])
        for i in np.flatnonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0):
            r = brentq(cov_at, grid[i], grid[i + 1], xtol=1e-14)
            roots += 1
            cd = np.array([[c[0, 0], c[0, 1]], [c[1, 0], r]])
            Q = _reversible_intra(pi, cd, a_E, a_F)
            if joint_density_rank(Q, LEVELS) != 1:
                offenders.append((pi.tolist(), cd.tolist(), a_E, a_F))

    print(f"\n--- intra-level Cov=0 search: {roots} roots, "
          f"{len(offenders)} with rank > 1 ---")
    assert roots > 30, f"only {roots} zero-crossings found; search too weak"
    assert not offenders, f"rank > 1 at rho = 0 with intra-level edges: {offenders[:2]}"