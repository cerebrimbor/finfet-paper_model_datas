"""Independent K(2,2) / K(3,3) blind-spot verification.

The construction is independent of topology.zero_correlation_partner().
For K(3,3), a centred rank-1 perturbation is chosen directly in the
transportation-polytope tangent space and made orthogonal to the covariance
functional. The joint density is then evaluated directly with matrix
exponentials.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stochastic_ctmc.generator import stationary_distribution


def generator_from_coupling(pi_e, pi_f, W):
    ne, nf = W.shape
    Q = np.zeros((ne + nf, ne + nf))
    Q[:ne, ne:] = W / pi_e[:, None]
    Q[ne:, :ne] = W.T / pi_f[:, None]
    Q[np.diag_indices_from(Q)] = -Q.sum(axis=1)
    return Q


def covariance_from_W(pi_e, pi_f, W):
    D, K, C = W.sum(1), W.sum(0), W.sum()
    mE, mF = pi_e / D, pi_f / K
    return float(mE @ (W - np.outer(D, K) / C) @ mF / C)


def direct_joint(Q, ne, tf, te):
    E = np.arange(ne)
    F = np.arange(ne, Q.shape[0])
    Qee, Qef = Q[np.ix_(E, E)], Q[np.ix_(E, F)]
    Qff, Qfe = Q[np.ix_(F, F)], Q[np.ix_(F, E)]
    pi = stationary_distribution(Q)
    phi = pi[E] @ Qef
    phi /= phi.sum()
    tail = Qef @ np.ones(len(F))
    return np.array([
        [phi @ expm(Qff*x) @ Qfe @ expm(Qee*y) @ tail
         for y in te]
        for x in tf
    ])


def main():
    rng = np.random.default_rng(77)

    # ---- K(2,2): the centred coupling has only one degree of freedom. ----
    for _ in range(100):
        W = rng.uniform(0.2, 2.0, (2, 2))
        D, K, C = W.sum(1), W.sum(0), W.sum()
        Wc = W - np.outer(D, K) / C
        template = np.array([[1., -1.], [-1., 1.]])
        assert np.max(np.abs(Wc - Wc[0, 0] * template)) < 1e-12

    # For generic unequal dwell means, rho=0 therefore forces Wc=0.
    # A product coupling is exactly the rank-1 case.
    pi_e = np.array([0.35, 0.65])
    pi_f = np.array([0.60, 0.40])
    D = np.array([1.0, 2.0])
    K = np.array([1.5, 1.5])
    W0 = np.outer(D, K) / D.sum()
    assert abs(covariance_from_W(pi_e, pi_f, W0)) < 1e-14

    # ---- K(3,3): nonzero centred perturbation with zero covariance. ----
    pi_e = rng.uniform(0.5, 1.5, 3); pi_e /= pi_e.sum()
    pi_f = rng.uniform(0.5, 1.5, 3); pi_f /= pi_f.sum()
    D = rng.uniform(0.7, 1.7, 3)
    K = rng.uniform(0.7, 1.7, 3); K *= D.sum() / K.sum()
    C = D.sum()
    W0 = np.outer(D, K) / C

    x, y = pi_e / D, pi_f / K
    U = np.array([[1., 0.], [0., 1.], [-1., -1.]])
    a, b = U.T @ x, U.T @ y

    # u is exactly orthogonal to a, hence a^T (u v^T) b = 0.
    u = np.array([-a[1], a[0]])
    v = np.array([0.7, -1.2])
    T = np.outer(u, v)
    assert np.linalg.norm(T) > 0
    assert abs(a @ T @ b) < 1e-14

    M = U @ T @ U.T
    neg = M < 0
    step = 0.90 * np.min(-W0[neg] / M[neg])
    W = W0 + step * M

    assert np.all(W > 0)
    assert np.allclose(W.sum(1), D)
    assert np.allclose(W.sum(0), K)

    cov = covariance_from_W(pi_e, pi_f, W)
    assert abs(cov) < 1e-12

    Q = generator_from_coupling(pi_e, pi_f, W)
    tf = np.linspace(0.02, 3.0, 14)
    te = np.linspace(0.02, 3.0, 14)
    J = direct_joint(Q, 3, tf, te)
    s = np.linalg.svd(J, compute_uv=False)
    ratios = s / s[0]
    rank_est = int(np.sum(ratios > 1e-7))

    print("K(2,2): centred coupling has one nontrivial degree of freedom -> no generic blind spot")
    print(f"K(3,3): covariance = {cov:.3e}")
    print(f"K(3,3): joint singular-value ratios = {ratios[:4]}")
    print(f"K(3,3): sampled joint rank estimate = {rank_est}")

    assert rank_est >= 2
    print("PASS")


if __name__ == "__main__":
    main()
