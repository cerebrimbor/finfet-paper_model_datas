"""Independent check of the adjacent-dwell covariance identity.

The implementation under audit is NOT used to calculate the reference
covariance.  We construct reversible generators from conductances and evaluate

    Cov(tf,te) = (1/C) m_E^T (W - D K^T/C) m_F

against a direct Colquhoun-Hawkes matrix expression.

Run:
    python verification/verify_covariance_independent.py
"""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stochastic_ctmc.generator import stationary_distribution
from stochastic_ctmc.correlation import analytic_adjacent_moments


def make_reversible(pi_e, pi_f, W):
    ne, nf = W.shape
    Q = np.zeros((ne + nf, ne + nf))
    Q[:ne, ne:] = W / pi_e[:, None]
    Q[ne:, :ne] = W.T / pi_f[:, None]
    Q[np.diag_indices_from(Q)] = -Q.sum(axis=1)
    return Q


def direct_covariance(Q, level_map):
    E = np.flatnonzero(np.asarray(level_map) == 0)
    F = np.flatnonzero(np.asarray(level_map) == 1)
    Q_EE = Q[np.ix_(E, E)]
    Q_EF = Q[np.ix_(E, F)]
    Q_FF = Q[np.ix_(F, F)]
    Q_FE = Q[np.ix_(F, E)]

    pi = stationary_distribution(Q)
    phi = pi[E] @ Q_EF
    phi = phi / phi.sum()

    invF = np.linalg.inv(Q_FF)
    invE = np.linalg.inv(Q_EE)
    oneF = np.ones(len(F))
    oneE = np.ones(len(E))

    mean_f = -(phi @ invF @ oneF)
    psi = phi @ (-invF) @ Q_FE
    mean_e = -(psi @ invE @ oneE)

    cross = (
        phi @ invF @ invF @ Q_FE @ invE @ invE @ Q_EF @ oneF
    )
    return float(cross - mean_f * mean_e)


def main():
    rng = np.random.default_rng(20260808)
    level_map = np.array([0, 0, 0, 1, 1, 1])

    max_err = 0.0
    for trial in range(30):
        pi_e = rng.uniform(0.2, 2.0, 3)
        pi_f = rng.uniform(0.2, 2.0, 3)
        W = rng.uniform(0.3, 2.0, (3, 3))
        Q = make_reversible(pi_e, pi_f, W)

        # Direct identity RHS.
        D = W.sum(axis=1)
        K = W.sum(axis=0)
        C = W.sum()

        # Mean dwell vectors from the actual sub-generator blocks.
        E = np.flatnonzero(level_map == 0)
        F = np.flatnonzero(level_map == 1)
        mE = np.linalg.solve(-Q[np.ix_(E, E)], np.ones(3))
        mF = np.linalg.solve(-Q[np.ix_(F, F)], np.ones(3))

        rhs = mE @ (W - np.outer(D, K) / C) @ mF / C
        lhs = direct_covariance(Q, level_map)
        err = abs(lhs - rhs)
        max_err = max(max_err, err)

    print(f"maximum absolute identity error over 30 cases: {max_err:.3e}")
    assert max_err < 1e-11
    print("PASS")


if __name__ == "__main__":
    main()
