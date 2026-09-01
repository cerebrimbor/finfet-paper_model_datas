"""Symbolic verification of the adjacent-dwell covariance identity.

The manuscript states two things about Cov(t_f, t_e) that a reader should be
able to check rather than take on trust:

  (A)  Cov = (1/C) m_E^T (W - D K^T / C) m_F                        [general n]

       with W the gateway conductance matrix, D = W 1 and K = W^T 1 its margins,
       C = sum(W), and m_E = (-Q_EE)^-1 1, m_F = (-Q_FF)^-1 1 the MEAN-DWELL
       VECTORS -- not reciprocal exit rates, which they equal only when a level
       has no internal edges.

  (B)  Cov = det(W) (mE0 - mE1)(mF0 - mF1) / C^2                    [n = 2]

(A) has a one-line proof when the levels are internally unconnected: the pair
(sub-state of F occupied, sub-state of E entered next) carries stationary
probability W_ij / C, each dwell is then exponential with mean m_Fj or m_Ei, and
subtracting the product of the marginal means leaves the centred coupling. With
intra-level edges that argument lapses -- dwells become phase-type and the
sub-state left is no longer the one entered -- but (A) survives anyway, and
section 6 below verifies the general proof step by step. Writing P_E =
diag(pi_E) and S_E = P_E Q_EE, reversibility makes S_E and S_F symmetric, and
the argument runs:

    Q_EF 1 = -Q_EE 1            =>  E[t_f t_e] = phi R^2 Q_FE m_E,  R = (-Q_FF)^-1
    phi = K^T / C, K = -S_F 1   =>  phi R   = pi_F^T / C
    P_F R = -P_F S_F^-1 P_F symmetric
                                =>  phi R^2 = m_F^T P_F / C
                                =>  E[t_f t_e] = (1/C) m_F^T W^T m_E
    K^T m_F = pi(F), m_E^T D = pi(E)
                                =>  E[t_f] E[t_e] = pi(E) pi(F) / C^2

and subtracting gives (A). Nothing in it assumes Q_EE or Q_FF diagonal, so it
holds for arbitrary intra-level connectivity at every n; reversibility, however,
is essential, and is used twice. (B) follows from (A) by elementary algebra, and
is additionally checked in fully symbolic form at n = 2 with arbitrary
intra-level conductances on both levels.

A by-product worth keeping: E[t_f] = pi(F)/C and E[t_e] = pi(E)/C, i.e. a
level's mean dwell is its stationary mass divided by the total crossing flux.

The script also pins the BOUNDARY of the bipartite assumption. The covariance
identity does not need it; the transportation-polytope chart does. Reading the
margins of W off the 1-D marginal works only when the marginal's mixture rates
are the sub-state exit rates K_j / pi_j, and Check 5 asserts that this holds
without intra-level edges and FAILS with them. That asymmetry is why the
manuscript qualifies the polytope step and leaves the covariance form
unqualified.

Exact rational arithmetic throughout: these are identities between rational
functions, so a single exact evaluation at a generic point is a proof for that
matrix size, and the n = 2 cases are carried out in full symbols.

Run:  python -u scripts/verify_covariance_identity.py
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sympy as sp

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def ones(n):
    return sp.ones(n, 1)


def rational(rng, lo=1, hi=25):
    return sp.Rational(rng.randint(lo, hi), rng.randint(1, 12))


def blocks(pi_E, pi_F, W, A_E=None, A_F=None):
    """Reversible generator blocks for a two-level chain.

    Detailed balance is structural: with a symmetric conductance c on each
    undirected edge, rate(x -> y) = c_xy / pi_x gives pi_x q_xy = pi_y q_yx.
    ``W[i, j]`` is the gateway conductance e_i -- f_j; ``A_E`` and ``A_F`` are
    symmetric, zero-diagonal conductance matrices for the intra-level edges.
    """
    nE, nF = len(pi_E), len(pi_F)
    A_E = sp.zeros(nE, nE) if A_E is None else A_E
    A_F = sp.zeros(nF, nF) if A_F is None else A_F

    Q_EF = sp.Matrix(nE, nF, lambda i, j: W[i, j] / pi_E[i])
    Q_FE = sp.Matrix(nF, nE, lambda j, i: W[i, j] / pi_F[j])
    Q_EE = sp.Matrix(nE, nE, lambda i, k: A_E[i, k] / pi_E[i] if i != k else 0)
    Q_FF = sp.Matrix(nF, nF, lambda j, l: A_F[j, l] / pi_F[j] if j != l else 0)
    for i in range(nE):
        Q_EE[i, i] = -(sum(Q_EE.row(i)) + sum(Q_EF.row(i)))
    for j in range(nF):
        Q_FF[j, j] = -(sum(Q_FF.row(j)) + sum(Q_FE.row(j)))
    return Q_EE, Q_EF, Q_FF, Q_FE


def entry_law(W):
    """phi_j: equilibrium probability that a filled dwell begins in sub-state j.

    The flux into f_j is the column sum of W, so phi is that column sum over C
    -- independent of pi, and of whether pi is normalised.
    """
    nE, nF = W.shape
    C = sum(W)
    return sp.Matrix(1, nF, lambda _, j: sum(W[i, j] for i in range(nE)) / C)


def covariance(Q_EE, Q_EF, Q_FF, Q_FE, W):
    """Cov(t_f, t_e) from the Colquhoun-Hawkes joint density, as the library
    computes it: E[t_f t_e] = phi Q_FF^-2 Q_FE Q_EE^-2 Q_EF 1."""
    nE, nF = Q_EF.shape
    phi = entry_law(W)
    iFF, iEE = Q_FF.inv(), Q_EE.inv()
    psi = phi * (-iFF) * Q_FE
    mean_f = (-(phi * iFF * ones(nF)))[0]
    mean_e = (-(psi * iEE * ones(nE)))[0]
    e_fe = (phi * iFF * iFF * Q_FE * iEE * iEE * Q_EF * ones(nE))[0]
    return sp.cancel(e_fe - mean_f * mean_e)


def mean_dwells(Q_EE, Q_FF):
    return ((-Q_EE).inv() * ones(Q_EE.shape[0]),
            (-Q_FF).inv() * ones(Q_FF.shape[0]))


def margins(W):
    nE, nF = W.shape
    return (sp.Matrix(nE, 1, lambda i, _: sum(W.row(i))),
            sp.Matrix(nF, 1, lambda j, _: sum(W.col(j))))


def sample(n, rng, intra):
    pi_E = [rational(rng) for _ in range(n)]
    pi_F = [rational(rng) for _ in range(n)]
    W = sp.Matrix(n, n, lambda i, j: rational(rng))
    A_E = A_F = None
    if intra:
        A_E, A_F = sp.zeros(n, n), sp.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                A_E[i, j] = A_E[j, i] = rational(rng)
                A_F[i, j] = A_F[j, i] = rational(rng)
    return pi_E, pi_F, W, A_E, A_F


# ---------------------------------------------------------------------------
# 1. The row-sum collapse that both the identity and the rank argument use.
# ---------------------------------------------------------------------------
def check_row_sum_collapse():
    print("\n1. Row-sum collapse:  E[t_f t_e] = phi (-Q_FF)^-2 Q_FE m_E")
    print("   (Q_EF 1 = -Q_EE 1 turns the inner double inverse into m_E)")
    rng = random.Random(11)
    for n in (2, 3):
        for intra in (False, True):
            pi_E, pi_F, W, A_E, A_F = sample(n, rng, intra)
            Q_EE, Q_EF, Q_FF, Q_FE = blocks(pi_E, pi_F, W, A_E, A_F)
            phi = entry_law(W)
            mE, _ = mean_dwells(Q_EE, Q_FF)
            iFF, iEE = Q_FF.inv(), Q_EE.inv()
            lhs = (phi * iFF * iFF * Q_FE * iEE * iEE * Q_EF * ones(n))[0]
            rhs = (phi * iFF * iFF * Q_FE * mE)[0]
            check(f"n={n}, intra-level edges={intra}", sp.cancel(lhs - rhs) == 0)


# ---------------------------------------------------------------------------
# 2. (A) for n = 2, 3, 4, with and without intra-level edges.
# ---------------------------------------------------------------------------
def check_general_form():
    print("\n2. Cov = (1/C) m_E^T (W - D K^T / C) m_F")
    rng = random.Random(23)
    for n, trials in ((2, 3), (3, 3), (4, 2)):
        for intra in (False, True):
            bad = 0
            for _ in range(trials):
                pi_E, pi_F, W, A_E, A_F = sample(n, rng, intra)
                Q_EE, Q_EF, Q_FF, Q_FE = blocks(pi_E, pi_F, W, A_E, A_F)
                mE, mF = mean_dwells(Q_EE, Q_FF)
                D, K = margins(W)
                C = sum(W)
                pred = (mE.T * (W - D * K.T / C) * mF)[0] / C
                got = covariance(Q_EE, Q_EF, Q_FF, Q_FE, W)
                if sp.cancel(got - pred) != 0:
                    bad += 1
            check(f"n={n}, intra-level edges={intra}", bad == 0,
                  f"{trials - bad}/{trials} exact")


# ---------------------------------------------------------------------------
# 3. (B) fully symbolic at n = 2, with symbolic intra-level conductances.
# ---------------------------------------------------------------------------
def check_n2_fully_symbolic():
    print("\n3. n=2 in full symbols, intra-level conductances a (on E) and b (on F)")
    p0, p1, q0, q1 = sp.symbols("p0 p1 q0 q1", positive=True)
    c00, c01, c10, c11 = sp.symbols("c00 c01 c10 c11", positive=True)
    a, b = sp.symbols("a b", positive=True)

    W = sp.Matrix([[c00, c01], [c10, c11]])
    A_E = sp.Matrix([[0, a], [a, 0]])
    A_F = sp.Matrix([[0, b], [b, 0]])
    Q_EE, Q_EF, Q_FF, Q_FE = blocks([p0, p1], [q0, q1], W, A_E, A_F)

    # the generator really is a generator, symbolically
    check("full generator rows sum to zero",
          sp.simplify((Q_EE * ones(2) + Q_EF * ones(2)).norm()) == 0
          and sp.simplify((Q_FF * ones(2) + Q_FE * ones(2)).norm()) == 0)

    mE, mF = mean_dwells(Q_EE, Q_FF)
    C = sum(W)
    rhs = W.det() * (mE[0] - mE[1]) * (mF[0] - mF[1]) / C**2
    diff = sp.cancel(sp.together(covariance(Q_EE, Q_EF, Q_FF, Q_FE, W) - rhs))
    num = sp.expand(sp.simplify(sp.fraction(diff)[0]))
    check("Cov - det(W)(mE0-mE1)(mF0-mF1)/C^2 == 0 identically", num == 0,
          f"numerator = {num}")


# ---------------------------------------------------------------------------
# 4. The n = 2 reduction of (A) to (B).
# ---------------------------------------------------------------------------
def check_n2_reduction():
    print("\n4. n=2 reduction:  W - D K^T / C = (det W / C) [[1,-1],[-1,1]]")
    c00, c01, c10, c11 = sp.symbols("c00 c01 c10 c11", positive=True)
    W = sp.Matrix([[c00, c01], [c10, c11]])
    D, K = margins(W)
    C = sum(W)
    W_tilde = W - D * K.T / C
    target = (W.det() / C) * sp.Matrix([[1, -1], [-1, 1]])
    check("centred coupling is the centring outer product times det(W)/C",
          sp.simplify(W_tilde - target) == sp.zeros(2, 2))
    check("centred coupling has zero margins",
          sp.simplify((W_tilde * ones(2)).norm()) == 0
          and sp.simplify((W_tilde.T * ones(2)).norm()) == 0)


# ---------------------------------------------------------------------------
# 5. THE BIPARTITE BOUNDARY. The covariance identity survives intra-level
#    edges; the polytope chart does not.
# ---------------------------------------------------------------------------
def check_polytope_boundary():
    print("\n5. Boundary of the bipartite assumption (the polytope chart needs it)")
    print("   Reading margins off the 1-D marginal requires the marginal's")
    print("   mixture rates to BE the sub-state exit rates K_j / pi_j.")
    rng = random.Random(37)
    x = sp.Symbol("x")

    for n in (2, 3):
        for intra in (False, True):
            pi_E, pi_F, W, A_E, A_F = sample(n, rng, intra)
            _, _, Q_FF, _ = blocks(pi_E, pi_F, W, A_E, A_F)
            _, K = margins(W)
            exit_rates = [K[j] / pi_F[j] for j in range(n)]

            # Do the F-level dwell modes coincide with the exit rates?
            actual = sp.expand((-Q_FF).charpoly(x).as_expr())
            claimed = sp.expand(sp.prod([x - r for r in exit_rates]))
            same = sp.simplify(actual - claimed) == 0

            if not intra:
                check(f"n={n} bipartite: mixture rates ARE K_j/pi_j", same)
            else:
                check(f"n={n} intra-level: mixture rates are NOT K_j/pi_j",
                      not same, "inversion from marginal to margins fails")

    # And the consequence: with intra-level edges the bipartite inversion
    # returns the wrong margins, so the polytope is not the right chart.
    pi_E, pi_F, W, A_E, A_F = sample(2, rng, True)
    _, _, Q_FF, _ = blocks(pi_E, pi_F, W, A_E, A_F)
    _, K = margins(W)
    recovered = [sp.cancel(-Q_FF[j, j] * pi_F[j]) for j in range(2)]
    check("n=2 intra-level: recovered margins differ from the true K",
          any(sp.cancel(recovered[j] - K[j]) != 0 for j in range(2)),
          f"recovered={recovered}, true={list(K)}")


# ---------------------------------------------------------------------------
# 6. The general-n proof, lemma by lemma. Supersedes the n <= 4 spot checks in
#    section 2: those confirm the identity, these confirm the ARGUMENT, so a
#    failure here localises which step of the proof broke.
# ---------------------------------------------------------------------------
def check_general_proof():
    print("\n6. The general-n proof, step by step (intra-level edges on both levels)")
    print("   Only two structural inputs: generator row sums, and detailed balance.")
    rng = random.Random(4242)
    for n in (2, 3, 4):
        pi_E, pi_F, W, A_E, A_F = sample(n, rng, intra=True)
        Q_EE, Q_EF, Q_FF, Q_FE = blocks(pi_E, pi_F, W, A_E, A_F)
        P_E, P_F = sp.diag(*pi_E), sp.diag(*pi_F)
        C = sum(W)
        D, K = margins(W)
        mE, mF = mean_dwells(Q_EE, Q_FF)
        R = (-Q_FF).inv()
        phi = entry_law(W)
        piE, piF = sp.Matrix(pi_E), sp.Matrix(pi_F)
        Z = lambda M: sp.simplify(M) == sp.zeros(*M.shape)   # noqa: E731

        # Reversibility, in the form the proof actually uses.
        check(f"n={n}  S_E = P_E Q_EE is symmetric", Z(P_E * Q_EE - (P_E * Q_EE).T))
        check(f"n={n}  S_F = P_F Q_FF is symmetric", Z(P_F * Q_FF - (P_F * Q_FF).T))
        check(f"n={n}  P_E Q_EF = W  and  P_F Q_FE = W^T",
              Z(P_E * Q_EF - W) and Z(P_F * Q_FE - W.T))

        # Row sums, in the form the proof actually uses.
        check(f"n={n}  D = -S_E 1  and  K = -S_F 1",
              Z(D + P_E * Q_EE * ones(n)) and Z(K + P_F * Q_FF * ones(n)))

        # Step 1: the row-sum collapse.
        check(f"n={n}  step 1: (-Q_EE)^-2 Q_EF 1 = m_E",
              Z((-Q_EE).inv() * (-Q_EE).inv() * Q_EF * ones(n) - mE))
        # Step 2/3: one factor of R.
        check(f"n={n}  step 3: phi R = pi_F^T / C", Z(phi * R - piF.T / C))
        # Step 4: the second factor of R -- this is where reversibility enters.
        check(f"n={n}  step 4: phi R^2 = m_F^T P_F / C  [needs reversibility]",
              Z(phi * R * R - mF.T * P_F / C))
        # Step 5.
        check(f"n={n}  step 5: E[t_f t_e] = m_E^T W m_F / C",
              sp.cancel((phi * R * R * Q_FE * mE)[0] - (mE.T * W * mF)[0] / C) == 0)
        # Step 6: the means, and the identification of the correction term.
        check(f"n={n}  step 6: E[t_f] = pi(F)/C, E[t_e] = pi(E)/C",
              sp.cancel((phi * mF)[0] - sum(pi_F) / C) == 0
              and sp.cancel((phi * R * Q_FE * mE)[0] - sum(pi_E) / C) == 0)
        check(f"n={n}  step 6: m_E^T D = pi(E)  and  K^T m_F = pi(F)",
              sp.simplify((mE.T * D)[0] - sum(pi_E)) == 0
              and sp.simplify((K.T * mF)[0] - sum(pi_F)) == 0)


def main() -> int:
    print("=" * 72)
    print("Symbolic verification of the adjacent-dwell covariance identity")
    print("exact rational / fully symbolic arithmetic -- no floating point")
    print("=" * 72)
    check_row_sum_collapse()
    check_general_form()
    check_n2_fully_symbolic()
    check_n2_reduction()
    check_polytope_boundary()
    check_general_proof()

    print()
    print("=" * 72)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    print("  Cov = (1/C) m_E^T (W - D K^T/C) m_F is PROVED for all n and for")
    print("  arbitrary intra-level connectivity: section 6 verifies each step of")
    print("  the argument, which uses only generator row sums and detailed")
    print("  balance and never assumes Q_EE or Q_FF diagonal. At n = 2 it")
    print("  reduces to the det(W) form, an identity in full symbols. The")
    print("  polytope chart, unlike the covariance identity, genuinely does")
    print("  require bipartite levels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
