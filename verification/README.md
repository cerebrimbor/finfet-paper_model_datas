# Independent verification suite

These scripts are intentionally separate from the research implementation.

## Purpose

They check the claims that matter to the manuscript:

1. CTMC generator + Gillespie stationary behaviour.
2. Adjacent-dwell covariance identity.
3. K(2,2) versus K(3,3) correlation blind spot.
4. Chi-squared detection of the blind spot's non-factorising joint.
5. BSIM-CMG Id-Vg offset provenance.

## Important

The scripts are **verification**, not substitutes for the paper's derivations.
A Monte Carlo PASS means the numerical implementation is consistent with the
reference calculation within the stated tolerance; it does not constitute a
mathematical proof.

Run from the repository root after copying this directory to `verification/`.
