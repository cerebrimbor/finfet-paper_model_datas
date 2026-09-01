"""End-to-end paper-figure pipeline (Phase 5).

Runs one shared generator through every phase and emits, into ``figures/`` and
``data/``:

* fig1_structural_identity.png -- Gillespie / EM / RTN telegraph on one generator.
* fig2_blind_spot_joints.png   -- K(3,3) degenerate pair: identical 1-D marginals,
                                  rho = 0 for both, joint rank 1 vs 3, and the
                                  standardised residual that separates them.
* fig3_chi2_power.png          -- chi-squared power against record length across
                                  the blind-spot family, and the bin-count
                                  trade-off that sets the optimum.
* data/*.csv, data/bundle.h5   -- the numbers behind every figure.

The previous draft's Figs. 2-4 (relaxation band, Euler-Maruyama convergence,
Hodgkin-Huxley spike) belonged to the superseded structural-identity argument and
were cut from the manuscript; their generators are gone with them. The HH and SDE
layers they exercised are still covered by the test suite.

Run:  python scripts/generate_figures.py
"""

from __future__ import annotations

import os
import sys

# Allow running as a plain script (python scripts/generate_figures.py): put the
# repo root on the path so the package imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from stochastic_ctmc.config import DEFAULT, HHConfig
from stochastic_ctmc.generator import two_state_generator, stationary_distribution
from stochastic_ctmc.gillespie import gillespie_ssa
from stochastic_ctmc.sde import channel_langevin_terms, euler_maruyama, deterministic_relaxation
from stochastic_ctmc.ensemble import ensemble_em
from stochastic_ctmc.hh import simulate_hh, gate_rates
from stochastic_ctmc.circuit import simulate_rtn
from stochastic_ctmc.config import RTNConfig
from stochastic_ctmc.plotting import (
    plot_trajectory_comparison, plot_error_band, plot_convergence, save_figure,
)
from stochastic_ctmc.export import export_columns_csv, export_bundle_hdf5, hdf5_available
from stochastic_ctmc.topology import (
    bipartite_generator_from_coupling, independence_partner, zero_correlation_partner,
)
from stochastic_ctmc.dwell import dwell_pdf_1d, dwell_pdf_2d, joint_density_rank
from stochastic_ctmc.correlation import analytic_adjacent_moments
from stochastic_ctmc.independence import (
    chi2_noncentrality, pairs_needed_chi2, analytic_binned_joint,
)

# plotting.py selects a non-interactive backend at import, so this must follow it.
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
DAT = os.path.join(ROOT, "data")


def _ensure_dirs():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(DAT, exist_ok=True)


def fig1_structural_identity(a=2.0, b=1.0, N=200):
    """One generator, three realisations -- stacked, not overlaid.

    Three changes from the earlier overlaid version, all of them things a
    referee would otherwise ask about:

    * Stacked panels. Overlaid, the Gillespie telegraph and the RTN telegraph
      are the same two levels and sit on top of one another, so neither is
      readable.
    * The Euler-Maruyama path starts at the stationary occupancy x* rather than
      at 0. Starting at 0 puts a visible relaxation transient in the first third
      of the panel, which contradicts the caption's claim that the path
      "fluctuates about x*".
    * Time is plotted in units of 1/(a+b), the relaxation time, which is what
      the caption says the axis is. Previously the axis was in raw simulation
      time and the caption was wrong.

    No in-panel title: the caption carries that text, and PRE does not want it
    duplicated inside the artwork.
    """
    Q = two_state_generator(a, b)
    x_star = a / (a + b)
    rate = a + b                      # 1/relaxation time

    t_max = 5.0
    ss = gillespie_ssa(Q, 0, t_max=t_max, rng=0)
    tg = np.linspace(0, t_max, 800)
    gill = np.array([ss.state_at(t) for t in tg])  # 0/1 telegraph

    drift, diffusion = channel_langevin_terms(Q, N)
    # Start on the stationary mean: we are illustrating the fluctuation, not the
    # approach to it.
    em = euler_maruyama(drift, diffusion, x_star, t_max, 2e-3, rng=0, clip=(0, 1))

    # Independent seed: an independent draw of the *same* process, so both
    # telegraphs are visible rather than the Gillespie trace hiding under it.
    rtn = simulate_rtn(RTNConfig(capture_rate=a, emission_rate=b, delta_vth=1.0),
                       t_max, rng=7)
    rtn_sig = rtn.vth_signal(tg)

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 5.4), sharex=True)
    panels = (
        (axes[0], tg * rate, gill, "C0", "(a) Gillespie CTMC", "state"),
        (axes[1], em.times * rate, em.X, "C1",
         f"(b) Euler-Maruyama, $N={N}$", "open fraction"),
        (axes[2], tg * rate, rtn_sig, "C2", "(c) FinFET trap", r"$\Delta V_{th}$ / step"),
    )
    for ax, t, y, colour, tag, ylab in panels:
        ax.plot(t, y, lw=1.2, color=colour)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.25)
        ax.text(0.012, 0.86, tag, transform=ax.transAxes, fontsize=9,
                va="top", ha="left")
    axes[1].axhline(x_star, color="0.4", ls=":", lw=1.0)
    axes[1].text(0.995, x_star, r"  $x^\star$", transform=axes[1].get_yaxis_transform(),
                 fontsize=8, color="0.35", va="bottom", ha="right")
    axes[0].set_ylim(-0.15, 1.15)
    axes[2].set_ylim(-0.15, 1.15)
    axes[2].set_xlabel(r"time  $(a+b)\,t$")
    fig.tight_layout()

    save_figure(fig, os.path.join(FIG, "fig1_structural_identity.png"))
    export_columns_csv(os.path.join(DAT, "fig1_gillespie.csv"), {"t": tg, "state": gill})
    export_columns_csv(os.path.join(DAT, "fig1_em.csv"), {"t": em.times, "x": em.X})
    return Q


LEV6 = [0, 0, 0, 1, 1, 1]          # K(3,3): states 0-2 empty, 3-5 filled


def _k33_degenerate_triple(seed=7, deviation_rank=2, partner_seed=3, margin=0.5):
    """A K(3,3) target, its rank-1 partner, and its blind-spot partner.

    All three share the same 1-D dwell marginals by construction: those fix pi
    and both margins of the flux matrix W, and every point of the resulting
    transportation polytope reproduces them. The rank-1 partner is the product
    coupling D K^T / C; the blind-spot partner is the point of the rho = 0
    hyperplane reached by a rank-``deviation_rank`` deviation from it.
    """
    rng = np.random.default_rng(seed)
    pi = rng.uniform(0.5, 1.5, 6)
    pi = pi / pi.sum()
    W = rng.uniform(0.4, 2.0, (3, 3))
    target = bipartite_generator_from_coupling(pi[:3], pi[3:], W)
    q_rank1 = independence_partner(target, LEV6)
    q_blind = zero_correlation_partner(target, LEV6, deviation_rank=deviation_rank,
                                       seed=partner_seed, margin=margin)
    return target, q_rank1, q_blind


def fig2_blind_spot_joints(n_bins=12, n_grid=160):
    """The blind spot, drawn.

    A raw density heatmap is the wrong picture here: the two joints differ by
    parts in 10^4, which no shared colour scale can show. The right picture is
    the one the test actually forms -- the standardised residual
    (P - P_F P_E)/sqrt(P_F P_E) on quantile bins, whose squares sum to the
    noncentrality lambda. On that view the rank-1 partner is identically zero
    and the blind-spot partner is not, which is the whole claim.
    """
    _, q1, qb = _k33_degenerate_triple(seed=11, deviation_rank=2, margin=0.9)

    mom = analytic_adjacent_moments(qb, LEV6, 1)
    tf = np.linspace(1e-3, 4.0 * mom.mean_f, n_grid)
    te = np.linspace(1e-3, 4.0 * mom.mean_e, n_grid)

    m1_f, mb_f = (dwell_pdf_1d(q, LEV6, tf, level=1) for q in (q1, qb))
    m1_e, mb_e = (dwell_pdf_1d(q, LEV6, te, level=0) for q in (q1, qb))
    marg_err = max(np.abs(m1_f - mb_f).max() / m1_f.max(),
                   np.abs(m1_e - mb_e).max() / m1_e.max())

    rho1 = analytic_adjacent_moments(q1, LEV6, 1).rho
    rhob = mom.rho
    r1, rb = joint_density_rank(q1, LEV6), joint_density_rank(qb, LEV6)

    def residual(Q):
        P = analytic_binned_joint(Q, LEV6, n_bins=n_bins, level=1)
        E = np.outer(P.sum(axis=1), P.sum(axis=0))
        return (P - E) / np.sqrt(E)

    z1, zb = residual(q1), residual(qb)
    lam1, lamb = float((z1**2).sum()), float((zb**2).sum())

    # Panel titles are deliberately terse tags: the caption carries the prose,
    # and PRE does not want it repeated inside the artwork. What stays in-panel
    # is only what the caption cannot cheaply state -- the numbers.
    fig, ax = plt.subplots(2, 2, figsize=(9.2, 7.2))

    a = ax[0, 0]
    a.semilogy(tf, m1_f, lw=3.2, alpha=0.4, label="rank 1: filled")
    a.semilogy(tf, mb_f, lw=1.2, ls="--", label="blind spot: filled")
    a.semilogy(te, m1_e, lw=3.2, alpha=0.4, label="rank 1: empty")
    a.semilogy(te, mb_e, lw=1.2, ls="--", label="blind spot: empty")
    a.set_ylim(1e-3 * max(m1_f.max(), m1_e.max()), 2.0 * max(m1_f.max(), m1_e.max()))
    a.set_xlabel("dwell time"); a.set_ylabel("density")
    a.set_title(f"(a)  max rel. difference {marg_err:.0e}", fontsize=10, loc="left")
    a.legend(fontsize=7, loc="lower left")

    a = ax[0, 1]
    J = dwell_pdf_2d(qb, LEV6, tf, te, level=1)
    im = a.imshow(J, origin="lower", aspect="auto", cmap="viridis",
                  extent=[te[0], te[-1], tf[0], tf[-1]])
    a.set_xlabel("empty dwell $t_e$"); a.set_ylabel("filled dwell $t_f$")
    a.set_title("(b)  joint density, blind-spot partner", fontsize=10, loc="left")
    fig.colorbar(im, ax=a, fraction=0.046)

    # One shared colorbar for (c) and (d). Panel (c) is identically zero, so
    # giving it its own bar implied a scale it does not have and hid the fact
    # that the two panels are drawn on the SAME scale -- which is the point.
    lim = np.abs(zb).max()
    for a, z, lab, rk, rho, lam in (
            (ax[1, 0], z1, "(c)  rank-1 partner", r1, rho1, lam1),
            (ax[1, 1], zb, "(d)  blind-spot partner", rb, rhob, lamb)):
        im = a.imshow(z, origin="lower", aspect="auto", cmap="RdBu_r",
                      vmin=-lim, vmax=lim, extent=[0, 1, 0, 1])
        a.set_xlabel("empty-dwell quantile"); a.set_ylabel("filled-dwell quantile")
        a.set_title(lab, fontsize=10, loc="left")
        # lambda is bin-count dependent, so say which bin count it is quoted at:
        # Fig. 3 and the text quote lambda at the 5-bin default, this panel at
        # n_bins, and the two are not comparable without the label.
        a.text(0.03, 0.955,
               f"rank {rk}\n" + r"$\rho$ = " + f"{rho:+.0e}\n"
               + r"$\lambda$ = " + f"{lam:.1e}  ({n_bins} bins)",
               transform=a.transAxes, fontsize=8, va="top", ha="left",
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))

    fig.tight_layout()
    fig.colorbar(im, ax=[ax[1, 0], ax[1, 1]], fraction=0.03, pad=0.02,
                 label="standardised residual")
    save_figure(fig, os.path.join(FIG, "fig2_blind_spot_joints.png"))
    export_columns_csv(os.path.join(DAT, "fig2_marginals.csv"),
                       {"t_f": tf, "rank1_filled": m1_f, "blind_filled": mb_f,
                        "t_e": te, "rank1_empty": m1_e, "blind_empty": mb_e})
    return marg_err, (r1, rb), (rho1, rhob), (lam1, lamb)


def fig3_chi2_power(n_bins=5, n_sigma=5.0):
    """What closes the blind spot: chi-squared power against record length."""
    dof = (n_bins - 1) ** 2
    crit = stats.chi2.isf(2.0 * stats.norm.sf(n_sigma), dof)
    alpha = float(2.0 * stats.norm.sf(n_sigma))

    # A family of blind-spot generators. The point of spanning several targets
    # rather than one: at comparable deviation size (||W~||/||W|| ~ 0.4 for all
    # of these) the noncentrality still ranges over three decades, so how long a
    # record the test needs is set by WHERE in the polytope the generator sits,
    # not merely how far from the product coupling.
    family = []
    for seed, dev_rank, style in ((19, 1, "-"), (7, 1, "-"),
                                  (7, 2, "--"), (11, 2, "--")):
        try:
            _, _, qb = _k33_degenerate_triple(seed=seed, deviation_rank=dev_rank,
                                              margin=0.9)
        except ValueError:
            continue
        lam = chi2_noncentrality(qb, LEV6, n_bins=n_bins, level=1)
        if lam <= 0.0:
            continue
        family.append((seed, dev_rank, style, lam, joint_density_rank(qb, LEV6),
                       pairs_needed_chi2(qb, LEV6, n_bins=n_bins, level=1,
                                         n_sigma=n_sigma), qb))
    family.sort(key=lambda f: f[3])

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    n = np.logspace(3, 10, 400)
    for _, _, style, lam, rk, need, _ in family:
        a.semilogx(n, stats.ncx2.sf(crit, dof, lam * n), style, lw=1.8,
                   label=(f"rank {rk},  " + r"$\lambda$ = " + f"{lam:.1e},  "
                          f"n = {need/1e6:.1f}M"))
        # Grey guide at the record length each generator needs for 90% power;
        # labelled in the caption so the reader is not left guessing.
        a.axvline(need, color="0.8", lw=0.7, zorder=0)
    a.axhline(0.9, color="0.4", ls=":", lw=1.0)
    a.text(1.3e3, 0.915, "90% power", fontsize=8, color="0.3")
    a.axhline(alpha, color="crimson", lw=2.0)
    a.text(1.3e3, 0.05, r"correlation: $\rho=0$ exactly," "\n"
                        r"power pinned at the test size $\alpha$",
           fontsize=8, color="crimson")
    a.set_xlabel("adjacent dwell pairs $n$")
    a.set_ylabel(f"power at {n_sigma:g}$\\sigma$")
    a.set_ylim(-0.03, 1.03)
    a.set_title(f"(a)  {n_bins} quantile bins per axis", fontsize=10, loc="left")
    a.legend(fontsize=7.5, loc="center right")

    # Binning is the one knob the experimenter controls. More bins resolve more
    # structure (lambda rises) but cost degrees of freedom, so the requirement
    # has a shallow minimum -- which is what justifies the default of 5.
    qb_ref = family[-1][6]
    bins = np.arange(3, 21)
    needs = np.array([pairs_needed_chi2(qb_ref, LEV6, n_bins=int(nb), level=1,
                                        n_sigma=n_sigma) for nb in bins], dtype=float)
    lams = np.array([chi2_noncentrality(qb_ref, LEV6, n_bins=int(nb), level=1)
                     for nb in bins])
    n_opt = int(bins[int(np.argmin(needs))])
    b.plot(bins, needs / 1e6, "o-", lw=1.6, ms=4)
    b.axvline(n_opt, color="0.6", ls=":", lw=1.0)
    b.axvline(n_bins, color="crimson", ls="-", lw=1.0, alpha=0.6)
    b.text(n_bins, b.get_ylim()[1], " default", fontsize=8, color="crimson",
           va="top", ha="left")
    b.set_xlabel("quantile bins per axis")
    b.set_ylabel("pairs for 90% power  (millions)")
    b.set_title(f"(b)  optimum near {n_opt} bins", fontsize=10, loc="left")
    b2 = b.twinx()
    b2.plot(bins, lams, "s--", color="0.55", lw=1.0, ms=3)
    b2.set_ylabel(r"noncentrality $\lambda$ (grey)", color="0.4", fontsize=9)

    fig.tight_layout()
    save_figure(fig, os.path.join(FIG, "fig3_chi2_power.png"))
    export_columns_csv(
        os.path.join(DAT, "fig3_chi2_power.csv"),
        {"target_seed": np.array([f[0] for f in family], dtype=float),
         "deviation_rank": np.array([f[1] for f in family], dtype=float),
         "noncentrality": np.array([f[3] for f in family]),
         "joint_rank": np.array([f[4] for f in family], dtype=float),
         "pairs_needed": np.array([f[5] for f in family], dtype=float)})
    export_columns_csv(os.path.join(DAT, "fig3_binning.csv"),
                       {"n_bins": bins.astype(float), "noncentrality": lams,
                        "pairs_needed": needs})
    return [f[:6] for f in family]


def main():
    _ensure_dirs()
    print("fig1 (structural identity) ...")
    Q = fig1_structural_identity()

    print("fig2 (blind spot: matched marginals, rho = 0, different joints) ...")
    marg_err, ranks, rhos, lams = fig2_blind_spot_joints()
    print(f"  marginals agree to {marg_err:.2e} relative; joint ranks {ranks}; "
          f"rho = {rhos[0]:+.1e}, {rhos[1]:+.1e}; lambda = {lams[0]:.1e}, {lams[1]:.1e}")

    print("fig3 (chi-squared power vs record length) ...")
    family = fig3_chi2_power()
    for seed, dev_rank, _, lam, rk, need in family:
        print(f"  target seed {seed}, deviation rank {dev_rank}: joint rank {rk}, "
              f"lambda = {lam:.3e}, pairs for 90% power at 5 sigma = {need:,}")

    if hdf5_available():
        export_bundle_hdf5(
            os.path.join(DAT, "bundle.h5"),
            {"fig3_noncentrality": np.array([f[3] for f in family]),
             "fig3_pairs_needed": np.array([f[5] for f in family], dtype=float)},
            attrs={"generator": "two_state(2,1)",
                   "stationary_p": float(stationary_distribution(Q)[1])},
        )
        print("  wrote data/bundle.h5")

    print(f"\nDone. Figures in {FIG}, data in {DAT}.")


if __name__ == "__main__":
    main()
