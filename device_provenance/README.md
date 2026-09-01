# Device provenance: the DVTSHIFT reference sweeps

These six files are the exact artifacts behind the claim in
`stochastic_ctmc/circuit.py` that a trap's effect on the device is an *exact*
gate-voltage offset — the justification for there being no SPICE simulator in
the RTN loop. They are a provenance record, not working source: nothing in the
Python package imports or reads them.

| File | What it is |
|---|---|
| `modelcard.nmos` | BSIM-MG 105 sample NMOS card, `DVTSHIFT = 0` (trap empty) |
| `modelcard_trap.nmos` | The same card, `DVTSHIFT = 5m` (trap filled) — the only difference |
| `idvg.sp` | Id–Vg sweep deck, baseline |
| `idvg_trap.sp` | Id–Vg sweep deck, trap |
| `idvg.csv` | Output of `idvg.sp` |
| `idvg_trap.csv` | Output of `idvg_trap.sp` |

The model card is Berkeley's published benchmarking card
(`BSIMCMG110.0.0_20160101.tar.gz`), not an extraction from any real technology.
It is fit for demonstrating that `DVTSHIFT` behaves as a rigid gate-voltage
offset, and not fit for quantitative claims about a specific device.

`.gitattributes` pins these paths with `-text` so the committed bytes — including
the files' original, inconsistent line endings — survive checkout on any
platform. The CSVs are committed through an explicit override of the repo-wide
`*.csv` ignore rule.

## Simulator

ngspice **46**, 64-bit Windows binary distribution, archive `ngspice-46_64.7z`,
from the `ng-spice-rework` file area of the ngspice SourceForge project
(<https://sourceforge.net/projects/ngspice/files/ng-spice-rework/>; see also
<https://ngspice.sourceforge.io/download.html>).

The compact model is **BSIM-CMG** loaded as an OSDI shared object,
`lib/ngspice/BSIMCMG.osdi`, which ships inside that same archive. The decks pull
it in at runtime with `pre_osdi`; there is nothing to compile.

Reported by `bin/ngspice_con.exe -v`:

```
** ngspice-46 : Circuit level simulation program
** Compiled with KLU Direct Linear Solver
```

The runs themselves report `Using SPARSE 1.3 as Direct Linear Solver` — KLU is
compiled in but not selected for a circuit this small.

## Required directory layout — this setup step is manual

The decks use **relative** paths, so they only resolve from one specific working
directory inside an extracted ngspice tree:

- `.include Modelcards/modelcard.nmos` → a `Modelcards/` subdirectory of the CWD
- `pre_osdi ../../../lib/ngspice/BSIMCMG.osdi` → CWD must be exactly three levels
  below `Spice64/`

That directory is `ngspice-46_64/Spice64/examples/osdi/bsimcmg/`, which the
archive already provides.

**The ngspice archive and its extracted tree are deliberately gitignored** —
`.gitignore` excludes both `*.7z` and `ngspice-46_64/`, because a 10.7 MB
third-party binary distribution does not belong in this repository. A fresh
clone therefore has the decks but no simulator and no `Spice64/` tree, and the
steps below will not run until you download and extract ngspice yourself. This
is intended, not a packaging oversight.

## Reproducing `idvg.csv` and `idvg_trap.csv` from scratch

From the repository root, with `ngspice-46_64.7z` downloaded there and extracted
so that `ngspice-46_64/Spice64/` exists:

```bash
cp device_provenance/idvg.sp device_provenance/idvg_trap.sp \
   ngspice-46_64/Spice64/examples/osdi/bsimcmg/
cp device_provenance/modelcard.nmos device_provenance/modelcard_trap.nmos \
   ngspice-46_64/Spice64/examples/osdi/bsimcmg/Modelcards/
```

Note that this overwrites the `modelcard.nmos` shipped in that example
directory. Work from a throwaway extraction if you want the stock tree left
untouched.

Then, from `ngspice-46_64/Spice64/examples/osdi/bsimcmg/`:

```bash
../../../bin/ngspice_con.exe -b idvg.sp
```

```bash
../../../bin/ngspice_con.exe -b idvg_trap.sp
```

Each writes its CSV into the current directory and reports `No. of Data Rows :
201`. Both outputs reproduce the committed files byte-for-byte.

The sweep is `dc VG 0 1.0 0.005` at `VD = 0.05 V` — 201 points from 0 to 1.0 V
in 5 mV steps. `wrdata` emits two whitespace-separated columns, gate voltage and
`i(VS)`; the current is written positive and increases monotonically across the
sweep, so it can be used directly with no sign correction.

### Re-extracting the 5 mV offset

The point of the two sweeps is that the shift between them, measured at *matched
drain current* rather than at matched gate voltage, is constant. Inverting the
baseline curve by interpolating gate voltage against log current:

```python
import numpy as np

vg0, i0 = np.loadtxt("idvg.csv", unpack=True)
vg1, i1 = np.loadtxt("idvg_trap.csv", unpack=True)

# Keep the trap points whose current the baseline sweep actually brackets.
mask = (i1 >= i0.min()) & (i1 <= i0.max())
shift = vg1[mask] - np.interp(np.log(i1[mask]), np.log(i0), vg0)

print(mask.sum(), shift.mean(), shift.max() - shift.min())
```

This yields **200** usable points, a mean shift of **5.000000 mV**, and a spread
of about **2 nV** — the interpolation error, not a physical variation. Plain
linear interpolation in current, rather than log current, gives the same mean and
the same spread.

The one discarded point is `Vg = 0`. The trap curve is displaced towards higher
gate voltage, so its current at the bottom of the sweep falls below the lowest
current the baseline curve reaches; matching it would require extrapolating past
the start of the baseline data. Every other point is bracketed and used.

A shift that is constant to nanovolts across roughly four and a half decades of
drain current (2.4e-10 A to 7.7e-6 A) is what licenses treating the transistor as a memoryless nonlinearity:
one DC sweep captures it completely, and the CTMC only has to decide *when* the
trap is filled.
