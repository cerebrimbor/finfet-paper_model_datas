"""Verify the committed BSIM-CMG provenance sweeps.

Reads the two committed Id-Vg CSVs and independently re-extracts the trap
offset at matched drain current.

Expected result for the supplied artifacts:
  200 usable points
  mean shift ~5 mV
  range ~2 nV

This verifies the data artifact; it does not prove the same offset holds for
other transistor cards, geometries, bias conditions, or technologies.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "device_provenance" / "idvg.csv"
trap = ROOT / "device_provenance" / "idvg_trap.csv"

if not base.exists() or not trap.exists():
    raise SystemExit("Expected device_provenance/idvg.csv and idvg_trap.csv")

vg0, i0 = np.loadtxt(base, unpack=True)
vg1, i1 = np.loadtxt(trap, unpack=True)

mask = (i1 >= i0.min()) & (i1 <= i0.max())
shift = vg1[mask] - np.interp(np.log(i1[mask]), np.log(i0), vg0)

print(f"usable points: {mask.sum()}")
print(f"mean shift:    {shift.mean()*1e3:.9f} mV")
print(f"spread:        {(shift.max()-shift.min())*1e9:.3f} nV")

assert mask.sum() == 200
assert abs(shift.mean() - 0.005) < 1e-10
assert (shift.max()-shift.min()) < 1e-8
print("PASS")
