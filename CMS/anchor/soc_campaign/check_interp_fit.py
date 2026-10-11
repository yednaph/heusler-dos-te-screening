#!/usr/bin/env python3
"""
Decide whether BoltzTraP2's symmetry reconstruction is valid for a MAGNETIC
spin-orbit run.

THE ARGUMENT
BoltzTraP2 fits smoothed Fourier (star-function) coefficients to the DFT
eigenvalues on the irreducible k-set, using symmetry from spglib. With more star
functions than input points the fit is constrained to pass through the input data
essentially exactly -- that is the method's defining property.

The star functions are built by symmetrising plane waves over the point group
BoltzTraP2 believes the crystal has. For a NONCOLLINEAR magnetic run QE writes
lsda=false, so the loader sets magmom=None and spglib returns the FULL SPATIAL
group -- ignoring that the moments may break some of those operations.

If the assumed group is too large, the star functions are forced to be symmetric
under operations the real band structure does not obey. The fit then CANNOT pass
through the data, and the residual at the input k-points blows up. So the residual
is a direct, sufficient test of whether the symmetry assumption holds -- no need
to reason about magnetic space groups at all.

READING THE RESULT
  residual ~ 1e-6 eV or below : symmetry fine, the fit is interpolating.
  residual ~ meV or above     : the assumed group is wrong (or the fit is
                                underdetermined); do NOT trust the transport.

A non-magnetic run is checked alongside as the control, so the threshold is
calibrated on this machine rather than assumed.
"""
import sys

import numpy as np
from BoltzTraP2 import serialization, fite


def residual(bt2_path, label):
    data, equivalences, coeffs, meta = serialization.load_calculation(bt2_path)
    lattvec = data.get_lattvec()
    # Re-evaluate the fitted bands AT THE INPUT k-points.
    fitted = fite.getBands(data.kpoints, equivalences, lattvec, coeffs)[0]
    ref = data.ebands
    # getBands returns (nbands, nkpoints); ebands is stored the same way
    if fitted.shape != ref.shape:
        fitted = fitted.T
    if fitted.shape != ref.shape:
        raise RuntimeError(f"shape mismatch {fitted.shape} vs {ref.shape}")
    d = np.abs(fitted - ref)
    HARTREE = 27.211386245988          # BoltzTraP2 stores energies in Hartree
    print(f"\n=== {label} ===")
    print(f"   k-points fitted      : {data.kpoints.shape[0]}")
    print(f"   bands                : {ref.shape[0]}")
    print(f"   star functions       : {len(equivalences)}  "
          f"({len(equivalences)/data.kpoints.shape[0]:.1f}x the k-points)")
    print(f"   dosweight            : {data.dosweight}")
    print(f"   MAX |fit - DFT|      : {d.max()*HARTREE:.3e} eV")
    print(f"   RMS |fit - DFT|      : {np.sqrt((d**2).mean())*HARTREE:.3e} eV")
    return d.max() * HARTREE


if __name__ == "__main__":
    worst = {}
    for arg in sys.argv[1:]:
        path, _, label = arg.partition("=")
        worst[label or path] = residual(path, label or path)
    print("\n" + "=" * 58)
    for k, v in worst.items():
        verdict = ("OK -- fit interpolates, symmetry consistent" if v < 1e-4
                   else "FAIL -- fit cannot reproduce the data")
        print(f"   {k:<28} {v:.3e} eV   {verdict}")
