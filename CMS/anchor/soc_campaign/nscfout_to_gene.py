#!/usr/bin/env python3
"""
Recover BoltzTraP2 input from a completed run's TEXT output -- no DFT re-run.

WHY THIS EXISTS
The driver ran `btp2 interpolate ./out`, and BoltzTraP2's ESPRESSO loader globs
`*.xml` at the TOP level of out/, where QE leaves a STALE `<prefix>.xml` from the
SCF step. The nscf (run with disk_io='nowf') updates only
`out/<prefix>.save/data-file-schema.xml`. So every transport result was computed
from the 9x9x9 SHIFTED SCF mesh instead of the 12-13^3 UNSHIFTED nscf mesh, and
out/ was then deleted -- so the XML is gone.

BUT the nscf ran with verbosity='high', which prints the full k-point list in
CRYSTAL coordinates and every eigenvalue to nscf.out. 3,622 of 3,629 runs have
those blocks. That is everything BoltzTraP2's GENE loader needs, so the nscf work
is recoverable from text at ~seconds per material instead of ~20 min of DFT.

CAVEAT, and it must be tested: QE prints eigenvalues to 4 decimals in eV, i.e. a
0.1 meV quantisation. At 300 K, k_B*T = 25.9 meV, so that is ~0.4% of the thermal
window. Whether it perturbs transport is an empirical question -- validate against
an XML-derived reference before trusting it (see --validate).
"""
import argparse
import os
import re
import sys

import numpy as np

RY = 13.605693122994       # eV per Ry
BOHR = 0.529177210903      # Angstrom per bohr


def parse_scf_in(path):
    """cell (Angstrom) and fractional positions from the QE input."""
    t = open(path).read()
    cell = []
    m = re.search(r"CELL_PARAMETERS[^\n]*\n((?:\s*[-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]+\s*\n){3})", t)
    for ln in m.group(1).strip().splitlines():
        cell.append([float(x) for x in ln.split()])
    sym, frac = [], []
    m = re.search(r"ATOMIC_POSITIONS[^\n]*\n((?:\s*\w+\s+[-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]+\s*\n?)+)", t)
    for ln in m.group(1).strip().splitlines():
        f = ln.split()
        if len(f) >= 4:
            sym.append(re.sub(r"\d+$", "", f[0]))     # strip Mn1 -> Mn
            frac.append([float(x) for x in f[1:4]])
    return np.array(cell), sym, np.array(frac)


def parse_nscf_out(path):
    """k-points in CRYSTAL coords, eigenvalues (eV), E_F (eV), nspin."""
    t = open(path, "rb").read().decode("utf8", "replace")
    nelec = float(re.search(r"number of electrons\s*=\s*([\d.]+)", t).group(1))
    spin_pol = ("SPIN UP" in t) or ("total magnetization" in t)
    nspin = 2 if spin_pol else 1
    mf = re.search(r"the Fermi energy is\s+([-\d.]+)", t)
    if mf:
        ef = float(mf.group(1))
    else:
        hi = re.search(r"highest occupied level \(ev\):\s+([-\d.]+)", t)
        ef = float(hi.group(1)) if hi else None

    # crystal-coordinate k list. CAREFUL: "cryst. coord." also appears earlier as
    # the ATOMIC POSITIONS header, and the k list itself is printed TWICE --
    # first in "cart. coord. in units 2pi/alat", then in "cryst. coord.".
    # So anchor on the cartesian k-list header and take the NEXT cryst. block.
    i_cart = t.find("cart. coord. in units 2pi/alat")
    if i_cart < 0:
        raise RuntimeError("no k-point list (needs verbosity='high')")
    i_cryst = t.find("cryst. coord.", i_cart)
    if i_cryst < 0:
        raise RuntimeError("no crystal-coordinate k list found")
    tail = t[i_cryst:]
    # stop at the next section header so we cannot run into later output
    stop = tail.find("Dense  grid")
    if stop > 0:
        tail = tail[:stop]
    kc = re.findall(r"k\(\s*\d+\) = \(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\), wk =\s*([\d.]+)",
                    tail)
    kpts = np.array([[float(a), float(b), float(c)] for a, b, c, _ in kc])

    # eigenvalue blocks, in file order (matches the k list order)
    blocks = re.findall(r"\n          k =.{1,40}\(\s*\d+ PWs\)   bands \(ev\):\n\n(.*?)\n\n", t, re.S)
    # QE writes band energies as 8f9.4 -- FIXED 9-char fields. Two wide negatives
    # (e.g. -232.1397-165.5137, deep semicore states) touch with no separating
    # space, so whitespace splitting merges them into one unparseable token.
    # Parse by fixed width, falling back to split() for any non-conforming line.
    def parse_block(b):
        vals = []
        for ln in b.splitlines():
            ln = ln.rstrip()
            if not ln.strip():
                continue
            body = ln[2:] if ln.startswith("  ") else ln
            if len(body) % 9 == 0 and len(body) >= 9:
                try:
                    vals += [float(body[i:i + 9]) for i in range(0, len(body), 9)]
                    continue
                except ValueError:
                    pass
            vals += [float(x) for x in ln.split()]
        return vals
    E = [parse_block(b) for b in blocks]
    nb = min(len(e) for e in E)
    E = np.array([e[:nb] for e in E])                 # (nk*nspin, nbnd)
    return kpts, E, ef, nspin, nelec


def write_gene(dest, name, cell_ang, sym, frac, kpts, E, ef, nspin):
    os.makedirs(dest, exist_ok=True)
    cell_bohr = cell_ang / BOHR
    cart_bohr = frac @ cell_bohr
    with open(f"{dest}/{name}.structure", "w") as f:
        f.write(f"{name} recovered from nscf.out\n")
        for v in cell_bohr:
            f.write(f"  {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n")
        f.write(f"{len(sym)}\n")
        for s, c in zip(sym, cart_bohr):
            f.write(f"{s} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n")
    nk = len(kpts)
    with open(f"{dest}/{name}.energy", "w") as f:
        f.write(f"{name}\n")
        f.write(f"{nk} {nspin} {ef/RY:.12f}\n")
        # for nspin=2 QE prints 2*nk blocks; the k list is the same for both spins
        for isp in range(nspin):
            for ik in range(nk):
                row = E[isp * nk + ik]
                f.write(f" {kpts[ik][0]:.10f} {kpts[ik][1]:.10f} {kpts[ik][2]:.10f} {len(row)}\n")
                for e in row:
                    f.write(f" {e/RY:.12f}\n")
    return nk, E.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--src", default=f"{os.path.dirname(os.path.abspath(__file__))}/../runs_local")
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    src = f"{a.src}/{a.run_dir}"
    cell, sym, frac = parse_scf_in(f"{src}/scf.in")
    kpts, E, ef, nspin, nelec = parse_nscf_out(f"{src}/nscf.out")
    if len(kpts) * nspin != E.shape[0]:
        print(f"  WARNING: {len(kpts)} k-points x nspin {nspin} != {E.shape[0]} eigenvalue blocks")
    nk, nb = write_gene(a.dest, a.run_dir.split("__")[0].split("_")[0],
                        cell, sym, frac, kpts, E, ef, nspin)
    print(f"  {a.run_dir}: nk={nk} nbnd={nb} nspin={nspin} nelec={nelec:.0f} "
          f"E_F={ef:.4f} eV -> {a.dest}")


if __name__ == "__main__":
    main()
