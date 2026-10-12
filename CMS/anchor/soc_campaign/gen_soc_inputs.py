#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The authors listed in CITATION.cff; see CMS/anchor/COPYING.
# =============================================================================
# Generate spin-orbit (noncollinear) QE inputs for a material that already has a
# completed SCALAR run in runs_local/.
#
# Design rules, each learned from the Ag2AuAs validation:
#
#  1. REUSE THE EXACT STRUCTURE from the original scf.in (cell + positions,
#     copied verbatim). The SOC run must differ from the scalar one ONLY in the
#     relativistic treatment, otherwise the comparison is confounded.
#
#  2. CUTOFFS ARE PER MATERIAL, taken as the max over its species of the values
#     each rel- PAW pseudo declares, times a safety factor. This is NOT optional:
#     Au alone demands rho_cutoff 423 Ry where the campaign used 280, and Cd
#     demands 638. A global cutoff would silently under-converge half the set.
#
#  3. nbnd SCALES WITH THE SPINOR COUNT. Under SOC there are `nelec` occupied
#     spinor bands (not nelec/2), so band counts must roughly double to span the
#     same energy window. Matching the scalar band COUNT would halve the window
#     and bias transport.
#
#  4. z_valence DIFFERS from the campaign's ONCV/USPP set (e.g. Ag 19 -> 11), so
#     nelec is recomputed from the actual rel- pseudos, never inherited.
# =============================================================================
import os as _os_htr  # release v1: project paths come from the environment
_HTR = _os_htr.environ.get('HEUSLER_TE_ROOT', _os_htr.path.abspath(_os_htr.path.join(_os_htr.path.dirname(__file__), '../../..')))
_HTE = _os_htr.environ.get('HEUSLER_TE_EXPORT', _os_htr.path.join(_HTR, 'DOS_EXPORT'))
import json, os, re, sys

A = _HTR + "/CMS/anchor"
PSD = f"{A}/pseudos_rel"
IDX = json.load(open(f"{PSD}/pseudo_index.json"))

WFC_SAFETY = 1.15     # margin over the pseudo's declared wfc_cutoff
RHO_SAFETY = 1.10     # margin over the declared rho_cutoff
BASE = lambda lab: re.sub(r"\d+$", "", lab)   # Fe1 -> Fe, pseudo lookup only

NBND_FACTOR = 2.4     # spinor bands per electron; matches the validated Ag2AuAs run
                      # (38 electrons -> 100 bands) and leaves ample empty states
                      # for BoltzTraP2's +/-0.5 eV reopt-doping window at 1200 K


def block(text, tag, stop):
    """Pull a card block (e.g. CELL_PARAMETERS ... up to the next card)."""
    m = re.search(rf"^{tag}.*$", text, re.M)
    if not m:
        return None, None
    start = m.start()
    rest = text[m.end():]
    end = len(text)
    for s in stop:
        mm = re.search(rf"^{s}", rest, re.M)
        if mm:
            end = min(end, m.end() + mm.start())
    return text[start:end].rstrip("\n"), m.group(0)


CARDS = ["ATOMIC_SPECIES", "ATOMIC_POSITIONS", "CELL_PARAMETERS", "K_POINTS"]


def parse(path):
    t = open(path).read()
    out = {}
    for c in CARDS:
        b, hdr = block(t, c, [x for x in CARDS if x != c])
        out[c] = b
    out["nat"] = int(re.search(r"nat\s*=\s*(\d+)", t).group(1))
    out["ntyp"] = int(re.search(r"ntyp\s*=\s*(\d+)", t).group(1))
    return out


def species_of(spec_block):
    """[(label, mass)] in the order QE will index them."""
    rows = []
    for ln in spec_block.splitlines()[1:]:
        p = ln.split()
        if len(p) >= 3:
            rows.append(p[0])
    return rows


def counts(pos_block, labels):
    n = {l: 0 for l in labels}
    for ln in pos_block.splitlines()[1:]:
        p = ln.split()
        if p and p[0] in n:
            n[p[0]] += 1
    return n


def gen(run_dir, outdir, kmesh_nscf=None):
    src = f"{A}/runs_local/{run_dir}"
    scf = parse(f"{src}/scf.in")
    labels = species_of(scf["ATOMIC_SPECIES"])
    missing = [l for l in labels if BASE(l) not in IDX]
    if missing:
        return None, f"no rel- pseudo for {sorted({BASE(l) for l in missing})}"
    nat = counts(scf["ATOMIC_POSITIONS"], labels)
    nelec = sum(IDX[BASE(l)]["z"] * nat[l] for l in labels)
    ecutwfc = max(IDX[BASE(l)]["wfc"] for l in labels) * WFC_SAFETY
    ecutrho = max(IDX[BASE(l)]["rho"] for l in labels) * RHO_SAFETY
    # keep ecutrho at least 8x ecutwfc, which PAW needs even if the pseudos
    # declare a lower dual
    ecutrho = max(ecutrho, 8.0 * ecutwfc)
    nbnd = int(round(NBND_FACTOR * nelec))

    spec = "ATOMIC_SPECIES\n" + "\n".join(
        f"  {l}  1.0  {IDX[BASE(l)]['file']}" for l in labels)

    # nscf k-mesh: reuse the original nscf's mesh so the transport grid matches
    kn = kmesh_nscf
    if kn is None and os.path.exists(f"{src}/nscf.in"):
        kn = parse(f"{src}/nscf.in")["K_POINTS"]
    if kn is None:
        kn = scf["K_POINTS"]

    os.makedirs(outdir, exist_ok=True)
    common = f"""  ibrav = 0
  nat = {scf['nat']}
  ntyp = {scf['ntyp']}
  ecutwfc = {ecutwfc:.1f}
  ecutrho = {ecutrho:.1f}
  noncolin = .true.
  lspinorb = .true.
  occupations = 'smearing'
  smearing = 'mv'
  degauss = 0.01
"""
    with open(f"{outdir}/scf.in", "w") as f:
        f.write(f"""&CONTROL
  calculation = 'scf'
  prefix = '{run_dir}'
  outdir = './out'
  pseudo_dir = '{PSD}'
  tprnfor = .true.
  disk_io = 'low'
/
&SYSTEM
{common}/
&ELECTRONS
  conv_thr = 1.0d-8
  mixing_mode = 'local-TF'
  mixing_beta = 0.3
  mixing_ndim = 12
  electron_maxstep = 300
/
{spec}
{scf['CELL_PARAMETERS']}
{scf['ATOMIC_POSITIONS']}
{scf['K_POINTS']}
""")
    with open(f"{outdir}/nscf.in", "w") as f:
        f.write(f"""&CONTROL
  calculation = 'nscf'
  prefix = '{run_dir}'
  outdir = './out'
  pseudo_dir = '{PSD}'
  verbosity = 'high'
  disk_io = 'nowf'
/
&SYSTEM
{common}  nbnd = {nbnd}
/
&ELECTRONS
  conv_thr = 1.0d-8
/
{spec}
{scf['CELL_PARAMETERS']}
{scf['ATOMIC_POSITIONS']}
{kn}
""")
    return dict(run_dir=run_dir, nelec=nelec, nbnd=nbnd,
                ecutwfc=round(ecutwfc, 1), ecutrho=round(ecutrho, 1),
                species=labels), None


if __name__ == "__main__":
    import csv
    tgt = sys.argv[1]; dest = sys.argv[2]
    rows = list(csv.DictReader(open(tgt)))
    print(f"{'run_dir':<14}{'nelec':>7}{'nbnd':>6}{'ecutwfc':>9}{'ecutrho':>9}  species")
    ok = 0
    for r in rows:
        info, err = gen(r["run_dir"], f"{dest}/{r['run_dir']}")
        if err:
            print(f"{r['run_dir']:<14}  SKIP: {err}"); continue
        ok += 1
        print(f"{info['run_dir']:<14}{info['nelec']:>7.0f}{info['nbnd']:>6}"
              f"{info['ecutwfc']:>9.1f}{info['ecutrho']:>9.1f}  {' '.join(info['species'])}")
    print(f"\ngenerated {ok}/{len(rows)}")
