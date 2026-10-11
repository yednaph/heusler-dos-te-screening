#!/usr/bin/env python3
# =============================================================================
# generate_qe_inputs.py  --  CIF -> QE vc-relax  ->  QE scf.in + nscf.in
#
#   TWO-STAGE pipeline (added after confirming, via Xiao & Tadano's own DXMag
#   paper, that these anchor structures -- novel MatterGen-generated compounds
#   through only "light MLIP relaxation" -- are NOT DFT-relaxed the way Xiao &
#   Tadano's own ternary-track database entries were; their quaternary/all-d
#   track uses the SAME kind of cheap MLIP relaxation and is explicitly the
#   lower-confidence tier in their own methodology. A real DFT relaxation is
#   required before the SCF/NSCF/BoltzTraP2 chain, not optional.):
#
#     Stage 1 (--stage relax):      CIF -> vc-relax.in
#         (relax BOTH atomic positions AND cell, matching Xiao & Tadano's own
#          "equilibrium lattice parameters ... obtained with GGA calculations"
#          full-cell-relaxation convention for their DFT track)
#     Stage 2 (--stage post_relax): converged relax.out -> scf.in + nscf.in
#         (built from the RELAXED structure, not the original CIF -- species/
#          cutoffs/nelec/magnetism are re-derived from the relaxed structure's
#          own element list, so stage 2 needs no reference back to the CIF)
#
#   Depends only on numpy (present in the qe env) -- no pymatgen. The prediction
#   CIFs are P1 with explicit atoms, so we parse cell + positions directly.
#
#   Usage (called in a loop by run_anchor.slurm):
#     python generate_qe_inputs.py --stage relax --cif RhNbSn_xxx.cif \
#         --sssp_dir SSSP --sssp_json SSSP_efficiency.json \
#         --outdir RhNbSn --prefix RhNbSn
#     python generate_qe_inputs.py --stage post_relax --relax_out RhNbSn/relax.out \
#         --sssp_dir SSSP --sssp_json SSSP_efficiency.json \
#         --outdir RhNbSn --prefix RhNbSn
# =============================================================================
import argparse, json, re, sys
from pathlib import Path
import numpy as np

MAG3D  = {"V": 0.2, "Cr": 0.4, "Mn": 0.4, "Fe": 0.4, "Co": 0.4, "Ni": 0.3}  # start moments
K_SCF  = 40.0     # k-grid density constant: n_i ~ K / a_i  (SCF/relax, moderate, symmetry ON)
# NSCF runs WITH symmetry (QE default -- see write_scf_nscf, no nosym/noinv:
# BoltzTraP2's QE loader requires an irreducible k-point set, see comment
# there), so QE only explicitly computes the symmetry-irreducible wedge of
# this nominal grid (a fraction of the full mesh for a high-symmetry cubic
# Heusler). BoltzTraP2 then Fourier-INTERPOLATES that IBZ mesh 15x denser --
# so a nominal ~12x12x12 grid is plenty for accurate transport while keeping
# the actual (much smaller, IBZ-only) NSCF cost low.
K_NSCF = 55.0     # ~13 per direction nominal grid => small IBZ point count


def read_cif(path):
    """Parse a generated (P1) CIF -> (a,b,c,al,be,ga in Ang/deg, [(El,x,y,z),...])."""
    t = Path(path).read_text(errors="replace")
    def g(key):
        m = re.search(rf"{re.escape(key)}(?!\w)\s+([\S]+)", t)
        return float(m.group(1)) if m else None
    a, b, c = g("_cell_length_a"), g("_cell_length_b"), g("_cell_length_c")
    al, be, ga = g("_cell_angle_alpha"), g("_cell_angle_beta"), g("_cell_angle_gamma")
    F = r"[-+]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][-+]?\d+)?"
    atoms = re.findall(rf"^\s*\w+\s+([A-Za-z]{{1,2}})\s+({F})\s+({F})\s+({F})", t, re.M)
    if not (a and b and c and atoms):
        sys.exit(f"  CIF parse failed for {path}")
    return a, b, c, al, be, ga, [(e, float(x), float(y), float(z)) for e, x, y, z in atoms]


def lattice_vectors(a, b, c, al, be, ga):
    """Crystallographic a,b,c,alpha,beta,gamma (Ang/deg) -> 3x3 Cartesian vectors (Ang)."""
    al, be, ga = np.radians([al, be, ga])
    v = np.zeros((3, 3))
    v[0] = [a, 0, 0]
    v[1] = [b * np.cos(ga), b * np.sin(ga), 0]
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    v[2] = [cx, cy, np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))]
    return v


def parse_relax_output(relax_out_path):
    """
    Parse a CONVERGED QE vc-relax output for its final structure. QE prints
    the relaxed geometry between 'Begin final coordinates' and 'End final
    coordinates', as a CELL_PARAMETERS (angstrom) block + an ATOMIC_POSITIONS
    (crystal) block -- this is the standard, version-stable QE convention for
    a successfully converged relax/vc-relax run.

    Returns: vecs (3x3 ndarray, Angstrom), atoms [(El,x,y,z) fractional, ...]
    Raises SystemExit with a clear message if the run did not converge (no
    final-coordinates block found) -- callers must not silently proceed with
    an unrelaxed/partial structure.
    """
    t = Path(relax_out_path).read_text(errors="replace")
    if "Begin final coordinates" not in t or "End final coordinates" not in t:
        sys.exit(f"  NOT CONVERGED: no 'Begin/End final coordinates' block in "
                 f"{relax_out_path} -- relax did not finish successfully; do "
                 f"NOT proceed to scf/nscf with this structure.")
    block = t.split("Begin final coordinates")[1].split("End final coordinates")[0]
    lines = block.splitlines()

    # Line-by-line, pattern-terminated parsing -- deliberately does NOT
    # assume a blank line separates CELL_PARAMETERS from what follows (an
    # earlier version did, via a regex requiring \n\s*\n as a terminator;
    # testing against a second, equally plausible QE output layout without
    # that blank line showed it silently over-captures past the cell block
    # into the atomic positions. Real QE's exact blank-line placement here
    # isn't something to bet a whole 20-material campaign on when a robust,
    # format-agnostic alternative is just as easy: a lattice ALWAYS has
    # exactly 3 vectors, so take exactly the next 3 numeric lines after the
    # header; positions block ends at the first line that doesn't parse as
    # "Element x y z", whatever that line is (blank, "End final
    # coordinates", or anything else).
    NUM = r"[-+]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][-+]?\d+)?"
    cell_start = None
    for i, l in enumerate(lines):
        if re.match(r"^\s*CELL_PARAMETERS", l):
            if "angstrom" not in l.lower():
                sys.exit(f"  PARSE FAILED: CELL_PARAMETERS header does not say "
                         f"'angstrom' ({l.strip()!r}) in {relax_out_path} -- "
                         f"refusing to guess the unit (could be bohr/alat).")
            cell_start = i + 1
            break
    if cell_start is None:
        sys.exit(f"  PARSE FAILED: no CELL_PARAMETERS block found in the final "
                 f"coordinates of {relax_out_path}")
    cell_lines = [l for l in lines[cell_start:] if l.strip()][:3]
    if len(cell_lines) != 3 or any(
        len(l.split()) < 3 or not re.match(rf"^\s*{NUM}\s+{NUM}\s+{NUM}", l)
        for l in cell_lines
    ):
        sys.exit(f"  PARSE FAILED: could not read 3 valid CELL_PARAMETERS lines "
                 f"after the header in {relax_out_path} (got: {cell_lines!r})")
    vecs = np.array([[float(x) for x in l.split()[:3]] for l in cell_lines])

    pos_start = None
    for i, l in enumerate(lines):
        if re.match(r"^\s*ATOMIC_POSITIONS", l):
            if "crystal" not in l.lower():
                sys.exit(f"  PARSE FAILED: ATOMIC_POSITIONS header does not say "
                         f"'crystal' ({l.strip()!r}) in {relax_out_path} -- "
                         f"refusing to guess the convention (could be angstrom/"
                         f"bohr Cartesian, which this code treats as fractional).")
            pos_start = i + 1
            break
    if pos_start is None:
        sys.exit(f"  PARSE FAILED: no ATOMIC_POSITIONS block found in the final "
                 f"coordinates of {relax_out_path}")
    atom_re = re.compile(rf"^\s*([A-Za-z]{{1,2}})\s+({NUM})\s+({NUM})\s+({NUM})\s*$")
    atoms = []
    for l in lines[pos_start:]:
        m = atom_re.match(l)
        if not m:
            break   # first non-matching line (blank, "End final coordinates",
                    # or anything else) ends the block -- no blank-line assumption
        el, x, y, z = m.groups()
        atoms.append((el, float(x), float(y), float(z)))
    if not atoms:
        sys.exit(f"  PARSE FAILED: ATOMIC_POSITIONS block was empty in {relax_out_path}")
    return vecs, atoms


def symmetrize_structure(vecs, atoms, symprec=1e-3):
    """
    Snap the relaxed structure onto its EXACT crystallographic symmetry via
    spglib, before scf/nscf are generated from it.

    WHY THIS IS NEEDED (found 2026-07-21 by reproducing a real HPC failure
    locally, end-to-end, with a self-built BoltzTraP2 26.3.1 matching the
    HPC exactly): BFGS relaxation converges to only an APPROXIMATELY
    symmetric structure -- residual noise of order 1e-6 in fractional
    coordinates, well within forc_conv_thr/press_conv_thr, but non-isotropic
    (e.g. biased along one body-diagonal direction). QE's own symmetry
    detection and BoltzTraP2's INDEPENDENT one (spglib-based, hardcoded
    symprec=1e-4 in BoltzTraP2.sphere.frontend.calc_reciprocal_iksubset) use
    DIFFERENT effective tolerances, and this specific noise band sits right
    in the gap between them. Confirmed directly for RhNbSn's real relaxed
    structure: QE's own log reported only "6 Sym. Ops." (should be 24, F-43m
    Td); spglib.get_symmetry_dataset on the exact same positions found only
    R3m/6-op at symprec=1e-5 but the correct F-43m/24-op at symprec>=1e-4.
    BoltzTraP2 (using its own, looser check) then finds QE's NSCF k-points
    -- already reduced by QE's own, degraded 6-op symmetry -- to be
    "inconsistent": some of QE's "irreducible" points are, under the TRUE
    24-op symmetry BoltzTraP2 correctly detects, actually equivalent to each
    other. Symmetrizing here (spglib.find_primitive, symprec looser than the
    observed noise but tighter than any real structural difference) makes
    the structure EXACTLY symmetric before QE ever sees it, so both QE and
    BoltzTraP2 detect the same (full, correct) point group and agree by
    construction. Verified end-to-end for RhNbSn: raw relaxed structure ->
    interpolate fails with "inconsistent sets of k points"; symmetrized
    structure -> QE finds the full 24 Sym. Ops, interpolate succeeds.
    """
    try:
        import spglib
    except ImportError:
        sys.exit("  spglib is required to symmetrize the relaxed structure "
                 "before scf/nscf (pip install spglib in the qe env) -- "
                 "without it, BFGS relaxation noise can make QE and "
                 "BoltzTraP2 disagree about the crystal symmetry and "
                 "btp2 interpolate will fail with 'inconsistent sets of "
                 "k points'. See symmetrize_structure()'s docstring.")
    species_order = sorted({e for e, *_ in atoms})
    sp_index = {e: i + 1 for i, e in enumerate(species_order)}
    positions = np.array([[x, y, z] for _, x, y, z in atoms])
    numbers = [sp_index[e] for e, *_ in atoms]
    prim = spglib.find_primitive((vecs, positions, numbers), symprec=symprec)
    if prim is None:
        sys.exit(f"  SYMMETRIZATION FAILED: spglib.find_primitive found no "
                 f"symmetry at symprec={symprec} -- refusing to proceed "
                 f"with an unsymmetrized structure.")
    new_vecs, new_pos, new_nums = prim
    # find_primitive is allowed to return FEWER atoms than the input -- some
    # of this project's CIFs are supplied as a non-primitive (e.g. 2-formula-
    # unit doubled) cell, and find_primitive correctly reducing that to its
    # true primitive representation is expected, not an error (found
    # 2026-07-21, real job 28146: Fe2NbAl/Fe2TaAl/Ru2ScSb all relaxed as
    # nat=8 with a non-cubic kSCF grid shape, i.e. a doubled cell, and
    # correctly reduce to the standard nat=4 full-Heusler primitive cell).
    # What actually must be guarded against is a reduction that DOESN'T
    # preserve stoichiometry -- every element's count must shrink by the
    # exact SAME integer factor, or something has gone genuinely wrong.
    if len(new_nums) > len(atoms) or len(atoms) % len(new_nums) != 0:
        sys.exit(f"  SYMMETRIZATION FAILED: spglib.find_primitive returned "
                 f"{len(new_nums)} atoms from an input of {len(atoms)} -- not "
                 f"a valid integer sub-multiple, refusing to proceed.")
    factor = len(atoms) // len(new_nums)
    inv_sp_check = {v: k for k, v in sp_index.items()}
    orig_counts, new_counts = {}, {}
    for n in numbers:
        orig_counts[n] = orig_counts.get(n, 0) + 1
    for n in new_nums:
        new_counts[n] = new_counts.get(n, 0) + 1
    for n, oc in orig_counts.items():
        nc = new_counts.get(n, 0)
        if oc != factor * nc:
            sys.exit(f"  SYMMETRIZATION FAILED: element {inv_sp_check[n]} count "
                     f"went from {oc} to {nc} (factor {oc/nc if nc else float('inf')}), "
                     f"not the uniform factor {factor} every other element "
                     f"reduced by -- this is NOT a simple primitive-cell "
                     f"reduction, refusing to silently change stoichiometry.")
    if set(orig_counts) != set(new_counts):
        sys.exit(f"  SYMMETRIZATION FAILED: element set changed "
                 f"({sorted(inv_sp_check[n] for n in orig_counts)} -> "
                 f"{sorted(inv_sp_check[n] for n in new_counts)}) -- refusing "
                 f"to silently change composition.")
    if factor != 1:
        print(f"  [symmetrize] input cell was {factor}x non-primitive "
              f"({len(atoms)} atoms) -- reduced to the true primitive cell "
              f"({len(new_nums)} atoms), stoichiometry verified preserved.")
    inv_sp = {v: k for k, v in sp_index.items()}
    new_atoms = [(inv_sp[n], float(p[0]), float(p[1]), float(p[2]))
                for n, p in zip(new_nums, new_pos)]
    return np.array(new_vecs), new_atoms


def kgrid(vecs, K):
    """uniform k-grid n_i = max(6, round(K / |a_i|)) per lattice vector length."""
    return [max(6, int(round(K / np.linalg.norm(vecs[i])))) for i in range(3)]


def read_zvalence(upf_path):
    """Read the number of valence electrons from a UPF header (v2 attribute or v1 line)."""
    t = Path(upf_path).read_text(errors="replace")
    m = re.search(r'z_valence\s*=\s*"?\s*([0-9.eE+\-]+)', t, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r'([0-9.eE+\-]+)\s+Z\s*valence', t, re.I)            # UPF v1: "16.00  Z valence"
    return float(m.group(1)) if m else 0.0


def species_pseudo_info(species, sssp, sssp_dir):
    """Per-element pseudo filename + cutoffs (tested values from SSSP). Shared
    by both stages so relax and scf/nscf always use IDENTICAL cutoffs/pseudos
    for the same composition."""
    ecutwfc = ecutrho = 0.0
    pinfo = {}
    for el in species:
        if el not in sssp:
            sys.exit(f"  {el} not in SSSP json -- cannot build inputs")
        rec = sssp[el]
        fn  = rec.get("filename") or rec.get("pseudopotential")
        wfc = float(rec.get("cutoff_wfc") or rec.get("cutoff") or 60)
        rho = float(rec.get("cutoff_rho") or rec.get("dual", 8) * wfc)
        z   = read_zvalence(Path(sssp_dir) / fn)
        pinfo[el] = (fn, z)
        ecutwfc = max(ecutwfc, wfc)
        ecutrho = max(ecutrho, rho)
    return pinfo, ecutwfc, ecutrho


def afm_split(species, atoms):
    """Build a FERRIMAGNETIC / ANTIFERROMAGNETIC starting guess.

    WHY: MAG3D gives every magnetic element a POSITIVE starting moment, i.e. a
    FERROMAGNETIC starting guess for every material. But Mn2YZ full-Heuslers are
    classically FERRIMAGNETS (the two Mn sublattices couple ANTIPARALLEL), and
    Cr-Co pairs likewise couple antiparallel. Forcing an FM start on such a
    material makes the SCF fight its way to the opposite spin topology, and it
    oscillates instead of converging -- the observed failure mode for
    Mn2TaBi/Mn2TaSb/Mn2NbSb (moment pinned ~3 muB instead of the ~6 muB two
    parallel Mn would give) and CoCrIn/CrCoIn (moment flipping +1.03 <-> -0.03).

    QE's starting_magnetization is PER SPECIES, so an antiparallel guess for two
    atoms of the SAME element requires splitting that element into two species
    (e.g. Mn and Mn1) that share one pseudopotential but get opposite moments.

    Returns (new_species, new_atoms, sign_map):
      new_species  ordered species labels (may include split labels like 'Mn1')
      new_atoms    atoms relabelled to those species
      sign_map     label -> +1/-1 multiplier for starting_magnetization
    """
    mag_idx = [i for i, (e, *_) in enumerate(atoms) if e in MAG3D]
    if len(mag_idx) < 2:
        return species, atoms, {el: +1 for el in species}   # nothing to anti-align

    new_atoms = list(atoms)
    sign_map = {el: +1 for el in species}
    new_species = list(species)
    # alternate the sign over magnetic atoms: 1st +, 2nd -, 3rd +, ...
    seen = {}
    for n, i in enumerate(mag_idx):
        el, x, y, z = atoms[i]
        want = +1 if n % 2 == 0 else -1
        seen.setdefault(el, [])
        if want == +1 or not seen[el]:
            # keep the plain element label for the first occurrence
            label = el
            sign_map[label] = want
        else:
            label = f"{el}1"                 # split species, same pseudopotential
            if label not in new_species:
                new_species.append(label)
            sign_map[label] = want
        seen[el].append(want)
        new_atoms[i] = (label, x, y, z)
    return new_species, new_atoms, sign_map


def build_blocks(species, pinfo, vecs, atoms):
    def species_block():
        L = ["ATOMIC_SPECIES"]
        for el in species:
            base = el[:-1] if (el not in pinfo and el[:-1] in pinfo) else el
            fn, _ = pinfo[base]
            L.append(f"  {el:3s} 1.0  {fn}")
        return "\n".join(L)

    def cell_block():
        L = ["CELL_PARAMETERS angstrom"]
        for i in range(3):
            L.append(f"  {vecs[i,0]:14.9f} {vecs[i,1]:14.9f} {vecs[i,2]:14.9f}")
        return "\n".join(L)

    def pos_block():
        L = ["ATOMIC_POSITIONS crystal"]
        for e, x, y, z in atoms:
            L.append(f"  {e:3s} {x:12.8f} {y:12.8f} {z:12.8f}")
        return "\n".join(L)

    return species_block, cell_block, pos_block


def system_block(nat, ntyp, ecutwfc, ecutrho, species, magnetic, degauss=0.01,
                 sign_map=None):
    def mag_lines():
        if not magnetic:
            return ""
        out = []
        for i, el in enumerate(species, start=1):
            base = el[:-1] if (el not in MAG3D and el[:-1] in MAG3D) else el
            m = MAG3D.get(base, 0.0) * (sign_map.get(el, 1) if sign_map else 1)
            out.append(f"  starting_magnetization({i}) = {m}")
        return "\n".join(out) + "\n"

    return (f"  ibrav = 0\n  nat = {nat}\n  ntyp = {ntyp}\n"
            f"  ecutwfc = {ecutwfc:.1f}\n  ecutrho = {ecutrho:.1f}\n"
            f"  occupations = 'smearing'\n  smearing = 'mv'\n  degauss = {degauss}\n"
            + (f"  nspin = 2\n{mag_lines()}" if magnetic else ""))


def write_relax(outdir, prefix, sssp_dir, species, pinfo, ecutwfc, ecutrho,
                magnetic, vecs, atoms):
    species_block, cell_block, pos_block = build_blocks(species, pinfo, vecs, atoms)
    system = system_block(len(atoms), len(species), ecutwfc, ecutrho, species, magnetic)
    electrons = ("  conv_thr = 1.0d-8\n  mixing_mode = 'local-TF'\n"
                 "  mixing_beta = 0.3\n  mixing_ndim = 12\n  electron_maxstep = 300\n")
    ks = kgrid(vecs, K_SCF)

    # vc-relax: BOTH atomic positions and the cell are optimized, matching
    # Xiao & Tadano's own DFT-track convention (they relax lattice parameters
    # too, not just positions). disk_io='low': only the charge density/xml is
    # needed downstream (by the SCF restart... actually not even that, since
    # stage 2 rebuilds scf.in from the RELAXED GEOMETRY, not from out/ -- so
    # 'low' here is pure disk savings, nothing downstream reads these
    # wavefunctions either).
    # forc_conv_thr / etot_conv_thr / press_conv_thr: standard, moderate QE
    # defaults for a first-principles Heusler-type intermetallic (not a
    # floppy/molecular system) -- bfgs typically converges such cells in a
    # modest number of ionic steps. nstep=100 caps runaway non-convergence
    # rather than silently consuming unbounded walltime.
    relax = (f"&CONTROL\n  calculation = 'vc-relax'\n  prefix = '{prefix}'\n"
             f"  outdir = './out'\n  pseudo_dir = '{sssp_dir}'\n"
             f"  disk_io = 'low'\n  tprnfor = .true.\n  tstress = .true.\n"
             f"  forc_conv_thr = 1.0d-3\n  etot_conv_thr = 1.0d-4\n  nstep = 100\n/\n"
             f"&SYSTEM\n{system}/\n&ELECTRONS\n{electrons}/\n"
             f"&IONS\n  ion_dynamics = 'bfgs'\n/\n"
             f"&CELL\n  cell_dynamics = 'bfgs'\n  press_conv_thr = 0.5\n/\n"
             f"{species_block()}\n{cell_block()}\n{pos_block()}\n"
             f"K_POINTS automatic\n  {ks[0]} {ks[1]} {ks[2]} 1 1 1\n")
    (outdir / "relax.in").write_text(relax)
    print(f"  {prefix:10s} [relax] nat={len(atoms)} ntyp={len(species)} "
          f"ecut={ecutwfc:.0f}/{ecutrho:.0f} mag={magnetic} kSCF={ks}")


def write_scf_nscf(outdir, prefix, sssp_dir, species, pinfo, ecutwfc, ecutrho,
                   magnetic, vecs, atoms, nelec, nscf_diag="david", mix_beta=0.3, degauss=0.01,
                   afm=False):
    sign_map = None
    if afm and magnetic:
        species, atoms, sign_map = afm_split(species, atoms)
    species_block, cell_block, pos_block = build_blocks(species, pinfo, vecs, atoms)
    system = system_block(len(atoms), len(species), ecutwfc, ecutrho, species, magnetic,
                          degauss, sign_map)
    # SCF MIXING (2026-07-24): mix_beta is tunable so the run script can RETRY a
    # material that failed SCF convergence with gentler mixing. Real failure this
    # fixes -- CoCrIn (magnetic, 5.37 muB) oscillated at ~1e-3 Ry for all 300
    # iterations against a 1e-8 target under the default beta=0.3, failing under
    # BOTH the relax and no-relax pipelines. The accuracy trace was oscillating,
    # not diverging, which is the signature of too-aggressive charge mixing on a
    # magnetic system; a smaller beta damps the charge sloshing.
    electrons = (f"  conv_thr = 1.0d-8\n  mixing_mode = 'local-TF'\n"
                 f"  mixing_beta = {mix_beta}\n  mixing_ndim = 12\n"
                 f"  electron_maxstep = 300\n")
    # +50 empty conduction bands (was +30): the temperature sweep is now
    # locked at up to 1200K (extract_transport.py), not 900K. Wider thermal
    # window (kT=0.103eV at 1200K vs 0.078eV at 900K) needs more converged
    # conduction-band states above E_F for BoltzTraP2's margin=9*kB*Tmax
    # trimming to stay safely within the computed band data -- cheap NSCF
    # cost increase versus the risk of silently under-converging the high-T
    # tail of the transport integral.
    nbnd = int(nelec / 2) + 50
    ks, kn = kgrid(vecs, K_SCF), kgrid(vecs, K_NSCF)

    # disk_io='low': NSCF only ever reads the SCF's converged charge density +
    # XML (outdir/prefix.save), never its wavefunctions -- writing them here
    # is pure wasted disk with zero downstream use. Same class of bug that
    # already blew the 500GB quota on Fe2HfSi/Fe2HfGe once (that fix covered
    # NSCF via disk_io='nowf' below; this covers the SCF side).
    scf = (f"&CONTROL\n  calculation = 'scf'\n  prefix = '{prefix}'\n"
           f"  outdir = './out'\n  pseudo_dir = '{sssp_dir}'\n  tprnfor = .true.\n"
           f"  disk_io = 'low'\n/\n"
           f"&SYSTEM\n{system}/\n&ELECTRONS\n{electrons}/\n"
           f"{species_block()}\n{cell_block()}\n{pos_block()}\n"
           f"K_POINTS automatic\n  {ks[0]} {ks[1]} {ks[2]} 1 1 1\n")

    # No nosym/noinv here: BoltzTraP2.io.parse_ESPRESSO_xml explicitly wants
    # an IRREDUCIBLE k-point set (its own docstring says so), and its sanity
    # check (calc_reciprocal_iksubset, BoltzTraP2/sphere/frontend.pyx) rebuilds
    # the symmetry-irreducible subset of whatever points it's given and raises
    # "inconsistent sets of k points" if that subset is smaller than the input
    # -- i.e. it fails whenever the input contains symmetry-redundant points.
    # nosym+noinv forces QE to emit the FULL, unreduced BZ mesh, which for a
    # high-symmetry cubic Heusler (Oh, 48 ops) is full of symmetry-duplicate
    # points and triggers this exact error on every material (confirmed via
    # real job 28144 + reading the actual BoltzTraP2 26.3.1 source). Leaving
    # symmetry reduction on (QE default) makes NSCF emit its own IBZ mesh,
    # which is what the loader is actually designed to consume.
    nscf_extra = ""
    # NSCF diagonalization robustness (2026-07-22): the default Davidson solver
    # crashes NSCF (MPI_ABORT / QE error banner) for the Ga-bearing materials in
    # this project -- 7/7 confirmed on the HPC (batch 1+2), an unresolved
    # "nscf_qe_error". Ga's semicore-3d pseudos give many closely-spaced empty
    # conduction-band states, which is exactly the regime where Davidson can
    # fail to converge at NSCF while conjugate-gradient ('cg') stays stable
    # (cg is slower but far more robust for many-empty-band NSCF; NSCF is not
    # the campaign's bottleneck -- vc-relax is). `diago_full_acc = .true.` makes
    # QE converge the EMPTY bands to full accuracy too (not just the occupied
    # ones), which BoltzTraP2 needs anyway for the conduction-band transport
    # integrals. Selected via --nscf_diag; the run_anchor slurm scripts start
    # with Davidson and auto-retry with cg on an nscf error, so this path is
    # taken automatically for whichever materials actually need it.
    if nscf_diag == "cg":
        nscf_elec = "  conv_thr = 1.0d-8\n  diagonalization = 'cg'\n  diago_full_acc = .true.\n"
    else:
        nscf_elec = "  conv_thr = 1.0d-8\n"
    nscf = (f"&CONTROL\n  calculation = 'nscf'\n  prefix = '{prefix}'\n"
            f"  outdir = './out'\n  pseudo_dir = '{sssp_dir}'\n  verbosity = 'high'\n"
            f"  disk_io = 'nowf'\n/\n"
            f"&SYSTEM\n{system}{nscf_extra}  nbnd = {nbnd}\n/\n&ELECTRONS\n{nscf_elec}/\n"
            f"{species_block()}\n{cell_block()}\n{pos_block()}\n"
            f"K_POINTS automatic\n  {kn[0]} {kn[1]} {kn[2]} 0 0 0\n")

    (outdir / "scf.in").write_text(scf)
    (outdir / "nscf.in").write_text(nscf)
    print(f"  {prefix:10s} [scf/nscf inputs written] nat={len(atoms)} "
          f"ntyp={len(species)} ecut={ecutwfc:.0f}/{ecutrho:.0f} nbnd={nbnd} "
          f"mag={magnetic} kSCF={ks} kNSCF={kn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["relax", "rerelax", "post_relax", "cif_scf_nscf"])
    ap.add_argument("--cif", help="required for --stage relax / cif_scf_nscf")
    ap.add_argument("--relax_out", help="required for --stage rerelax / post_relax")
    ap.add_argument("--sssp_dir", required=True)
    ap.add_argument("--sssp_json", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--afm", action="store_true",
                    help="Antiferromagnetic/ferrimagnetic STARTING GUESS: alternate the sign "
                         "of starting_magnetization over magnetic atoms (splitting a species "
                         "when the same element needs both signs). For Mn2YZ ferrimagnets and "
                         "Cr-Co pairs an FM start makes the SCF oscillate; see afm_split().")
    ap.add_argument("--degauss", type=float, default=0.01,
                    help="Smearing width (Ry). Default 0.01. Larger values damp the "
                         "Fermi-surface charge oscillation that stalls SCF for metallic "
                         "magnetic systems; 0.02 is standard for metals.")
    ap.add_argument("--mix_beta", type=float, default=0.3,
                    help="SCF mixing_beta. Default 0.3; the run script retries a "
                         "material that failed SCF convergence with a smaller value "
                         "(gentler charge mixing) -- fixes magnetic charge-sloshing.")
    ap.add_argument("--nscf_diag", choices=["david", "cg"], default="david",
                    help="NSCF diagonalization solver. 'david' (default) is fast; 'cg' is the "
                         "robust fallback for materials whose NSCF crashes under Davidson "
                         "(e.g. the Ga-bearing semicore-3d materials). post_relax stage only.")
    args = ap.parse_args()

    sssp = json.loads(Path(args.sssp_json).read_text())
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.stage == "relax":
        if not args.cif:
            sys.exit("  --cif is required for --stage relax")
        a, b, c, al, be, ga, atoms = read_cif(args.cif)
        vecs = lattice_vectors(a, b, c, al, be, ga)
        species = sorted({e for e, *_ in atoms})
        pinfo, ecutwfc, ecutrho = species_pseudo_info(species, sssp, args.sssp_dir)
        magnetic = any(e in MAG3D for e in species)
        write_relax(outdir, args.prefix, args.sssp_dir, species, pinfo,
                   ecutwfc, ecutrho, magnetic, vecs, atoms)

    elif args.stage == "rerelax":
        # PULAY-STRESS CONTINUATION (2026-07-23): re-relax from a PREVIOUS
        # converged relax.out. A single vc-relax fixes the plane-wave basis to
        # the STARTING cell, so once the cell moves the basis is slightly
        # incomplete for the new cell (Pulay stress) and the reported "converged"
        # cell is a little off. Restarting vc-relax from the relaxed structure
        # regenerates the G-vector basis for the new cell; iterating until a
        # restart converges WITHOUT MOVING (QE: "... 1 bfgs steps") removes the
        # Pulay stress. This stage writes a FRESH relax.in from the relaxed
        # geometry. Deliberately NO symmetrisation here -- this is a pure
        # continuation of the same relaxation (QE preserves the space group
        # during vc-relax); the one-time snap-to-symmetry happens later in
        # post_relax, right before scf/nscf.
        if not args.relax_out:
            sys.exit("  --relax_out is required for --stage rerelax")
        vecs, atoms = parse_relax_output(args.relax_out)
        species = sorted({e for e, *_ in atoms})
        pinfo, ecutwfc, ecutrho = species_pseudo_info(species, sssp, args.sssp_dir)
        magnetic = any(e in MAG3D for e in species)
        write_relax(outdir, args.prefix, args.sssp_dir, species, pinfo,
                   ecutwfc, ecutrho, magnetic, vecs, atoms)

    elif args.stage == "cif_scf_nscf":
        # NO-RELAX PATH (2026-07-23): build scf.in + nscf.in DIRECTLY from the
        # (ML-relaxed) CIF, skipping the DFT vc-relax entirely. Rationale:
        #   * descriptor<->transport consistency -- transport is computed on the
        #     SAME structure the DOS screen scored, so structure is no longer a
        #     confounding variable in the ranking validation;
        #   * avoids relax non-convergence failures (CoCrIn, Mn2TaBi) and the
        #     BFGS-noise/Pulay complications entirely;
        #   * the symmetrisation below still runs -- essential for clean,
        #     BoltzTraP2-compatible k-points and primitive-cell reduction of
        #     non-primitive CIFs (e.g. CrTiTl's 6-atom I-4m2 cell) -- but on a
        #     clean CIF with NO BFGS noise it is robust, not the tolerance-
        #     mismatch minefield it was on relaxed structures.
        # SCOPE NOTE: results are transport "at the ML-relaxed geometry", NOT
        # full-DFT-relaxed ground state -- label them that way. For an absolute
        # zT on a single champion material, relax that one properly (--stage relax).
        if not args.cif:
            sys.exit("  --cif is required for --stage cif_scf_nscf")
        a, b, c, al, be, ga, atoms = read_cif(args.cif)
        vecs = lattice_vectors(a, b, c, al, be, ga)
        vecs, atoms = symmetrize_structure(vecs, atoms)
        species = sorted({e for e, *_ in atoms})
        pinfo, ecutwfc, ecutrho = species_pseudo_info(species, sssp, args.sssp_dir)
        magnetic = any(e in MAG3D for e in species)
        nelec = sum(pinfo[el][1] for el, *_ in atoms)
        write_scf_nscf(outdir, args.prefix, args.sssp_dir, species, pinfo,
                       ecutwfc, ecutrho, magnetic, vecs, atoms, nelec,
                       nscf_diag=args.nscf_diag, mix_beta=args.mix_beta,
                       degauss=args.degauss, afm=args.afm)

    else:  # post_relax
        if not args.relax_out:
            sys.exit("  --relax_out is required for --stage post_relax")
        vecs, atoms = parse_relax_output(args.relax_out)
        vecs, atoms = symmetrize_structure(vecs, atoms)
        species = sorted({e for e, *_ in atoms})
        pinfo, ecutwfc, ecutrho = species_pseudo_info(species, sssp, args.sssp_dir)
        magnetic = any(e in MAG3D for e in species)
        nelec = sum(pinfo[el][1] for el, *_ in atoms)
        write_scf_nscf(outdir, args.prefix, args.sssp_dir, species, pinfo,
                       ecutwfc, ecutrho, magnetic, vecs, atoms, nelec,
                       nscf_diag=args.nscf_diag, mix_beta=args.mix_beta,
                       degauss=args.degauss, afm=args.afm)


if __name__ == "__main__":
    main()
