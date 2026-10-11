"""
DOSDataset -- PyTorch Dataset for ALIGNN d-projected DOS prediction on Heusler alloys.

TARGET: d-projected spin-resolved DOS (NOT total DOS).
REASON: d-electrons dominate thermoelectric transport in transition-metal Heuslers.
        Mott formula: S propto (1/g_d) * dg_d/dE at E_F
        d-projected DOS has sharper features -> better TE discrimination.
        Total DOS includes featureless s,p contributions near E_F.

JSON key used: data['spd_dos']['d']['densities']['1'/'-1']
  (confirmed identical to summing pdos[site][d-orbital] across all sites
   and all d-suborbitals dxy/dyz/dz2/dxz/dx2; validated on 200 materials)

Data journey per sample:
    material_id (string from split CSV, column 'material_id')
        -> load {dos_dir}/{material_id}.json  [pymatgen CompleteDos format]
        -> Structure.from_dict(data['structure'])      -> pymatgen Structure (ordered)
        -> jarvis.core.atoms.pmg_to_atoms()
        -> alignn.graphs.Graph.atom_dgl_multigraph()   -> (atom_graph, line_graph)
        -> lattice = torch.tensor(atoms.lattice_mat)   -> shape (3,3)
        -> D-projected DOS preprocessing pipeline (see below)
        -> return {'graph': (atom_graph, line_graph, lattice),
                   'dos': tensor(4002,), 'id': mid,
                   'efermi': tensor([efermi]), 'n_atoms': tensor([n_atoms])}

Note: CIF files are NOT used. Structure is loaded from the embedded 'structure' key
in the DOS JSON. This is the exact geometry from the DFT calculation — guaranteed
ordered, consistent with the DOS data, and avoids the broken DxMag CIF encoding.

D-PROJECTED DOS preprocessing pipeline (all confirmed from 51,257-file audit):

  1. EXTRACT D-ORBITAL DOS: from data['spd_dos']['d']['densities']['1'/'-1']
     This is the total d-projected DOS summed over all sites and d-suborbitals.
     Confirmed: spd_dos['d'] == sum over sites of pdos[site]['dxy'/'dyz'/...] (max diff=0)
     Validated on 200 full_cubic materials: 0 failures, 0 zero-DOS materials.

  2. PER-ATOM NORMALIZATION: divide by n_atoms from structure.sites
     Units: states/eV/atom (removes cell-size dependence)
     Diagnostic: 100x dynamic range before normalization (8 to 808 states/eV/cell)
     Source of n_atoms: len(data['structure']['sites']) in DOS JSON

  3. FERMI ALIGNMENT: subtract efermi from energy axis
     Result: index 1000 = E_F = 0.0 eV for ALL materials
     Diagnostic: efermi range 2.2-12.6 eV across datasets;
                 (energies[0]-efermi) = -12.000 +/- 0.04 eV (consistent)

  4. INTERPOLATE TO COMMON GRID: linspace(-10, 10, 2001) at 0.01 eV/pt
     Coverage: 100% of all 51,257 materials (diagnostic confirmed)
     Output: 4002-dim = [d_dos_up(2001), d_dos_down(2001)]

  5. CLIP TO NON-NEGATIVE: physical constraint DOS >= 0
     Diagnostic: 0/51,257 files had negative DFT DOS; clip handles interp noise only

  6. SPIN CONVENTION: spd_dos['d']['densities']['1']=up, ['-1']=down
     Diagnostic: 100% consistent across all 6 datasets

  7. CONCATENATE: [d_dos_up(2001), d_dos_down(2001)] -> (4002,)

Physics validation (200-material sample):
  D-band center UP:   mean=-3.44 eV, std=0.69 eV  (expected < 0 for Heuslers)
  D-band center DOWN: mean=-3.16 eV, std=0.81 eV
  Exchange splitting: mean=-0.28 eV

Batch collation (DOSDataset.collate):
    N sample dicts
        -> dgl.batch(atom_graphs)         -> batched atom_graph
        -> dgl.batch(line_graphs)         -> batched line_graph
        -> torch.stack(lattice tensors)   -> shape (N, 3, 3)
        -> torch.stack(dos tensors)       -> shape (N, 4002)
        -> torch.stack(efermi tensors)    -> shape (N, 1)
        -> torch.stack(n_atoms tensors)   -> shape (N, 1)
        -> return {'graph': (batched_g, batched_lg, lattices),
                   'dos': ..., 'id': [...], 'efermi': ..., 'n_atoms': ...}

ALIGNN.forward() expects batch['graph'] to be a 3-tuple (g, lg, lat).
It unpacks as: g, lg, lat = batch['graph']
The lat element is unpacked but unused in computation when extra_features=0;
it must still be present in the tuple to avoid ValueError.
"""

# =============================================================================
# PHYSICS CONSISTENCY NOTES -- do not remove
# =============================================================================
# STRUCTURE SOURCE: data['structure'] in DOS JSON (NOT CIF files).
# All DxMag CIFs are broken due to encoding bug (conventional-cell atom lists
# under non-P1 space groups -> pymatgen computes occupancies > 1 -> parse fails).
# The JSON-embedded structure is the authoritative DFT geometry and is ordered.
#
# D-PROJECTED DOS preprocessing pipeline (confirmed from 51,257-file audit):
#
# TARGET: d-projected spin-resolved DOS via data['spd_dos']['d']['densities']
# REASON: Mott formula S propto dg_d/dE at E_F; d-electrons dominate in Heuslers
#
# 1. EXTRACT D-DOS: spd_dos['d']['densities']['1']=up, ['-1']=down
#    Confirmed identical to summing pdos site d-suborbitals (max diff=0)
#
# 2. PER-ATOM NORMALIZATION: divide by n_atoms from structure.sites
#    Units: states/eV/atom (removes cell-size dependence)
#
# 3. FERMI ALIGNMENT: subtract efermi from energy axis
#    Result: index 1000 = E_F = 0.0 eV for ALL materials
#
# 4. COMMON GRID: interpolate to linspace(-10, 10, 2001)
#    Coverage: 100% of all materials (diagnostic confirmed)
#    Output: 4002-dim = [d_dos_up(2001), d_dos_down(2001)]
#
# 5. NON-NEGATIVITY: clip to >= 0 (physical constraint)
#    Diagnostic: 0/51,257 files had negative DFT DOS
#
# 6. SPIN CONVENTION: '1'=spin-up, '-1'=spin-down (100% consistent)
#
# These choices enable physically meaningful TE screening via Mott formula.
# =============================================================================

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

import dgl
from scipy.interpolate import interp1d
from jarvis.core.atoms import pmg_to_atoms
from alignn.graphs import Graph
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element as PmgElement


# ---------------------------------------------------------------------------
# Heusler-specific node features (Group 1 enrichment — 2026-03-27)
# ---------------------------------------------------------------------------
# ATOM_INPUT_FEATURES = base ALIGNN features + 5 Heusler-specific features.
# Base ALIGNN/CGCNN features per atom: 92 (one-hot encoding over 92 elements,
# as produced by jarvis Graph.atom_dgl_multigraph with atom_features='cgcnn').
# After appending 5 Heusler-specific scalar features: 92 + 5 = 97.
# This constant must match the final input dimension seen by ALIGNN's first
# linear layer. Set in config.yaml under model.atom_input_features.
ATOM_INPUT_FEATURES = 97  # 92 base CGCNN (one-hot, 92 elements) + 5 Heusler-specific; updated 2026-03-27


def _get_heusler_node_features(atomic_number: int) -> list:
    """
    Return 5 Heusler-specific node features for a given atomic number.
    These are appended to ALIGNN's default base elemental features.

    Features (all normalised to roughly [0, 1]):
      F10  n_d_electrons : d-electron count / 10
      F11  vec_contrib   : max |oxidation state| / 10  (VEC proxy)
      F12  d_filling     : n_d / 10  (filling fraction)
      F13  hunds_moment  : Hund's-rule spin estimate = |5-n_d| if n_d<=5
                           else |10-n_d|, scaled / 5
      F14  d_row         : principal quantum number of d-shell
                           (4 for 3d, 5 for 4d, 6 for 5d, else 0), scaled / 6

    Physics basis:
      - Slater-Pauling rule: M_t = VEC - 24 (full) / VEC - 18 (half Heusler)
      - Half-metallicity requires integer M_t -> correct VEC critical
      - d-bandwidth scales with d-row (3d < 4d < 5d orbital extension)
      - d-band DOS shape near E_F determines Mott Seebeck coefficient
    """
    try:
        el = PmgElement.from_Z(atomic_number)
        ec = el.electronic_structure   # e.g. "[Ar] 3d7 4s2"
        d_match = re.search(r'(\d+)d(\d+)', ec)
        n_d = int(d_match.group(2)) if d_match else 0
        d_row_raw = int(d_match.group(1)) + 1 if d_match else 0  # 3d shell -> row 4

        ox = el.common_oxidation_states
        vec_val = float(max(abs(v) for v in ox)) if ox else float(el.group or 0)

        hunds = abs(5 - n_d) if n_d <= 5 else abs(10 - n_d)

        return [
            n_d / 10.0,               # F10: d-electron count
            min(vec_val, 10.0) / 10.0, # F11: VEC proxy
            n_d / 10.0,               # F12: d-filling fraction
            hunds / 5.0,              # F13: Hund's moment estimate
            d_row_raw / 6.0,          # F14: d-shell row (3d=4/6, 4d=5/6, 5d=1.0)
        ]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0, 0.0]


def _load_structure_from_json_data(data: dict, json_path: str):
    """
    Load a jarvis Atoms object from a pre-parsed DOS JSON dict.

    Reads structure from data['structure'] (pymatgen Structure serialization).
    This is the exact geometry used in the DFT calculation that produced the
    DOS — guaranteed to be ordered and consistent with the DOS data.

    Args:
        data:      Already-parsed JSON dict (from json.load)
        json_path: Path string used only for error messages

    Raises:
        KeyError:   if 'structure' key is missing from data
        AssertionError: if structure is somehow disordered (should never happen
                    for DxMag DFT-derived structures)
    """
    if 'structure' not in data:
        raise KeyError(
            f"'structure' key missing in {os.path.basename(json_path)}. "
            f"Available keys: {list(data.keys())}"
        )
    structure = Structure.from_dict(data['structure'])
    assert structure.is_ordered, (
        f"Unexpected disordered structure in JSON {json_path} — "
        f"this should never happen for DFT-derived CompleteDos files"
    )
    return pmg_to_atoms(structure)


def _build_graph(atoms, cutoff: float = 8.0, max_neighbors: int = 12) -> Tuple:
    """
    Build ALIGNN atom graph and line graph from a jarvis Atoms object.

    Returns:
        (atom_graph, line_graph, lattice_tensor)
        atom_graph:    DGLGraph -- nodes=atoms, edges=bonds
                                   ndata['atom_features'] extended with 5
                                   Heusler-specific features (total 14 dims)
        line_graph:    DGLGraph -- nodes=bonds, edges=bond-angle pairs
                                  edata['h'] = bond-angle cosines
        lattice_tensor: torch.Tensor shape (3,3), dtype float32

    Note: ALIGNN.forward() does  g, lg, lat = g  (3-tuple unpack).
    lat is unused in computation when extra_features=0 but must be present.
    """
    g, lg = Graph.atom_dgl_multigraph(
        atoms,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        compute_line_graph=True,
    )
    lattice = torch.tensor(atoms.lattice_mat, dtype=torch.float32)

    # Append 5 Heusler-specific features to each atom node
    # atoms.atomic_numbers is a numpy array of Z values, one per atom
    extra = torch.tensor(
        [_get_heusler_node_features(int(z)) for z in atoms.atomic_numbers],
        dtype=torch.float32,
    )  # shape (N_atoms, 5)
    g.ndata['atom_features'] = torch.cat(
        [g.ndata['atom_features'], extra], dim=1
    )  # shape (N_atoms, base_features + 5)

    return g, lg, lattice


# ---------------------------------------------------------------------------
# Confirmed constants from physics diagnostic (March 2026, 51,257 files)
# ---------------------------------------------------------------------------
_COMMON_GRID = np.linspace(-10.0, 10.0, 2001)   # Fermi-aligned common grid


_EXPECTED_RAW_POINTS = 2401  # confirmed from 51,257-file diagnostic


def _process_dos_data(data: dict, json_path: str) -> Tuple[np.ndarray, float, int]:
    """
    Process a pre-parsed CompleteDos JSON dict into d-projected physics-consistent
    4002-dim float32 array, plus metadata.

    Accepts an already-parsed dict so the caller can open the JSON once and
    share it with _load_structure_from_json_data (avoids double I/O).

    TARGET: d-projected spin-resolved DOS for thermoelectric screening.
    SOURCE: data['spd_dos']['d']['densities']['1'/'-1']
            (confirmed identical to summing pdos d-suborbitals over all sites)

    Processing pipeline (in order):
      1. Validate required fields
      2. Extract d-orbital projected DOS: spd_dos['d']['densities']['1'/'-1']
      3. Per-atom normalization: divide by n_atoms (states/eV/cell -> states/eV/atom)
      4. Fermi alignment: shift energy axis by -efermi
      5. Interpolate onto common grid: linspace(-10, 10, 2001) at 0.01 eV/pt
      6. Clip to non-negative (physical constraint; handles interpolation noise)
      7. Concatenate: [d_dos_up(2001), d_dos_down(2001)] -> (4002,)

    Returns:
        target  : np.ndarray, shape (4002,), dtype float32 -- d-projected DOS
        efermi  : float, Fermi energy in eV (raw DFT value, for metadata)
        n_atoms : int, number of atoms in the unit cell

    Raises:
        KeyError:   if required JSON fields are missing
        ValueError: if array dimensions are wrong or n_atoms <= 0
    """

    # ── Mandatory field presence checks ─────────────────────────────────
    _fname = os.path.basename(json_path) if json_path else '<unknown>'
    for field in ('efermi', 'energies', 'structure', 'spd_dos'):
        if field not in data:
            raise KeyError(
                f"Missing field '{field}' in {_fname}. "
                f"Available keys: {list(data.keys())}"
            )

    if 'd' not in data['spd_dos']:
        raise KeyError(
            f"spd_dos has no 'd' key in {_fname}. "
            f"Available: {list(data['spd_dos'].keys())}"
        )

    if 'densities' not in data['spd_dos']['d']:
        raise KeyError(
            f"spd_dos['d'] has no 'densities' key in {_fname}. "
            f"Check path: spd_dos['d']['densities']['1'] (NOT spd_dos['d']['1'])."
        )

    for spin in ('1', '-1'):
        if spin not in data['spd_dos']['d']['densities']:
            raise KeyError(
                f"spd_dos['d']['densities'] missing spin key '{spin}' in {_fname}"
            )

    # Step 1 -- Load raw data
    energies = np.array(data['energies'], dtype=np.float64)
    efermi   = float(data['efermi'])
    n_atoms  = len(data['structure']['sites'])

    # ── Array dimension checks ───────────────────────────────────────────
    if len(energies) != _EXPECTED_RAW_POINTS:
        raise ValueError(
            f"Expected {_EXPECTED_RAW_POINTS} energy points, "
            f"got {len(energies)} in {_fname}"
        )

    if n_atoms <= 0:
        raise ValueError(f"n_atoms={n_atoms} is invalid in {_fname}")

    # Step 2 -- Extract d-orbital projected DOS
    # Source: data['spd_dos']['d']['densities']['1'/'−1']
    # Confirmed via 200-material validation:
    #   spd_dos['d'] == sum of pdos[site][dxy/dyz/dz2/dxz/dx2] over all sites (max diff=0)
    #   D-band center: mean=-3.44 eV (up), -3.16 eV (down) — physically correct for Heuslers
    spd_d    = data['spd_dos']['d']['densities']
    dos_up   = np.array(spd_d['1'],  dtype=np.float64)
    dos_down = np.array(spd_d['-1'], dtype=np.float64)

    for arr, label in ((dos_up, 'dos_up'), (dos_down, 'dos_down')):
        if len(arr) != _EXPECTED_RAW_POINTS:
            raise ValueError(
                f"{label} has {len(arr)} points, "
                f"expected {_EXPECTED_RAW_POINTS} in {_fname}"
            )

    # Step 3 -- Per-atom normalization
    # n_atoms from the embedded pymatgen structure (confirmed present in all files)
    dos_up   = dos_up   / n_atoms
    dos_down = dos_down / n_atoms

    # Step 4 -- Fermi alignment
    e_shifted = energies - efermi
    # e_shifted[0] approx -12.0 eV, e_shifted[-1] approx +12.0 eV (diagnostic confirmed)

    # ── Fermi-alignment range sanity check ───────────────────────────────
    e_min_shifted = float(e_shifted[0])
    if not (-13.0 < e_min_shifted < -11.0):
        warnings.warn(
            f"Unexpected Fermi-shifted energy start: {e_min_shifted:.3f} eV "
            f"(expected ~-12.0) in {_fname}. "
            f"efermi={efermi:.4f} eV. Proceeding with interpolation.",
            RuntimeWarning, stacklevel=2
        )

    # Step 5 -- Interpolate onto common grid
    # fill_value=0.0 for extrapolation (outside DFT window); bounds_error=False
    f_up   = interp1d(e_shifted, dos_up,   kind='linear',
                      fill_value=0.0, bounds_error=False)
    f_down = interp1d(e_shifted, dos_down, kind='linear',
                      fill_value=0.0, bounds_error=False)

    dos_up_aligned   = f_up(_COMMON_GRID)
    dos_down_aligned = f_down(_COMMON_GRID)

    # Step 6 -- Clip to non-negative (physical constraint)
    # Diagnostic: 0/51,257 files have negative DFT d-DOS; clip handles interp noise only
    dos_up_aligned   = np.clip(dos_up_aligned,   0.0, None)
    dos_down_aligned = np.clip(dos_down_aligned, 0.0, None)

    # ── Physics sanity: d-DOS should not be all zeros ────────────────────
    if dos_up_aligned.sum() < 1e-8 and dos_down_aligned.sum() < 1e-8:
        warnings.warn(
            f"Both spin channels near-zero after interpolation in {_fname}. "
            f"File may be corrupt.",
            RuntimeWarning, stacklevel=2
        )

    # Step 7 -- Concatenate spin channels
    target = np.concatenate([dos_up_aligned, dos_down_aligned])   # shape (4002,)
    assert target.shape == (4002,), f"Output shape {target.shape} != (4002,)"
    assert (target >= 0).all(), "Negative values after clip — should never happen"

    return target.astype(np.float32), efermi, n_atoms


def _load_dos(json_path: str) -> Tuple[np.ndarray, float, int]:
    """
    Load a DOS JSON file by path and return processed d-projected DOS.

    Thin wrapper around _process_dos_data for callers that only need the DOS
    and don't share the JSON dict with structure loading.

    Raises:
        FileNotFoundError: if json_path does not exist
        KeyError:          if required JSON fields are missing
        ValueError:        if array dimensions are wrong or n_atoms <= 0
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"DOS JSON not found: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return _process_dos_data(data, json_path)


def validate_dos_file_sample(dataset_dir: str, n_sample: int = 10,
                              seed: int = 42) -> dict:
    """
    Randomly sample n_sample DOS JSON files from a dataset directory and
    validate them. Returns a summary dict suitable for logging.

    Called by post_upload_checklist.sh to verify data integrity on HPC.

    Args:
        dataset_dir: path to dos_json/ directory (flat — {material_id}.json files directly inside)
        n_sample:    number of files to validate (default 10)
        seed:        random seed for reproducibility

    Returns:
        {
          'n_tested':           int,
          'n_passed':           int,
          'n_failed':           int,
          'failures':           list of (filename, error_msg),
          'd_center_up_mean':   float (eV),
          'd_center_down_mean': float (eV),
        }

    D-band center sanity: should be negative for Heuslers
    (expected: mean ~-3.4 eV up, ~-3.2 eV down for full_cubic).
    """
    import glob
    import random
    random.seed(seed)

    files = glob.glob(os.path.join(dataset_dir, '*.json'))
    if not files:
        return {
            'n_tested': 0, 'n_passed': 0, 'n_failed': 0,
            'failures': [('N/A', 'No JSON files found')],
            'd_center_up_mean': float('nan'),
            'd_center_down_mean': float('nan'),
        }

    sample = random.sample(files, min(n_sample, len(files)))
    passed = 0
    failed = []
    d_centers_up, d_centers_down = [], []

    for fp in sample:
        try:
            dos, _, _ = _load_dos(fp)
            assert dos.shape == (4002,), f"Shape {dos.shape} != (4002,)"
            assert (dos >= 0).all(), "Negative DOS values"
            e = _COMMON_GRID
            up_ch   = dos[:2001].astype(np.float64)
            down_ch = dos[2001:].astype(np.float64)
            if up_ch.sum() > 1e-6:
                d_centers_up.append(
                    float(np.trapz(up_ch * e, e) / np.trapz(up_ch, e))
                )
            if down_ch.sum() > 1e-6:
                d_centers_down.append(
                    float(np.trapz(down_ch * e, e) / np.trapz(down_ch, e))
                )
            passed += 1
        except Exception as ex:
            failed.append((os.path.basename(fp), str(ex)))

    return {
        'n_tested':           len(sample),
        'n_passed':           passed,
        'n_failed':           len(failed),
        'failures':           failed,
        'd_center_up_mean':   float(np.mean(d_centers_up))   if d_centers_up   else float('nan'),
        'd_center_down_mean': float(np.mean(d_centers_down)) if d_centers_down else float('nan'),
    }


class DOSDataset(Dataset):
    """
    PyTorch Dataset for predicting d-projected electronic DOS from crystal CIF structures.

    TARGET: d-projected spin-resolved DOS (NOT total DOS).
    SOURCE: data['spd_dos']['d']['densities']['1'/'-1'] in pymatgen CompleteDos JSON.
    REASON: Mott formula S propto (1/g_d) * dg_d/dE at E_F; d-electrons dominate
            thermoelectric transport in transition-metal Heuslers.

    D-projected DOS preprocessing pipeline (applied in _load_dos):
      1. Extract d-DOS: spd_dos['d']['densities']['1']=up, ['-1']=down
         (confirmed identical to summing pdos d-suborbitals; validated 200 materials)
      2. Per-atom normalization: divide by n_atoms from structure.sites
         Units: states/eV/atom (removes cell-size dependence)
      3. Fermi alignment: shift energy axis by -efermi
         Result: E=0 corresponds to Fermi level for all materials
      4. Interpolate to common grid: linspace(-10, 10, 2001) at 0.01 eV
         Coverage: 100% of all 51,257 materials confirmed by diagnostic
      5. Clip to non-negative: physical constraint DOS >= 0
      6. Concatenate: [d_dos_up(2001), d_dos_down(2001)] -> (4002,)

    Spin convention (confirmed from diagnostic):
      spd_dos['d']['densities']['1']  = d-band spin up  (index 0:2001 in target)
      spd_dos['d']['densities']['-1'] = d-band spin down (index 2001:4002 in target)

    Energy axis (same for all samples after preprocessing):
      E = linspace(-10.0, 10.0, 2001)  [eV, Fermi-centered]
      E[1000] = 0.0 eV = Fermi level exactly
      E[800]  = -2.0 eV, E[1200] = +2.0 eV
      E[900]  = -1.0 eV, E[1100] = +1.0 eV

    Each sample:
        Input:  Structure embedded in DOS JSON -> DGL atomistic graph (atom_graph, line_graph, lattice)
        Target: d-DOS tensor shape (4002,) = [d_spin_up(2001), d_spin_down(2001)]
                Per-atom, Fermi-aligned, clipped to non-negative.

    Batch dict keys (match train_hpc.py and run_inference.py):
        'graph':   (batched_atom_graph, batched_line_graph, stacked_lattice)
                   -- 3-tuple as required by ALIGNN.forward()
        'dos':     Tensor shape (batch_size, 4002)   -- d-projected, per-atom, Fermi-aligned
        'id':      List[str] of material IDs
        'efermi':  Tensor shape (batch_size, 1)      -- raw DFT Fermi energy (eV)
        'n_atoms': Tensor shape (batch_size, 1)      -- atoms per unit cell
    """

    # Physics-confirmed constants (do not modify)
    COMMON_GRID      = _COMMON_GRID                # linspace(-10, 10, 2001)
    GRID_SPACING     = 0.01                        # eV per point
    FERMI_IDX        = 1000                        # index of E=0.0 eV exactly
    N_ENERGY_POINTS  = 2001                        # points per spin channel
    N_SPIN_CHANNELS  = 2
    TARGET_DIM       = 4002                        # total output dimension
    TARGET_TYPE      = 'd_projected_spin_resolved' # d-orbital projected, NOT total DOS
    NORMALIZATION    = 'per_atom'                  # documented convention
    ATOM_INPUT_FEATURES = ATOM_INPUT_FEATURES      # 97 = 92 base CGCNN + 5 Heusler-specific

    def __init__(
        self,
        dos_dir: str,
        ids_file: str,
        id_column: str = 'material_id',
        cutoff: float = 8.0,
        max_neighbors: int = 12,
    ):
        """
        Args:
            dos_dir:       Directory containing {material_id}.json files
                           (pymatgen CompleteDos JSON format). Structure is
                           read from data['structure'] inside each JSON —
                           no separate CIF directory is needed.
            ids_file:      CSV file with one column of material IDs.
            id_column:     Column name in the CSV (default: 'material_id').
            cutoff:        Cutoff radius in Angstroms for graph edges.
            max_neighbors: Max neighbours per atom in the atom graph.
        """
        df = pd.read_csv(ids_file)
        if id_column not in df.columns:
            raise ValueError(
                f"Column '{id_column}' not found in {ids_file}. "
                f"Available: {df.columns.tolist()}"
            )
        all_ids = df[id_column].tolist()

        # Support per-sample dos_path column for multi-dataset pre-training.
        # If present, each row's dos_path is used directly instead of dos_dir/id.json.
        has_dos_path = 'dos_path' in df.columns
        if has_dos_path:
            all_paths = df['dos_path'].tolist()
        else:
            all_paths = [os.path.join(dos_dir, f"{mid}.json") for mid in all_ids]

        # Keep only materials where the JSON exists
        self.ids:   List[str] = []
        self.paths: List[str] = []
        skipped:    List[str] = []
        for mid, path in zip(all_ids, all_paths):
            if os.path.exists(path):
                self.ids.append(mid)
                self.paths.append(path)
            else:
                skipped.append(mid)

        if skipped:
            print(f"WARNING: {len(skipped)} materials skipped (missing JSON):")
            for mid in skipped[:5]:
                print(f"  {mid}")
            if len(skipped) > 5:
                print(f"  ... and {len(skipped) - 5} more")

        self.dos_dir       = dos_dir
        self.cutoff        = cutoff
        self.max_neighbors = max_neighbors

        print(
            f"DOSDataset: {len(self.ids)} valid / {len(all_ids)} total "
            f"from {os.path.basename(ids_file)}"
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        mid       = self.ids[idx]
        json_path = self.paths[idx]

        # -- Load JSON once (structure + DOS share the same dict) --------------
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[WARNING] Skipping {mid} (JSON load failed): {exc}", flush=True)
            return None

        # -- Graph construction ------------------------------------------------
        # Structure is read from data['structure'] — the exact DFT geometry.
        # No CIF file needed; all DxMag CIFs are broken due to encoding bug.
        try:
            atoms = _load_structure_from_json_data(data, json_path)
        except Exception as exc:
            print(f"[WARNING] Skipping {mid} (structure load failed): {exc}", flush=True)
            return None

        try:
            g, lg, lattice = _build_graph(atoms, self.cutoff, self.max_neighbors)
        except Exception as exc:
            print(f"[WARNING] Skipping {mid} (graph build failed): {exc}", flush=True)
            return None

        # -- DOS target --------------------------------------------------------
        try:
            dos_array, efermi, n_atoms = _process_dos_data(data, json_path)
        except Exception as exc:
            print(f"[WARNING] Skipping {mid} (DOS processing failed): {exc}", flush=True)
            return None
        dos_tensor = torch.tensor(dos_array, dtype=torch.float32)

        # -- Return sample -----------------------------------------------------
        # 'graph' is a 3-tuple: ALIGNN.forward() unpacks as: g, lg, lat = g
        return {
            'graph':   (g, lg, lattice),
            'dos':     dos_tensor,                                  # shape (4002,)
            'id':      mid,
            'efermi':  torch.tensor([efermi],  dtype=torch.float32),  # shape (1,)
            'n_atoms': torch.tensor([n_atoms], dtype=torch.float32),  # shape (1,)
        }

    @staticmethod
    def collate(batch: list):
        """
        Custom collate function for DataLoader.

        Required because DGL graphs cannot use torch's default collation.
        Pass to DataLoader as: collate_fn=DOSDataset.collate

        Filters out None entries produced by __getitem__ when a sample is
        skipped (structure/graph build failure, etc.).
        Returns None if every sample in the batch was skipped — callers
        (train_one_epoch, validate, variance_aware_output_init) must guard:
            if batch is None: continue

        Returns:
            None  — if all samples in this batch were skipped
            {
                'graph':   (batched_atom_graph, batched_line_graph, lattice_stack),
                'dos':     Tensor shape (N, 4002),
                'id':      List[str] of length N,
                'efermi':  Tensor shape (N, 1),
                'n_atoms': Tensor shape (N, 1),
            }
        """
        valid = [b for b in batch if b is not None]
        n_skipped = len(batch) - len(valid)
        if n_skipped > 0:
            print(f"[WARNING] collate: {n_skipped}/{len(batch)} samples skipped in this batch",
                  flush=True)
        if not valid:
            return None

        atom_graphs = [b['graph'][0] for b in valid]
        line_graphs = [b['graph'][1] for b in valid]
        lattices    = [b['graph'][2] for b in valid]   # each (3, 3)
        doses       = [b['dos']      for b in valid]   # each (4002,)
        ids         = [b['id']       for b in valid]
        efermis     = [b['efermi']   for b in valid]   # each (1,)
        n_atoms_l   = [b['n_atoms']  for b in valid]   # each (1,)

        return {
            'graph': (
                dgl.batch(atom_graphs),           # batched DGLGraph
                dgl.batch(line_graphs),           # batched DGLGraph
                torch.stack(lattices),            # (N, 3, 3)
            ),
            'dos':     torch.stack(doses),        # (N, 4002)
            'id':      ids,
            'efermi':  torch.stack(efermis),      # (N, 1)
            'n_atoms': torch.stack(n_atoms_l),    # (N, 1)
        }
