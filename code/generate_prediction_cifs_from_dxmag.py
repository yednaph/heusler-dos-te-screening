#!/usr/bin/env python3
"""
generate_prediction_cifs_from_dxmag.py
=======================================
Reads the DXMag Heusler Excel database, generates CIF files for all materials
NOT already in training data, and organises them into per-category folders.

Run with the conda Python that has openpyxl:
    C:/Users/YOUR_USERNAME/Conda/python.exe code/generate_prediction_cifs_from_dxmag.py

Output: data/predictions/{full_cubic,full_tetra,inverse_cubic,inverse_tetra,half_cubic,half_tetra}/

CIF format exactly matches the training CIFs:
    data_{composition}_{uuid}
    # Generated from DXMag Heusler Database
    # UUID: ...
    # Composition: ...
    _cell_length_a    ...
    ...
    _symmetry_space_group_name_H-M '...'
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x  _atom_site_fract_y  _atom_site_fract_z
    El1  El  x  y  z
    ...
"""

import argparse
import ast
import collections
import math
import pathlib
import re
import sys


# ---------------------------------------------------------------------------
# Periodic table: atomic number -> element symbol
# ---------------------------------------------------------------------------

ELEMENT_SYMBOLS = {
    1: 'H',  2: 'He', 3: 'Li', 4: 'Be', 5: 'B',  6: 'C',  7: 'N',  8: 'O',
    9: 'F',  10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
    23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
    30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
    37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
    44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
    58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
    65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
    72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
    79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
    86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U',
    93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm',
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cell_params_from_matrix(matrix):
    """
    Convert 3x3 lattice matrix (rows = lattice vectors) to
    (a, b, c, alpha_deg, beta_deg, gamma_deg).
    """
    v1, v2, v3 = matrix[0], matrix[1], matrix[2]

    def norm(v):
        return math.sqrt(sum(x * x for x in v))

    def dot(va, vb):
        return sum(a * b for a, b in zip(va, vb))

    def angle_deg(va, vb):
        n1, n2 = norm(va), norm(vb)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        cos_a = dot(va, vb) / (n1 * n2)
        cos_a = max(-1.0, min(1.0, cos_a))   # clamp for floating-point noise
        return math.degrees(math.acos(cos_a))

    a = norm(v1)
    b = norm(v2)
    c = norm(v3)
    alpha = angle_deg(v2, v3)   # angle between b and c
    beta  = angle_deg(v1, v3)   # angle between a and c
    gamma = angle_deg(v1, v2)   # angle between a and b
    return a, b, c, alpha, beta, gamma


def parse_matrix(s):
    """Parse a 3x3 list-of-lists string, e.g. '[[0.0,3.5,3.5],[3.5,0.0,3.5],...]'."""
    return ast.literal_eval(s.strip())


def parse_positions(s):
    """Parse fractional positions string: list of [x,y,z] lists."""
    return ast.literal_eval(s.strip())


def parse_numbers(s):
    """
    Parse atomic numbers string.  Handles both:
      - Python list with commas: '[21, 21, 21, 13]'
      - Numpy repr without commas: '[21 21 21 13]'
    Returns list of int.
    """
    s = s.strip()
    if ',' in s:
        return ast.literal_eval(s)
    # numpy-style: extract all digit sequences
    return [int(x) for x in re.findall(r'\d+', s)]


def get_category(mat_type: str, space_group: str) -> str:
    """
    Map (type, space_group) to one of the 6 dataset categories.
    mat_type: 'full', 'half', 'inverse'
    space_group examples: 'Fm-3m', 'F-43m', 'I4/mmm', 'I-4m2'
    """
    sg = space_group.strip()
    # Cubic: Fm-3m (full), F-43m (half or inverse)
    if 'Fm' in sg or 'F-4' in sg:
        geo = 'cubic'
    # Tetragonal: I4/mmm (full), I-4m2 (half or inverse)
    elif 'I4' in sg or 'I-4' in sg:
        geo = 'tetra'
    else:
        return 'unclassified'

    t = mat_type.lower().strip()
    if t not in ('full', 'half', 'inverse'):
        return 'unclassified'
    return f'{t}_{geo}'


def build_trained_uuid_index(data_root: pathlib.Path) -> set:
    """
    Return set of UUIDs (lowercase) that already have training CIFs.
    UUID pattern: 8-4-4-4-12 hex characters.
    """
    uuid_re = re.compile(
        r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})',
        re.I
    )
    trained = set()
    for ds in ('full_cubic', 'full_tetra', 'inverse_cubic',
               'inverse_tetra', 'half_cubic', 'half_tetra'):
        cif_dir = data_root / ds / 'cif'
        if cif_dir.exists():
            for f in cif_dir.glob('*.cif'):
                m = uuid_re.search(f.stem)
                if m:
                    trained.add(m.group(1).lower())
    return trained


def write_cif(path: pathlib.Path, uuid: str, composition: str,
              a, b, c, alpha, beta, gamma, space_group: str,
              elements, positions):
    """Write a CIF file in the standard training format."""
    lines = [
        f'data_{composition}_{uuid}',
        f'# Generated from DXMag Heusler Database',
        f'# UUID: {uuid}',
        f'# Composition: {composition}',
        '',
        f'_cell_length_a    {a:.6f}',
        f'_cell_length_b    {b:.6f}',
        f'_cell_length_c    {c:.6f}',
        f'_cell_angle_alpha {alpha:.3f}',
        f'_cell_angle_beta  {beta:.3f}',
        f'_cell_angle_gamma {gamma:.3f}',
        f"_symmetry_space_group_name_H-M '{space_group}'",
        '',
        'loop_',
        '_atom_site_label',
        '_atom_site_type_symbol',
        '_atom_site_fract_x',
        '_atom_site_fract_y',
        '_atom_site_fract_z',
    ]
    # Count per element for labels (e.g. Ag1, Ag2, ...)
    elem_count = collections.Counter()
    for el, pos in zip(elements, positions):
        elem_count[el] += 1
        idx = elem_count[el]
        x, y, z = pos
        label = f'{el}{idx}'
        lines.append(f'{label:<6s} {el:<4s} {x:.6f}   {y:.6f}   {z:.6f}')

    path.write_text('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CATEGORIES = [
    'full_cubic', 'full_tetra',
    'inverse_cubic', 'inverse_tetra',
    'half_cubic', 'half_tetra',
    'unclassified',
]


def main():
    parser = argparse.ArgumentParser(
        description='Generate prediction CIFs from DXMag Excel database.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--excel', type=pathlib.Path,
                        default=pathlib.Path('data/DXMag_HeuslerDB_2025-09-09.xlsx'),
                        help='Path to DXMag Excel file')
    parser.add_argument('--data_root', type=pathlib.Path, default=pathlib.Path('data'),
                        help='Root data directory (contains full_cubic/, etc.)')
    parser.add_argument('--output_dir', type=pathlib.Path,
                        default=pathlib.Path('data/predictions'),
                        help='Output directory for prediction CIFs')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print what would happen without writing any files')
    parser.add_argument('--limit', type=int, default=0,
                        help='Process at most N rows (0 = all; for quick testing)')
    args = parser.parse_args()

    if not args.excel.exists():
        print(f'ERROR: Excel file not found: {args.excel}', file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 1. Build trained UUID index                                          #
    # ------------------------------------------------------------------ #
    print('Building trained UUID index...')
    trained_uuids = build_trained_uuid_index(args.data_root)
    print(f'  Found {len(trained_uuids):,} trained UUIDs to skip')

    # ------------------------------------------------------------------ #
    # 2. Create output folders                                             #
    # ------------------------------------------------------------------ #
    if not args.dry_run:
        for cat in CATEGORIES:
            (args.output_dir / cat).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 3. Read Excel                                                        #
    # ------------------------------------------------------------------ #
    try:
        import openpyxl
    except ImportError:
        print('ERROR: openpyxl not installed.  Run:', file=sys.stderr)
        print('  pip install openpyxl', file=sys.stderr)
        sys.exit(1)

    print(f'Loading Excel: {args.excel} ...', flush=True)
    wb = openpyxl.load_workbook(args.excel, read_only=True, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    data_rows = rows[1:]
    print(f'  {len(data_rows):,} data rows, {len(header)} columns')
    print(f'  Columns: {list(header[:10])}')

    # Map column names to indices (case-insensitive, strip whitespace)
    col = {str(h).strip().lower(): i for i, h in enumerate(header) if h is not None}
    print(f'  Column map (first 10): {dict(list(col.items())[:10])}')

    # Expected columns (try to find them)
    # Common names in DXMag: uuid, composition/formula, type, space_group/sg, cell, positions, numbers
    def find_col(*names):
        for n in names:
            if n.lower() in col:
                return col[n.lower()]
        return None

    i_uuid  = find_col('uuid', 'id')
    i_comp  = find_col('composition', 'formula', 'chemical_formula')
    i_type  = find_col('type', 'structure_type', 'heusler_type')
    i_sg    = find_col('space_group', 'spacegroup', 'sg', 'space_group_symbol',
                       'space group symbol', 'space group')
    i_cell  = find_col('cell', 'lattice', 'cell_matrix', 'lattice_matrix')
    i_pos   = find_col('positions', 'fractional_positions', 'frac_coords', 'coords')
    i_nums  = find_col('numbers', 'atomic_numbers', 'species')

    print(f'\n  Column indices resolved:')
    print(f'    uuid={i_uuid}  composition={i_comp}  type={i_type}  '
          f'space_group={i_sg}  cell={i_cell}  positions={i_pos}  numbers={i_nums}')

    missing = [name for name, idx in [
        ('uuid', i_uuid), ('composition', i_comp), ('type', i_type),
        ('space_group', i_sg), ('cell', i_cell), ('positions', i_pos),
        ('numbers', i_nums),
    ] if idx is None]
    if missing:
        print(f'\nERROR: Could not find columns: {missing}', file=sys.stderr)
        print('Available columns:', list(col.keys()), file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 4. Process rows                                                      #
    # ------------------------------------------------------------------ #
    counts   = collections.Counter()
    skipped  = 0
    errors   = []
    written  = 0

    limit = args.limit if args.limit > 0 else len(data_rows)

    for row_idx, row in enumerate(data_rows[:limit]):
        if row_idx % 5000 == 0 and row_idx > 0:
            print(f'  ...processed {row_idx:,} rows', flush=True)

        uuid        = str(row[i_uuid]).strip().lower() if row[i_uuid] else ''
        composition = str(row[i_comp]).strip()         if row[i_comp] else ''
        mat_type    = str(row[i_type]).strip().lower() if row[i_type] else ''
        sg          = str(row[i_sg]).strip()            if row[i_sg]   else ''
        cell_str    = str(row[i_cell]).strip()          if row[i_cell] else ''
        pos_str     = str(row[i_pos]).strip()           if row[i_pos]  else ''
        nums_str    = str(row[i_nums]).strip()          if row[i_nums] else ''

        if not uuid or uuid == 'none':
            continue

        # Skip already-trained materials
        if uuid in trained_uuids:
            skipped += 1
            continue

        # Determine category
        category = get_category(mat_type, sg)
        counts[category] += 1

        if args.dry_run:
            continue

        # Parse geometry
        try:
            matrix    = parse_matrix(cell_str)
            positions = parse_positions(pos_str)
            numbers   = parse_numbers(nums_str)
        except Exception as e:
            errors.append((uuid, f'parse error: {e}'))
            continue

        if len(numbers) != len(positions):
            errors.append((uuid, f'atom count mismatch: {len(numbers)} numbers vs {len(positions)} positions'))
            continue

        # Convert lattice matrix -> cell parameters
        try:
            a, b, c, alpha, beta, gamma = cell_params_from_matrix(matrix)
        except Exception as e:
            errors.append((uuid, f'cell param error: {e}'))
            continue

        # Map atomic numbers -> element symbols
        try:
            elements = [ELEMENT_SYMBOLS[int(n)] for n in numbers]
        except KeyError as e:
            errors.append((uuid, f'unknown atomic number: {e}'))
            continue

        # Write CIF
        out_path = args.output_dir / category / f'{composition}_{uuid}.cif'
        try:
            write_cif(out_path, uuid, composition, a, b, c, alpha, beta, gamma, sg, elements, positions)
            written += 1
        except Exception as e:
            errors.append((uuid, f'write error: {e}'))

    wb.close()

    # ------------------------------------------------------------------ #
    # 5. Summary                                                           #
    # ------------------------------------------------------------------ #
    print(f'\n=== Generation results ===')
    for cat in CATEGORIES:
        n = counts.get(cat, 0)
        if n > 0:
            print(f'  {cat:22s}: {n:6,}')
    print(f'  {"TOTAL to predict":22s}: {sum(counts.values()):6,}')
    print(f'  {"skipped (trained)":22s}: {skipped:6,}')
    if errors:
        print(f'  {"errors":22s}: {len(errors):6,}')
        for uid, msg in errors[:10]:
            print(f'    {uid}: {msg}')

    if not args.dry_run:
        print(f'\nDone. {written:,} CIFs written to {args.output_dir}/')
    else:
        print(f'\nDry run complete. No files written.')


if __name__ == '__main__':
    main()
