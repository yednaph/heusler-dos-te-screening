#!/usr/bin/env python3
"""dataset_characterization.py -- descriptive statistics of the training and prediction sets (paper Data section).

Sets (membership by UUID, the join key documented in DOS_EXPORT/metadata/FAMILY_TAXONOMY.md):
  training   = DXMag entries with a DFT spin-resolved DOS file, i.e. a JSON in data/{family}/dos_json/
               (file name {composition}_{uuid}.json; cross-checked against DOS_EXPORT/metadata/splits)
  prediction = DXMag entries without DOS, i.e. a CIF in data/predictions/{family}/ (generated from the xlsx)
Metadata: DXMag HeuslerDB release 2025-09-09 (data/DXMag_HeuslerDB_2025-09-09.xlsx, 106,235 rows; byte-identical
copy in DOS_EXPORT/metadata). Family = DXMag `type` (full/half/inverse) x lattice from `space group symbol`
(Fm-3m/F-43m -> cubic, I4/mmm/I-4m2 -> tetra), checked against the directory each file sits in.
VEC: reuses VAL/vec() from code/paper/screen_eval.py (s+d electrons for transition metals, s+p for main group).

Reads existing files only (no DFT, no ML). Outputs in paper_rebuild/v7_research_rewrite/figures/data/:
  dataset_characterization_by_family.csv     one row per (set, family) incl. 'all' -- wide table of metrics
  dataset_characterization_distributions.csv long table: atoms/cell, f.u./cell, VEC, elements per compound, element use
  dataset_characterization_overlap.csv       prediction-vs-training composition/energy overlap per family
  dataset_characterization.txt               short human-readable summary
"""
import os as _os_htr  # release v1: project paths come from the environment
_HTR = _os_htr.environ.get('HEUSLER_TE_ROOT', _os_htr.path.abspath(_os_htr.path.join(_os_htr.path.dirname(__file__), '../../../..')))
_HTE = _os_htr.environ.get('HEUSLER_TE_EXPORT', _os_htr.path.join(_HTR, 'DOS_EXPORT'))
import re                                                    # composition-string parsing
import sys                                                   # path insertion to reuse the project VEC function
from functools import reduce                                 # gcd over several stoichiometric counts
from math import gcd                                         # greatest common divisor for formula reduction
from pathlib import Path                                     # file-system paths
import numpy as np                                           # numerics
import pandas as pd                                          # tables

# ---------------------------------------------------------------- paths (all inputs read-only)
REPO = Path(_HTR)  # project root
XLSX = REPO / 'data/DXMag_HeuslerDB_2025-09-09.xlsx'                  # DXMag metadata, release 2025-09-09
SPLITS = Path(_HTE + '/metadata/splits')  # exported CV splits (training IDs only)
OUT = Path(__file__).resolve().parents[2] / 'figures' / 'data'        # paper_rebuild/v7_research_rewrite/figures/data
OUT.mkdir(parents=True, exist_ok=True)                                # create output folder if absent
FAMILIES = ['full_cubic', 'full_tetra', 'half_cubic', 'half_tetra', 'inverse_cubic', 'inverse_tetra']  # six families
EXPECTED = {'training': dict(zip(FAMILIES, [9396, 8047, 9396, 6909, 9072, 8437])),       # counts quoted in the paper
            'prediction': dict(zip(FAMILIES, [4504, 13479, 4914, 10896, 6702, 14483]))}
E, H, EF = 'energy (eV/atom)', 'hull distance (eV/atom)', 'formation energy (eV/atom)'      # DXMag energy columns
M, MC, NAT = 'total magnetization (muB/f.u.)', 'total magnetization (muB/cell)', 'number of atoms in the cell'
MEV = 1e-3                                                            # 1 meV/atom in eV/atom

# ---------------------------------------------------------------- VEC: reuse the project definition
sys.path.insert(0, str(REPO / 'code/paper'))                          # make screen_eval importable
from screen_eval import VAL, vec                                      # noqa: E402  (VAL table and vec(formula))


def parse(formula):
    """Composition string -> {element: integer count} (DXMag strings are integer-stoichiometric, e.g. Co2MnSi)."""
    d = {}                                                            # element -> count accumulator
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', formula):         # element symbol followed by optional count
        d[el] = d.get(el, 0) + (int(n) if n else 1)                   # an absent count means 1
    return d


def reduced(formula):
    """Order-independent reduced formula: counts divided by their gcd, elements sorted alphabetically."""
    d = parse(formula)                                                # integer counts
    g = reduce(gcd, d.values())                                       # common divisor of all counts
    return ''.join(f'{el}{d[el] // g if d[el] // g > 1 else ""}' for el in sorted(d))  # canonical string


def vec_safe(formula):
    """Project vec() per formula unit, NaN if an element is missing from the VAL table."""
    try:
        return vec(formula)                                           # s+d (TM) / s+p (main group) electron count
    except KeyError:
        return np.nan                                                 # element not covered by VAL


def lattice(sg):
    """Space-group symbol -> lattice class used in the family names."""
    sg = str(sg).strip()                                              # tolerate stray whitespace
    if sg in ('Fm-3m', 'F-43m'):
        return 'cubic'                                                # L2_1 / C1_b / XA
    if sg in ('I4/mmm', 'I-4m2'):
        return 'tetra'                                                # tetragonal variants
    return 'unclassified'                                             # anything else (none expected)


# ---------------------------------------------------------------- load metadata
dx = pd.read_excel(XLSX)                                              # 106,235 rows expected
dx['uuid'] = dx['UUID'].astype(str).str.strip().str.lower()          # lowercase UUID = join key
dx['family_dx'] = dx['type'].str.strip() + '_' + dx['space group symbol'].map(lattice)  # family from metadata
dx['red'] = dx['composition'].map(reduced)                            # canonical reduced formula
dx['n_el'] = dx['composition'].map(lambda f: len(parse(f)))           # number of distinct elements per compound
dx['atoms_fu'] = dx['composition'].map(lambda f: sum(parse(f).values()))  # atoms per formula unit as written
dx['fu_cell'] = dx[NAT] / dx['atoms_fu']                              # formula units in the listed cell
dx['vec'] = dx['composition'].map(vec_safe)                           # valence electrons per formula unit

# ---------------------------------------------------------------- set membership from the file system
rows = []                                                             # (set, dir_family, material_id)
for f in FAMILIES:
    for p in (REPO / f'data/{f}/dos_json').glob('*.json'):           # DFT DOS present -> training
        rows.append(('training', f, p.stem))
    for p in (REPO / f'data/predictions/{f}').glob('*.cif'):         # no DOS, CIF generated from xlsx -> prediction
        rows.append(('prediction', f, p.stem))
mem = pd.DataFrame(rows, columns=['set', 'family', 'material_id'])    # membership table
mem['uuid'] = mem['material_id'].str.rsplit('_', n=1).str[1].str.lower()   # UUID part of {formula}_{uuid}
mem['formula_file'] = mem['material_id'].str.rsplit('_', n=1).str[0]       # formula part of the file name

# cross-check: union of the exported Exp A fold-0 train/val/test IDs must equal the DOS JSON IDs per family
split_ok = {}                                                         # family -> bool
for f in FAMILIES:
    ids = set(pd.concat([pd.read_csv(SPLITS / f / f'expA_fold0_{s}.csv') for s in ('train', 'val', 'test')])
              ['material_id'])                                        # all IDs in fold 0 (covers whole family)
    split_ok[f] = ids == set(mem.query('set == "training" and family == @f')['material_id'])  # exact equality

# ---------------------------------------------------------------- integrity checks
chk = {}                                                              # name -> value, reported in the summary
chk['xlsx_rows'] = len(dx)                                            # total DXMag rows
chk['xlsx_uuid_duplicates'] = int(dx['uuid'].duplicated().sum())      # repeated UUIDs in the metadata
chk['file_uuid_duplicates_training'] = int(mem.query('set=="training"')['uuid'].duplicated().sum())
chk['file_uuid_duplicates_prediction'] = int(mem.query('set=="prediction"')['uuid'].duplicated().sum())
chk['uuid_in_both_sets'] = len(set(mem.query('set=="training"').uuid) & set(mem.query('set=="prediction"').uuid))
chk['file_uuids_not_in_xlsx'] = len(set(mem.uuid) - set(dx.uuid))    # files without metadata
chk['xlsx_uuids_in_no_set'] = len(set(dx.uuid) - set(mem.uuid))      # metadata rows without a file
chk['split_union_equals_dos_json'] = all(split_ok.values())          # exported splits agree with DOS archive
j = mem.merge(dx[['uuid', 'family_dx', 'composition']], on='uuid', how='left')  # attach metadata to files
chk['family_dir_vs_xlsx_mismatch'] = int((j['family'] != j['family_dx']).sum())  # directory vs metadata family
chk['formula_file_vs_xlsx_mismatch'] = int((j['formula_file'] != j['composition']).sum())  # file name vs metadata
chk['unclassified_space_groups'] = int(dx['family_dx'].str.endswith('unclassified').sum())
setmap = mem.drop_duplicates('uuid').set_index('uuid')['set']        # UUID -> set
dx['set'] = dx['uuid'].map(setmap).fillna('none')                    # set label on every metadata row
dx['family'] = dx['family_dx']                                        # use metadata family (identical if mismatch = 0)
counts = dx.groupby(['set', 'family']).size()                         # observed counts
cmp_rows = [(s, f, EXPECTED[s][f], int(counts.get((s, f), 0))) for s in EXPECTED for f in FAMILIES]
count_check = pd.DataFrame(cmp_rows, columns=['set', 'family', 'quoted', 'observed'])  # quoted vs observed
count_check['match'] = count_check['quoted'] == count_check['observed']

# Two composition keys are used throughout:
#   'composition' = DXMag string as written; for half Heuslers its element ORDER encodes the site assignment
#                   (AgAuAl and AuAgAl are different site permutations), so it is a site-ordered label;
#   'red'         = order-independent reduced formula (merges those permutations; also merges full vs inverse).
KEYS = {'composition': 'site-ordered DXMag composition string', 'red': 'order-independent reduced formula'}
tr_all = dx[dx['set'] == 'training']                                  # training rows
for k in KEYS:
    mn = dx.groupby(['family', k])[E].transform('min')                # lowest energy in (family, key) group
    chk[f'training_is_group_min_E[{k}]'] = int((tr_all[E] == mn[tr_all.index]).sum())   # training entries at minimum
    chk[f'training_groups_with_gt1_training[{k}]'] = int((tr_all.groupby(['family', k]).size() > 1).sum())
chk['distinct_composition_strings_vs_reduced_formulas'] = (dx['composition'].nunique(), dx['red'].nunique())
chk['families_where_strings_alias_reduced'] = {f: int(n) for f, n in
                                               (dx.groupby(['family', 'red'])['composition'].nunique() > 1)
                                               .groupby('family').sum().items() if n}  # half families only expected


# ---------------------------------------------------------------- per (set, family) metrics
def metrics(s):
    """Wide-table metrics for one subset of rows."""
    m = s[M]                                                          # magnetisation per f.u. (may be missing)
    r = {
        'entries': len(s),                                            # rows
        'unique_uuid': s['uuid'].nunique(),                           # distinct UUIDs
        'unique_composition_strings': s['composition'].nunique(),     # as written in DXMag
        'unique_reduced_formulas': s['red'].nunique(),                # order-independent reduced formulas
        'distinct_elements': len(set().union(*[set(parse(c)) for c in s['composition'].unique()])),
        'share_binary': (s['n_el'] == 2).mean(),                      # e.g. Sc3Al (X = Y)
        'share_ternary': (s['n_el'] == 3).mean(),                     # X != Y != Z
        'atoms_cell_median': s[NAT].median(),                         # listed-cell size
        'atoms_cell_min': s[NAT].min(),
        'atoms_cell_max': s[NAT].max(),
        'share_1fu_cell': (s['fu_cell'] == 1).mean(),                 # one formula unit = primitive cell
        'mag_available': m.notna().mean(),                            # share with a DXMag moment
        'mag_missing_n': int(m.isna().sum()),
        'share_absM_gt_0.01_of_available': (m.abs() > 0.01).sum() / max(m.notna().sum(), 1),  # magnetic, loose bar
        'share_absM_gt_0.1_of_available': (m.abs() > 0.1).sum() / max(m.notna().sum(), 1),    # magnetic, firm bar
        'hull_available': s[H].notna().mean(),                        # hull distance availability
        'hull_missing_n': int(s[H].isna().sum()),
        'hull_median_eV': s[H].median(),
        'hull_q25_eV': s[H].quantile(0.25),
        'hull_q75_eV': s[H].quantile(0.75),
        'share_on_hull_le_1e-6': (s[H] <= 1e-6).mean(),               # on the convex hull
        'share_hull_lt_0.05': (s[H] < 0.05).mean(),
        'share_hull_lt_0.10': (s[H] < 0.10).mean(),
        'formation_available': s[EF].notna().mean(),                  # formation energy availability
        'formation_median_eV': s[EF].median(),
        'share_formation_lt_0': (s[EF] < 0).mean(),                   # exothermic vs elements
        'vec_available': s['vec'].notna().mean(),
        'vec_median': s['vec'].median(),
        'share_vec_8': (s['vec'] == 8).mean(),
        'share_vec_18': (s['vec'] == 18).mean(),
        'share_vec_24': (s['vec'] == 24).mean(),
        'share_vec_magic': (s['vec'] == np.where(s['family'].str.startswith('half'), 18, 24)).mean(),  # 18 half / 24 full,inv
    }
    return r


sub = dx[dx['set'].isin(['training', 'prediction'])]                  # rows belonging to a set
tab = []                                                              # wide-table rows
for st in ['training', 'prediction', 'all']:
    ss = sub if st == 'all' else sub[sub['set'] == st]                # 'all' = union of the two sets
    for f in FAMILIES + ['all']:
        s = ss if f == 'all' else ss[ss['family'] == f]               # per family or pooled
        tab.append({'set': st, 'family': f, **metrics(s)})
by_fam = pd.DataFrame(tab)                                            # wide table

# duplicates within a set: more than one entry with the same composition key in the same family
for i, r in by_fam.iterrows():
    ss = sub if r['set'] == 'all' else sub[sub['set'] == r['set']]   # set rows
    s = ss if r['family'] == 'all' else ss[ss['family'] == r['family']]   # family rows
    for k, tag in (('composition', 'str'), ('red', 'red')):           # both composition keys
        g = s.groupby(['family', k]).size()                           # entries per (family, key)
        by_fam.loc[i, f'family_{tag}_groups'] = len(g)                # distinct (family, key) pairs
        by_fam.loc[i, f'family_{tag}_groups_with_dup'] = int((g > 1).sum())   # pairs with >1 entry
        by_fam.loc[i, f'entries_in_dup_{tag}_groups'] = int(g[g > 1].sum())   # entries in such pairs
        by_fam.loc[i, f'max_entries_per_family_{tag}'] = int(g.max())          # largest multiplicity
    by_fam.loc[i, 'uuid_duplicates'] = int(s['uuid'].duplicated().sum())      # repeated UUIDs (expected 0)
by_fam.to_csv(OUT / 'dataset_characterization_by_family.csv', index=False, float_format='%.6g')

# ---------------------------------------------------------------- distributions (long format)
dist = []                                                             # (set, family, variable, value, count)
for st in ['training', 'prediction']:
    for f in FAMILIES + ['all']:
        s = sub[sub['set'] == st]                                     # one set
        s = s if f == 'all' else s[s['family'] == f]                  # one family or pooled
        for var in [NAT, 'fu_cell', 'vec', 'n_el']:
            for v, c in s[var].value_counts(dropna=False).sort_index().items():   # small-integer histogram
                dist.append((st, f, {NAT: 'atoms_in_cell'}.get(var, var), v, int(c)))
        el = pd.Series([e for c in s['composition'] for e in parse(c)]).value_counts()  # entries containing element
        for e, c in el.items():
            dist.append((st, f, 'entries_containing_element', e, int(c)))
pd.DataFrame(dist, columns=['set', 'family', 'variable', 'value', 'count']).to_csv(
    OUT / 'dataset_characterization_distributions.csv', index=False)

# ---------------------------------------------------------------- overlap prediction -> training (both keys)
tr = sub[sub['set'] == 'training']                                    # training rows
pr = sub[sub['set'] == 'prediction'].copy()                           # prediction rows
ov = []                                                               # overlap rows per (key, family)
for k in KEYS:
    in_any = pr[k].isin(set(tr[k]))                                   # key seen in training, any family
    tr_fk = set(zip(tr['family'], tr[k]))                             # training (family, key) pairs
    in_fam = pd.Series([x in tr_fk for x in zip(pr['family'], pr[k])], index=pr.index)  # same family too
    # same-family sibling energy; for 'composition' there is exactly one training entry per group,
    # for 'red' several may exist (half families) and the NEAREST in energy is used
    trE_f = tr.groupby(['family', k])[E].apply(lambda x: np.sort(x.values))   # training energies per (family, key)
    trE_a = tr.groupby(k)[E].apply(lambda x: np.sort(x.values))               # training energies per key, any family
    near_f = np.array([np.min(np.abs(trE_f[(f, c)] - e)) / MEV if (f, c) in trE_f.index else np.nan
                       for f, c, e in zip(pr['family'], pr[k], pr[E])])       # |dE| nearest same-family, meV/atom
    near_a = np.array([np.min(np.abs(trE_a[c] - e)) / MEV if c in trE_a.index else np.nan
                       for c, e in zip(pr[k], pr[E])])                        # |dE| nearest any family, meV/atom
    lo_f = np.array([(e - trE_f[(f, c)][0]) / MEV if (f, c) in trE_f.index else np.nan
                     for f, c, e in zip(pr['family'], pr[k], pr[E])])         # signed dE above lowest sibling
    pr[f'absdE_fam_{k}'], pr[f'absdE_any_{k}'], pr[f'dE_low_{k}'] = near_f, near_a, lo_f   # keep for subsetting
    for f in FAMILIES + ['all']:
        msk = np.ones(len(pr), bool) if f == 'all' else (pr['family'] == f).values   # family mask
        n = int(msk.sum())                                            # prediction entries in scope
        nf, na, lo = near_f[msk], near_a[msk], lo_f[msk]              # subset arrays
        has = int(np.isfinite(nf).sum())                              # entries with a same-family sibling
        ov.append({
            'key': k, 'key_meaning': KEYS[k], 'family': f, 'prediction_entries': n,
            'n_in_training_any_family': int(in_any.values[msk].sum()),
            'share_in_training_any_family': in_any.values[msk].mean(),
            'n_in_training_same_family': int(in_fam.values[msk].sum()),
            'share_in_training_same_family': in_fam.values[msk].mean(),
            'n_within_1meV_same_family': int(np.nansum(nf < 1)),
            'share_within_1meV_same_family_of_all': np.nansum(nf < 1) / n,
            'share_within_1meV_same_family_of_with_sibling': np.nansum(nf < 1) / max(has, 1),
            'n_within_1meV_any_family': int(np.nansum(na < 1)),
            'share_within_1meV_any_family_of_all': np.nansum(na < 1) / n,
            'share_within_10meV_same_family_of_all': np.nansum(nf < 10) / n,
            'dE_above_lowest_sibling_median_meV': np.nanmedian(lo),
            'dE_above_lowest_sibling_min_meV': np.nanmin(lo),
            'n_below_lowest_sibling': int(np.nansum(lo < -1e-6)),     # prediction entry lower than training sibling
        })
overlap = pd.DataFrame(ov)                                            # overlap table
overlap.to_csv(OUT / 'dataset_characterization_overlap.csv', index=False, float_format='%.6g')

# near-duplicates inside the prediction set: same family + key, energy within 1 meV/atom of another prediction entry
for k in KEYS:
    ps = pr.sort_values(['family', k, E])                             # sort so neighbours are adjacent
    gp = ps.groupby(['family', k])[E].diff().abs()                    # gap to the next-lower entry
    gn = ps.groupby(['family', k])[E].diff(-1).abs()                  # gap to the next-higher entry
    chk[f'prediction_within_1meV_of_another_prediction[{k}]'] = int(((gp < MEV) | (gn < MEV)).sum())

# ---------------------------------------------------------------- text summary
A = by_fam.set_index(['set', 'family'])                               # convenient lookup
L = ['DATASET CHARACTERISATION (generated by scripts/figures/dataset_characterization.py)',
     f'Metadata: {XLSX}  (DXMag HeuslerDB 2025-09-09)',
     f'Training = UUIDs with data/{{family}}/dos_json/*.json; prediction = UUIDs with data/predictions/{{family}}/*.cif',
     f'Energy column for overlap: "{E}"; hull "{H}"; formation "{EF}"; moment "{M}"; cell size "{NAT}"',
     'VEC: VAL/vec() from code/paper/screen_eval.py (s+d for TM, s+p for main group), per formula unit', '',
     'INTEGRITY']
L += [f'  {k}: {v}' for k, v in chk.items()]
L += [f'  splits fold0 union == dos_json, per family: {split_ok}', '', 'COUNTS (quoted vs observed)']
L += ['  ' + count_check.to_string(index=False).replace('\n', '\n  ')]
L += [f'  totals: training {int(A.loc[("training", "all"), "entries"])}, '
      f'prediction {int(A.loc[("prediction", "all"), "entries"])}', '', 'PER SET (all families)']
for st in ['training', 'prediction', 'all']:
    r = A.loc[(st, 'all')]                                            # pooled row
    L.append(f'  {st}: n={int(r.entries)}, composition strings={int(r.unique_composition_strings)}, '
             f'reduced formulas={int(r.unique_reduced_formulas)}, '
             f'elements={int(r.distinct_elements)}, atoms/cell median {r.atoms_cell_median:g} '
             f'[{r.atoms_cell_min:g}-{r.atoms_cell_max:g}], 1 f.u./cell {r.share_1fu_cell:.4f}, '
             f'moment available {r.mag_available:.4f}, |M|>0.1 {r["share_absM_gt_0.1_of_available"]:.4f}, '
             f'hull<0.05 {r["share_hull_lt_0.05"]:.4f}, hull<0.10 {r["share_hull_lt_0.10"]:.4f}, '
             f'on hull {r["share_on_hull_le_1e-6"]:.4f}, hull median {r.hull_median_eV:.3f} eV, '
             f'formation avail {r.formation_available:.4f}, VEC18 {r.share_vec_18:.4f}, VEC24 {r.share_vec_24:.4f}, '
             f'VEC8 {r.share_vec_8:.4f}, magic {r.share_vec_magic:.4f}')
L += ['', f'OVERLAP (prediction -> training, pooled; energy column "{E}")']
for k in KEYS:
    o = overlap[(overlap['key'] == k) & (overlap['family'] == 'all')].iloc[0]   # pooled row for this key
    L += [f'  key = {k} ({KEYS[k]})',
          f'    in training (any family): {int(o.n_in_training_any_family)}/{int(o.prediction_entries)} = '
          f'{o.share_in_training_any_family:.4%}',
          f'    in training (same family): {int(o.n_in_training_same_family)} = {o.share_in_training_same_family:.4%}',
          f'    |dE|<1 meV/atom to nearest same-family training entry: {int(o.n_within_1meV_same_family)} = '
          f'{o.share_within_1meV_same_family_of_all:.4%} of all prediction; '
          f'{o.share_within_1meV_same_family_of_with_sibling:.4%} of those with a sibling',
          f'    |dE|<1 meV/atom to nearest training entry of same key, any family: '
          f'{int(o.n_within_1meV_any_family)} = {o.share_within_1meV_any_family_of_all:.4%}',
          f'    dE above lowest same-family training entry: median {o.dE_above_lowest_sibling_median_meV:.2f} '
          f'meV/atom, min {o.dE_above_lowest_sibling_min_meV:.3g}, below it {int(o.n_below_lowest_sibling)}']
L += ['', 'Per-family detail: dataset_characterization_by_family.csv, _overlap.csv, _distributions.csv']
(OUT / 'dataset_characterization.txt').write_text('\n'.join(L) + '\n')   # write summary
print('\n'.join(L))                                                   # echo to console
