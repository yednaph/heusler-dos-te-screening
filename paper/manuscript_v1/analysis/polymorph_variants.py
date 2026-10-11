#!/usr/bin/env python3
"""Structural variants of the same composition: energetic ordering vs calculated electronic class and transport.

EXISTING results only (no DFT, no transport, no model):
  results/paper_analysis/paper_dataset.pkl   one row per DFT run (class, gap, best PF/tau, peak undoped |S|, criteria)
  figures/data/fig2v9_entries.csv            DXMag total energy per atom for all 106,235 entries (cached from the xlsx)
Definitions
  reference crystal : the lowest-DXMag-energy calculated entry of a composition (any set, any family)
  variant           : another calculated entry of the same composition at least 5 meV/atom above the reference
                      (entries within 5 meV/atom are near-duplicates, results/paper_analysis/twins.txt)
The 42 runs of UUIDs present in both sets ('ambiguous') are excluded. Writes polymorph_variants.csv and
polymorph_summary.txt.
"""
import os as _os_htr  # release v1: project paths come from the environment
_HTR = _os_htr.environ.get('HEUSLER_TE_ROOT', _os_htr.path.abspath(_os_htr.path.join(_os_htr.path.dirname(__file__), '../../..')))
_HTE = _os_htr.environ.get('HEUSLER_TE_EXPORT', _os_htr.path.join(_HTR, 'DOS_EXPORT'))
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = Path(_HTR)
TWIN_MEV = 5.0                                                      # near-duplicate energy tolerance (meV/atom)

d = pd.read_pickle(REPO / 'results/paper_analysis/paper_dataset.pkl').copy()
dx = pd.read_csv(HERE.parent / 'figures/data/fig2v9_entries.csv')     # uuid, set, family, composition, energy
d['uuid'] = d.UUID.astype(str).str.lower().str.strip()
d['E'] = d.uuid.map(dx.set_index('uuid')['energy (eV/atom)'])        # DXMag energy (eV/atom)
d = d[d.data_set != 'ambiguous'].copy()
assert d.E.notna().all() and len(d) == 3583
d['qualified'] = d.winner.fillna(False).astype(bool)

ref = d.loc[d.groupby('composition').E.idxmin()].set_index('composition')      # lowest-energy calculated crystal
d['dE'] = 1e3 * (d.E - d.composition.map(ref.E))                              # meV/atom above it
v = d[d.dE >= TWIN_MEV].copy()
for c in ['dft_class', 'gap', 'qualified', 'certified', 'best_PF', 'max_S_intr', 'family', 'data_set', 'space group symbol']:
    v['ref_' + c] = v.composition.map(ref[c])
v['pf_ratio'] = v.best_PF / v.ref_best_PF                           # variant / reference, best PF/tau 300-1200 K
v['S_ratio'] = v.max_S_intr / v.ref_max_S_intr                      # peak undoped |S|
v['same_class'] = v.dft_class == v.ref_dft_class
cols = ['composition', 'UUID', 'data_set', 'family', 'space group symbol', 'dE', 'hull', 'dft_class', 'gap', 'qualified',
        'certified', 'best_PF', 'max_S_intr', 'soc_status', 'ref_family', 'ref_space group symbol', 'ref_data_set', 'ref_dft_class',
        'ref_gap', 'ref_qualified', 'ref_certified', 'ref_best_PF', 'ref_max_S_intr', 'pf_ratio', 'S_ratio', 'same_class']
v[cols].sort_values(['composition', 'dE']).to_csv(HERE / 'polymorph_variants.csv', index=False)

S = [f'variants (>= {TWIN_MEV:g} meV/atom above the composition\'s lowest calculated crystal): {len(v)} entries, '
     f'{v.composition.nunique()} compositions; by set {v.data_set.value_counts().to_dict()}; '
     f'same family {(v.family == v.ref_family).sum()}, other family {(v.family != v.ref_family).sum()}',
     f'dE median {v.dE.median():.1f} meV/atom, IQR [{v.dE.quantile(.25):.1f}, {v.dE.quantile(.75):.1f}]',
     'class reference (rows) vs variant (cols):\n' + pd.crosstab(v.ref_dft_class, v.dft_class, margins=True).to_string(),
     f'same class: {v.same_class.mean():.1%}',
     'qualified reference (rows) vs variant (cols):\n' + pd.crosstab(v.ref_qualified, v.qualified, margins=True).to_string()]
bins = pd.cut(v.dE, [TWIN_MEV, 25, 50, 100, 200, 5000])
S.append('by energy above reference:\n' + v.groupby(bins, observed=True).agg(
    n=('pf_ratio', 'size'), median_pf_ratio=('pf_ratio', 'median'), median_S_ratio=('S_ratio', 'median'),
    same_class=('same_class', 'mean')).round(3).to_string())
cc = set(d[d.certified].composition)
vc = v[v.composition.isin(cc)]
S.append(f'certified compositions with a calculated variant: {vc.composition.nunique()} {sorted(set(vc.composition))}')
S.append(vc[['composition', 'ref_space group symbol', 'space group symbol', 'data_set', 'dE', 'ref_gap', 'gap', 'dft_class',
             'qualified', 'hull', 'soc_status', 'pf_ratio', 'S_ratio']].round(3).to_string(index=False))
hv = v[(v.dft_class == 'gapped') & v.ref_dft_class.eq('gapped')]
S.append(f'gapped reference and gapped variant: {len(hv)} {sorted(set(hv.composition))}')
up = v[v.qualified & ~v.ref_qualified]
S.append(f'variant qualified but reference not: {len(up)} {up[["composition", "dft_class", "dE"]].round(1).to_dict("records")}')
open(HERE / 'polymorph_summary.txt', 'w').write('\n'.join(S) + '\n')
print('\n'.join(S))
assert len(v) == 824 and vc.composition.nunique() == 3
