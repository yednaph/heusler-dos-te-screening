#!/usr/bin/env python3
"""DXMag phonon (dynamical-stability) status of the certified compositions, from existing records only.
DXMag reports phonons (commensurate supercell q-points) for 8,198 training entries; prediction entries have none. For each certified
composition the training-set crystal (the certified entry itself, or the near-duplicate training twin of a certified prediction
entry) is checked. Writes phonon_status.txt."""
import os as _os_htr  # release v1: project paths come from the environment
_HTR = _os_htr.environ.get('HEUSLER_TE_ROOT', _os_htr.path.abspath(_os_htr.path.join(_os_htr.path.dirname(__file__), '../../..')))
_HTE = _os_htr.environ.get('HEUSLER_TE_EXPORT', _os_htr.path.join(_HTR, 'DOS_EXPORT'))
from pathlib import Path
import pandas as pd
H = Path(__file__).resolve().parent
d = pd.read_pickle(_HTR + '/results/paper_analysis/paper_dataset.pkl')
cc = set(d[d.certified].composition)
t = d[(d.data_set == 'training') & d.composition.isin(cc) & (d['phonon computed'] == 'yes')]
st = t.groupby('composition')['phonon stable (commensurate)'].apply(lambda s: set(s))
ok = [c for c in cc if c in st.index and st[c] == {'Yes'}]
allt = d[(d.data_set == 'training') & (d['phonon computed'] == 'yes')]
L = [f'certified compositions: {len(cc)}; with a DXMag phonon result for a training-set crystal: {len(st)}; all such crystals stable: {len(ok)}',
     f'calculated training runs with phonons: {len(allt)}; unstable: {(allt["phonon stable (commensurate)"] == "No").sum()}',
     'by QE-PBE class (stable / unstable):\n' + pd.crosstab(allt.dft_class, allt['phonon stable (commensurate)']).to_string(),
     'note: "no clear phonon instability" was part of a priority filter that ordered selection (Methods), so this is consistent but not independent.']
(H / 'phonon_status.txt').write_text('\n'.join(L) + '\n'); print('\n'.join(L))
assert len(cc) == 80 and len(ok) == 80
