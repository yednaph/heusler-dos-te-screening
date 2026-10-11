"""Audit of the DOS-archive normalisation (read-only; no model or DFT run).

For magnetic training entries (|M| >= 1 muB/f.u. in DXMag), integrate the archive's spin-resolved TOTAL DOS up to E_F to
get the moment of the stored cell, divide by the number of formula units of the stored structure, and compare with the
DXMag total magnetisation per formula unit. A ratio of 1 means the DOS is per stored cell; a ratio of 2 means the
archive DOS carries a factor of two relative to the stored structure (and so does the per-atom d-DOS target).
Output: figures/data/dos_normalisation_check.csv (per entry) and printed per-family medians.
"""
import json, glob, random, re
import numpy as np
import pandas as pd
from common import REPO, DATA, FAMILIES

x = pd.read_excel(REPO / 'data/DXMag_HeuslerDB_2025-09-09.xlsx',
                  usecols=['UUID', 'total magnetization (muB/f.u.)', 'composition'])   # DXMag moment per f.u.
x['uuid'] = x.UUID.str.lower(); x = x.set_index('uuid')
random.seed(3)                                                        # fixed sample for reproducibility
rows = []
for f in FAMILIES:
    k = 0
    for p in random.sample(glob.glob(str(REPO / f'data/{f}/dos_json/*.json')), 600):
        u = p.rsplit('_', 1)[1][:-5].lower(); M = x.loc[u, 'total magnetization (muB/f.u.)']
        if not np.isfinite(M) or abs(M) < 1.0:                       # magnetic entries only
            continue
        d = json.load(open(p)); e = np.array(d['energies']) - d['efermi']
        up, dn = np.array(d['densities']['1']), np.array(d['densities']['-1'])   # total DOS per spin
        m = e <= 0
        mom = abs(np.trapz((up - dn)[m], e[m]))                         # moment of the stored cell (muB)
        nat = len(d['structure']['sites'])                             # atoms in the stored structure
        apf = sum(int(n or 1) for _, n in re.findall(r'([A-Z][a-z]?)(\d*)', x.loc[u, 'composition']))  # atoms per f.u.
        rows.append(dict(family=f, uuid=u, nat_json=nat, ratio=mom / (nat / apf) / abs(M)))
        k += 1
        if k >= 25:                                                     # 25 per family
            break
df = pd.DataFrame(rows); df.to_csv(DATA / 'dos_normalisation_check.csv', index=False)
print(df.groupby('family').ratio.describe()[['count', '25%', '50%', '75%']].round(3)); print('all', df.ratio.median().round(3))
print('overall quartiles', df.ratio.quantile([.25, .5, .75]).round(3).tolist())
