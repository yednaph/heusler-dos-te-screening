"""Re-derive every number that v7 added to the manuscript from stored files (read-only; no training, inference or DFT).
Writes figures/data/v7_claim_checks.csv (claim, value, source) and asserts the values quoted in the text."""
import pickletools, zipfile
import numpy as np
import pandas as pd
from common import REPO, EXPORT, DATA, screen_pools, dft_runs

out = []                                                              # (claim, value, source)
e = pd.read_csv(DATA / 'fig2_entries.csv')                            # per-entry set/family/composition/energy (fig2 cache)
x = pd.read_excel(REPO / 'data/DXMag_HeuslerDB_2025-09-09.xlsx',
                  usecols=['UUID', 'total magnetization (muB/f.u.)', 'number of atoms in the cell', 'phonon computed'])
x['uuid'] = x.UUID.str.lower()
e = e.merge(x.drop(columns='UUID'), on='uuid')

# 1. prediction entries without a same-family training sibling: all tetragonal, all with a cubic training sibling
tr = e[e.set == 'training']; pr = e[e.set == 'prediction']
key = set(zip(tr.family, tr.composition))
no = pr[[(f, c) not in key for f, c in zip(pr.family, pr.composition)]]
cub = sum((f.replace('_tetra', '_cubic'), c) in key for f, c in zip(no.family, no.composition))
assert len(no) == 3875 and no.family.str.endswith('tetra').all() and cub == 3875
out.append(('prediction entries without same-family sibling (all tetragonal, cubic sibling exists)', len(no), 'DXMag xlsx + set membership'))

# 2. moment and cell differences to the training sibling
t = tr.set_index(['family', 'composition'])
p = pr.join(t[['energy (eV/atom)', 'total magnetization (muB/f.u.)', 'number of atoms in the cell']], on=['family', 'composition'],
            rsuffix='_t').dropna(subset=['energy (eV/atom)_t'])
dM = (p['total magnetization (muB/f.u.)'].abs() - p['total magnetization (muB/f.u.)_t'].abs()).abs().dropna()
big = (p['number of atoms in the cell'] > p['number of atoms in the cell_t']).mean()
assert abs((dM > 0.1).mean() - 0.485) < 1e-3 and abs(big - 0.090) < 1e-3
out += [('share of prediction entries with sibling differing in |M| by >0.1 muB/f.u.', round((dM > 0.1).mean(), 4), 'DXMag xlsx'),
        ('share of prediction entries with sibling listed in a larger cell', round(big, 4), 'DXMag xlsx')]

# 3. phonon results only in the training set
ph = e.groupby('set')['phonon computed'].apply(lambda s: int((s.astype(str) == 'True').sum()))
assert ph['training'] == 8198 and ph['prediction'] == 0
out.append(('entries with phonon results (training / prediction)', f"{ph['training']} / {ph['prediction']}", 'DXMag xlsx'))

# 4. semiconductor labels with zero d-DOS gap width
s = screen_pools()
z = {k: int(((g.etype == 'semiconductor') & (g.gap_eV == 0)).sum()) for k, g in s.groupby('pool')}
assert z == {'prediction': 1156, 'training': 541}
out.append(('semiconductor labels with zero gap width (training / prediction)', f"{z['training']} / {z['prediction']}", 'gapfix screen files'))

# 5. prediction-only certified compositions = training winners with failed SOC runs
d = dft_runs(); c = d[d.certified == True]
ponly = set(c[c.data_set == 'prediction'].composition) - set(c[c.data_set == 'training'].composition)
trw = d[(d.data_set == 'training') & d.composition.isin(ponly)]
assert len(ponly) == 5 and (trw.soc_status == 'failed').all()
out.append(('prediction-only certified compositions whose training entry failed SOC', len(ponly), 'paper_dataset.pkl'))


# 6. epoch stored in the surviving best.pt checkpoints (pickle opcodes only; nothing is unpickled)
def stored_epoch(path):
    z_ = zipfile.ZipFile(path); data = z_.read([n for n in z_.namelist() if n.endswith('data.pkl')][0])
    ops = list(pickletools.genops(data))
    for i, (op, arg, _) in enumerate(ops):
        if op.name == 'BINUNICODE' and arg == 'epoch':
            for op2, arg2, _ in ops[i + 1:i + 4]:
                if op2.name.startswith('BININT'):
                    return arg2


ep = {k: stored_epoch(REPO / f'output/full_cubic/expA/{k}/checkpoints/best.pt') for k in ('fold0', 'fold1')}
assert ep == {'fold0': 0, 'fold1': 2}
out.append(('epoch of best.pt, full cubic fold0 / fold1', f"{ep['fold0']} / {ep['fold1']}", 'output/full_cubic/expA/*/checkpoints/best.pt'))

# 7. stored values in a deployed checkpoint
zf = zipfile.ZipFile(EXPORT / 'checkpoints/full_tetra/fold0/swa_model.pt')
sz = [i.file_size for i in zf.infolist() if '/data/' in i.filename and not i.filename.endswith('.pkl')]
vals = (sum(sz) - 8 * sz.count(8)) / 4
assert vals == 42572954
out.append(('float values stored in a deployed checkpoint (+53 int64 counters)', int(vals), 'DOS_EXPORT/checkpoints/full_tetra/fold0'))

# 8. output-head parameters
head = 512 * 1024 + 1024 + 1024 * 2048 + 2048 + 2048 * 4002 + 4002
out.append(('output-head parameters', head, 'train_hpc.py:363-371 (layer sizes)'))

pd.DataFrame(out, columns=['claim', 'value', 'source']).to_csv(DATA / 'v7_claim_checks.csv', index=False)
print(pd.DataFrame(out, columns=['claim', 'value', 'source']).to_string(index=False))
