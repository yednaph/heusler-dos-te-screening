"""Fig. 2 (v7) -- the two sets and the d-DOS target (data figure, paper-figures skill).

Question: what data does the model learn from and what must it predict?
5-second takeaway: the training set holds the lowest-energy entry of each family/composition with a DFT d-DOS;
the prediction set holds the remaining, higher-energy entries of the same compositions; the target is a 2 x 2,001-point
spin-resolved d-DOS whose Fermi-level region is weighted by the loss and read by the screen.

Inputs (read-only): DXMag metadata xlsx, data/<family>/dos_json (training membership), data/predictions/<family>/*.cif
(prediction membership), DOS_EXPORT held-out dumps (one real DFT target). Counts asserted against the paper.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
SKILL = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SKILL))
from figkit import style as S, qa                                     # noqa: E402
from common import REPO, EXPORT, DATA, OUT, FAMILIES, E_GRID, per_sample   # noqa: E402

S.apply()
CACHE = DATA / 'fig2_entries.csv'                                     # per-entry cache (xlsx read is slow)

# ---------------------------------------------------------------- entries: set, family, composition, energy
if not CACHE.exists():
    dx = pd.read_excel(REPO / 'data/DXMag_HeuslerDB_2025-09-09.xlsx',
                       usecols=['UUID', 'type', 'space group symbol', 'composition', 'energy (eV/atom)'])
    dx['uuid'] = dx['UUID'].astype(str).str.strip().str.lower()                     # join key
    lat = dx['space group symbol'].str.strip().map({'Fm-3m': 'cubic', 'F-43m': 'cubic', 'I4/mmm': 'tetra', 'I-4m2': 'tetra'})
    dx['family'] = dx['type'].str.strip() + '_' + lat                               # family from metadata
    mem = {}                                                                          # uuid -> set
    for f in FAMILIES:
        for p in (REPO / f'data/{f}/dos_json').glob('*.json'):
            mem[p.stem.rsplit('_', 1)[1].lower()] = 'training'                       # has DFT DOS
        for p in (REPO / f'data/predictions/{f}').glob('*.cif'):
            mem[p.stem.rsplit('_', 1)[1].lower()] = 'prediction'                     # no DOS -> predicted
    dx['set'] = dx['uuid'].map(mem)
    dx[['uuid', 'set', 'family', 'composition', 'energy (eV/atom)']].to_csv(CACHE, index=False)
dx = pd.read_csv(CACHE)
cnt = dx.groupby(['set', 'family']).size().unstack(0)                 # family x set counts
assert cnt['training'].sum() == 51257 and cnt['prediction'].sum() == 54978
cnt.to_csv(DATA / 'fig2_counts.csv')

# energy of each prediction entry above the (unique) training entry of the same family and composition string
tr = dx[dx.set == 'training'].set_index(['family', 'composition'])['energy (eV/atom)']
assert not tr.index.duplicated().any()                                # one training entry per family/composition
pr = dx[dx.set == 'prediction'].copy()
pr['E_tr'] = [tr.get((f, c), np.nan) for f, c in zip(pr.family, pr.composition)]
pr['dE_meV'] = 1e3 * (pr['energy (eV/atom)'] - pr['E_tr'])           # meV/atom above the training sibling
sib = pr.dropna(subset=['dE_meV'])
share_1meV = (sib.dE_meV < 1).sum() / len(pr)                         # denominator: all prediction entries
assert len(sib) == 51103 and abs(share_1meV - 0.1508) < 5e-4 and (sib.dE_meV >= 0).all()
pd.Series({'with_sibling': len(sib), 'share_lt_1meV_all': share_1meV, 'median_meV': sib.dE_meV.median(),
           'min_meV': sib.dE_meV.min()}).to_csv(DATA / 'fig2_dE_summary.csv')

# ---------------------------------------------------------------- one real DFT target: a gapped held-out half-Heusler
ps = per_sample()
cand = ps[(ps.family == 'half_cubic') & (ps.true_gapped == True)]
pick = cand.iloc[(cand.pearson_r - cand.pearson_r.median()).abs().argsort().iloc[0]]
dd = EXPORT / 'runtime_assets/heldout_dumps' / pick.family / pick.fold
j = [l.strip() for l in open(dd / 'test_ids.txt')].index(pick.material_id)
T = np.load(dd / 'test_targets.npy', mmap_mode='r')[j]
up, dn = np.asarray(T[:2001]), np.asarray(T[2001:])
pd.Series({'material_id': pick.material_id, 'formula': pick.formula, 'fold': pick.fold}).to_csv(DATA / 'fig2_example.csv')

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(183 * S.MM, 62 * S.MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.9, 1.45], left=0.095, right=0.985, bottom=0.17, top=0.88, wspace=0.42)
T_C, P_C = S.SET_COLOR['training'], S.SET_COLOR['prediction']

# (a) entries per family, mirrored: training left, prediction right
ax = fig.add_subplot(gs[0])
y = np.arange(len(FAMILIES))[::-1]
ax.barh(y, -cnt.loc[FAMILIES, 'training'], color=T_C, height=0.62)
ax.barh(y, cnt.loc[FAMILIES, 'prediction'], color=P_C, height=0.62)
for yi, f in zip(y, FAMILIES):
    ax.text(-cnt.loc[f, 'training'] - 300, yi, f"{cnt.loc[f, 'training']:,}", ha='right', va='center', fontsize=5.4, color=T_C)
    ax.text(cnt.loc[f, 'prediction'] + 300, yi, f"{cnt.loc[f, 'prediction']:,}", ha='left', va='center', fontsize=5.4, color=P_C)
ax.set_yticks(y); ax.set_yticklabels([f.replace('_', ' ') for f in FAMILIES])
ax.axvline(0, color=S.INK2, lw=0.5)
ax.set_xlim(-16500, 21500); ax.set_xticks([-10000, 0, 10000]); ax.set_xticklabels(['10k', '0', '10k'])
ax.set_xlabel('entries')
ax.text(-9500, 6.05, 'training set\nDFT d-DOS', ha='center', va='center', fontsize=5.6, color=T_C, fontweight='bold')
ax.text(11500, 6.05, 'prediction set\nno DOS', ha='center', va='center', fontsize=5.6, color=P_C, fontweight='bold')
ax.set_ylim(-0.6, 6.7)
S.panel(ax, 'a', x=-0.40)

# (b) how far above its training sibling each prediction entry lies
ax = fig.add_subplot(gs[1])
bins = np.logspace(-2, 3.3, 44)
ax.hist(sib.dE_meV.clip(lower=1e-2), bins=bins, color=P_C, alpha=0.9)
ax.set_xscale('log'); ax.set_ylim(0, 4700); ax.plot([1, 1], [0, 3700], color=S.INK2, lw=0.6, ls=':')
ax.set_xlabel('$E-E_\\mathrm{training\\ sibling}$ (meV atom$^{-1}$)'); ax.set_ylabel('prediction entries')
ax.text(0.03, 0.985, f'{share_1meV:.1%} of the prediction set within 1 meV atom$^{{-1}}$;\nmedian {sib.dE_meV.median():.0f} meV atom$^{{-1}}$ above the sibling',
        transform=ax.transAxes, ha='left', va='top', fontsize=5.0, color=S.INK)
S.panel(ax, 'b', x=-0.3)

# (c) the target: spin-resolved d-DOS on the 0.01 eV grid, with the windows the loss and the screen use
ax = fig.add_subplot(gs[2])
ax.axvspan(-3, 3, color=S.TINT[S.NAVY], lw=0, zorder=0)
ax.axvspan(-2, 2, color='#C5D9EE', lw=0, zorder=0)
ax.fill_between(E_GRID, 0, up, color='#9AA7B5', lw=0)
ax.fill_between(E_GRID, 0, -dn, color='#9AA7B5', lw=0)
ax.plot(E_GRID, up, color=S.INK2, lw=0.45); ax.plot(E_GRID, -dn, color=S.INK2, lw=0.45)
beta = 1 / (8.617333e-5 * 700)                                        # 1/(k_B T) at 700 K, eV^-1
w = (beta / 4) / np.cosh(beta * E_GRID / 2) ** 2                       # -df/dE at 700 K (loss weight shape)
top = max(up.max(), dn.max())
ax.plot(E_GRID, -1.25 * top + 0.35 * top * w / w.max(), color=S.NAVY, lw=0.8)
ax.axhline(0, color=S.INK2, lw=0.4)
ax.set_xlim(-6, 6); ax.set_ylim(-1.32 * top, 1.18 * top)
ax.set_xlabel('$E-E_F$ (eV)'); ax.set_ylabel('d-DOS (states eV$^{-1}$ atom$^{-1}$)')
ax.text(-5.8, 1.05 * top, 'spin up', fontsize=5.4, color=S.INK2, va='top')
ax.text(-5.8, -0.95 * top, 'spin down', fontsize=5.4, color=S.INK2, va='bottom')
ax.text(0, 1.12 * top, '±2 eV: Fermi-window loss and $r_{\\pm2\\,\\mathrm{eV}}$', fontsize=5.0, color=S.NAVY, ha='center', va='bottom')
ax.text(2.55, 0.62 * top, '±3 eV:\nslope term\n(spin up)', fontsize=5.0, color=S.NAVY, ha='left', va='center')
ax.text(5.9, -1.12 * top, '$-\\partial f/\\partial E$ (700 K)\nloss weight', fontsize=5.0, color=S.NAVY, ha='right', va='center')
ax.text(5.8, 0.25 * top, f'{pick.formula}\n(held-out, DFT)', fontsize=5.0, color=S.INK2, ha='right', va='bottom')
S.panel(ax, 'c', x=-0.16)

qa.check(fig, 'fig2_data_target')
S.save(fig, OUT / 'fig2_data_target', width_mm=183)
print('example', pick.formula, pick.family, pick.fold, 'share<1meV', round(share_1meV, 4), 'median', round(sib.dE_meV.median(), 1))
