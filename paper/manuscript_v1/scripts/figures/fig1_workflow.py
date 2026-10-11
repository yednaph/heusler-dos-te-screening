"""Fig. 1 (v12) -- (a) screening and assessment strategy; (b, c) the spin-resolved d-DOS the screen reads, for two real
held-out training-set entries with their DFT reference and the matched model prediction (best-validation checkpoint of the fold
in which the entry was a test entry).

Question: what population is screened, how does it move from d-DOS to first-principles assessment, and what does a d-DOS look like?
5-second takeaway: indicators decide what is calculated, calculations decide what is certified; the d-DOS distinguishes a gap in both
spin channels (18-electron half-Heusler ZrNiSn) from a gap in one channel (half-metal Co2MnSi), and the prediction reproduces the
shape while leaving weight inside the gap.
Examples: chosen as established compounds whose held-out Pearson r lies near the population median (0.911); they illustrate the
representation, not population accuracy (Fig. 5). Counts are read from results files and asserted.
"""
import os as _os_htr  # release v1: project paths come from the environment
_HTR = _os_htr.environ.get('HEUSLER_TE_ROOT', _os_htr.path.abspath(_os_htr.path.join(_os_htr.path.dirname(__file__), '../../../..')))
_HTE = _os_htr.environ.get('HEUSLER_TE_EXPORT', _os_htr.path.join(_HTR, 'DOS_EXPORT'))
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.patches as mp
SKILL = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SKILL))
from figkit import style as S, glyphs as G, qa          # noqa: E402
from common import dft_runs, per_sample, EXPORT, DATA, OUT, E_GRID   # noqa: E402

S.apply()
import importlib.util as _ilu                                         # the screen's own label code (code/te_rescreen.py)
_sp = _ilu.spec_from_file_location('ter', _HTR + '/code/te_rescreen.py')
TER = _ilu.module_from_spec(_sp); _sp.loader.exec_module(TER)
d = dft_runs(); ps = per_sample()
N = {}
for t in ['training', 'prediction']:
    g = d[d.data_set == t]
    N[t] = dict(runs=len(g), gapped=int((g.dft_class == 'gapped').sum()), metal=int((g.dft_class == 'metal').sum()),
                narrow=int((g.dft_class == 'narrow').sum()), hm=int((g.dft_class == 'half_metal').sum()), cert=int(g.certified.sum()))
pd.DataFrame(N).to_csv(DATA / 'fig1_counts.csv')
assert N['training']['runs'] == 1562 and N['prediction']['runs'] == 2021 and N['training']['cert'] == 76 and N['prediction']['cert'] == 79
W, H, HB = 183, 82, 50                                                # diagram height, DOS-row height (mm)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(W * S.MM, (H + HB) * S.MM))
ax = fig.add_axes([0, HB / (H + HB), 1, H / (H + HB)]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
T_C, P_C = S.SET_COLOR['training'], S.SET_COLOR['prediction']
yb, yt = 6, 60

# ---------------------------------------------------------------- role brackets (the key message)
def bracket(x0, x1, y, label, sub, col):
    ax.plot([x0, x0, x1, x1], [y - 1.6, y, y, y - 1.6], color=col, lw=1.1, solid_capstyle='butt')
    G.text(ax, (x0 + x1) / 2, y + 2.3, label, fs=6.6, w='bold', c=col)
    G.text(ax, (x0 + x1) / 2, y + 5.6, sub, fs=5.4, c=S.INK2)


bracket(36, 119, 67.5, 'SCREENING — prioritises', 'all 106,235 entries · decides what is computed', S.BLUE)
bracket(123.5, 181, 67.5, 'ASSESSMENT — certifies and ranks', '3,625 calculations · decides what is certified', S.TEAL)

# ---------------------------------------------------------------- data column
G.stat_tile(ax, 2, 44, 30, 16, '106,235', 'DXMag Heusler entries', S.NAVY, fs_count=9.5)
G.text(ax, 17, 41.0, 'divided by DOS availability', fs=5.2, c=S.INK2)
G.stat_tile(ax, 2, 23.5, 30, 14.5, '51,257', 'training set · DFT d-DOS', T_C)
G.stat_tile(ax, 2, 6, 30, 14.5, '54,978', 'prediction set · no DOS', P_C)

# ---------------------------------------------------------------- 1 DOS model
G.card(ax, 36, 27, 47, 33, S.NAVY, heading='d-DOS model', number=1)
G.text(ax, 38.2, 52.4, 'ALIGNN · spin-resolved d-DOS', fs=5.4, ha='left')
G.text(ax, 38.2, 48.9, '2 × 2,001 points, −10…10 eV', fs=5.2, ha='left', c=S.INK2)
G.node(ax, 49.5, 44.2, 23, 3.4, 'pretrained: all spectra', S.NAVY, fs=5.0)
G.chevron(ax, 49.5, 41.8, S.ARROW, size=0.7, direction='down')
G.node(ax, 49.5, 39.2, 23, 3.4, 'fine-tuned: 6 families', S.TEAL, fs=5.0)
G.text(ax, 49.5, 35.4, '× 5 folds = 30 models;\nprediction = 5-model mean', fs=5.0, c=S.INK2)
G.text(ax, 72.5, 42.0, 'held-out\nspectra:\nFig. 2', fs=5.0, c=S.INK2)
G.text(ax, 59.5, 30.0, 'spectral shape reproduced; gaps mostly missed', fs=5.0, c=S.NAVY, w='bold')
G.arrow(ax, 32, 31.0, 36, 31.0, color=T_C, lw=1.0); G.text(ax, 34.0, 29.0, 'train', fs=5.0, c=T_C)
G.card(ax, 36, 6, 47, 16, P_C, heading=None, fill='#E3EDF6')
G.text(ax, 59.5, 16.6, 'predicted d-DOS · 54,978 spectra', fs=6.2, w='bold', c=P_C)
G.text(ax, 59.5, 11.0, 'ensemble mean and spread', fs=5.4)
G.arrow(ax, 32, 13.0, 36, 13.0, color=P_C, lw=1.0)
G.arrow(ax, 59.5, 27, 59.5, 22, color=P_C, lw=1.0); G.text(ax, 60.6, 24.4, 'predict', fs=5.0, c=P_C, ha='left')
ax.plot([32, 34.0, 34.0, 95.5, 95.5], [36.0, 36.0, 62.2, 62.2, 60.9], color=T_C, lw=1.0, zorder=3, solid_joinstyle='round')
G.chevron(ax, 95.5, 61.0, T_C, size=1.0, direction='down')
G.text(ax, 64.0, 63.8, 'training set: screened on its own DFT d-DOS', fs=5.2, c=T_C, w='bold')

# ---------------------------------------------------------------- 2 screen
G.card(ax, 87.2, yb, 32, yt - yb, S.BLUE, heading='DOS indicators', number=2)
G.text(ax, 89.4, 52.3, 'same definitions, each\nset scored separately', fs=5.0, ha='left', va='top', c=S.INK2)
G.node(ax, 103.2, 41.6, 29, 3.8, 'Cutler–Mott moments, 900 K', S.BLUE, fs=5.1)
x = 89.4
for lab in ['$S^{\\mathrm{d}}_{\\max}$ Seebeck', '$P^{\\mathrm{d}}_{\\max}$ power f.']:
    x = G.chip(ax, x, 32.4, lab, S.BLUE, w=13.9, h=4.0, fs=5.1) + 0.6
x = 89.4
for lab in ['$\\xi$ sharpness', '$L_0^{*}$ weight']:
    x = G.chip(ax, x, 27.6, lab, S.BLUE, w=13.9, h=4.0, fs=5.1) + 0.6
G.text(ax, 103.2, 21.8, 'd-DOS gap label +\npriority filters (stability,\nlabel, ensemble spread)', fs=5.0)
G.text(ax, 103.2, 14.3, 'output: 9 score-ordered\nselection rounds', fs=5.0, c=S.PURPLE)
G.text(ax, 103.2, 8.8, 'semiconductor-labelled:\n1,550 training · 2,394 prediction', fs=5.0, c=S.BLUE, w='bold')
G.arrow(ax, 83, 13.0, 87.2, 13.0, color=P_C, lw=1.0)

# ---------------------------------------------------------------- 3 DFT assessment
G.card(ax, 123.5, yb, 29, yt - yb, S.TEAL, heading='DFT assessment', number=3)
G.arrow(ax, 119.2, 32.5, 123.5, 32.5, color=S.ARROW, lw=1.0)
G.pills(ax, 125.6, 47.4, ['SCF', 'NSCF'], S.TEAL, h=3.8, widths=[10.5, 11.5], fs=5.1)
G.pills(ax, 125.6, 42.2, ['BoltzTraP2 (CRTA)'], S.TEAL, h=3.8, widths=[24.6], fs=5.1)
G.pills(ax, 125.6, 37.0, ['+ SOC, Z ≥ 72'], S.TEAL, h=3.8, widths=[24.6], fs=5.1)
G.text(ax, 138.0, 32.4, '3,625 runs', fs=7.4, w='bold', c=S.TEAL)
G.text(ax, 138.0, 25.6, f'{S.SET_LABEL["training"]}: {N["training"]["runs"]:,}', fs=5.4, c=S.SET_COLOR['training'], w='bold')
G.text(ax, 138.0, 21.4, f'{S.SET_LABEL["prediction"]}: {N["prediction"]["runs"]:,}', fs=5.4, c=S.SET_COLOR['prediction'], w='bold')
G.text(ax, 138.0, 17.2, '42 not attributable to a set', fs=5.0, c=S.INK2)
G.text(ax, 138.0, 10.6, 'QE-PBE class, Seebeck,\nPF/τ at 300–1200 K', fs=5.0, c=S.INK2)

# ---------------------------------------------------------------- 4 certification criteria -> certified (v11: final protocol, no revision history)
G.card(ax, 156.4, 25, 25, 35, S.GREEN, heading='Certification', number=4)
for i, lab in enumerate(['(i) $|S(E_F)|>60$ μV K$^{-1}$', '(ii) best PF/τ > PF$_0$', '(iii) hull < 0.10 eV atom$^{-1}$',
                         '(iv) QE-PBE gap ≥ 0.136 eV', '(v) SOC re-check, Z ≥ 72']):
    G.text(ax, 158.4, 50.6 - i * 3.9, lab, fs=5.0, c=S.INK, ha='left')
G.text(ax, 168.9, 28.6, 'robustness: SOC and\nstructural variants', fs=5.0, c=S.GREEN, w='bold')
G.arrow(ax, 152.5, 42, 156.4, 42, color=S.ARROW, lw=1.0)
G.stat_tile(ax, 156.4, 6, 25, 16.5, '', '', S.GREEN)
G.text(ax, 168.9, 18.4, '80', fs=10.5, w='bold', c='white')
G.text(ax, 168.9, 13.4, 'certified compositions', fs=5.3, c='white')
G.text(ax, 168.9, 9.4, '74 both sets · 5 P · 1 T', fs=5.0, c='white')
G.arrow(ax, 168.9, 25, 168.9, 22.5, color=S.GREEN, lw=1.0)

G.text(ax, 59.5, 2.4, 'held-out metrics are upper bounds (pretraining saw all 51,257 spectra)', fs=5.0, c=S.PURPLE)
G.text(ax, 151.0, 2.4, 'calculated sample follows the scores (recall unmeasured)', fs=5.0, c=S.PURPLE)

G.text(ax, 2.5, 79.5, 'a', fs=9, w='bold', ha='left')

# ---------------------------------------------------------------- b  the materials space: entries per family, by state and DOS availability
dx = pd.read_csv(DATA / 'fig2v9_entries.csv'); dx['type'] = dx.family.str.split('_').str[0]
dx['gs'] = False; dx.loc[dx.groupby(['type', 'composition'])['energy (eV/atom)'].idxmin(), 'gs'] = True
dx['fammin'] = False; dx.loc[dx.groupby(['family', 'composition'])['energy (eV/atom)'].idxmin(), 'fammin'] = True
assert len(dx) == 106235 and dx.gs.sum() == 27864 and (dx[dx.set == 'training'].gs).sum() == 27860
FAMS = ['full_cubic', 'full_tetra', 'half_cubic', 'half_tetra', 'inverse_cubic', 'inverse_tetra']
axb = fig.add_axes([0.075, 0.075, 0.43, (HB - 14) / (H + HB)])
T_C, P_C = S.SET_COLOR['training'], S.SET_COLOR['prediction']
T_L, P_L = '#E9B877', '#9DBBDA'                                   # lighter tints: metastable (training) / family minima (prediction)
rows = []
for i, f in enumerate(FAMS):
    y = len(FAMS) - 1 - i; t = dx[(dx.family == f) & (dx.set == 'training')]; p = dx[(dx.family == f) & (dx.set == 'prediction')]
    tg, tm = int(t.gs.sum()), int((~t.gs).sum()); pf, pm = int(p.fammin.sum()), int((~p.fammin).sum())
    axb.barh(y, -tg, color=T_C, height=0.7, lw=0); axb.barh(y, -tm, left=-tg, color=T_L, height=0.7, lw=0)
    axb.barh(y, pf, color=P_L, height=0.7, lw=0); axb.barh(y, pm, left=pf, color=P_C, height=0.7, lw=0)
    axb.text(-(tg + tm) - 300, y, f'{tg + tm:,}', ha='right', va='center', fontsize=5.0, color=T_C)
    axb.text(pf + pm + 300, y, f'{pf + pm:,}', ha='left', va='center', fontsize=5.0, color=P_C)
    rows.append((f, tg, tm, pf, pm))
axb.axvline(0, color=S.INK, lw=0.6)
axb.set_yticks(range(len(FAMS))); axb.set_yticklabels([f.replace('_', ' ') for f in FAMS[::-1]], fontsize=5.6)
axb.set_xlim(-17500, 19500); axb.set_xticks([-15000, -10000, -5000, 0, 5000, 10000, 15000])
axb.set_xticklabels(['15k', '10k', '5k', '0', '5k', '10k', '15k'], fontsize=5.4); axb.set_xlabel('DXMag entries', labelpad=1)
axb.text(0.23, 1.04, 'training set · DFT d-DOS (51,257)', transform=axb.transAxes, ha='center', va='bottom', fontsize=5.6, color=T_C, fontweight='bold')
axb.text(0.74, 1.04, 'prediction set · no DOS (54,978)', transform=axb.transAxes, ha='center', va='bottom', fontsize=5.6, color=P_C, fontweight='bold')
h = [mp.Patch(color=T_C, label='ground state of composition and prototype'), mp.Patch(color=T_L, label='cubic minimum above a tetragonal ground state'),
     mp.Patch(color=P_L, label='tetragonal minimum without a DFT spectrum'), mp.Patch(color=P_C, label='other metastable configuration')]
axb.legend(handles=h, loc='upper left', bbox_to_anchor=(1.005, 1.0), fontsize=5.0, frameon=False, handlelength=1.0, borderaxespad=0.0, labelspacing=0.35)
axb.text(-0.11, 1.08, 'b', transform=axb.transAxes, fontsize=9, fontweight='bold')
pd.DataFrame(rows, columns=['family', 'train_ground', 'train_metastable', 'pred_family_min', 'pred_metastable']).to_csv(DATA / 'fig1v13_population.csv', index=False)
# ---------------------------------------------------------------- c  energy of each prediction entry above its same-composition training entry
tr_ = dx[dx.set == 'training'].set_index(['family', 'composition'])['energy (eV/atom)']
pr_ = dx[dx.set == 'prediction'].copy()
pr_['dE'] = 1e3 * (pr_['energy (eV/atom)'] - [tr_.get((f, c_), np.nan) for f, c_ in zip(pr_.family, pr_.composition)])
sib = pr_.dropna(subset=['dE'])
assert len(sib) == 51103 and abs((sib.dE < 1).sum() / len(pr_) - 0.1508) < 5e-4
axc = fig.add_axes([0.835, 0.075, 0.15, (HB - 14) / (H + HB)])
axc.hist(sib.dE.clip(lower=1e-2), bins=np.logspace(-2, 3.3, 40), color=P_C, lw=0)
axc.set_xscale('log'); axc.set_xlabel('energy above same-composition\ntraining entry (meV atom$^{-1}$)', labelpad=1, fontsize=5.4)
axc.set_ylabel('prediction entries', labelpad=1, fontsize=5.6)
axc.text(0.03, 0.97, f'median {sib.dE.median():.0f} meV atom$^{{-1}}$\n15% within 1 meV', transform=axc.transAxes, va='top', fontsize=5.0, color=S.INK2)
axc.text(-0.42, 1.08, 'c', transform=axc.transAxes, fontsize=9, fontweight='bold')
qa.check(fig, 'fig1_workflow')
S.save(fig, OUT / 'fig1_workflow', width_mm=183)
