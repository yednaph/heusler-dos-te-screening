"""Fig. 7 (v9) -- certified candidates, their robustness and the known-compound check (data figure, paper-figures skill).

Question: what survives first-principles certification, how firmly, and does it include the established half-Heusler thermoelectrics?
5-second takeaway: the indicator A does not order the certified entries by power factor; the certified count depends on the
transport convention and the bar; the Pareto front and the established half-Heusler thermoelectrics lie in the same region of the
computed |S|-PF/tau map.

Inputs (existing results only): paper_dataset.pkl, sensitivity.xlsx, ranking_fixed_n.xlsx, figures/data/known_hh_check.csv.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from scipy.stats import spearmanr
SKILL = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SKILL))
from figkit import style as S, qa                                     # noqa: E402
from common import dft_runs, ANALYSIS, DATA, OUT, PF_BAR              # noqa: E402

S.apply()
d_all = dft_runs(); d = d_all[d_all.data_set.isin(['training', 'prediction'])]
CERT = d_all[d_all.certified == True]
c = CERT.copy()
pf900 = np.where(c.heavy & c.soc_reopt900_PF.notna(), c.soc_reopt900_PF, c.reopt900_PF)
conv = [('best,\n300–1200 K', c.final_best_PF.values), ('900 K', pf900), ('900 K,\n$|n|\\leq10^{21}$', c.final_cap1e21_any_900_PF.values),
        ('900 K,\n$|n|\\leq10^{20}$', c.final_cap1e20_any_900_PF.values)]
known = pd.read_csv(DATA / 'known_hh_check.csv').query('outcome == "certified"').reset_index(drop=True)
KN = {comp: (i + 1, name) for i, (comp, name) in enumerate(zip(known.composition, known.name))}
assert len(KN) == 10

fig = plt.figure(figsize=(S.W2, 100 * S.MM))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.15], hspace=0.6, wspace=1.35, left=0.065, right=0.985, top=0.95, bottom=0.11)

# a, b  indicator rank vs DFT best power factor among certified entries, per population (no cross-set axis)
rows = []
for k, t in enumerate(['training', 'prediction']):
    ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
    cg = d[(d.data_set == t) & (d.certified == True)].copy()
    rho = spearmanr(cg.SCREEN_A_seebeck_uVK, cg.final_best_PF).correlation; rows.append((t, len(cg), rho))
    cg['r'] = 100 * cg.SCREEN_A_seebeck_uVK.rank(pct=True)
    ax.scatter(cg.r, cg.final_best_PF / 1e12, s=8, marker=S.SET_MARKER[t], lw=0, color=S.SET_COLOR[t])
    ax.set_xlim(0, 102); ax.set_ylim(0.9, 3.0)
    ax.set_xlabel(r'rank of $S^{\mathrm{d}}_{\max}$ among certified (%)', labelpad=1)
    ax.set_ylabel(r'DFT best PF/$\tau$ ($10^{12}$ W m$^{-1}$K$^{-2}$s$^{-1}$)', labelpad=1, fontsize=5.8)
    ax.text(0.03, 0.97, f'{S.SET_LABEL[t]}, n = {len(cg)}\nSpearman ρ = {rho:.2f}', transform=ax.transAxes, fontsize=5.4,
            va='top', color=S.SET_COLOR[t], fontweight='bold')
    S.panel(ax, 'ab'[k], x=-0.24, y=1.02)
pd.DataFrame(rows, columns=['set', 'n_certified', 'spearman_A_bestPF']).to_csv(DATA / 'fig7_spearman.csv', index=False)
assert [round(r_[2], 2) for r_ in rows] == [0.13, 0.29]

# c  v13: where the 80 certified compositions stand in the literature (conventions panel moved to the SI)
LIT = pd.read_csv(Path(__file__).resolve().parents[2] / 'literature/literature_status.csv').set_index('composition')
LC = {'established': S.INK, 'experimentally investigated': S.TEAL, 'computationally investigated': '#A7B4C2', 'underexplored': S.PURPLE}
LS = {'established': 'established', 'experimentally investigated': 'measured', 'computationally investigated': 'computed only',
      'underexplored': 'no study found'}
assert len(LIT) == 80
ax = fig.add_subplot(gs[0, 4:6])
par = ['CoNbGe', 'CoNbSn', 'CoTiAs', 'CoVSn', 'RhNbSn', 'RhVGe', 'CoHfAs', 'CoZrAs']
rows_lit = []
for yy, (lab, comps) in enumerate([('all 80 certified compositions', list(LIT.index)), ('8 on the first Pareto layer', par)][::-1]):
    x0 = 0.0; N_ = len(comps)
    for k_ in LC:
        sub_ = [c_ for c_ in comps if LIT.loc[c_, 'class'] == k_]; n_ = len(sub_)
        if not n_: continue
        npoly = sum(1 for c_ in sub_ if str(LIT.loc[c_, 'polymorph']) not in ('', 'nan'))
        w_ = 100.0 * n_ / N_; wp = 100.0 * npoly / N_
        ax.barh(yy, w_, left=x0, color=LC[k_], height=0.56, lw=0.4, edgecolor='white')
        if npoly:
            ax.barh(yy, wp, left=x0 + w_ - wp, color='none', height=0.56, lw=0.0, hatch='//////', edgecolor='white')
        if w_ >= 6:
            ax.text(x0 + (w_ - wp) / 2 if npoly and w_ - wp > 6 else x0 + w_ / 2, yy, f'{n_}', ha='center', va='center', fontsize=5.6,
                    color='white' if k_ != 'computationally investigated' else S.INK, fontweight='bold')
        rows_lit.append((lab, k_, n_, npoly)); x0 += w_
    ax.text(0, yy + 0.36, lab, fontsize=5.4, color=S.INK2, va='bottom')
pd.DataFrame(rows_lit, columns=['group', 'class', 'n', 'n_orthorhombic_flag']).to_csv(DATA / 'fig5v13_literature.csv', index=False)
ax.set_ylim(-0.5, 1.8); ax.set_yticks([]); ax.set_xlim(0, 100)
ax.set_xlabel('share of compositions (%)', labelpad=1)
for s_ in ('left', 'top', 'right'): ax.spines[s_].set_visible(False)
h = [mp.Patch(color=LC[k], label=LS[k]) for k in LC] + [mp.Patch(facecolor='#A7B4C2', hatch='//////', edgecolor='white', label='orthorhombic phase reported')]
ax.legend(handles=h, loc='upper left', bbox_to_anchor=(0.0, -0.3), ncol=2, fontsize=5.0, handlelength=1.1, columnspacing=0.7, frameon=False)
ax.text(0.0, 1.06, 'prior thermoelectric evidence', transform=ax.transAxes, fontsize=5.6, va='bottom')
S.panel(ax, 'c', x=-0.06, y=1.06)

# d, e  Pareto maps with the established half-Heusler thermoelectrics numbered
prov = CERT.groupby('composition').data_set.apply(lambda x: '+'.join(sorted({'P' if v_ == 'prediction' else 'T' for v_ in x})))
PC = {'P+T': S.INK, 'P': S.SET_COLOR['prediction'], 'T': S.SET_COLOR['training']}
PL = {'P+T': 'certified in both sets', 'P': 'prediction set only', 'T': 'training set only'}
OFFS = {'RhNbSn': (-4, 3, 'right'), 'CoNbGe': (4, 3, 'left'), 'CoNbSn': (6, -1, 'left'), 'RhVGe': (-6, -7, 'right'),
        'CoTiAs': (5, 0, 'left'), 'CoVSn': (4, -8, 'left'), 'CoHfAs': (-4, 3, 'right'), 'CoZrAs': (-5, 2, 'right')}
for k, (sheet, lab, lt) in enumerate([('n1e21_p_900K', 'p-type', 'd'), ('n1e21_n_900K', 'n-type', 'e')]):  # noqa
    ax = fig.add_subplot(gs[1, 3 * k:3 * k + 3])
    rk = pd.read_excel(ANALYSIS / 'ranking_fixed_n.xlsx', sheet_name=sheet)
    rk = rk[~rk.wrong_sign.astype(bool)].copy(); rk['prov'] = rk.composition.map(prov)
    rk['lit'] = rk.composition.map(LIT['class']); rk['poly'] = rk.composition.map(LIT['polymorph']).fillna('').astype(str) != ''
    assert rk.lit.notna().all()
    for k_ in ['computationally investigated', 'experimentally investigated', 'underexplored', 'established']:
        for poly_ in (False, True):
            hh = rk[(rk.lit == k_) & (rk.poly == poly_)]
            if not len(hh): continue
            ax.scatter(hh.S_uVK.abs(), hh.PF / 1e12, s=13 if k_ != 'computationally investigated' else 9, lw=0.35, edgecolor='white',
                       color=LC[k_], marker='D' if poly_ else 'o', zorder=4 if k_ != 'computationally investigated' else 3)
    kn = rk[rk.composition.isin(KN)]
    fr = rk[rk.tier == 1].sort_values('S_uVK', key=abs)
    ax.plot(fr.S_uVK.abs(), fr.PF / 1e12, color=S.GREEN, lw=0.9, zorder=2)
    for _, row in fr.iterrows():
        dx, dy, ha = OFFS.get(row.composition, (-3, 3, 'right'))
        if k == 1 and row.composition == 'CoTiAs': dx, dy, ha = (-2, -9, 'right')
        ax.annotate(row.composition + (' (8)' if row.composition == 'CoNbSn' else ''), (abs(row.S_uVK), row.PF / 1e12), xytext=(dx, dy), textcoords='offset points', fontsize=5.3,
                    ha=ha, color=S.GREEN, fontweight='bold')
    ax.axhline(PF_BAR / 1e12, color=S.INK, lw=0.5, ls='--')
    if k == 0:                                                        # number the established compounds (key in the empty lower right)
        NUM = {'ZrNiSn': (1, (0, -8)), 'TiNiSn': (2, (0, -8)), 'NbFeSb': (3, (-7, 4)), 'TaFeSb': (4, (-7, -5)), 'VFeSb': (5, (0, -8)),
               'ZrCoBi': (6, (0, -8)), 'ZrCoSb': (7, (-9, 0)), 'NbCoSn': (8, (8, -5)), 'HfCoSb': (9, (2, -8)), 'TiCoSb': (10, (7, -6))}
        for _, row in kn.iterrows():
            n_, (dx, dy) = NUM[KN[row.composition][1]]
            if row.composition == 'CoNbSn': continue                      # labelled on the Pareto front as 'CoNbSn (8)'
            ax.annotate(str(n_), (abs(row.S_uVK), row.PF / 1e12), xytext=(dx, dy), textcoords='offset points', fontsize=5.0,
                        ha='center', va='center', color=S.INK2, fontweight='bold')
        order = sorted(NUM.items(), key=lambda kv: kv[1][0])
        col1 = '\n'.join(f'{v[0]}  {nm}' for nm, v in order[:5]); col2 = '\n'.join(f'{v[0]}  {nm}' for nm, v in order[5:])
        ax.text(0.68, 0.05, col1, transform=ax.transAxes, fontsize=5.0, color=S.INK, va='bottom', ha='left', linespacing=1.2)
        ax.text(0.85, 0.05, col2, transform=ax.transAxes, fontsize=5.0, color=S.INK, va='bottom', ha='left', linespacing=1.2)
        assert len(kn) == 10
    ax.set_xlabel(f'computed |S| (μV K$^{{-1}}$), {lab}', labelpad=1)
    ax.set_ylabel(r'computed PF/$\tau$ ($10^{12}$ W m$^{-1}$K$^{-2}$s$^{-1}$)', labelpad=1, fontsize=5.6)
    ax.text(0.02, 0.97, '900 K · |n| ≤ 10$^{21}$ cm$^{-3}$', transform=ax.transAxes, fontsize=5.4, va='top', color=S.INK2)
    ax.set_ylim(0.2, 2.05)
    if k == 0:
        from matplotlib.lines import Line2D
        hl = [Line2D([], [], ls='', marker='o', mfc=LC[k], mec='white', ms=4, label=LS[k]) for k in LC]
        hl.append(Line2D([], [], ls='', marker='D', mfc='#A7B4C2', mec='white', ms=3.8, label='orthorhombic phase reported'))
        hl.append(Line2D([], [], color=S.GREEN, lw=0.9, label='first Pareto layer'))
        ax.legend(handles=hl, loc='upper left', bbox_to_anchor=(0.0, 0.92), fontsize=5.0, handletextpad=0.2, borderaxespad=0.2)
    S.panel(ax, lt, x=-0.135, y=1.02)
qa.check(fig, 'fig7_candidates')
S.save(fig, OUT / 'fig7_candidates', width_mm=183)
print(rows)
