"""v4 data figures (Figs. 2, 4, 5, 6) in the approved Fig. 1 visual language (paper-figures skill, figkit tokens).

Existing result files only; every plotted count is written to figures/data/ and key values are asserted.
Sets are never pooled: one panel row (or column) per set, set colour + marker + the words 'training set'/'prediction set'.
Sizes are print sizes (183 mm double column); fonts >= 4.5 pt.
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from scipy.stats import spearmanr
SKILL = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SKILL))
from figkit import style as S, qa                                     # noqa: E402
from common import (dft_runs, per_sample, screen_pools, EXPORT, DATA, OUT, ANALYSIS, PF_BAR, FAMILIES, E_GRID)  # noqa

S.apply()
SETS = ['training', 'prediction']
d_all = dft_runs()
d = d_all[d_all.data_set.isin(SETS)].copy()
CERT = d_all[d_all.certified == True]
LABELS = ['metal', 'half-metal', '<0.10', '0.10–0.40', '0.40–1.00', '1.00–1.40', '>1.40']
LAB_TXT = ['M', 'HM', '<0.1', '0.1–0.4', '0.4–1.0', '1.0–1.4', '>1.4']


def ddos_label(e, t):
    if e == 'metal': return 'metal'
    if e == 'half_metal': return 'half-metal'
    return {'<0.10': '<0.10', '0.10-0.40': '0.10–0.40', '0.40-1.00': '0.40–1.00', '1.00-1.40': '1.00–1.40', '>1.40': '>1.40'}.get(t, 'n/a')


d['lab'] = [ddos_label(e, t) for e, t in zip(d.SCREEN_etype, d.SCREEN_gap_tier)]


def set_tag(ax, t, x=0.0, y=1.02, fs=6.2):
    """Set identity tag (Fig. 1 language): filled rounded label in the set colour."""
    ax.text(x, y, f'{S.SET_LABEL[t]} · {S.SET_SOURCE[t]}', transform=ax.transAxes, fontsize=fs, fontweight='bold',
            color='white', ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.22,rounding_size=0.25', fc=S.SET_COLOR[t], ec='none'))


def soft_spines(ax):
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(S.INK2)


# ======================================================================= Fig. 2  DOS model
ps = per_sample()
CM = {('True', 'True'): 'gapped · flagged', ('True', 'False'): 'gapped · missed',
      ('False', 'True'): 'not gapped · flagged', ('False', 'False'): 'not gapped · not flagged'}
ps['cm'] = [CM[(str(a), str(b))] for a, b in zip(ps.true_gapped, ps.pred_gapped)]
STY = {'gapped · missed': (S.PURPLE, 1.6, '-'), 'not gapped · not flagged': (S.NAVY, 1.6, '-'),
       'gapped · flagged': (S.CLASS_COLOR['gapped'], 0.8, '-'), 'not gapped · flagged': (S.SUPERSEDED, 0.8, (0, (3, 1.5)))}


def gap_at_ef(v):
    return (v[..., 997:1004].min(-1) < 0.15) & (v[..., 2998:3005].min(-1) < 0.15)


rows = []
for fam in FAMILIES:
    for k in range(5):
        dd = EXPORT / 'runtime_assets/heldout_dumps' / fam / f'fold{k}'
        ids = [l.strip() for l in open(dd / 'test_ids.txt')]
        gp = gap_at_ef(np.asarray(np.load(dd / 'test_predictions.npy', mmap_mode='r')))
        gt = gap_at_ef(np.asarray(np.load(dd / 'test_targets.npy', mmap_mode='r')))
        rows += [(fam, k, ids[j], j, bool(gt[j])) for j in np.where(gp != gt)[0]]
ex = pd.DataFrame(rows, columns=['family', 'fold', 'material_id', 'j', 'dft_gap']).merge(ps[['material_id', 'pearson_r', 'formula']], on='material_id')
fig = plt.figure(figsize=(S.W2, 76 * S.MM))
gs = fig.add_gridspec(2, 3, width_ratios=[0.62, 1, 1], height_ratios=[1, 1.05], hspace=0.72, wspace=0.42,
                      left=0.065, right=0.985, top=0.93, bottom=0.17)
for k, (flag, ttl, lt) in enumerate([(True, 'real gap filled by the prediction', 'b'), (False, 'spurious gap opened in a metal', 'c')]):
    ax = fig.add_subplot(gs[0, 1 + k])
    g = ex[ex.dft_gap == flag]
    pick = g.iloc[(g.pearson_r - g.pearson_r.median()).abs().argsort().iloc[0]]
    dd = EXPORT / 'runtime_assets/heldout_dumps' / pick.family / f'fold{pick.fold}'
    Pv = np.load(dd / 'test_predictions.npy', mmap_mode='r')[pick.j]; Tv = np.load(dd / 'test_targets.npy', mmap_mode='r')[pick.j]
    m = (E_GRID >= -3) & (E_GRID <= 3)
    ax.axvspan(-1, 1, color='none', hatch='////', ec='#E2E6EB', lw=0)
    ax.axvspan(-0.03, 0.03, color='#D9CFC1', lw=0)
    ax.fill_between(E_GRID[m], 0, Tv[:2001][m], color='#C9D2DC', lw=0, label='DFT (held out)')
    ax.fill_between(E_GRID[m], 0, -Tv[2001:][m], color='#C9D2DC', lw=0)
    ax.plot(E_GRID[m], Pv[:2001][m], color=S.SET_COLOR['prediction'], lw=0.8, label='ML prediction')
    ax.plot(E_GRID[m], -Pv[2001:][m], color=S.SET_COLOR['prediction'], lw=0.8)
    ax.axhline(0, color=S.INK2, lw=0.35)
    f = re.sub(r'(\d+)', r'$_{\1}$', pick.formula)
    ax.set_title(ttl, loc='left', fontsize=6.4, color=S.NAVY, pad=2)
    ax.text(0.99, 0.95, f'{f} · r = {pick.pearson_r:.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=5.6)
    if k == 0:
        ax.legend([plt.Rectangle((0, 0), 1, 1, fc='#C9D2DC'), Line2D([], [], color=S.SET_COLOR['prediction'], lw=0.9),
                   plt.Rectangle((0, 0), 1, 1, fc='#D9CFC1'), plt.Rectangle((0, 0), 1, 1, fc='white', hatch='////', ec='#B8C2CC')],
                  ['DFT (held out)', 'ML prediction', 'label window ±0.03 eV', 'gap-flag window ±1 eV'], loc='upper left',
                  bbox_to_anchor=(-1.02, 1.06), fontsize=5.0, handlelength=1.2, borderaxespad=0.1, frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)
    ax.text(0.01, 0.95, '↑', transform=ax.transAxes, fontsize=6, color=S.INK2, va='top'); ax.text(0.01, 0.04, '↓', transform=ax.transAxes, fontsize=6, color=S.INK2)
    ax.set_xlabel(r'$E-E_F$ (eV)', labelpad=1)
    if k == 0:
        ax.set_ylabel('d-DOS (eV$^{-1}$ atom$^{-1}$)', labelpad=1)
    S.panel(ax, lt, x=-0.21 if k == 0 else -0.07, y=1.02)
# confusion matrix (left, spanning both rows)
ax = fig.add_subplot(gs[:, 0])
cnt = np.array([[(ps.cm == 'gapped · flagged').sum(), (ps.cm == 'gapped · missed').sum()],
                [(ps.cm == 'not gapped · flagged').sum(), (ps.cm == 'not gapped · not flagged').sum()]])
assert cnt.sum() == 51257 and cnt[0, 1] == 4352
rowp = cnt / cnt.sum(1, keepdims=True)
cols = [[S.TINT[S.GREEN], S.TINT[S.PURPLE]], [S.TINT[S.SUPERSEDED], S.TINT[S.NAVY]]]
edge = [[S.CLASS_COLOR['gapped'], S.PURPLE], [S.SUPERSEDED, S.NAVY]]
for i in range(2):
    for j in range(2):
        ax.add_patch(FancyBboxPatch((j + 0.04, 1 - i + 0.04), 0.92, 0.92, boxstyle='round,pad=0,rounding_size=0.06',
                                    fc=cols[i][j], ec=edge[i][j], lw=0.9))
        ax.text(j + 0.5, 1 - i + 0.6, f'{cnt[i, j]:,}', ha='center', va='center', fontsize=7.2, fontweight='bold', color=edge[i][j])
        ax.text(j + 0.5, 1 - i + 0.33, f'{100 * rowp[i, j]:.1f}% of row', ha='center', va='center', fontsize=5.2, color=S.INK2)
ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect('equal')
ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(['flagged', 'not flagged'], fontsize=5.8)
ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(['gapped', 'not\ngapped'], fontsize=5.8)
ax.set_xlabel('ML prediction', labelpad=2); ax.set_ylabel('DFT d-DOS (held out)', labelpad=2)
for s_ in ax.spines.values(): s_.set_visible(False)
ax.tick_params(length=0)
ax.text(1.0, -0.62, 'balanced accuracy 0.57\nMCC 0.25 · recall 0.15', ha='center', va='top', fontsize=5.8, fontweight='bold', color=S.NAVY)
ax.text(1.0, 2.12, 'held-out gap flag (±1 eV)', ha='center', va='bottom', fontsize=5.4, color=S.INK2)
S.panel(ax, 'a', x=-0.34, y=1.08)
med = ps.groupby('cm')[['pearson_r', 'fermi_pearson_2eV']].median()
med.to_csv(DATA / 'v4_fig2_medians.csv')
for k, (col, lab, lt) in enumerate([('pearson_r', 'Pearson r, full spectrum', 'd'), ('fermi_pearson_2eV', r'Pearson r, $|E-E_F|\leq2$ eV', 'e')]):
    ax = fig.add_subplot(gs[1, 1 + k])
    for key in ['not gapped · flagged', 'gapped · flagged', 'not gapped · not flagged', 'gapped · missed']:
        x = np.sort(ps.loc[ps.cm == key, col].dropna().values)
        c_, lw_, ls_ = STY[key]
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c_, lw=lw_, ls=ls_, label=f'{key} ({len(x):,})')
    ax.text(0.03, 0.93, f"median r\nmissed gap {med.loc['gapped · missed', col]:.3f}\ncorrect non-gap {med.loc['not gapped · not flagged', col]:.3f}",
            transform=ax.transAxes, fontsize=5.4, va='top')
    lo_ = np.floor(10 * min(ps[col].min(), 0)) / 10                 # v11: full range, nothing hidden below r = 0.2
    ax.set_xlim(lo_, 1.0); ax.set_ylim(0, 1.02); ax.set_xlabel(lab, labelpad=1)
    if k == 0: ax.set_ylabel('cumulative fraction', labelpad=1)
    S.panel(ax, lt, x=-0.15 if k == 0 else -0.07, y=1.02)
    if k == 1:
        ax.legend(loc='upper center', bbox_to_anchor=(-0.12, -0.3), ncol=4, fontsize=5.2, handlelength=1.6, columnspacing=1.0,
                  title='held-out gap flag: DFT · prediction (n)', title_fontsize=5.2)
qa.check(fig, 'fig2', ignore=('↑', '↓'))
S.save(fig, OUT / 'fig2_dos_model', width_mm=183)

# ======================================================================= Fig. 4  what DFT found (labels, gaps, winner rate)
fig = plt.figure(figsize=(S.W2, 86 * S.MM))
gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 0.9], hspace=0.62, wspace=0.3, left=0.06, right=0.985, top=0.87, bottom=0.1)
STACK = ['gapped', 'narrow', 'half_metal', 'metal']
GB = [0.136, 0.3, 0.6, 0.9, 1.2, 1.5]
rate_rows = []
for r, t in enumerate(SETS):
    g = d[d.data_set == t]
    lt = 'abc' if r == 0 else 'def'
    ax = fig.add_subplot(gs[r, 0])
    ct = pd.crosstab(g.lab, g.dft_class).reindex(index=LABELS, columns=STACK).fillna(0)
    n = ct.sum(1); frac = ct.div(n.replace(0, np.nan), axis=0).fillna(0) * 100
    bottom = np.zeros(len(LABELS))
    for c in STACK:
        ax.bar(range(len(LABELS)), frac[c].values, bottom=bottom, color=S.CLASS_COLOR[c], width=0.74, edgecolor='white', lw=0.5,
               label=S.CLASS_LABEL[c])
        bottom += frac[c].values
    for i, (ni, gfr) in enumerate(zip(n, frac['gapped'])):
        ax.text(i, 101.5, f'{int(ni)}', ha='center', va='bottom', fontsize=5.0, color=S.INK2)
        if gfr >= 9: ax.text(i, gfr / 2, f'{gfr:.0f}%', ha='center', va='center', fontsize=5.4, color='white', fontweight='bold')
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LAB_TXT, fontsize=5.4)
    ax.set_ylim(0, 110); ax.set_yticks([0, 50, 100]); ax.set_ylabel('QE-PBE class (% runs)', labelpad=1)
    ax.set_xlabel('screen label: M, HM, or semiconductor gap tier (eV)', fontsize=5.6, labelpad=1)
    set_tag(ax, t, y=1.07); S.panel(ax, lt[0], x=-0.12, y=1.07)
    ax = fig.add_subplot(gs[r, 1])
    s_ = g[g.SCREEN_etype == 'semiconductor']; gg, g0 = s_[s_.gap > 0], s_[s_.gap == 0]
    rng = np.random.default_rng(1)
    ax.scatter(g0.SCREEN_gap_eV.clip(upper=2.0), -0.14 + rng.uniform(-0.06, 0.06, len(g0)), s=1.4, lw=0, alpha=0.5, color=S.CLASS_COLOR['metal'])
    ax.scatter(gg.SCREEN_gap_eV.clip(upper=2.0), gg.gap, s=2.6, lw=0, alpha=0.8,
               c=[S.CLASS_COLOR['gapped'] if v >= 0.136 else S.CLASS_COLOR['narrow'] for v in gg.gap])
    ax.plot([0, 1.5], [0, 1.5], color=S.INK2, lw=0.5, ls='--'); ax.axhline(-0.04, color=S.INK2, lw=0.35)
    ax.text(1.98, 0.02, f'no QE gap: {len(g0):,} of {len(s_):,}', ha='right', va='bottom', fontsize=5.2, color=S.INK2,
            bbox=dict(fc='white', ec='none', pad=0.4))
    ax.set_xlim(0, 2.02); ax.set_ylim(-0.22, 1.5); ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.set_xlabel('screen d-DOS gap (eV; >2 shown at 2)', labelpad=1); ax.set_ylabel('QE-PBE gap (eV)', labelpad=1)
    S.panel(ax, lt[1], x=-0.18, y=1.07)
    ax = fig.add_subplot(gs[r, 2])
    gp = g[g.gap >= 0.136]; b = pd.cut(gp.gap, GB); rr = gp.groupby(b).winner.agg(['size', 'mean'])
    xs = np.arange(len(rr))
    ax.axvspan(1.5, 3.5, color=S.TINT[S.GREEN], lw=0)
    ax.bar(xs, 100 * rr['mean'], width=0.66, color=S.SET_COLOR[t])
    for xi, (sz, mm_) in zip(xs, rr.values):
        ax.text(xi, 100 * mm_ + 2, f'{100 * mm_:.0f}%', ha='center', va='bottom', fontsize=5.4, fontweight='bold', color=S.SET_COLOR[t])
        ax.text(xi, 2.5, f'{int(sz)}', ha='center', va='bottom', fontsize=5.0, color='white')
        rate_rows.append((t, str(b.cat.categories[xi]), int(sz), mm_))
    ax.set_xticks(xs); ax.set_xticklabels(['.14–.3', '.3–.6', '.6–.9', '.9–1.2', '1.2–1.5'], fontsize=5.2)
    ax.set_ylim(0, 78); ax.set_ylabel('PF- and stability-qualified (%)', labelpad=1); ax.set_xlabel('QE-PBE gap of gapped runs (eV)', labelpad=1)
    S.panel(ax, lt[2], x=-0.32, y=1.07)
pd.DataFrame(rate_rows, columns=['set', 'gap_bin_eV', 'n', 'winner_rate']).to_csv(DATA / 'v4_winner_rate_by_gap.csv', index=False)
h = [plt.Rectangle((0, 0), 1, 1, fc=S.CLASS_COLOR[c]) for c in STACK]
fig.legend(h, [S.CLASS_LABEL[c] for c in STACK], loc='upper center', bbox_to_anchor=(0.53, 1.0), ncol=4, fontsize=5.6,
           title=None, handlelength=1.0, columnspacing=1.2)
qa.check(fig, 'fig4')
S.save(fig, OUT / 'fig4_dft_verdict', width_mm=183)

# ======================================================================= Fig. 5  descriptors: decisions 1, 2, 7
se = pd.concat([pd.read_excel(ANALYSIS / 'screen_eval.xlsx', sheet_name=s_) for s_ in ('screen_eval', 'prime_window')])
bt = pd.read_excel(ANALYSIS / 'screen_eval.xlsx', sheet_name='bootstrap_dAUC')
SC = ['A_seebeck', 'B_pf', 'C_sharp', 'D_dosmass', 'D34_july', 'VEC_rule', 'random']
SCL = {'A_seebeck': '$S^{\\mathrm{d}}_{\\max}$ Seebeck', 'B_pf': '$P^{\\mathrm{d}}_{\\max}$ power factor', 'C_sharp': '$\\xi$ sharpness', 'D_dosmass': '$L_0^{*}$ weight',
       'D34_july': 'initial composite', 'VEC_rule': 'electron count 18/24', 'random': 'random'}
SUB = [('all computed', 'all computed', 'o'), ('prime window 0.10-1.40 eV', 'gap window', 's'), ('VEC 18/24 only', '18/24 e$^-$', '^'),
       ('half_cubic only', 'half cubic', 'D')]
# top-100 composition, prediction set (si/B15_d34_postmortem.txt)
TOP = pd.DataFrame({'initial composite': [5, 2, 7, 3, 0], '$P^{\\mathrm{d}}_{\\max}$': [7, 3, 19, 4, 1], '$S^{\\mathrm{d}}_{\\max}$': [83, 41, 98, 69, 35]},
                   index=['semicond.\nlabel', 'hull\n< 0.1', 'computed', 'DFT-\ngapped', 'certified'])
fig = plt.figure(figsize=(S.W2, 92 * S.MM))
gs = fig.add_gridspec(2, 4, width_ratios=[0.9, 1, 1, 1.05], hspace=0.62, wspace=0.52, left=0.075, right=0.985, top=0.88, bottom=0.1)
ax = fig.add_subplot(gs[:, 0])
yy = np.arange(len(TOP.columns))[::-1]
for k, (rowname, vals) in enumerate(TOP.T.iterrows()):
    pass
cats = list(TOP.index)
ypos = np.arange(len(cats))[::-1] * 1.0
for j, (sc_, col) in enumerate([('$S^{\\mathrm{d}}_{\\max}$', S.BLUE), ('$P^{\\mathrm{d}}_{\\max}$', S.SUPERSEDED), ('initial composite', '#C9D2DC')]):
    ax.barh(ypos + (1 - j) * 0.27, TOP[sc_].values, height=0.25, color=col, label=sc_)
    for y_, v in zip(ypos + (1 - j) * 0.27, TOP[sc_].values):
        ax.text(v + 1.5, y_, f'{v}', va='center', fontsize=5.0, color=S.INK)
ax.set_yticks(ypos); ax.set_yticklabels(cats, fontsize=5.6); ax.set_xlim(0, 112); ax.set_xlabel('of the top 100 entries', labelpad=1)
ax.legend(loc='lower right', fontsize=5.4, title='ranked by', title_fontsize=5.2)
ax.text(0.0, 1.01, 'prediction set:\ntop 100 by each score', transform=ax.transAxes, fontsize=5.8, color=S.NAVY, fontweight='bold', va='bottom')
S.panel(ax, 'a', x=-0.4, y=1.09)
sp_rows = []
for r, t in enumerate(SETS):
    g = d[d.data_set == t]; cg = g[g.certified == True]
    lts = 'bcd' if r == 0 else 'efg'
    ax = fig.add_subplot(gs[r, 1])
    for c in ['metal', 'half_metal', 'narrow', 'gapped']:
        hh = g[g.dft_class == c]
        ax.scatter(hh.SCREEN_A_seebeck_uVK, hh.max_S_intr, s=1.6 if c == 'metal' else 2.4, lw=0, alpha=0.55 if c == 'metal' else 0.85,
                   color=S.CLASS_COLOR[c])
    ax.scatter(cg.SCREEN_A_seebeck_uVK, cg.max_S_intr, s=6, facecolor='none', edgecolor=S.INK, lw=0.4)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.05, 3000); ax.axhline(60, color=S.INK2, lw=0.45, ls=':')
    ax.set_xlabel(r'indicator $S^{\mathrm{d}}_{\max}$ ($\mu$V K$^{-1}$)', labelpad=1); ax.set_ylabel(r'DFT max $|S|$ ($\mu$V K$^{-1}$)', labelpad=1)
    set_tag(ax, t, y=1.08); S.panel(ax, lts[0], x=-0.32, y=1.08)
    ax = fig.add_subplot(gs[r, 2])
    rho = spearmanr(cg.SCREEN_A_seebeck_uVK, cg.final_best_PF).correlation
    sp_rows.append((t, len(cg), rho))
    ax.scatter(cg.SCREEN_A_seebeck_uVK, cg.final_best_PF / 1e12, s=7, marker=S.SET_MARKER[t], color=S.SET_COLOR[t], lw=0)
    ax.set_xlabel(r'indicator $S^{\mathrm{d}}_{\max}$ ($\mu$V K$^{-1}$)', labelpad=1); ax.set_ylabel(r'DFT best PF/$\tau$ ($10^{12}$ SI)', labelpad=1)
    ax.text(0.02, 1.02, f'certified entries (n = {len(cg)}); Spearman ρ = {rho:.2f}', transform=ax.transAxes, fontsize=5.4, va='bottom')
    S.panel(ax, lts[1], x=-0.28, y=1.08)
    ax = fig.add_subplot(gs[r, 3])
    for sub_, sl, mk in SUB:
        gsub = se[(se.subset == sub_) & (se.data_set == t)].set_index('score').reindex(SC)
        allc = sub_ == 'all computed'
        ax.scatter(gsub.ROC_AUC, np.arange(len(SC)), marker=mk, s=12 if allc else 8, facecolor=S.SET_COLOR[t] if allc else 'none',
                   edgecolor=S.SET_COLOR[t] if allc else S.INK2, lw=0.7, zorder=3, label=sl)
    ax.axvline(0.5, color=S.INK2, lw=0.45, ls=':')
    ax.set_yticks(range(len(SC))); ax.set_yticklabels([SCL[s_] for s_ in SC], fontsize=5.4); ax.invert_yaxis()
    ax.set_xlim(0.4, 1.0); ax.set_xlabel('ROC-AUC for certified entries', labelpad=1)

    def dtxt(c_):
        b_ = bt[(bt.comparison == c_) & (bt.data_set == t) & (bt.subset == 'all computed')].iloc[0]
        return f'{b_.delta_AUC:+.3f} [{b_.CI95_low:+.3f}, {b_.CI95_high:+.3f}]'.replace('-', '−')
    ax.text(1.0, 1.02, f'ΔAUC S−P {dtxt("A_seebeck - B_pf")}\nΔAUC S−count {dtxt("A_seebeck - VEC_rule")}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=5.0)
    S.panel(ax, lts[2], x=-0.55, y=1.08)
pd.DataFrame(sp_rows, columns=['set', 'n_certified', 'spearman_A_bestPF']).to_csv(DATA / 'v4_spearman_A_PF.csv', index=False)
hc = [Line2D([], [], ls='', marker='o', ms=3, color=S.CLASS_COLOR[c]) for c in S.CLASS_ORDER] + \
     [Line2D([], [], ls='', marker='o', ms=3.6, mfc='none', mec=S.INK, mew=0.5)]
fig.legend(hc, [S.CLASS_LABEL[c] for c in S.CLASS_ORDER] + ['certified'], loc='upper left', bbox_to_anchor=(0.27, 1.0), ncol=5,
           fontsize=5.4, handletextpad=0.1, columnspacing=0.8)
hs = [Line2D([], [], ls='', marker=mk, ms=3.4, mfc=S.INK if sl == 'all computed' else 'none', mec=S.INK2, mew=0.7) for _, sl, mk in SUB]
fig.legend(hs, [sl for _, sl, _ in SUB], loc='upper right', bbox_to_anchor=(0.995, 1.0), ncol=4, fontsize=5.4, handletextpad=0.1,
           columnspacing=0.7)
qa.check(fig, 'fig5')
S.save(fig, OUT / 'fig5_descriptors', width_mm=183)

# ======================================================================= Fig. 6  what survives
c = CERT.copy()
pf900 = np.where(c.heavy & c.soc_reopt900_PF.notna(), c.soc_reopt900_PF, c.reopt900_PF)
conv = [('best\n300–1200 K', c.final_best_PF.values), ('900 K', pf900), ('900 K\n$|n|\\leq10^{21}$', c.final_cap1e21_any_900_PF.values),
        ('900 K\n$|n|\\leq10^{20}$', c.final_cap1e20_any_900_PF.values)]
fig = plt.figure(figsize=(S.W2, 98 * S.MM))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.1], hspace=0.55, wspace=1.3, left=0.06, right=0.985, top=0.95, bottom=0.11)
# a SOC re-check
ax = fig.add_subplot(gs[0, 0:2])
hv = d[(d.winner == True) & (d.dft_class == 'gapped') & (d.heavy == True) & (d.soc_status == 'done')]
soc_rows = []
for t in SETS:
    hh = hv[hv.data_set == t]; ok = hh.soc_best_PF > PF_BAR
    col = np.where(ok, S.SET_COLOR[t], '#AAB4BF')
    if t == 'training':
        ax.scatter(hh.best_PF / 1e12, hh.soc_best_PF / 1e12, s=10, marker='^', lw=0, color=col, zorder=3,
                   label=f'training set: {int(ok.sum())} of {len(hh)} stay')
    else:
        ax.scatter(hh.best_PF / 1e12, hh.soc_best_PF / 1e12, s=14, marker='o', facecolor='none', edgecolor=col, lw=0.7, zorder=4,
                   label=f'prediction set: {int(ok.sum())} of {len(hh)} stay')
    soc_rows.append((t, len(hh), int(ok.sum())))
pd.DataFrame(soc_rows, columns=['set', 'heavy_gapped_winners_soc_done', 'above_bar_with_soc']).to_csv(DATA / 'v4_soc_recheck.csv', index=False)
ax.plot([0.3, 5], [0.3, 5], color=S.INK2, lw=0.5, ls='--')
ax.axhline(PF_BAR / 1e12, color=S.INK, lw=0.5, ls=':'); ax.axvline(PF_BAR / 1e12, color=S.INK, lw=0.5, ls=':')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.9, 4); ax.set_ylim(0.4, 4)
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
for a_ in (ax.xaxis, ax.yaxis):
    a_.set_major_locator(FixedLocator([0.5, 1, 2, 4])); a_.set_major_formatter(FixedFormatter(['0.5', '1', '2', '4'])); a_.set_minor_locator(NullLocator())
ax.set_xlabel(r'best PF/$\tau$ scalar ($10^{12}$ SI)', labelpad=1); ax.set_ylabel(r'with SOC ($10^{12}$ SI)', labelpad=1)
ax.legend(loc='lower right', fontsize=5.0, handletextpad=0.2, title='heavy, gapped, qualified runs', title_fontsize=5.0)
S.panel(ax, 'a', x=-0.22, y=1.02)
# b conventions
ax = fig.add_subplot(gs[0, 2:4]); rng = np.random.default_rng(0)
for i, (lab, v) in enumerate(conv):
    v = np.asarray(v, float)
    for t, off in [('training', -0.18), ('prediction', 0.18)]:
        mm_ = (c.data_set == t).values
        ax.scatter(i + off + rng.uniform(-0.11, 0.11, mm_.sum()), v[mm_], s=1.8, lw=0, alpha=0.85, marker=S.SET_MARKER[t],
                   color=np.where(v[mm_] > PF_BAR, S.SET_COLOR[t], S.MUTED))
    ax.text(i, 7.5e12, f'{int((v > PF_BAR).sum())}', ha='center', fontsize=6.6, fontweight='bold', color=S.GREEN)
ax.text(-0.45, 1.25e13, 'certified entries above the threshold (of 155)', fontsize=5.2, color=S.INK2)
ax.axhline(PF_BAR, color=S.INK, lw=0.6, ls='--'); ax.set_yscale('log'); ax.set_ylim(5e10, 1.8e13)
ax.set_xticks(range(4)); ax.set_xticklabels([l for l, _ in conv], fontsize=5.3); ax.set_ylabel(r'PF/$\tau$ (W m$^{-1}$ K$^{-2}$ s$^{-1}$)', labelpad=1)
S.panel(ax, 'b', x=-0.24, y=1.02)
# c bar sensitivity
ax = fig.add_subplot(gs[0, 4:6])
sv = pd.read_excel(ANALYSIS / 'sensitivity.xlsx').set_index('case')
mult = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
key = {0.8: 'bar x0.8', 0.9: 'bar x0.9', 1.0: 'declared (baseline)', 1.1: 'bar x1.1', 1.2: 'bar x1.2', 1.5: 'bar x1.5'}
ax.plot(mult, [sv.loc[key[m_], 'certified_compounds'] for m_ in mult], marker='o', ms=2.8, color=S.GREEN, lw=1.0)
corr = [i for i in sv.index if 'corrected' in i][0]
ax.scatter([1.154e12 / PF_BAR], [sv.loc[corr, 'certified_compounds']], marker='D', s=14, color=S.ORANGE, zorder=3)
ax.annotate('threshold from\ncorrected runs: 77', (1.154e12 / PF_BAR, sv.loc[corr, 'certified_compounds']), xytext=(1.16, 96),
            fontsize=5.2, color=S.ORANGE, arrowprops=dict(arrowstyle='-', lw=0.4, color=S.ORANGE))
ax.set_xlabel('threshold / declared threshold', labelpad=1); ax.set_ylabel('certified compositions', labelpad=1); ax.set_ylim(0, 118)
S.panel(ax, 'c', x=-0.26, y=1.02)
# d,e Pareto maps
prov = CERT.groupby('composition').data_set.apply(lambda x: '+'.join(sorted({'P' if v_ == 'prediction' else 'T' for v_ in x})))
PC = {'P+T': S.INK, 'P': S.SET_COLOR['prediction'], 'T': S.SET_COLOR['training']}
PL = {'P+T': 'certified in both sets', 'P': 'prediction set only', 'T': 'training set only'}
OFFS = {'RhNbSn': (-4, 3, 'right'), 'CoNbGe': (4, 3, 'left'), 'CoNbSn': (-6, 3, 'right'), 'RhVGe': (-6, -7, 'right'),
        'CoTiAs': (5, 0, 'left'), 'CoVSn': (4, -8, 'left'), 'CoHfAs': (-4, 3, 'right'), 'CoZrAs': (-5, 2, 'right')}
for k, (sheet, lab, lt) in enumerate([('n1e21_p_900K', 'p-type', 'd'), ('n1e21_n_900K', 'n-type', 'e')]):
    ax = fig.add_subplot(gs[1, 3 * k:3 * k + 3])
    rk = pd.read_excel(ANALYSIS / 'ranking_fixed_n.xlsx', sheet_name=sheet)
    rk = rk[~rk.wrong_sign.astype(bool)].copy(); rk['prov'] = rk.composition.map(prov)
    for p_ in ['P+T', 'P', 'T']:
        hh = rk[rk.prov == p_]
        ax.scatter(hh.S_uVK.abs(), hh.PF / 1e12, s=11 if p_ != 'P+T' else 8, lw=0.3, edgecolor='white', color=PC[p_], zorder=3,
                   label=f'{PL[p_]} ({len(hh)})')
    # v8: established half-Heusler thermoelectric compositions (known_hh_check.py) as open squares
    kn = rk[rk.composition.isin(set(pd.read_csv(DATA / 'known_hh_check.csv').query('outcome == "certified"').composition))]
    ax.scatter(kn.S_uVK.abs(), kn.PF / 1e12, s=24, marker='s', facecolor='none', edgecolor=S.INK2, lw=0.6, zorder=4,
               label=f'established half-Heusler thermoelectrics ({len(kn)})')
    fr = rk[rk.tier == 1].sort_values('S_uVK', key=abs)
    ax.plot(fr.S_uVK.abs(), fr.PF / 1e12, color=S.GREEN, lw=0.9, zorder=2)
    for _, row in fr.iterrows():
        dx, dy, ha = OFFS.get(row.composition, (-3, 3, 'right'))
        if k == 1 and row.composition == 'CoTiAs': dx, dy, ha = (-2, -9, 'right')
        ax.annotate(row.composition, (abs(row.S_uVK), row.PF / 1e12), xytext=(dx, dy), textcoords='offset points', fontsize=5.4,
                    ha=ha, color=S.GREEN, fontweight='bold')
    ax.axhline(PF_BAR / 1e12, color=S.INK, lw=0.5, ls='--')
    ax.set_xlabel(f'|S| (μV K$^{{-1}}$), {lab}', labelpad=1); ax.set_ylabel(r'PF/$\tau$ ($10^{12}$ W m$^{-1}$ K$^{-2}$ s$^{-1}$)', labelpad=1)
    ax.text(0.02, 0.97, '900 K · |n| ≤ 10$^{21}$ cm$^{-3}$', transform=ax.transAxes, fontsize=5.4, va='top', color=S.INK2)
    ax.set_ylim(0.2, 2.05)
    if k == 0: ax.legend(loc='lower right', fontsize=5.2, handletextpad=0.2)
    S.panel(ax, lt, x=-0.11, y=1.02)
qa.check(fig, 'fig6')
S.save(fig, OUT / 'fig6_what_survives', width_mm=183)
print('v4 data figures done')
