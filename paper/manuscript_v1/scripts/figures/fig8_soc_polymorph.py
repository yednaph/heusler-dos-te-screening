"""Fig. 8 (v11) -- robustness of the first-principles assessment to spin-orbit coupling and to structural variation.

Question: do the electronic class and calculated transport of a candidate survive (i) adding spin-orbit coupling (SOC) for
heavy-element compounds and (ii) moving to a higher-energy structural variant of the same composition?
5-second takeaway: SOC narrows gaps and lowers PF/tau almost everywhere; it moves a third of the heavy-element candidates below the
power-factor threshold, almost all Au-containing ones, without closing their gaps. Higher-energy variants keep the electronic class of
their composition and comparable transport up to ~100 meV/atom; the three certified compositions with a distinct variant show
variants that are gapped but no better than the lowest-energy crystal.

Inputs (existing results only): analysis/soc_paired.csv (soc_paired.py), analysis/polymorph_variants.csv (polymorph_variants.py).
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
SKILL = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SKILL))
from figkit import style as S, qa                                     # noqa: E402
from common import DATA, OUT, PF_BAR, REPO                                  # noqa: E402

S.apply()
AN = Path(__file__).resolve().parents[2] / 'analysis'
p = pd.read_csv(AN / 'soc_paired.csv')
p = p[p.data_set.isin(['training', 'prediction'])]                    # 6 'ambiguous' metal pairs not drawn
v = pd.read_csv(AN / 'polymorph_variants.csv')
EG = 0.136                                                            # gapped/narrow boundary (smearing width, eV)
el = lambda s: set(re.findall(r'[A-Z][a-z]?', s))

fig = plt.figure(figsize=(S.W2, 112 * S.MM))
gt = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.78, 1.45], wspace=0.5, left=0.06, right=0.985, top=0.94, bottom=0.6)
gb = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.3, left=0.06, right=0.985, top=0.44, bottom=0.09)

# a  band gap, scalar-relativistic vs SOC, same structure and k-mesh
ax = fig.add_subplot(gt[0, 0])
g = p[(p.dft_class != 'metal') | (p.soc_class != 'metal')]
for t in ['training', 'prediction']:
    x = g[g.data_set == t]
    z = (x.dft_class == 'metal') | (x.soc_class == 'metal')            # metallic in one calculation: gap plotted as 0
    ax.scatter(x[~z].gap, x[~z].soc_gap, s=5, marker=S.SET_MARKER[t], lw=0, alpha=0.8, color=S.SET_COLOR[t], label=S.SET_LABEL[t], zorder=3)
    ax.scatter(x[z].gap, x[z].soc_gap, s=6, marker=S.SET_MARKER[t], facecolor='none', edgecolor=S.SET_COLOR[t], lw=0.4, alpha=0.8, zorder=3)
ax.text(0.97, 0.205, f'n = {len(g)}; open: metallic\nin one calculation', transform=ax.transAxes, ha='right', va='bottom', fontsize=5.0, color=S.INK2)
ax.plot([0, 1.6], [0, 1.6], color=S.INK, lw=0.6, zorder=2)
ax.axhline(EG, color=S.INK2, lw=0.5, ls=':'); ax.axvline(EG, color=S.INK2, lw=0.5, ls=':')
ax.set_xlim(0, 1.5); ax.set_ylim(0, 1.5)
ax.set_xlabel('band gap, scalar-relativistic (eV)', labelpad=1, fontsize=6.2); ax.set_ylabel('band gap with SOC (eV)', labelpad=1, fontsize=6.2)
both = p[(p.dft_class == 'gapped') & (p.soc_class == 'gapped')]
ax.text(0.04, 0.97, f'gapped in both: {len(both)}\nnarrower with SOC: {(both.dgap < 0).mean():.0%}\n'
        f'median $\\Delta E_g$ = {both.dgap.median() * 1e3:.0f} meV', transform=ax.transAxes, va='top', fontsize=5.4, color=S.INK)
ax.text(EG + 0.03, 0.5, f'$E_g$ = {EG} eV', rotation=90, ha='left', va='bottom', fontsize=5.0, color=S.INK2)
ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.33), fontsize=5.2, handletextpad=0.2, borderaxespad=0.2, markerscale=1.4)
S.panel(ax, 'a', x=-0.24, y=1.03)

# b  electronic class, scalar (rows) -> SOC (columns), all 607 paired runs incl. the 6 ambiguous metals
ax = fig.add_subplot(gt[0, 1])
pa = pd.read_csv(AN / 'soc_paired.csv')
order = ['gapped', 'narrow', 'metal']
M = pd.crosstab(pa.dft_class, pa.soc_class).reindex(index=order, columns=order, fill_value=0)
frac = M.div(M.sum(1), axis=0).values
ax.imshow(frac, cmap='Greys', vmin=0, vmax=1.6, aspect='auto')
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{M.values[i, j]}', ha='center', va='center', fontsize=6.6,
                color='white' if frac[i, j] > 0.55 else S.INK, fontweight='bold' if i == j else 'normal')
lab = ['gapped', 'narrow', 'metal']
ax.set_xticks(range(3)); ax.set_xticklabels(lab, fontsize=5.4); ax.set_yticks(range(3))
ax.set_yticklabels([f'{l}\n(n = {n})' for l, n in zip(lab, M.sum(1).values)], fontsize=5.4)
ax.set_xlabel('class with SOC', labelpad=1, fontsize=6.2); ax.set_ylabel('class, scalar-relativistic', labelpad=1, fontsize=6.2)
ax.tick_params(length=0)
ax.text(0.5, 1.03, f'n = {len(pa)} paired runs', transform=ax.transAxes, ha='center', va='bottom', fontsize=5.2, color=S.INK2)
S.panel(ax, 'b', x=-0.42, y=1.03)
M.to_csv(DATA / 'fig8_class_transition.csv')

# c  v13: carrier-resolved SOC effect on the scalar candidates (900 K, |n| <= 1e21 cm^-3), by heavy element
D_ = pd.read_pickle(REPO / 'results/paper_analysis/paper_dataset.pkl')
cnd = D_[D_.winner.fillna(False).astype(bool) & D_.gapped & D_.heavy & (D_.soc_status == 'done') & D_.data_set.isin(['training', 'prediction'])].copy()
cnd['rp'] = cnd.soc_w1_cap1e21_p_900_PF / cnd.w1_cap1e21_p_900_PF; cnd['rn'] = cnd.soc_w1_cap1e21_n_900_PF / cnd.w1_cap1e21_n_900_PF
cnd['lost'] = cnd.soc_best_PF <= PF_BAR
GROUPS = [('Au', lambda s_: 'Au' in el(s_)), ('Pt', lambda s_: 'Pt' in el(s_) and 'Au' not in el(s_)), ('Ir', lambda s_: 'Ir' in el(s_)),
          ('Hf', lambda s_: 'Hf' in el(s_) and not ({'Au', 'Pt', 'Ir'} & el(s_))), ('Ta', lambda s_: 'Ta' in el(s_) and not ({'Au', 'Pt', 'Ir', 'Hf'} & el(s_))),
          ('Bi', lambda s_: 'Bi' in el(s_) and not ({'Au', 'Pt', 'Ir', 'Hf', 'Ta'} & el(s_))), ('other', None)]
assigned = pd.Series('', index=cnd.index)
for gname, fn in GROUPS[:-1]:
    m_ = cnd.composition.map(fn) & (assigned == ''); assigned[m_] = gname
assigned[assigned == ''] = 'other'; cnd['grp'] = assigned
ax = fig.add_subplot(gt[0, 2]); rng = np.random.default_rng(3); rows = []
PC_, NC_ = S.PURPLE, S.TEAL
for i, (gname, _) in enumerate(GROUPS):
    x = cnd[cnd.grp == gname]
    if not len(x): continue
    for off, col_, cc in [(-0.17, 'rp', PC_), (0.17, 'rn', NC_)]:
        xx = i + off + rng.uniform(-0.07, 0.07, len(x))
        for t_ in ['training', 'prediction']:
            mm_ = (x.data_set == t_).values
            ax.scatter(xx[mm_], x[col_].values[mm_], s=5, marker=S.SET_MARKER[t_], color=cc, lw=0, alpha=0.85, zorder=3)
        ax.plot([i + off - 0.13, i + off + 0.13], [x[col_].median()] * 2, color=S.INK, lw=1.0, zorder=4)
    rows.append((gname, len(x), int(x.lost.sum()), x.rp.median(), x.rn.median()))
ax.axhline(1, color=S.INK2, lw=0.5, ls=':')
ax.set_yscale('log'); ax.set_ylim(0.2, 2.6); ax.set_yticks([0.25, 0.5, 1, 2]); ax.set_yticklabels(['0.25', '0.5', '1', '2']); ax.minorticks_off()
ax.set_xticks(range(len(rows))); ax.set_xticklabels([f'{g}\n{l}/{n}' for g, n, l, _, _ in rows], fontsize=5.0); ax.set_xlim(-0.6, len(GROUPS) - 0.4)
ax.set_xlabel('heavy element of the candidate', labelpad=1, fontsize=6.0)
ax.set_ylabel(r'PF/$\tau$ with SOC / scalar, 900 K', labelpad=1, fontsize=6.2)
ax.text(0.5, 1.02, 'per group: runs lost below PF$_0$ / candidate runs', transform=ax.transAxes, ha='center', va='bottom', fontsize=5.0, color=S.INK2)
h = [Line2D([], [], ls='', marker='o', mfc=PC_, mec='none', ms=3, label='p-type'), Line2D([], [], ls='', marker='o', mfc=NC_, mec='none', ms=3, label='n-type'),
     Line2D([], [], color=S.INK, lw=1.0, label='median')]
ax.legend(handles=h, loc='lower left', fontsize=5.0, handletextpad=0.2, borderaxespad=0.2, ncol=3, columnspacing=0.6)
S.panel(ax, 'c', x=-0.3, y=1.03)
cr = pd.DataFrame(rows, columns=['group', 'runs', 'lost', 'median_p_ratio', 'median_n_ratio']); cr.to_csv(DATA / 'fig6v13_soc_carrier.csv', index=False)
assert len(cnd) == 115 and int(cnd.lost.sum()) == 40 and abs(cnd.rp.median() - 0.758) < 0.005 and abs(cnd.rn.median() - 0.997) < 0.005

# d  best PF/tau of the scalar candidates (qualified + gapped, heavy), scalar vs SOC
ax = fig.add_subplot(gb[0, 0])
c = p[p.qualified & (p.dft_class == 'gapped')].copy()
c['Au'] = c.composition.map(lambda s: 'Au' in el(s))
rows = []
for t in ['training', 'prediction']:
    x = c[c.data_set == t]
    for au in (False, True):
        y = x[x.Au == au]
        ax.scatter(y.best_PF / 1e12, y.soc_best_PF / 1e12, s=9 if au else 6, marker=S.SET_MARKER[t], zorder=3,
                   facecolor='none' if au else S.SET_COLOR[t], edgecolor=S.SET_COLOR[t], lw=0.6 if au else 0)
    rows.append((t, len(x), int((x.soc_best_PF > PF_BAR).sum()), int((x.soc_best_PF <= PF_BAR).sum()),
                 int(x.Au.sum()), int((x.Au & (x.soc_best_PF <= PF_BAR)).sum())))
lo, hi = 0.7, 3.0
ax.plot([lo, hi], [lo, hi], color=S.INK, lw=0.6)
ax.axhline(PF_BAR / 1e12, color=S.INK, lw=0.6, ls='--'); ax.axvline(PF_BAR / 1e12, color=S.INK2, lw=0.5, ls=':')
ax.axhspan(lo, PF_BAR / 1e12, color=S.MUTED, alpha=0.35, lw=0, zorder=0)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xticks([0.7, 1, 2, 3]); ax.set_xticklabels(['0.7', '1', '2', '3']); ax.set_yticks([0.7, 1, 2, 3]); ax.set_yticklabels(['0.7', '1', '2', '3'])
ax.minorticks_off()
ax.set_xlabel(r'best PF/$\tau$, scalar ($10^{12}$ W m$^{-1}$K$^{-2}$s$^{-1}$)', labelpad=1, fontsize=6.2)
ax.set_ylabel(r'best PF/$\tau$ with SOC ($10^{12}$ W m$^{-1}$K$^{-2}$s$^{-1}$)', labelpad=1, fontsize=6.2)
kept, lost = sum(r[2] for r in rows), sum(r[3] for r in rows)
ax.text(0.04, 0.97, f'heavy-element candidates: {len(c)}\nabove PF$_0$ with SOC: {kept}', transform=ax.transAxes, va='top',
        fontsize=5.4, color=S.INK)
ax.text(0.04, 0.62, f'below PF$_0$: {lost}\n(Au: {sum(r[5] for r in rows)} of {sum(r[4] for r in rows)})',
        transform=ax.transAxes, ha='left', va='top', fontsize=5.4, color=S.INK)
ax.text(hi * 0.96, PF_BAR / 1e12 * 1.04, 'PF$_0$', ha='right', va='bottom', fontsize=5.0, color=S.INK2)
h = [Line2D([], [], ls='', marker='o', mfc=S.INK2, mec=S.INK2, ms=3, label='no Au'),
     Line2D([], [], ls='', marker='o', mfc='none', mec=S.INK2, mew=0.6, ms=3.4, label='Au-containing')]
ax.legend(handles=h, loc='upper left', bbox_to_anchor=(0.0, 0.82), fontsize=5.2, handletextpad=0.2, borderaxespad=0.2)
S.panel(ax, 'd', x=-0.16, y=1.03)
pd.DataFrame(rows, columns=['set', 'candidates', 'above_PF0_soc', 'below_PF0_soc', 'Au', 'Au_below']).to_csv(DATA / 'fig8_soc_candidates.csv', index=False)
assert (kept, lost) == (75, 40) and [r[1] for r in rows] == [56, 59]

# d  structural variants: energy above the composition's lowest-energy calculated crystal vs PF/tau ratio
ax = fig.add_subplot(gb[0, 1])
mm = (v.dft_class == 'metal') & (v.ref_dft_class == 'metal')          # metal in both crystals: grey background
ax.scatter(v[mm].dE, v[mm].pf_ratio, s=2.2, marker='o', lw=0, color=S.MUTED, zorder=2)
bins = np.array([5, 12, 25, 50, 100, 200, 1000]); mid, med, nb = [], [], []
for a_, b_ in zip(bins[:-1], bins[1:]):
    y = v[mm & (v.dE >= a_) & (v.dE < b_)].pf_ratio
    if len(y) >= 10: mid.append(np.sqrt(a_ * b_)); med.append(y.median()); nb.append(len(y))
ax.plot(mid, med, color=S.INK2, lw=0.9, marker='s', ms=2.0, zorder=3)
nm = v[~mm]                                                           # at least one crystal not metallic
for _, r in nm.iterrows():
    ax.scatter(r.dE, r.pf_ratio, s=13, marker=S.SET_MARKER[r.data_set], facecolor=S.CLASS_COLOR[r.ref_dft_class],
               edgecolor=S.INK if r.dft_class != r.ref_dft_class else 'white', lw=0.6, zorder=5)
lab = {'IrTaSn': (1.25, 0.2), 'NiVGa': (0.75, 0.2), 'PdScAs': (1.0, 4.5), 'LuZnBi': (1.0, 0.25), 'La2IrAs': (2.1, 1.0), 'LaCdSb': (0.55, 0.4)}
done = set()
for _, r in nm[nm.composition.isin(lab)].iterrows():
    if r.composition in done: continue
    done.add(r.composition); fx, fy = lab[r.composition]
    ax.annotate(r.composition.replace('La2', 'La$_2$'), (r.dE, r.pf_ratio), xytext=(r.dE * fx, r.pf_ratio * fy), fontsize=5.1, ha='center', color=S.INK,
                bbox=dict(fc='white', ec='none', pad=0.4, alpha=0.85), arrowprops=dict(arrowstyle='-', lw=0.4, color=S.INK2), zorder=7)
ax.axhline(1, color=S.INK2, lw=0.5, ls=':')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(4.5, 1000); ax.set_ylim(1e-3, 1e3)
ax.set_xlabel(r'energy above the reference crystal (meV atom$^{-1}$)', labelpad=1, fontsize=6.2)
ax.set_ylabel(r'best PF/$\tau$, variant / reference crystal', labelpad=1, fontsize=6.2)
ax.text(0.04, 0.97, f'{len(v)} variants: {int(mm.sum())} metal pairs (grey;\nline: their median), {len(nm)} other pairs', transform=ax.transAxes,
        va='top', fontsize=5.1, color=S.INK)
h = [Line2D([], [], ls='', marker='o', mfc=S.CLASS_COLOR[k], mec='white', ms=3.4, label=f'reference {S.CLASS_LABEL[k] if k != "narrow" else "narrow"}')
     for k in ['gapped', 'narrow', 'half_metal', 'metal']]
h.append(Line2D([], [], ls='', marker='o', mfc='white', mec=S.INK, mew=0.6, ms=3.4, label='class changes'))
leg = ax.legend(handles=h, loc='upper right', bbox_to_anchor=(1.0, 0.86), fontsize=5.0, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25, ncol=1, frameon=True)
leg.get_frame().set(facecolor='white', edgecolor='none', alpha=0.85)
S.panel(ax, 'e', x=-0.12, y=1.03)
pd.DataFrame({'dE_mid': mid, 'median_pf_ratio_metal_pairs': med, 'n': nb}).to_csv(DATA / 'fig8_variant_median.csv', index=False)
nm[['composition', 'data_set', 'ref_dft_class', 'dft_class', 'dE', 'pf_ratio']].to_csv(DATA / 'fig8_nonmetal_variants.csv', index=False)
assert len(v) == 824 and len(nm) == 24 and int(mm.sum()) == 800

probs = qa.check(fig)
print('QA problems:', len(probs)); [print(' ', x) for x in probs]
S.save(fig, OUT / 'fig8_soc_polymorph', width_mm=183)
