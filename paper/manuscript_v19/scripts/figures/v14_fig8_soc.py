"""Fig. 8 (v14) -- spin-orbit coupling in paired scalar-relativistic / SOC calculations (data figure, paper-figures skill).

Question: what does a fully relativistic recalculation change for heavy-element runs and candidates?
5-second takeaway: SOC narrows gaps, rarely changes the class of a gapped run, and lowers the p-type but not the n-type power factor of
heavy-element candidates, which removes almost all Au-containing ones.

Built from scripts/figures/fig8_soc_polymorph.py (v13 panels a-d); the variant panel moved to Fig. 9 (v14_fig9_variants.py).
v14 fixes: no text over data, no in-panel title, no bold inside the graphic, training-set triangles drawn on top of prediction-set circles
(coincident twins show a triangle inside a circle), metallic gaps drawn on the axis.
v16 (V16_R2 sec. 4, colour roles): panel c no longer uses purple/teal for the carrier type (those are the half-metal and gapped class
colours); carrier type is encoded by marker fill -- p-type filled dark grey, n-type open (white fill, dark-grey edge) -- set by shape.
Inputs (existing results only): analysis/soc_paired.csv (soc_paired.py; 607 pairs), results/paper_analysis/paper_dataset.pkl (115 candidates).
Outputs: figures/v14_fig8_soc.pdf/.png; figures/data/v14_fig8_gap.csv, v14_fig8_class_matrix.csv, v14_fig8_carrier.csv, v14_fig8_candidates.csv
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

SKILL = Path(__file__).resolve().parents[4]   # repository root, which holds figkit/
sys.path.insert(0, str(SKILL))
from figkit import style as S, qa                                                # noqa: E402
from common import DATA, OUT, PF_BAR, ANALYSIS                                   # noqa: E402  (read-only import)

S.apply()

# v16 Pass-6: sub- and superscripts (units such as K$^{-1}$, E$_F$, S$^d_{max}$) print at >= 4.5 pt. matplotlib draws them at
# SHRINK_FACTOR x the text size (default 0.7, i.e. 3.5 pt for 5 pt text); 0.82 with every scripted text >= 5.5 pt gives >= 4.51 pt.
import re as _re                                                      # noqa: E402
import matplotlib._mathtext as _mt                                    # noqa: E402
_mt.SHRINK_FACTOR = 0.82


def assert_scripts(fig, min_pt=4.5):
    """Fail if any visible text with a mathtext sub/superscript would print it below min_pt."""
    bad = []
    for t in fig.findobj(lambda o: hasattr(o, 'get_text') and hasattr(o, 'get_fontsize')):
        try:
            s = t.get_text()
        except Exception:
            continue
        if s and t.get_visible() and any(_re.search(r'[\^_]', m) for m in s.split('$')[1::2]) \
                and _mt.SHRINK_FACTOR * t.get_fontsize() < min_pt - 1e-6:
            bad.append((s[:50], t.get_fontsize()))
    assert not bad, f'sub/superscripts below {min_pt} pt: {sorted(set(bad))}'
AN = Path(__file__).resolve().parents[2] / 'analysis'
pa = pd.read_csv(AN / 'soc_paired.csv')                                           # all 607 pairs (incl. 6 'ambiguous' metal pairs)
assert len(pa) == 607 and pa.data_set.value_counts().to_dict() == {'training': 351, 'prediction': 250, 'ambiguous': 6}
assert pa.same_cell.all() and pa.same_positions.all() and pa.same_scf_kmesh.all() and pa.same_nscf_kmesh.all()
p = pa[pa.data_set.isin(['training', 'prediction'])]                             # set-attributable pairs (drawn per set)
EG = 0.136                                                                        # gapped / narrow boundary (eV)
el = lambda s: set(re.findall(r'[A-Z][a-z]?', s))                                 # element set of a composition
SETS = ['prediction', 'training']                                                 # draw order: circles first, triangles on top
MS = {'prediction': 9, 'training': 6.5}                                           # circle slightly larger so a coincident twin shows a rim
ZO = {'prediction': 3, 'training': 4}


def setscatter(ax, x, y, t, open_=False, **kw):
    """Set-encoded scatter: prediction = blue circle, training = orange triangle on top; open_ = hollow marker."""
    c = S.SET_COLOR[t]
    if open_:
        ax.scatter(x, y, s=MS[t] + 2, marker=S.SET_MARKER[t], facecolor='white', edgecolor=c, lw=0.6, zorder=ZO[t] + 2, **kw)
    else:
        ax.scatter(x, y, s=MS[t], marker=S.SET_MARKER[t], facecolor=c, edgecolor='white', lw=0.25, zorder=ZO[t], **kw)


fig = plt.figure(figsize=(S.W2, 65 * S.MM))
gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.78, 1.42, 1.0], wspace=0.52, left=0.058, right=0.988, top=0.93, bottom=0.2)

# ------------------------------------------------------------------ a  band gap, scalar-relativistic vs SOC (351 runs gapped/narrow in one)
ax = fig.add_subplot(gs[0, 0])
g = p[(p.dft_class != 'metal') | (p.soc_class != 'metal')].copy()
assert len(g) == 351
g['x'] = np.where(g.dft_class == 'metal', 0.0, g.gap); g['y'] = np.where(g.soc_class == 'metal', 0.0, g.soc_gap)   # metal -> gap 0
ax.plot([0, 1.6], [0, 1.6], color=S.INK, lw=0.6, zorder=2)
ax.axhline(EG, color=S.INK2, lw=0.5, ls=':', zorder=1); ax.axvline(EG, color=S.INK2, lw=0.5, ls=':', zorder=1)
for t in SETS:
    x = g[g.data_set == t]
    setscatter(ax, x.x, x.y, t, clip_on=False)
ax.set_xlim(0, 1.5); ax.set_ylim(0, 1.5)
ax.set_xticks([0, 0.5, 1.0, 1.5]); ax.set_yticks([0, 0.5, 1.0, 1.5])
ax.set_xlabel('gap, scalar-relativistic (eV)', labelpad=1); ax.set_ylabel('gap with SOC (eV)', labelpad=1)
both = p[(p.dft_class == 'gapped') & (p.soc_class == 'gapped')]
assert len(both) == 264 and round((both.dgap < 0).mean(), 2) == 0.93 and round(both.dgap.median() * 1e3) == -56
ax.text(0.112, 0.98, f'{len(both)} gapped in both\n{(both.dgap < 0).mean():.0%} narrower with SOC\n'
        f'median $\\Delta E_g$ {both.dgap.median() * 1e3:.0f} meV'.replace('-', '\u2212'), transform=ax.transAxes, ha='left', va='top',
        fontsize=S.FS['small'], color=S.INK, linespacing=1.25)
S.panel(ax, 'a', x=-0.27, y=1.0)
g[['UUID', 'composition', 'data_set', 'dft_class', 'soc_class', 'x', 'y']].rename(columns={'x': 'gap_scalar_eV', 'y': 'gap_soc_eV'}) \
    .to_csv(DATA / 'v14_fig8_gap.csv', index=False)

# ------------------------------------------------------------------ b  class transition, scalar (rows) -> SOC (columns), 607 pairs
ax = fig.add_subplot(gs[0, 1])
order = ['gapped', 'narrow', 'metal']
M = pd.crosstab(pa.dft_class, pa.soc_class).reindex(index=order, columns=order, fill_value=0)
assert int(M.values.sum()) == 607 and M.loc['gapped', 'gapped'] == 264 and tuple(M.sum(1)) == (297, 42, 268)
frac = M.div(M.sum(1), axis=0).values
ax.imshow(frac, cmap='Greys', vmin=0, vmax=1.6, aspect='auto')
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{M.values[i, j]}', ha='center', va='center', fontsize=S.FS['label'],
                color='white' if frac[i, j] > 0.55 else S.INK)
ax.set_xticks(range(3)); ax.set_xticklabels(order, fontsize=S.FS['small'])
ax.set_yticks(range(3)); ax.set_yticklabels([f'{l}\n({n})' for l, n in zip(order, M.sum(1).values)], fontsize=S.FS['small'])
ax.set_xlabel(f'class with SOC ({len(pa)} runs)', labelpad=1.5); ax.set_ylabel('class, scalar-relativistic', labelpad=1)
ax.tick_params(length=0)
for s_ in ax.spines.values():
    s_.set_visible(False)
S.panel(ax, 'b', x=-0.56, y=1.0)
M.to_csv(DATA / 'v14_fig8_class_matrix.csv')

# ------------------------------------------------------------------ c  carrier-resolved PF ratio of the 115 candidates by heavy element
D_ = pd.read_pickle(ANALYSIS / 'paper_dataset.pkl')
cnd = D_[D_.winner.fillna(False).astype(bool) & D_.gapped & D_.heavy & (D_.soc_status == 'done')
         & D_.data_set.isin(['training', 'prediction'])].copy()
cnd['rp'] = cnd.soc_w1_cap1e21_p_900_PF / cnd.w1_cap1e21_p_900_PF                # 900 K, |n| <= 1e21 cm^-3, p-type
cnd['rn'] = cnd.soc_w1_cap1e21_n_900_PF / cnd.w1_cap1e21_n_900_PF                # same, n-type
cnd['lost'] = cnd.soc_best_PF <= PF_BAR
assert len(cnd) == 115 and int(cnd.lost.sum()) == 40
assert round(cnd.rp.median(), 2) == 0.76 and round(cnd.rn.median(), 2) == 1.00
GROUPS = [('Au', {'Au'}), ('Pt', {'Pt'}), ('Ir', {'Ir'}), ('Hf', {'Hf'}), ('Ta', {'Ta'}), ('Bi', {'Bi'})]   # first matching group wins
cnd['grp'] = [next((gn for gn, es in GROUPS if es & el(c)), 'other') for c in cnd.composition]
ax = fig.add_subplot(gs[0, 2]); rng = np.random.default_rng(3); rows = []
CARRIER_INK = '#3A4652'                                                          # v16: neutral; fill encodes carrier type
CSTY = {'rp': dict(facecolor=CARRIER_INK, edgecolor='white', lw=0.2),            # p-type: filled
        'rn': dict(facecolor='white', edgecolor=CARRIER_INK, lw=0.5)}            # n-type: open
names = [gn for gn, _ in GROUPS] + ['other']
for i, gname in enumerate(names):
    x = cnd[cnd.grp == gname]
    for off, col_ in [(-0.19, 'rp'), (0.19, 'rn')]:
        xx = i + off + rng.uniform(-0.085, 0.085, len(x))
        for t_ in SETS:
            mm_ = (x.data_set == t_).values
            ax.scatter(xx[mm_], x[col_].values[mm_], s=MS[t_] - 1.5, marker=S.SET_MARKER[t_], zorder=ZO[t_], **CSTY[col_])
        ax.plot([i + off - 0.17, i + off + 0.17], [x[col_].median()] * 2, color=S.INK, lw=1.1, zorder=6, solid_capstyle='butt',
                path_effects=[pe.Stroke(linewidth=2.1, foreground='white'), pe.Normal()])   # white rim: visible over dark markers
    rows.append((gname, len(x), int(x.lost.sum()), x.rp.median(), x.rn.median()))
ax.axhline(1, color=S.INK2, lw=0.5, ls=':', zorder=1)
ax.set_yscale('log'); ax.set_ylim(0.3, 1.9); ax.set_yticks([0.3, 0.5, 0.7, 1, 1.5]); ax.set_yticklabels(['0.3', '0.5', '0.7', '1', '1.5'])
ax.minorticks_off()
ax.set_xticks(range(len(rows))); ax.set_xticklabels([f'{g_}\n{l}/{n}' for g_, n, l, _, _ in rows], fontsize=S.FS['small'])
ax.set_xlim(-0.55, len(rows) - 0.45)
ax.set_xlabel(r'heavy element (runs below PF$_0$ / runs)', labelpad=1)
ax.set_ylabel(r'PF/$\tau$ with SOC / scalar, 900 K', labelpad=1)
hs = [Line2D([], [], ls='', marker=S.SET_MARKER[t], mfc='#8A96A3', mec='white', mew=0.25, ms=3.6, label=S.SET_LABEL[t])
      for t in ['training', 'prediction']]
h = [Line2D([], [], ls='', marker='s', mfc=CARRIER_INK, mec=CARRIER_INK, mew=0.5, ms=3.2, label='p-type (filled)'), hs[0],
     Line2D([], [], ls='', marker='s', mfc='white', mec=CARRIER_INK, mew=0.6, ms=3.2, label='n-type (open)'), hs[1],
     Line2D([], [], color=S.INK, lw=1.0, label='median')]                      # column-major: row 1 p / n / median, row 2 the sets
ax.legend(handles=h, loc='upper left', bbox_to_anchor=(0.0, 1.03), fontsize=S.FS['small'], handletextpad=0.3, borderaxespad=0.1, ncol=3,
          columnspacing=0.9, handlelength=1.2, labelspacing=0.25)
S.panel(ax, 'c', x=-0.17, y=1.0)
cr = pd.DataFrame(rows, columns=['group', 'runs', 'lost_below_PF0', 'median_p_ratio', 'median_n_ratio'])
cr.loc[len(cr)] = ['all', len(cnd), int(cnd.lost.sum()), cnd.rp.median(), cnd.rn.median()]
cr.to_csv(DATA / 'v14_fig8_carrier.csv', index=False)
assert tuple(cr.set_index('group').loc['Au', ['runs', 'lost_below_PF0']]) == (22, 20)
assert round(cr.set_index('group').loc['Au', 'median_p_ratio'], 2) == 0.49

# ------------------------------------------------------------------ d  best PF/tau without vs with SOC, same 115 candidates
ax = fig.add_subplot(gs[0, 3])
c = p[p.qualified & (p.dft_class == 'gapped')].copy()
c['Au'] = c.composition.map(lambda s: 'Au' in el(s))
assert len(c) == 115 and set(c.UUID) == set(cnd.UUID)
lo, hi = 0.7, 3.0
ax.axhspan(lo, PF_BAR / 1e12, color='#ECEFF3', lw=0, zorder=0)   # opaque = MUTED at 0.35 over white
ax.plot([lo, 4.5], [lo, 4.5], color=S.INK, lw=0.6, zorder=1)
ax.axhline(PF_BAR / 1e12, color=S.INK, lw=0.6, ls=(0, (3.2, 1.6)), zorder=1)
rows = []
for t in SETS:
    x = c[c.data_set == t]
    for au in (False, True):
        y = x[x.Au == au]
        setscatter(ax, y.best_PF / 1e12, y.soc_best_PF / 1e12, t, open_=au)
    rows.append((t, len(x), int((x.soc_best_PF > PF_BAR).sum()), int((x.soc_best_PF <= PF_BAR).sum()),
                 int(x.Au.sum()), int((x.Au & (x.soc_best_PF <= PF_BAR)).sum())))
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1.05, 4.5); ax.set_ylim(lo, hi)
ax.set_xticks([1.1, 1.5, 2, 3, 4]); ax.set_xticklabels(['1.1', '1.5', '2', '3', '4']); ax.set_yticks([0.7, 1, 1.5, 2, 3])
ax.set_yticklabels(['0.7', '1', '1.5', '2', '3']); ax.minorticks_off()
ax.set_xlabel('best PF/$\\tau$, scalar\n($10^{12}$ W m$^{-1}$K$^{-2}$s$^{-1}$)', labelpad=1, linespacing=1.1)
ax.set_ylabel(r'best PF/$\tau$ with SOC ($10^{12}$ W m$^{-1}$K$^{-2}$s$^{-1}$)', labelpad=1)
kept, lost = sum(r[2] for r in rows), sum(r[3] for r in rows)
nau, nau_lost = sum(r[4] for r in rows), sum(r[5] for r in rows)
assert (kept, lost) == (75, 40) and (nau, nau_lost) == (22, 20)
assert [r[1] for r in rows] == [59, 56] and [r[3] for r in rows] == [19, 21]
ax.text(4.35, PF_BAR / 1e12 * 1.035, f'above PF$_0$: {kept}', ha='right', va='bottom', fontsize=S.FS['small'], color=S.INK)
ax.text(4.35, PF_BAR / 1e12 / 1.035, f'below PF$_0$: {lost}\nAu: {nau_lost} of {nau}', ha='right', va='top', fontsize=S.FS['small'],
        color=S.INK, linespacing=1.25)
h = [Line2D([], [], ls='', marker='o', mfc='white', mec=S.INK2, mew=0.6, ms=3.6, label='Au-containing')]
ax.legend(handles=h, loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=S.FS['small'], handletextpad=0.2, borderaxespad=0.1)
S.panel(ax, 'd', x=-0.25, y=1.0)
pd.DataFrame(rows, columns=['set', 'candidates', 'above_PF0_soc', 'below_PF0_soc', 'Au', 'Au_below']).to_csv(
    DATA / 'v14_fig8_candidates.csv', index=False)

assert_scripts(fig)                                                   # v16 Pass-6: scripts >= 4.5 pt
probs = qa.check(fig, 'v14_fig8_soc')
assert not probs, probs
S.save(fig, OUT / 'v14_fig8_soc', width_mm=183)
print(cr.to_string())
