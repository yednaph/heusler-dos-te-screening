"""v14 Fig. 5 -- the screen against first principles (the v13 Fig. 3, tightened; paper-figures skill).

Question: does the d-DOS label identify gapped compounds, and does the Seebeck indicator concentrate the certified ones?
5-second takeaway: the semiconductor label of the d-DOS is mostly wrong (a,d); along the within-population rank of S^d_max,
metals spread over the whole calculated range while gapped and certified entries crowd into the top few per cent (b,e); keeping
the top fifth of the calculated entries by S^d_max recovers most certified entries (c,f).

Copied from fig6_screening_vs_dft.py (v13 Fig. 3, left untouched) and changed only in presentation, per review V14_R2 Sec. 1.1/3.1:
  * c,f  -- the recovery annotation sat on the curves; the top-20 % values now live in the legend (one row per score);
  * b,e  -- the "share ... calculated" label crossed the step line; it is now a header above the strip; the strip spans
            >= two decades with 5.5 pt ticks; the IQR bars sit above each lane so the median tick is visible over dense metals;
  * a,d  -- horizontal two-line tier labels instead of 35 deg rotated 5 pt ticks; "< 20 runs" is a hatch with a key, not a
            pale tint that collided with the half-metal purple; no bold inside the panels;
  * the x-labels of row 1 are shared with row 2, and the figure is 100 mm high instead of 126 mm.
All numbers are recomputed exactly as in v13 and asserted (1,170/738 and 1,972/1,678 labelled-semiconductor runs/metals;
1,562 and 2,016 calculated entries; 76 and 79 certified; c,f values at 10/20/30 % equal to screen_eval.xlsx / Table S1).

Inputs (existing results only): screen_pools() (indicator of every screened entry), paper_dataset.pkl (outcomes),
screen_eval.xlsx. Outputs: figures/v14_fig5_screening.pdf/png, figures/data/v14_fig5_*.csv.
"""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
SKILL = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SKILL))
from figkit import style as S, qa                                     # noqa: E402
from common import dft_runs, screen_pools, REPO, ANALYSIS, DATA, OUT  # noqa: E402  (read-only use)

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
# the pre-DFT score evaluation code of the paper (captured_top = share of certified kept in the top fraction)
spec = importlib.util.spec_from_file_location('se', REPO / 'code/paper/screen_eval.py')
se = importlib.util.module_from_spec(spec); spec.loader.exec_module(se)
tab = pd.read_excel(ANALYSIS / 'screen_eval.xlsx', sheet_name='screen_eval').query('subset == "all computed"').set_index(['data_set', 'score'])
SETS = ['training', 'prediction']
d_all = dft_runs(); d = d_all[d_all.data_set.isin(SETS)].copy()       # 3,583 attributable runs (42 ambiguous excluded)
pools = screen_pools()                                                 # 51,257 + 54,978 screened entries
LABELS = ['metal', 'half-metal', '<0.10', '0.10–0.40', '0.40–1.00', '1.00–1.40', '>1.40']
LAB_TXT = ['M', 'HM', '<0.1', '0.1–\n0.4', '0.4–\n1.0', '1.0–\n1.4', '>1.4']   # horizontal, two lines (review: no rotation)
TIER = {'<0.10': '<0.10', '0.10-0.40': '0.10–0.40', '0.40-1.00': '0.40–1.00', '1.00-1.40': '1.00–1.40', '>1.40': '>1.40'}
d['lab'] = [('metal' if e == 'metal' else 'half-metal' if e == 'half_metal' else TIER.get(t, 'n/a'))
            for e, t in zip(d.SCREEN_etype, d.SCREEN_gap_tier)]          # the d-DOS gap label of each run
STACK = ['gapped', 'narrow', 'half_metal', 'metal']
LANES = [('metal', 'metal', S.CLASS_COLOR['metal']), ('half_metal', 'half-metal', S.CLASS_COLOR['half_metal']),
         ('narrow', 'narrow gap', S.CLASS_COLOR['narrow']), ('gapped', 'gapped', S.CLASS_COLOR['gapped']),
         ('certified', 'certified', S.GREEN)]
SCORES = [('SCREEN_A_seebeck_uVK', 'A_seebeck', r'$S^{\mathrm{d}}_{\max}$', S.NAVY, '-', 1.3),
          ('SCREEN_B_pf', 'B_pf', r'$P^{\mathrm{d}}_{\max}$', S.SUPERSEDED, '-', 0.9),
          ('SCREEN_pred_D34_TE_score', 'D34_july', 'initial composite', '#3A4652', (0, (3.2, 1.2, 1.0, 1.2)), 0.9)]   # v16: dark grey dash-dot (purple = half-metal)
FS_T = 5.5                                                            # tick / annotation size used throughout (>= 5 pt)
HATCH = '//////'                                                      # marks bars built on fewer than 20 runs


def set_tag(ax, t, y):
    """Set badge in the set colour (regular weight: no bold inside the graphic)."""
    ax.text(0.0, y, f'{S.SET_LABEL[t]} · {S.SET_SOURCE[t]}', transform=ax.transAxes, fontsize=5.9, color='white',
            ha='left', va='bottom', bbox=dict(boxstyle='round,pad=0.22,rounding_size=0.25', fc=S.SET_COLOR[t], ec='none'))


fig = plt.figure(figsize=(S.W2, 100 * S.MM))
outer = fig.add_gridspec(2, 3, width_ratios=[0.92, 1.62, 0.92], hspace=0.36, wspace=0.40, left=0.058, right=0.99,
                         top=0.875, bottom=0.085)
rows_out, rows_curve, rows_lane, rows_bar = [], [], [], []
for r, t in enumerate(SETS):
    g = d[d.data_set == t]; lt = 'abc' if r == 0 else 'def'; last = r == len(SETS) - 1
    # ---------------------------------------------------------------- a,d: QE-PBE class per d-DOS gap label
    ax = fig.add_subplot(outer[r, 0])
    ct = pd.crosstab(g.lab, g.dft_class).reindex(index=LABELS, columns=STACK).fillna(0)
    n = ct.sum(1); frac = ct.div(n.replace(0, np.nan), axis=0).fillna(0) * 100
    bottom = np.zeros(len(LABELS))
    for c in STACK:
        for i in range(len(LABELS)):
            ax.bar(i, frac[c].values[i], bottom=bottom[i], color=S.CLASS_COLOR[c], width=0.76, edgecolor='white', lw=0.5,
                   hatch=HATCH if n.values[i] < 20 else None)            # hatch (white) = fewer than 20 runs
        bottom += frac[c].values
    for i, (ni, gfr) in enumerate(zip(n, frac['gapped'])):
        ax.text(i, 101.5, f'{int(ni):,}', ha='center', va='bottom', fontsize=5.0, color=S.INK2)
        if gfr >= 12 and ni >= 20:
            ax.text(i, gfr / 2, f'{gfr:.0f}%', ha='center', va='center', fontsize=5.2, color='white')
        rows_bar.append((t, LABELS[i], int(ni), *[int(ct.loc[LABELS[i], c]) for c in STACK]))
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LAB_TXT, fontsize=FS_T, linespacing=0.95)
    ax.tick_params(axis='x', length=0, pad=2)
    ax.set_xlim(-0.55, len(LABELS) - 0.45)
    ax.set_ylim(0, 112); ax.set_yticks([0, 50, 100]); ax.set_ylabel('QE-PBE class (% of runs)', labelpad=1)
    if last:
        ax.set_xlabel('d-DOS gap label (M, HM, or gap tier in eV)', fontsize=5.8, labelpad=1.5)   # v16 Pass-6 term
    sc_n = int(n[LABELS[2:]].sum()); sc_m = int(ct.loc[LABELS[2:], 'metal'].sum()); rows_out.append((t, sc_n, sc_m))
    set_tag(ax, t, y=1.07); S.panel(ax, lt[0], x=-0.25, y=1.07)

    # ---------------------------------------------------------------- b,e: population rank of the indicator vs outcome
    p = pools[pools.pool == t][['UUID', 'A_seebeck_uVK']].copy()
    p['q'] = 1.0 - p.A_seebeck_uVK.rank(pct=True, method='average') + 1.0 / len(p)    # top fraction (1/N ... 1)
    u = g.drop_duplicates('UUID').merge(p[['UUID', 'q']], on='UUID', how='left')       # one row per calculated entry
    assert u.q.notna().all()
    q20 = np.quantile(u.q, 0.20)                                        # population rank where the top 20% of calculated ends
    sub = outer[r, 1].subgridspec(2, 1, height_ratios=[0.36, 1], hspace=0.08)
    axm = fig.add_subplot(sub[0]); ax = fig.add_subplot(sub[1], sharex=axm)
    bins = np.logspace(np.log10(1.0 / len(p)), 0, 46)
    nb_all, _ = np.histogram(p.q, bins=bins); nb_calc, _ = np.histogram(u.q, bins=bins)
    share = np.where(nb_all > 0, 100.0 * nb_calc / np.maximum(nb_all, 1), np.nan)
    cen = np.sqrt(bins[:-1] * bins[1:])
    axm.fill_between(cen, 0.5, share, step='mid', color=S.TINT[S.NAVY], lw=0)
    axm.step(cen, share, where='mid', color=S.NAVY, lw=0.8)
    axm.set_yscale('log'); axm.set_ylim(0.5, 200); axm.set_yticks([1, 10, 100])
    axm.set_yticklabels(['1', '10', '100'], fontsize=FS_T); axm.minorticks_off()
    axm.set_ylabel('% calc.', labelpad=1, fontsize=5.8); plt.setp(axm.get_xticklabels(), visible=False)
    axm.tick_params(axis='x', which='both', length=0)
    axm.text(1.0, 1.04, f'share of the {len(p):,} screened entries that was calculated', transform=axm.transAxes,
             va='bottom', ha='right', fontsize=FS_T, color=S.INK2)     # header above the strip, never on the line
    pd.DataFrame({'bin_center_top_fraction': cen, 'screened': nb_all, 'calculated': nb_calc, 'share_pct': share}).to_csv(
        DATA / f'v14_fig5_calc_share_{t}.csv', index=False)
    rng = np.random.default_rng(7)
    for k, (key, lab, col) in enumerate(LANES):
        sel = u[u.certified == True] if key == 'certified' else u[(u.dft_class == key) & (u.certified != True)]
        y = k - 0.06 + rng.uniform(-0.24, 0.24, len(sel))
        if key == 'certified':
            ax.scatter(sel.q, y, s=4.5, color=S.GREEN, edgecolor=S.INK, lw=0.25, zorder=3, rasterized=True)
        else:
            ax.scatter(sel.q, y, s=2.2, color=col, alpha=0.55 if key == 'metal' else 0.85, lw=0, zorder=2, rasterized=True)
        if len(sel) >= 5:                                              # IQR bar above the lane, median as an ink tick
            q1, q2, q3 = np.quantile(sel.q, [0.25, 0.5, 0.75])
            ax.plot([q1, q3], [k + 0.32, k + 0.32], color=S.INK, lw=1.0, zorder=4, solid_capstyle='butt')
            ax.plot([q2, q2], [k + 0.2, k + 0.44], color=S.INK, lw=1.0, zorder=5)
        ax.text(1.015, k, f'{len(sel):,}', transform=ax.get_yaxis_transform(), fontsize=FS_T, va='center', ha='left', color=S.INK2)
        rows_lane.append((t, key, len(sel), float(np.median(sel.q)) if len(sel) else np.nan))
    ax.axvline(q20, color=S.NAVY, lw=0.6, ls='--', zorder=1)
    ax.text(q20, 4.9, ' top 20% of calculated', fontsize=FS_T, color=S.NAVY, va='center', ha='left')
    ax.set_xscale('log'); ax.set_xlim(1.0, 1.0 / len(p) * 0.8)        # inverted: best-ranked to the right
    ax.set_xticks([1, 0.1, 0.01, 0.001, 1e-4]); ax.set_xticklabels(['100%', '10%', '1%', '0.1%', '0.01%'])
    ax.set_yticks(range(len(LANES))); ax.set_yticklabels([l for _, l, _ in LANES], fontsize=5.8, linespacing=0.9)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-0.45, 5.08)
    if last:
        ax.set_xlabel(r'rank within the own population by $S^{\mathrm{d}}_{\max}$ (top fraction, log scale)', labelpad=1)
    S.panel(axm, lt[1], x=-0.255, y=1.0)

    # ---------------------------------------------------------------- c,f: prioritisation curves
    ax = fig.add_subplot(outer[r, 2])
    y_ = u.certified.astype(bool).values; fr = np.linspace(0.005, 1, 200)
    ax.plot([0, 1], [0, 1], color=S.INK2, lw=0.6, ls=':')
    ax.axvline(0.2, color=S.NAVY, lw=0.5, ls='--', zorder=1)
    hl, ll = [], []
    for col, key, lab, colr, ls, lw in SCORES:
        ok = u[col].notna().values
        if ok.sum() < 100:                                             # the composite exists for the prediction set only
            continue
        cap = [se.captured_top(y_[ok], u[col].values[ok], f) for f in fr]
        h_, = ax.plot(np.r_[0, fr], np.r_[0, cap], color=colr, lw=lw, ls=ls)
        for f_ in (0.1, 0.2, 0.3):
            v = se.captured_top(y_[ok], u[col].values[ok], f_)
            assert abs(v - tab.loc[(t, key), f'captured_top{int(100 * f_)}pct']) < 1e-9
            rows_curve.append((t, key, f_, v))
        v20 = tab.loc[(t, key), 'captured_top20pct']
        ax.plot([0.2], [v20], marker='o', ms=2.4, color=colr, zorder=5)
        hl.append(h_); ll.append(f'{lab}   {100 * v20:.0f}%')
    ax.text(1.0, 1.03, f'{len(u):,} calculated · {int(y_.sum())} certified', transform=ax.transAxes, ha='right', va='bottom',
            fontsize=FS_T, color=S.INK2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.2, 0.5, 1]); ax.set_xticklabels(['0', '20', '50', '100'])
    ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(['0', '50', '100'])
    if last:
        ax.set_xlabel('calculated entries kept (%)', labelpad=1)
    ax.set_ylabel('certified recovered (%)', labelpad=1)
    leg = ax.legend(hl, ll, loc='lower right', fontsize=FS_T, handlelength=1.5, borderaxespad=0.15, labelspacing=0.3,
                    title='recovered at top 20%', title_fontsize=FS_T, alignment='right')
    leg.get_title().set_color(S.INK2)
    S.panel(ax, lt[2], x=-0.3, y=1.07)

# ---------------------------------------------------------------- asserts: identical numbers to v13 Fig. 3
o = pd.DataFrame(rows_out, columns=['set', 'n', 'metal']).set_index('set')
assert (o.loc['training', 'n'], o.loc['training', 'metal'], o.loc['prediction', 'n'], o.loc['prediction', 'metal']) == (1170, 738, 1972, 1678)
lanes = pd.DataFrame(rows_lane, columns=['set', 'lane', 'n', 'median_top_fraction'])
n_ent = lanes.groupby('set').n.sum()
assert n_ent['training'] == 1562 and n_ent['prediction'] == 2016      # calculated entries (2,021 prediction runs, 5 repeats)
ncert = lanes[lanes.lane == 'certified'].set_index('set').n
assert ncert['training'] == 76 and ncert['prediction'] == 79
curve = pd.DataFrame(rows_curve, columns=['set', 'score', 'fraction_kept', 'certified_recovered'])
c20 = curve[(curve.score == 'A_seebeck') & (curve.fraction_kept == 0.2)].set_index('set').certified_recovered
assert round(100 * c20['training']) == 75 and round(100 * c20['prediction']) == 91
curve.to_csv(DATA / 'v14_fig5_prioritisation.csv', index=False)
lanes.to_csv(DATA / 'v14_fig5_lanes.csv', index=False)
pd.DataFrame(rows_bar, columns=['set', 'screen_label', 'runs', *STACK]).to_csv(DATA / 'v14_fig5_label_class.csv', index=False)

h = [plt.Rectangle((0, 0), 1, 1, fc=S.CLASS_COLOR[c]) for c in STACK] + \
    [Line2D([], [], ls='', marker='o', ms=3.4, mfc=S.GREEN, mec=S.INK, mew=0.4),
     plt.Rectangle((0, 0), 1, 1, fc=S.INK2, ec='white', hatch=HATCH, lw=0)]
fig.legend(h, [S.CLASS_LABEL[c] for c in STACK] + ['certified', 'bar of < 20 runs'], loc='upper center',
           bbox_to_anchor=(0.5, 1.0), ncol=6, fontsize=5.8, handlelength=1.0, columnspacing=1.4)
assert_scripts(fig)                                                   # v16 Pass-6: scripts >= 4.5 pt
probs = qa.check(fig, 'v14_fig5_screening')
assert not probs, probs
S.save(fig, OUT / 'v14_fig5_screening', width_mm=183)
print(lanes.round(4).to_string(index=False)); print(curve.round(4).to_string(index=False))
