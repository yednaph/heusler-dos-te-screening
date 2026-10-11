"""Fig. 6 (v10) -- screening space against first-principles outcomes (central figure, paper-figures skill).

Question: where do the calculated entries lie within each whole population ranked by the d-DOS Seebeck indicator, and how
do their first-principles outcomes (metal, half-metal, narrow gap, gapped, certified) depend on that rank?
5-second takeaway: the d-DOS semiconductor label is mostly wrong (a,d); along the indicator rank, metals spread over the whole
calculated range while gapped and especially certified entries crowd into the top few per cent of the population (b,e); keeping
the top fifth of the calculated entries by the indicator recovers most certified entries (c,f) -- strong prioritisation,
not perfect separation.

x axis of b,e: rank within the entry's OWN population (training or prediction), so raw indicator values are never compared
across sets. Inputs (existing results only): screen_pools() (indicator of every screened entry), paper_dataset.pkl (outcomes),
screen_eval.xlsx (the c,f values at 10/20/30% are asserted equal to Supplementary Table S1).
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
from common import dft_runs, screen_pools, REPO, ANALYSIS, DATA, OUT  # noqa: E402

S.apply()
spec = importlib.util.spec_from_file_location('se', REPO / 'code/paper/screen_eval.py')
se = importlib.util.module_from_spec(spec); spec.loader.exec_module(se)
tab = pd.read_excel(ANALYSIS / 'screen_eval.xlsx', sheet_name='screen_eval').query('subset == "all computed"').set_index(['data_set', 'score'])
SETS = ['training', 'prediction']
d_all = dft_runs(); d = d_all[d_all.data_set.isin(SETS)].copy()
pools = screen_pools()
LABELS = ['metal', 'half-metal', '<0.10', '0.10–0.40', '0.40–1.00', '1.00–1.40', '>1.40']
LAB_TXT = ['M', 'HM', '<0.1', '0.1–0.4', '0.4–1.0', '1.0–1.4', '>1.4']
TIER = {'<0.10': '<0.10', '0.10-0.40': '0.10–0.40', '0.40-1.00': '0.40–1.00', '1.00-1.40': '1.00–1.40', '>1.40': '>1.40'}
d['lab'] = [('metal' if e == 'metal' else 'half-metal' if e == 'half_metal' else TIER.get(t, 'n/a'))
            for e, t in zip(d.SCREEN_etype, d.SCREEN_gap_tier)]
STACK = ['gapped', 'narrow', 'half_metal', 'metal']
LANES = [('metal', 'metal', S.CLASS_COLOR['metal']), ('half_metal', 'half-metal', S.CLASS_COLOR['half_metal']),
         ('narrow', 'narrow gap', S.CLASS_COLOR['narrow']), ('gapped', 'gapped, not cert.', S.CLASS_COLOR['gapped']),
         ('certified', 'certified', S.GREEN)]
SCORES = [('SCREEN_A_seebeck_uVK', 'A_seebeck', r'$S^{\mathrm{d}}_{\max}$ (Seebeck)', S.NAVY, '-', 1.3),
          ('SCREEN_B_pf', 'B_pf', r'$P^{\mathrm{d}}_{\max}$ (power factor)', S.SUPERSEDED, '-', 0.9),
          ('SCREEN_pred_D34_TE_score', 'D34_july', 'initial composite score', S.PURPLE, (0, (3, 1.4)), 0.9)]


def set_tag(ax, t, y=1.05):
    ax.text(0.0, y, f'{S.SET_LABEL[t]} · {S.SET_SOURCE[t]}', transform=ax.transAxes, fontsize=5.9, fontweight='bold', color='white',
            ha='left', va='bottom', bbox=dict(boxstyle='round,pad=0.22,rounding_size=0.25', fc=S.SET_COLOR[t], ec='none'))


fig = plt.figure(figsize=(S.W2, 126 * S.MM))
outer = fig.add_gridspec(2, 3, width_ratios=[0.95, 1.55, 0.95], hspace=0.62, wspace=0.42, left=0.06, right=0.985, top=0.9, bottom=0.085)
rows_out, rows_curve, rows_lane = [], [], []
for r, t in enumerate(SETS):
    g = d[d.data_set == t]; lt = 'abc' if r == 0 else 'def'
    # ---- a,d: QE-PBE class per screen label
    ax = fig.add_subplot(outer[r, 0])
    ct = pd.crosstab(g.lab, g.dft_class).reindex(index=LABELS, columns=STACK).fillna(0)
    n = ct.sum(1); frac = ct.div(n.replace(0, np.nan), axis=0).fillna(0) * 100
    bottom = np.zeros(len(LABELS))
    for c in STACK:
        alpha = np.where(n.values < 20, 0.45, 1.0)
        for i in range(len(LABELS)):
            ax.bar(i, frac[c].values[i], bottom=bottom[i], color=S.CLASS_COLOR[c], width=0.76, edgecolor='white', lw=0.5, alpha=alpha[i])
        bottom += frac[c].values
    for i, (ni, gfr) in enumerate(zip(n, frac['gapped'])):
        ax.text(i, 101.5, f'{int(ni):,}', ha='center', va='bottom', fontsize=5.0, color=S.INK2)
        if gfr >= 12:
            ax.text(i, gfr / 2, f'{gfr:.0f}%', ha='center', va='center', fontsize=5.0, color='white', fontweight='bold')
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LAB_TXT, fontsize=5.0, rotation=35, ha='right', rotation_mode='anchor')
    ax.set_ylim(0, 112); ax.set_yticks([0, 50, 100]); ax.set_ylabel('QE-PBE class (% of runs)', labelpad=1)
    ax.set_xlabel('screen label (gap tiers in eV)', fontsize=5.6, labelpad=1)
    sc_n = int(n[LABELS[2:]].sum()); sc_m = int(ct.loc[LABELS[2:], 'metal'].sum()); rows_out.append((t, sc_n, sc_m))
    set_tag(ax, t, y=1.08); S.panel(ax, lt[0], x=-0.27, y=1.08)

    # ---- b,e: screening space -- population rank of the indicator vs first-principles outcome
    p = pools[pools.pool == t][['UUID', 'A_seebeck_uVK']].copy()
    p['q'] = 1.0 - p.A_seebeck_uVK.rank(pct=True, method='average') + 1.0 / len(p)    # top fraction (1/N ... 1)
    u = g.drop_duplicates('UUID').merge(p[['UUID', 'q']], on='UUID', how='left')
    assert u.q.notna().all()
    q20 = np.quantile(u.q, 0.20)                                        # population rank where the top 20% of calculated entries ends
    sub = outer[r, 1].subgridspec(2, 1, height_ratios=[0.32, 1], hspace=0.06)
    axm = fig.add_subplot(sub[0]); ax = fig.add_subplot(sub[1], sharex=axm)
    bins = np.logspace(np.log10(1.0 / len(p)), 0, 46)
    # v13: share of the population calculated in each rank bin (how closely the calculations followed the indicator)
    nb_all, _ = np.histogram(p.q, bins=bins); nb_calc, _ = np.histogram(u.q, bins=bins)
    share = np.where(nb_all > 0, 100.0 * nb_calc / np.maximum(nb_all, 1), np.nan)
    cen = np.sqrt(bins[:-1] * bins[1:])
    axm.fill_between(cen, 0, share, step='mid', color=S.TINT[S.NAVY], lw=0)
    axm.step(cen, share, where='mid', color=S.NAVY, lw=0.8)
    axm.set_yscale('log'); axm.set_ylim(0.3, 150); axm.set_yticks([1, 10, 100]); axm.set_yticklabels(['1', '10', '100'], fontsize=5.0)
    axm.set_ylabel('% calc.', labelpad=1, fontsize=5.6); plt.setp(axm.get_xticklabels(), visible=False)
    axm.text(0.99, 0.08, f'share of {len(p):,} entries calculated', transform=axm.transAxes, va='bottom', ha='right',
             fontsize=5.0, color=S.INK2)
    pd.DataFrame({'bin_center_top_fraction': cen, 'screened': nb_all, 'calculated': nb_calc, 'share_pct': share}).to_csv(
        DATA / f'fig3v13_calc_share_{t}.csv', index=False)
    rng = np.random.default_rng(7)
    for k, (key, lab, col) in enumerate(LANES):
        sel = u[u.certified == True] if key == 'certified' else u[(u.dft_class == key) & (u.certified != True)]
        y = k + rng.uniform(-0.28, 0.28, len(sel))
        if key == 'certified':
            ax.scatter(sel.q, y, s=4.5, color=S.GREEN, edgecolor=S.INK, lw=0.25, zorder=3, rasterized=True)
        else:
            ax.scatter(sel.q, y, s=2.2, color=col, alpha=0.55 if key == 'metal' else 0.85, lw=0, zorder=2, rasterized=True)
        if len(sel) >= 5:
            q1, q2, q3 = np.quantile(sel.q, [0.25, 0.5, 0.75])
            ax.plot([q1, q3], [k, k], color=S.INK, lw=1.6, zorder=4, solid_capstyle='butt')
            ax.plot([q2], [k], marker='|', ms=6, mew=1.2, color='white', zorder=5)
        ax.text(1.04, k, f'{len(sel):,}', transform=ax.get_yaxis_transform(), fontsize=5.0, va='center', ha='left', color=S.INK2)
        rows_lane.append((t, key, len(sel), float(np.median(sel.q)) if len(sel) else np.nan))
    ax.axvline(q20, color=S.NAVY, lw=0.6, ls='--')
    ax.text(q20, 4.62, ' top 20% of calculated', fontsize=5.0, color=S.NAVY, va='center', ha='left')
    ax.set_xscale('log'); ax.set_xlim(1.0, 1.0 / len(p) * 0.8)        # inverted: best-ranked to the right
    ax.set_xticks([1, 0.1, 0.01, 0.001, 1e-4]); ax.set_xticklabels(['100%', '10%', '1%', '0.1%', '0.01%'])
    ax.set_yticks(range(len(LANES))); ax.set_yticklabels([l for _, l, _ in LANES], fontsize=5.6)
    ax.set_ylim(-0.6, 4.95)
    ax.set_xlabel(r'rank within the population by $S^{\mathrm{d}}_{\max}$ (top fraction, log scale)', labelpad=1)
    S.panel(axm, lt[1], x=-0.16, y=1.15)

    # ---- c,f: prioritisation curves
    ax = fig.add_subplot(outer[r, 2])
    y_ = u.certified.astype(bool).values; fr = np.linspace(0.005, 1, 200)
    ax.plot([0, 1], [0, 1], color=S.INK2, lw=0.6, ls=':')
    for col, key, lab, colr, ls, lw in SCORES:
        ok = u[col].notna().values
        if ok.sum() < 100:
            continue
        cap = [se.captured_top(y_[ok], u[col].values[ok], f) for f in fr]
        ax.plot(np.r_[0, fr], np.r_[0, cap], color=colr, lw=lw, ls=ls, label=lab)
        for f_ in (0.1, 0.2, 0.3):
            v = se.captured_top(y_[ok], u[col].values[ok], f_)
            assert abs(v - tab.loc[(t, key), f'captured_top{int(100 * f_)}pct']) < 1e-9
            rows_curve.append((t, key, f_, v))
    a20 = tab.loc[(t, 'A_seebeck'), 'captured_top20pct']
    ax.plot([0.2, 0.2], [0, a20], color=S.NAVY, lw=0.5, ls='--'); ax.plot([0, 0.2], [a20, a20], color=S.NAVY, lw=0.5, ls='--')
    ax.text(0.24, a20 - 0.04, f'top 20% recovers\n{100 * a20:.0f}% of certified', fontsize=5.0, color=S.NAVY, va='top')
    ax.text(1.0, 1.03, f'{len(u):,} calculated · {int(y_.sum())} certified', transform=ax.transAxes, ha='right', va='bottom', fontsize=5.0, color=S.INK2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(['0', '50', '100']); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(['0', '50', '100'])
    ax.set_xlabel('calculated entries kept (%)', labelpad=1); ax.set_ylabel('certified recovered (%)', labelpad=1)
    ax.legend(loc='lower right', fontsize=5.0, handlelength=1.5, borderaxespad=0.2,
              frameon=True, facecolor='white', edgecolor='none', framealpha=1.0)    # v13: legend in both rows
    S.panel(ax, lt[2], x=-0.3, y=1.08)

o = pd.DataFrame(rows_out, columns=['set', 'n', 'metal']).set_index('set')
assert (o.loc['training', 'n'], o.loc['training', 'metal'], o.loc['prediction', 'n'], o.loc['prediction', 'metal']) == (1170, 738, 1972, 1678)
pd.DataFrame(rows_curve, columns=['set', 'score', 'fraction_kept', 'certified_recovered']).to_csv(DATA / 'fig6_prioritisation.csv', index=False)
lanes = pd.DataFrame(rows_lane, columns=['set', 'lane', 'n', 'median_top_fraction']); lanes.to_csv(DATA / 'fig6_lanes.csv', index=False)
h = [plt.Rectangle((0, 0), 1, 1, fc=S.CLASS_COLOR[c]) for c in STACK] + [Line2D([], [], ls='', marker='o', ms=3.4, mfc=S.GREEN, mec=S.INK, mew=0.4)]
fig.legend(h, [S.CLASS_LABEL[c] for c in STACK] + ['certified'], loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=5,
           fontsize=5.6, handlelength=1.0, columnspacing=1.2)
qa.check(fig, 'fig6_screening_vs_dft', ignore=tuple(l for l in LAB_TXT if '–' in l or '<' in l))   # rotated tick boxes only
S.save(fig, OUT / 'fig6_screening_vs_dft', width_mm=183)
print(lanes.round(4).to_string(index=False))
