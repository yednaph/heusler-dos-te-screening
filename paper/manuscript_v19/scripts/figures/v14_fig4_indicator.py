"""v14 Fig. 4 -- what the d-DOS Seebeck indicator measures (new; paper-figures skill).

Question: what does S^d_max capture about the computed transport, and what does it miss?
5-second takeaway: across all calculated runs the indicator tracks the computed thermopower scale (Spearman 0.73 training set,
0.61 prediction set; a,b); for ZrNiSn the Cutler-Mott S(mu) of the DFT and the predicted d-DOS bracket the calculated S (c);
among the certified entries the indicator rank says almost nothing about the calculated power factor (rho 0.13 / 0.29; d).

Panels
  a,b  SCREEN_A_seebeck_uVK (S^d_max, the screen's own value) against max_S_intr (QE+BoltzTraP2 |S| at mu = E_F, maximum over
       300-1200 K) for every attributable calculated run, one panel per set (raw indicator values never share an axis across
       sets), colour = QE-PBE class, marker = set; dotted line = the 60 uV/K dopability criterion on the computed |S|.
  c    ZrNiSn (DXMag NiZrSn, run NiZrSn__0eeb5376): Cutler-Mott S(mu) at 900 K from the held-out DFT d-DOS and the model d-DOS
       (recomputed with the screen's own code, exactly as v13 Fig. 2c), with the QE+BoltzTraP2 S at E_F and at the PF optimum
       (transport_v2.json, 900 K) -- the same numbers as figures/data/fig2v13_indicator.csv, asserted.
  d,e  (v16, V16_R2 sec. 2c link A) the Goldsmid-Sharp test: S^d_max against the QE-PBE gap of every gapped run, one panel per set,
       with the line E_g = 2 e S T at T = 900 K (S = E_g / 2eT); Spearman rho 0.90 / 0.40 and median E_g/(2 e S^d_max T) 0.78 / 1.87.
  f    rank of S^d_max within the certified entries of each set against the calculated best PF/tau (final_best_PF), the same
       data as v13 Fig. 5a,b (fig7_candidates.py) -- ranks are within-set, so both sets can share the rank axis.
  v16 also marks in a the 97 training-set metals with S^d_max > 400 uV/K (85 of them 16- or 17-electron half-Heuslers).

Inputs (existing results only): results/paper_analysis/paper_dataset.pkl; DOS_EXPORT held-out dumps and per-sample metrics;
code/te_rescreen.py; CMS/anchor/runs_local/NiZrSn__0eeb5376/transport_v2.json. No new DFT, no BoltzTraP2 re-interpolation.
Outputs: figures/v14_fig4_indicator.pdf/png (183 x 110 mm, two rows), figures/data/v14_fig4_*.csv.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mp
from scipy.stats import spearmanr
SKILL = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SKILL))
from figkit import style as S, qa                                     # noqa: E402
from common import REPO, EXPORT, DATA, OUT, RECORDS, dft_runs         # noqa: E402  (read-only use)

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
_sp = importlib.util.spec_from_file_location('ter', REPO / 'code/te_rescreen.py')
TER = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(TER)   # the screen's own Cutler-Mott code
SETS = ['training', 'prediction']
FS_T = 5.5                                                            # annotation size (>= 5 pt at print size)
S_DOP = 60.0                                                          # dopability criterion on computed undoped |S| (uV/K)
ZR_RUN = 'NiZrSn__0eeb5376'                                           # ZrNiSn, training-set entry 0eeb5376-...

d_all = dft_runs()
assert len(d_all) == 3625 and (d_all.data_set == 'training').sum() == 1562 and (d_all.data_set == 'prediction').sum() == 2021 \
    and (d_all.data_set == 'ambiguous').sum() == 42                   # 3,625 runs = 1,562 + 2,021 + 42
d = d_all[d_all.data_set.isin(SETS)].copy()
assert d.SCREEN_A_seebeck_uVK.notna().all() and (d.max_S_intr > 0).all()   # every attributable run carries both values
zr = d[d.run_dir == ZR_RUN].iloc[0]

fig = plt.figure(figsize=(S.W2, 110 * S.MM))                          # v16: two rows of three panels
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.12], wspace=0.36, hspace=0.42, left=0.065, right=0.99, top=0.905, bottom=0.084)   # Pass-6: room for 0.82 subscripts
rows_rho, rows_quad = [], []
CLS_DRAW = ['metal', 'half_metal', 'narrow', 'gapped']                # metals first (underneath)
VAL = {**{e: 3 for e in ('Sc', 'Y', 'La', 'Lu', 'B', 'Al', 'Ga', 'In', 'Tl')}, **{e: 4 for e in ('Si', 'Ge', 'Sn', 'Pb')},
       **{e: 5 for e in ('P', 'As', 'Sb', 'Bi')},                     # main group (and Sc/Y/La/Lu): 3, 4, 5
       **{e: g_ for g_, es in {4: 'Ti Zr Hf', 5: 'V Nb Ta', 6: 'Cr Mo W', 7: 'Mn Tc Re', 8: 'Fe Ru Os', 9: 'Co Rh Ir',
                               10: 'Ni Pd Pt', 11: 'Cu Ag Au', 12: 'Zn Cd Hg'}.items() for e in es.split()}}   # d block: group number


def n_valence(comp):
    """Valence electrons per formula unit (d block = group number; Sc/Y/La/Lu and group 13 = 3; group 14 = 4; group 15 = 5)."""
    return sum(VAL[e] * (int(n) if n else 1) for e, n in re.findall(r'([A-Z][a-z]?)(\d*)', comp))

# ---------------------------------------------------------------- a,b  indicator against computed |S|, one panel per set
for k, t in enumerate(SETS):
    ax = fig.add_subplot(gs[0, k]); g = d[d.data_set == t]
    x, y = g.SCREEN_A_seebeck_uVK.values, g.max_S_intr.values
    rho = spearmanr(x, y).correlation
    rows_rho.append(('all calculated runs', t, len(g), 'SCREEN_A_seebeck_uVK', 'max_S_intr', rho))
    for c in CLS_DRAW:
        s_ = g[g.dft_class == c]
        rows_rho.append((f'runs, QE-PBE {c}', t, len(s_), 'SCREEN_A_seebeck_uVK', 'max_S_intr',
                         spearmanr(s_.SCREEN_A_seebeck_uVK, s_.max_S_intr).correlation if len(s_) > 2 else np.nan))
        ax.scatter(s_.SCREEN_A_seebeck_uVK, s_.max_S_intr, s=3.2 if c == 'metal' else 4.0, marker=S.SET_MARKER[t],
                   color=S.CLASS_COLOR[c], lw=0, alpha=0.6 if c == 'metal' else 0.9, zorder=2, rasterized=True)
        rows_quad.append((t, c, len(s_), int((s_.max_S_intr > S_DOP).sum())))
    ax.axhline(S_DOP, color=S.INK2, lw=0.6, ls=':', zorder=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    lo_x = 10 ** np.floor(np.log10(x.min())); ax.set_xlim(lo_x * 0.9, 1.4e3)
    ax.set_ylim(0.05, 3e3)
    ax.set_xticks([10, 100, 1000]); ax.set_xticklabels(['10', '100', '1000'])
    ax.set_yticks([0.1, 1, 10, 100, 1000]); ax.set_yticklabels(['0.1', '1', '10', '100', '1000'])
    ax.set_xlabel(r'$S^{\mathrm{d}}_{\max}$ of the screen ($\mu$V K$^{-1}$)', labelpad=1)
    if k == 0:
        ax.set_ylabel(r'calculated max $|S|$ ($\mu$V K$^{-1}$)', labelpad=1)
    ax.text(0.03, 0.98, f'{S.SET_LABEL[t]}\n{len(g):,} runs, ρ = {rho:.2f}', transform=ax.transAxes, va='top', ha='left',
            fontsize=5.8, color=S.SET_COLOR[t], linespacing=1.1)
    ax.text(0.02, S_DOP * 1.12, f'{S_DOP:.0f}', transform=ax.get_yaxis_transform(), va='bottom', ha='left', fontsize=FS_T,
            color=S.INK2)
    if t == 'training':                                                # the panel-c crystal, located in its own set
        ax.scatter([zr.SCREEN_A_seebeck_uVK], [zr.max_S_intr], s=22, facecolor='none', edgecolor=S.INK, lw=0.6, zorder=4)
        ax.annotate('ZrNiSn', (zr.SCREEN_A_seebeck_uVK, zr.max_S_intr), xytext=(-24, 7), textcoords='offset points',
                    fontsize=FS_T, color=S.INK, ha='right', va='center',
                    arrowprops=dict(arrowstyle='-', lw=0.45, color=S.INK, shrinkA=0.5, shrinkB=2.2))
    if t == 'training':                                                # v16: the high-S^d_max metals (16/17-electron half-Heuslers)
        hm = g[(g.dft_class == 'metal') & (g.SCREEN_A_seebeck_uVK > 400)]
        hm_vec = hm.composition.map(n_valence)
        n_hm, n_hm1617 = len(hm), int(hm_vec.isin([16, 17]).sum())
        assert (n_hm, n_hm1617) == (97, 85) and hm.family.str.startswith('half').all()
        hm.assign(valence_electrons=hm_vec)[['UUID', 'composition', 'family', 'SCREEN_A_seebeck_uVK', 'max_S_intr',
                                             'valence_electrons']].to_csv(DATA / 'v14_fig4_high_Sd_metals.csv', index=False)
        lx0, lx1 = np.log10(hm.SCREEN_A_seebeck_uVK.min()) - 0.04, np.log10(hm.SCREEN_A_seebeck_uVK.max()) + 0.04
        ly0, ly1 = np.log10(hm.max_S_intr.min()) - 0.09, np.log10(hm.max_S_intr.max()) + 0.09
        ax.add_patch(mp.Rectangle((10 ** lx0, 10 ** ly0), 10 ** lx1 - 10 ** lx0, 10 ** ly1 - 10 ** ly0, fill=False, ec=S.INK2,
                                  lw=0.5, ls=(0, (2.0, 1.2)), zorder=5))
        ax.annotate(f'{n_hm} metals, mostly\n16–17-electron\nhalf-Heuslers', (10 ** ((lx0 + lx1) / 2), 10 ** ly0), xytext=(0.985, 0.03),
                    textcoords='axes fraction', ha='right', va='bottom', fontsize=FS_T, color=S.INK2, linespacing=1.05,
                    arrowprops=dict(arrowstyle='-', lw=0.45, color=S.INK2, shrinkA=0.5, shrinkB=0))
    S.panel(ax, 'ab'[k], x=-0.27 if k == 0 else -0.22, y=1.02)
rho = pd.DataFrame(rows_rho, columns=['subset', 'set', 'n_runs', 'x', 'y', 'spearman_rho'])
r_all = rho[rho.subset == 'all calculated runs'].set_index('set').spearman_rho
assert round(r_all['training'], 2) == 0.73 and round(r_all['prediction'], 2) == 0.61
pd.DataFrame(rows_quad, columns=['set', 'dft_class', 'runs', 'runs_absS_above_60']).to_csv(DATA / 'v14_fig4_class_counts.csv', index=False)

# ---------------------------------------------------------------- c  ZrNiSn: d-DOS -> S(mu) -> calculated S
dd = EXPORT / 'runtime_assets/heldout_dumps/half_cubic/fold4'
mid = 'NiZrSn_0eeb5376-6970-44bf-af09-0f50d7670343'
j = [l.strip() for l in open(dd / 'test_ids.txt')].index(mid)
Pz = np.array(np.load(dd / 'test_predictions.npy', mmap_mode='r')[j]); Tz = np.array(np.load(dd / 'test_targets.npy', mmap_mode='r')[j])
up, dn = slice(0, 2001), slice(2001, 4002)
gT, gP = Tz[up] + Tz[dn], Pz[up] + Pz[dn]                             # total d-DOS (both spins), as the screen uses it
mu = TER.MU_GRID


def s_of_mu(gx):
    """Cutler-Mott S(mu) at T_OP = 900 K on the screen's mu grid (identical to fig2_representation.py)."""
    out = []
    for m_ in mu:
        w = TER.fermi_window(TER.E_GRID - m_); gw = gx * w
        out.append(-1e6 * ((gw * (TER.E_GRID - m_)).sum() / gw.sum()) / TER.T_OP)
    return np.array(out)


ST, SP = s_of_mu(gT), s_of_mu(gP)
AT, AP = TER.transport_descriptors(gT)['A_seebeck'], TER.transport_descriptors(gP)['A_seebeck']
tj = json.load(open(REPO / 'CMS/anchor/runs_local' / ZR_RUN / 'transport_v2.json'))
p900 = [p for p in tj['per_temperature'] if int(round(p['T_K'])) == 900][0]
S_int, mu_opt, S_opt = p900['intrinsic']['S_uVK'], p900['reopt_doping']['mu_eV'], p900['reopt_doping']['S_uVK']
ref = pd.read_csv(RECORDS / 'fig2v13_indicator.csv', index_col=0).iloc[:, 0]   # v13 Fig. 2c values: must be reproduced exactly
for key, v in [('A_dft', AT), ('A_model', AP), ('qe_S_EF_900K', S_int), ('qe_mu_opt_900K', mu_opt), ('qe_S_opt_900K', S_opt)]:
    assert abs(ref[key] - v) < 1e-6, (key, ref[key], v)
assert abs(AT - zr.SCREEN_A_seebeck_uVK) < 5                           # held-out DFT target vs stored training-set indicator
ax = fig.add_subplot(gs[0, 2]); PRED_C, BT_C = S.SET_COLOR['prediction'], S.TEAL
ax.axhspan(-S_DOP, S_DOP, color='#EEF1F4', lw=0, zorder=0)
ax.plot(mu, ST, color=S.INK, lw=1.1, zorder=3)
ax.plot(mu, SP, color=PRED_C, lw=1.0, ls=(0, (4, 1.5)), zorder=3)
kT, kP = int(np.argmax(np.abs(ST))), int(np.argmax(np.abs(SP)))
ax.plot([mu[kT]], [ST[kT]], marker='o', ms=3.2, color=S.INK, zorder=5)
ax.plot([mu[kP]], [SP[kP]], marker='o', ms=3.2, color=PRED_C, zorder=5)
ax.plot([0.0], [S_int], marker='s', ms=3.8, color=BT_C, ls='', zorder=6)
ax.plot([mu_opt], [S_opt], marker='D', ms=3.6, color=BT_C, mfc='white', mew=0.9, ls='', zorder=6)
# key for the two curves (values = S^d_max of each), direct labels for the two calculated points
ax.legend([Line2D([], [], color=S.INK, lw=1.1, marker='o', ms=3.0), Line2D([], [], color=PRED_C, lw=1.0, ls=(0, (4, 1.5)), marker='o', ms=3.0)],
          [f'DFT d-DOS ({AT:.0f})', f'model d-DOS ({AP:.0f})'], loc='upper left', fontsize=FS_T, handlelength=1.6,
          borderaxespad=0.2, labelspacing=0.25, title=r'curve ($S^{\mathrm{d}}_{\max}$, $\mu$V K$^{-1}$)', title_fontsize=FS_T,
          alignment='left')
neg = lambda v: f'{v:.0f}'.replace('-', '\u2212')                    # typographic minus
ax.annotate(f'QE+BoltzTraP2\nat $E_F$: {neg(S_int)}', (0.0, S_int), xytext=(-5, 0), textcoords='offset points', fontsize=FS_T,
            color=BT_C, va='center', ha='right', linespacing=1.05)
ax.annotate(f'PF opt.\n{neg(S_opt)}', (mu_opt, S_opt), xytext=(-5, 0), textcoords='offset points', fontsize=FS_T,
            color=BT_C, va='center', ha='right', linespacing=1.05)
ax.text(0.98, 0.03, 'ZrNiSn, 900 K', transform=ax.transAxes, ha='right', va='bottom', fontsize=FS_T, color=S.INK2)
ax.axhline(0, color=S.INK2, lw=0.35); ax.axvline(0, color=S.INK2, lw=0.35, ls=':')
ax.set_xlim(-1.04, 1.04); ax.set_ylim(-330, 590); ax.set_yticks([-300, -200, -100, 0, 100, 200, 300, 400])
ax.set_xlabel(r'chemical potential $\mu-E_F$ (eV)', labelpad=1)
ax.set_ylabel(r'$S$ at 900 K ($\mu$V K$^{-1}$)', labelpad=1)
S.panel(ax, 'c', x=-0.22, y=1.02)
pd.DataFrame({'mu_eV': mu, 'S_dft_dDOS_uVK': ST, 'S_model_dDOS_uVK': SP}).to_csv(DATA / 'v14_fig4_zrnisn_S_of_mu.csv', index=False)
pd.Series({'A_dft': AT, 'A_model': AP, 'A_stored_training_screen': zr.SCREEN_A_seebeck_uVK, 'mu_Amax_dft_eV': mu[kT],
           'mu_Amax_model_eV': mu[kP], 'qe_S_EF_900K': S_int, 'qe_mu_opt_900K': mu_opt, 'qe_S_opt_900K': S_opt,
           'max_S_intr_300_1200K': zr.max_S_intr, 'qe_gap_eV': zr.gap}).to_csv(DATA / 'v14_fig4_zrnisn_points.csv')

# ---------------------------------------------------------------- d,e  Goldsmid-Sharp test on the gapped runs, one panel per set
T_GS = 900.0                                                          # K, the screen's operating temperature
gs_rows, gs_sum = [], []
for k, t in enumerate(SETS):
    ax = fig.add_subplot(gs[1, k]); g = d[(d.data_set == t) & (d.dft_class == 'gapped')]
    eg, sd = g.gap.values, g.SCREEN_A_seebeck_uVK.values
    ratio = eg / (2 * sd * 1e-6 * T_GS)                               # E_g / (2 e S^d_max T): S in V/K, E_g in eV, e = 1
    r_gs = spearmanr(sd, eg).correlation
    gs_sum.append((t, len(g), r_gs, np.median(ratio), np.median(sd)))
    gs_rows += [(t, u_, e_, s_, q_) for u_, e_, s_, q_ in zip(g.UUID, eg, sd, ratio)]
    xg = np.array([0.11, 1.7])
    ax.plot(xg, xg / (2 * T_GS) * 1e6, color=S.INK, lw=0.8, zorder=2)   # S = E_g / (2 e T) in uV/K
    ax.scatter(eg, sd, s=5.0 if t == 'training' else 4.4, marker=S.SET_MARKER[t], color=S.CLASS_COLOR['gapped'], lw=0, zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.12, 1.6); ax.set_ylim(28, 1700)   # Pass-6: headroom for the taller stat text
    ax.set_xticks([0.2, 0.5, 1.0]); ax.set_xticklabels(['0.2', '0.5', '1']); ax.minorticks_off()
    ax.set_yticks([50, 100, 200, 500, 1000]); ax.set_yticklabels(['50', '100', '200', '500', '1000'])
    ax.set_xlabel(r'QE-PBE gap $E_g$ (eV)', labelpad=1)
    if k == 0:
        ax.set_ylabel(r'$S^{\mathrm{d}}_{\max}$ of the screen ($\mu$V K$^{-1}$)', labelpad=1)
    ax.legend([Line2D([], [], color=S.INK, lw=0.8)], [r'$E_g = 2eS\,T$ at 900 K'], loc='lower right', fontsize=6.4,   # 6.4 pt: subscripts >= 4.5 pt
              handlelength=1.2, borderaxespad=0.12, handletextpad=0.35)        # key in the empty lower right (no label on the data)
    ax.text(0.03, 0.98, f'{S.SET_LABEL[t]}\n{len(g)} gapped runs, ρ = {r_gs:.2f}\nmedian $E_g/(2eS^{{\mathrm{{d}}}}_{{\max}}T)$ = {np.median(ratio):.2f}',
            transform=ax.transAxes, va='top', ha='left', fontsize=6.4, color=S.SET_COLOR[t], linespacing=1.15)   # sub >= 4.5 pt
    S.panel(ax, 'de'[k], x=-0.27 if k == 0 else -0.22, y=1.02)
gsd = pd.DataFrame(gs_sum, columns=['set', 'gapped_runs', 'spearman_Sd_vs_gap', 'median_Eg_over_2eST', 'median_Sd_uVK']).set_index('set')
assert tuple(gsd.gapped_runs) == (334, 241)                            # training, prediction
assert tuple(gsd.spearman_Sd_vs_gap.round(2)) == (0.90, 0.40) and tuple(gsd.median_Eg_over_2eST.round(2)) == (0.78, 1.87)
assert tuple(gsd.median_Sd_uVK.round(0)) == (313, 163)
gsd.to_csv(DATA / 'v14_fig4_goldsmid_sharp_summary.csv')
pd.DataFrame(gs_rows, columns=['set', 'UUID', 'qe_gap_eV', 'Sd_max_uVK', 'Eg_over_2eST_900K']).to_csv(DATA / 'v14_fig4_goldsmid_sharp.csv',
                                                                                                         index=False)

# ---------------------------------------------------------------- f  indicator rank against calculated PF among certified
ax = fig.add_subplot(gs[1, 2]); rows_d, txt = [], []
for t in SETS:
    cg = d[(d.data_set == t) & (d.certified == True)].copy()
    r_ = spearmanr(cg.SCREEN_A_seebeck_uVK, cg.final_best_PF).correlation
    cg['rank_pct'] = 100 * cg.SCREEN_A_seebeck_uVK.rank(pct=True)     # within the certified entries of this set
    ax.scatter(cg.rank_pct, cg.final_best_PF / 1e12, s=7 if t == 'training' else 6, marker=S.SET_MARKER[t], color=S.SET_COLOR[t],
               lw=0, alpha=0.85, zorder=3 if t == 'training' else 2)   # training triangles on top (never hidden)
    rows_rho.append(('certified entries', t, len(cg), 'SCREEN_A_seebeck_uVK', 'final_best_PF', r_))
    rows_d += [(t, u_, rk, pf) for u_, rk, pf in zip(cg.UUID, cg.rank_pct, cg.final_best_PF)]
    txt.append((t, len(cg), r_))
assert [n_ for _, n_, _ in txt] == [76, 79]                           # certified 76 T / 79 P entries
assert [round(r_, 2) for _, _, r_ in txt] == [0.13, 0.29]             # = figures/data/fig7_spearman.csv
for i, (t, n_, r_) in enumerate(txt):
    ax.text(0.03, 0.98 - 0.09 * i, f'{S.SET_LABEL[t]} ({n_}), ρ = {r_:.2f}', transform=ax.transAxes, va='top',
            ha='left', fontsize=5.8, color=S.SET_COLOR[t])
ax.set_xlim(0, 102); ax.set_ylim(0.9, 3.05); ax.set_xticks([0, 50, 100])
ax.set_xlabel(r'$S^{\mathrm{d}}_{\max}$ rank among certified (%)', labelpad=1)
ax.set_ylabel(r'calculated best PF/$\tau$ ($10^{12}$ W m$^{-1}$K$^{-2}$s$^{-1}$)', labelpad=1, fontsize=6.2)
S.panel(ax, 'f', x=-0.24, y=1.02)
pd.DataFrame(rows_d, columns=['set', 'UUID', 'rank_pct_within_certified', 'final_best_PF']).to_csv(DATA / 'v14_fig4_certified_rank.csv', index=False)
pd.DataFrame(rows_rho, columns=['subset', 'set', 'n', 'x', 'y', 'spearman_rho']).to_csv(DATA / 'v14_fig4_spearman.csv', index=False)

# ---------------------------------------------------------------- shared keys above the panels
hc = [Line2D([], [], ls='', marker='s', ms=3.6, color=S.CLASS_COLOR[c]) for c in S.CLASS_ORDER]
hs = [Line2D([], [], ls='', marker=S.SET_MARKER[t], ms=3.6, color=S.INK2) for t in SETS]
fig.text(0.012, 0.975, 'QE-PBE class', fontsize=5.8, color=S.INK2, va='center', ha='left')
fig.legend(hc + hs, [S.CLASS_LABEL[c] for c in S.CLASS_ORDER] +
           [S.SET_LABEL[t] for t in SETS], loc='upper left', bbox_to_anchor=(0.075, 1.0), ncol=6, fontsize=5.8,
           handlelength=0.9, handletextpad=0.3, columnspacing=1.2)
assert_scripts(fig)                                                   # v16 Pass-6: scripts >= 4.5 pt
probs = qa.check(fig, 'v14_fig4_indicator')
assert not probs, probs
S.save(fig, OUT / 'v14_fig4_indicator', width_mm=183)
print(pd.read_csv(DATA / 'v14_fig4_spearman.csv').round(3).to_string(index=False))
print(pd.read_csv(DATA / 'v14_fig4_class_counts.csv').to_string(index=False))
print(gsd.round(3).to_string())
