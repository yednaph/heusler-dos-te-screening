"""Fig. 2 (v13) -- the electronic-structure signal and how far the model reproduces it (paper-figures skill).

Question: what does the screen read from a d-DOS, how does that become the Seebeck indicator, and how faithfully does the model
reproduce it across the whole held-out population?
5-second takeaway: real spectra show the gap a semiconductor and a half-metal present to the screen; the Cutler-Mott curve from the
d-DOS peaks at the band edge but does not reproduce the calculated thermopower; across 51,257 held-out spectra the model reproduces
shape but misses most gaps, and spectral similarity does not reveal the misses.

Inputs (existing results only): held-out dumps and per-sample metrics (DOS_EXPORT, best-validation checkpoints); the screen's own
code (code/te_rescreen.py: classify, transport_descriptors, fermi_window); DXMag/QE run NiZrSn__0eeb5376 transport_v2.json (900 K).
Replaces v12 Fig. 1b,c (examples), Fig. 3 (CoTiSb indicator, now SI) and Fig. 5a,e (population fidelity; b,c now SI).
"""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.lines import Line2D
SKILL = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SKILL))
from figkit import style as S, qa                                     # noqa: E402
from common import REPO, EXPORT, DATA, OUT, E_GRID, dft_runs, per_sample   # noqa: E402

S.apply()
_sp = importlib.util.spec_from_file_location('ter', REPO / 'code/te_rescreen.py')
TER = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(TER)
ps = per_sample()
DFT_C, PRED_C, BT_C = '#C9D2DC', S.SET_COLOR['prediction'], S.TEAL
LABEL_STRIP, GAP_TINT = '#D9CFC1', S.TINT[S.TEAL]

EX = [('half_cubic', 'fold4', 'NiZrSn_0eeb5376-6970-44bf-af09-0f50d7670343', 'ZrNiSn', 'half-Heusler, 18 valence electrons'),
      ('full_cubic', 'fold4', 'Co2MnSi_de46805a-bd68-446f-963c-c23c60fbf357', 'Co$_2$MnSi', 'full-Heusler, half-metal')]
spec = []
for fam, fold, mid, name, sub in EX:                                   # real held-out spectra, no smoothing
    dd = EXPORT / 'runtime_assets/heldout_dumps' / fam / fold
    j = [l.strip() for l in open(dd / 'test_ids.txt')].index(mid)
    P = np.array(np.load(dd / 'test_predictions.npy', mmap_mode='r')[j]); T = np.array(np.load(dd / 'test_targets.npy', mmap_mode='r')[j])
    r = float(ps.loc[ps.material_id == mid, 'pearson_r'].iloc[0])
    spec.append(dict(name=name, sub=sub, T=T, P=P, r=r, pct=float((ps.pearson_r < r).mean())))
assert abs(ps.pearson_r.median() - 0.911) < 1e-3 and all(abs(x['r'] - 0.911) < 0.01 for x in spec)

fig = plt.figure(figsize=(S.W2, 118 * S.MM))
top = fig.add_gridspec(1, 2, left=0.075, right=0.985, top=0.9, bottom=0.6, wspace=0.18)
bot = fig.add_gridspec(1, 3, left=0.075, right=0.985, top=0.44, bottom=0.075, wspace=0.42, width_ratios=[1.25, 0.72, 1.0])
EG = (E_GRID >= -3) & (E_GRID <= 3); up, dn = slice(0, 2001), slice(2001, 4002)
rows = []
for k, x in enumerate(spec):                                           # a, b  spin-resolved d-DOS, DFT vs prediction
    ax = fig.add_subplot(top[0, k]); T, P = x['T'], x['P']
    ax.axvspan(-0.03, 0.03, color=LABEL_STRIP, lw=0, zorder=0.5)
    for ch, sgn in [(up, 1), (dn, -1)]:
        z = T[ch] < 0.15; i0 = int(np.argmin(abs(E_GRID))); lo = hi = i0
        if z[i0]:
            while lo > 0 and z[lo - 1]: lo -= 1
            while hi < 2000 and z[hi + 1]: hi += 1
            ax.axvspan(E_GRID[lo], E_GRID[hi], ymin=0.5 if sgn > 0 else 0, ymax=1 if sgn > 0 else 0.5, color=GAP_TINT, lw=0, zorder=0)
        ax.fill_between(E_GRID[EG], 0, sgn * T[ch][EG], color=DFT_C, lw=0, zorder=1)
        ax.plot(E_GRID[EG], sgn * P[ch][EG], color=PRED_C, lw=0.8, zorder=3)
    ax.axhline(0, color=S.INK, lw=0.4); ax.axvline(0, color=S.INK2, lw=0.6, ls='--', zorder=2)
    ym = 1.08 * max(np.abs(T[up][EG]).max(), np.abs(T[dn][EG]).max(), P[up][EG].max(), P[dn][EG].max())
    ax.set_xlim(-3, 3); ax.set_ylim(-ym, ym)
    ax.set_xlabel(r'$E-E_F$ (eV)', labelpad=1)
    ax.set_ylabel('d-DOS (states eV$^{-1}$ atom$^{-1}$,\narchive units)', labelpad=1, fontsize=5.8)
    lab = []
    for which, X in [('DFT', T), ('model', P)]:
        et, gw, _, _, pol = TER.classify(np.asarray(X[:2001]), np.asarray(X[2001:]))
        lab.append(f'{which}: {et.replace("_", "-")}' + (f', d-DOS gap {gw:.2f} eV' if et == 'semiconductor' else f', |P| = {abs(pol):.2f}'))
        rows.append((x['name'], which, et, gw, pol))
    ax.text(0.0, 1.14, f'{x["name"]}, {x["sub"]}', transform=ax.transAxes, va='bottom', fontsize=6.0, fontweight='bold')
    ax.text(1.0, 1.14, f'held-out r = {x["r"]:.3f} ({100 * x["pct"]:.0f}th percentile)', transform=ax.transAxes, va='bottom', ha='right',
            fontsize=5.2, color=S.INK2)
    ax.text(0.0, 1.02, 'screen label  ' + lab[0] + '   |   ' + lab[1], transform=ax.transAxes, va='bottom', fontsize=5.0, color=S.INK2)
    ax.text(0.985, 0.95, 'spin ↑', transform=ax.transAxes, ha='right', va='top', fontsize=5.2, color=S.INK2)
    ax.text(0.985, 0.04, 'spin ↓', transform=ax.transAxes, ha='right', va='bottom', fontsize=5.2, color=S.INK2)
    S.panel(ax, 'ab'[k], x=-0.115, y=1.14)
    if k == 0:
        h = [mp.Patch(color=DFT_C, label='DFT d-DOS (archive)'), Line2D([], [], color=PRED_C, lw=0.8, label='model, held out'),
             mp.Patch(color=GAP_TINT, label='DFT channel below label threshold'), mp.Patch(color=LABEL_STRIP, label='label window ±0.03 eV')]
        ax.legend(handles=h, loc='lower right', bbox_to_anchor=(1.0, 0.1), fontsize=5.0, frameon=True, facecolor='white',
                  edgecolor='none', framealpha=0.9, handlelength=1.2, borderaxespad=0.2)
pd.DataFrame(rows, columns=['example', 'spectrum', 'label', 'dDOS_gap_eV', 'spin_pol']).to_csv(DATA / 'fig2v13_labels.csv', index=False)

# c  from d-DOS to the Seebeck indicator, ZrNiSn: DFT d-DOS, predicted d-DOS and first-principles transport of the same crystal
ax = fig.add_subplot(bot[0, 0]); z = spec[0]
mu = TER.MU_GRID
def s_of_mu(g):
    out = []
    for m_ in mu:
        w = TER.fermi_window(TER.E_GRID - m_); gw = g * w
        out.append(-1e6 * ((gw * (TER.E_GRID - m_)).sum() / gw.sum()) / TER.T_OP)
    return np.array(out)
gT, gP = z['T'][up] + z['T'][dn], z['P'][up] + z['P'][dn]
ST, SP = s_of_mu(gT), s_of_mu(gP)
AT, AP = TER.transport_descriptors(gT)['A_seebeck'], TER.transport_descriptors(gP)['A_seebeck']
d = dft_runs(); row = d[d.run_dir == 'NiZrSn__0eeb5376'].iloc[0]
assert abs(AT - row.SCREEN_A_seebeck_uVK) < 5, (AT, row.SCREEN_A_seebeck_uVK)   # held-out target vs stored training-set indicator
tj = json.load(open(REPO / 'CMS/anchor/runs_local' / row.run_dir / 'transport_v2.json'))
p900 = [p for p in tj['per_temperature'] if int(round(p['T_K'])) == 900][0]
S_int, mu_opt, S_opt = p900['intrinsic']['S_uVK'], p900['reopt_doping']['mu_eV'], p900['reopt_doping']['S_uVK']
ax.plot(mu, ST, color=S.INK, lw=1.1, label=r'$S(\mu)$, DFT d-DOS')
ax.plot(mu, SP, color=PRED_C, lw=1.0, ls=(0, (4, 1.5)), label=r'$S(\mu)$, model d-DOS')
for Sv, c_ in [(ST, S.INK), (SP, PRED_C)]:
    kk = int(np.argmax(np.abs(Sv))); ax.plot([mu[kk]], [Sv[kk]], marker='o', ms=3.2, color=c_, zorder=5)
ax.plot([0.0], [S_int], marker='s', ms=3.6, color=BT_C, ls='', zorder=6, label='QE+BoltzTraP2, $E_F$')
ax.plot([mu_opt], [S_opt], marker='D', ms=3.4, color=BT_C, mfc='white', mew=0.9, ls='', zorder=6, label='QE+BoltzTraP2, PF opt.')
ax.axhline(0, color=S.INK2, lw=0.35); ax.axvline(0, color=S.INK2, lw=0.35, ls=':')
ax.set_xlim(-1.05, 1.05); ax.set_xlabel(r'chemical potential $\mu-E_F$ (eV)', labelpad=1)
ax.set_ylabel(r'$S$ at 900 K ($\mu$V K$^{-1}$)', labelpad=1)
ax.text(0.0, 1.03, f'ZrNiSn: $S^{{\\mathrm{{d}}}}_{{\\max}}$ = {AT:.0f} (DFT) and {AP:.0f} μV K$^{{-1}}$ (model)', transform=ax.transAxes,
        va='bottom', fontsize=5.4)
ax.set_ylim(-330, 360)
ax.legend(loc='lower left', fontsize=5.0, handlelength=1.5, borderaxespad=0.2, labelspacing=0.25)
S.panel(ax, 'c', x=-0.16, y=1.03)
pd.Series({'A_dft': AT, 'A_model': AP, 'A_stored_training_screen': row.SCREEN_A_seebeck_uVK, 'qe_S_EF_900K': S_int,
           'qe_mu_opt_900K': mu_opt, 'qe_S_opt_900K': S_opt, 'qe_gap_eV': row.gap}).to_csv(DATA / 'fig2v13_indicator.csv')

# d  held-out gap flag: confusion matrix over all 51,257 held-out spectra
CM = {('True', 'True'): 'gapped · flagged', ('True', 'False'): 'gapped · missed',
      ('False', 'True'): 'not gapped · flagged', ('False', 'False'): 'not gapped · not flagged'}
ps['cm'] = [CM[(str(a), str(b))] for a, b in zip(ps.true_gapped, ps.pred_gapped)]
ax = fig.add_subplot(bot[0, 1])
cnt = np.array([[(ps.cm == 'gapped · flagged').sum(), (ps.cm == 'gapped · missed').sum()],
                [(ps.cm == 'not gapped · flagged').sum(), (ps.cm == 'not gapped · not flagged').sum()]])
assert cnt.sum() == 51257 and cnt[0, 1] == 4352 and cnt[0, 0] == 778
frac = cnt / cnt.sum(1, keepdims=True)
EDGE = [[S.CLASS_COLOR['gapped'], S.PURPLE], [S.SUPERSEDED, S.NAVY]]
for i in range(2):
    for jj in range(2):
        ax.add_patch(mp.FancyBboxPatch((jj + 0.04, 1 - i + 0.04), 0.92, 0.92, boxstyle='round,pad=0,rounding_size=0.06',
                                       fc=S.TINT.get(EDGE[i][jj], '#EEF1F4'), ec=EDGE[i][jj], lw=0.9))
        ax.text(jj + 0.5, 1 - i + 0.6, f'{cnt[i, jj]:,}', ha='center', va='center', fontsize=6.8, fontweight='bold', color=EDGE[i][jj])
        ax.text(jj + 0.5, 1 - i + 0.32, f'{100 * frac[i, jj]:.1f}% of row', ha='center', va='center', fontsize=5.0, color=S.INK2)
ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect('equal')
ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(['flagged', 'missed /\nnot flagged'], fontsize=5.2)
ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(['gapped', 'not\ngapped'], fontsize=5.4)
ax.set_xlabel('model', labelpad=1); ax.set_ylabel('DFT (held out)', labelpad=1)
for s_ in ax.spines.values(): s_.set_visible(False)
ax.tick_params(length=0)
ax.text(1.0, 2.06, 'gap within ±1 eV, 51,257 spectra', ha='center', va='bottom', fontsize=5.0, color=S.INK2)
S.panel(ax, 'd', x=-0.42, y=1.03)

# e  spectral similarity does not reveal missed gaps (cumulative distributions of Pearson r within 2 eV of E_F)
ax = fig.add_subplot(bot[0, 2])
STY = {'gapped · missed': (S.PURPLE, 1.5, '-'), 'not gapped · not flagged': (S.NAVY, 1.5, '-'),
       'gapped · flagged': (S.CLASS_COLOR['gapped'], 0.9, '-'), 'not gapped · flagged': (S.SUPERSEDED, 0.9, (0, (3, 1.5)))}
med = ps.groupby('cm')['fermi_pearson_2eV'].median()
for key in ['not gapped · flagged', 'gapped · flagged', 'not gapped · not flagged', 'gapped · missed']:
    v = np.sort(ps.loc[ps.cm == key, 'fermi_pearson_2eV'].dropna().values); c_, lw_, ls_ = STY[key]
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c_, lw=lw_, ls=ls_, label=f'{key} (median {med[key]:.2f})')
lo_ = np.floor(10 * min(ps.fermi_pearson_2eV.min(), 0)) / 10
ax.set_xlim(lo_, 1.0); ax.set_ylim(0, 1.02)
ax.set_xlabel(r'Pearson $r$ within 2 eV of $E_F$', labelpad=1); ax.set_ylabel('cumulative fraction', labelpad=1)
ax.legend(loc='upper left', fontsize=4.9 if False else 5.0, handlelength=1.4, borderaxespad=0.2, labelspacing=0.3)
S.panel(ax, 'e', x=-0.2, y=1.03)
med.to_csv(DATA / 'fig2v13_medians.csv')

probs = qa.check(fig, 'fig2_representation', ignore=('↑', '↓'))
S.save(fig, OUT / 'fig2_representation', width_mm=183)
print(pd.read_csv(DATA / 'fig2v13_labels.csv').to_string(index=False)); print('A dft/model', round(AT), round(AP), 'stored', round(row.SCREEN_A_seebeck_uVK),
      'QE S(EF)', round(S_int), 'mu_opt', round(mu_opt, 3), 'S_opt', round(S_opt))
