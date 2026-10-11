"""Fig. 3 (v10, single column) -- what is predicted and how it becomes the Seebeck indicator (paper-figures skill).

Question: what does a predicted d-DOS look like, how is the indicator S^d_max obtained from it, and how does it compare with the
first-principles transport of the same entry?
5-second takeaway: the prediction resolves a gap-like minimum at E_F; the d-DOS Seebeck curve peaks next to it (S^d_max); the
first-principles Seebeck coefficient of the same entry is larger and of different sign at E_F -- the indicator flags the band edge,
it does not reproduce S.

Example: the prediction-set CoTiSb entry (certified; an established half-Heusler thermoelectric). Inputs: saved ensemble mean and
spread (results/screening/<family>/novel_expA_ensemble); S(mu) recomputed with code/te_rescreen.py (A asserted equal to the stored
value); QE-PBE/BoltzTraP2 values at 900 K from CMS/anchor/runs_local/CoTiSb/transport_v2.json (intrinsic and mu-optimised points).
"""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
SKILL = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SKILL))
from figkit import style as S, qa                                     # noqa: E402
from common import REPO, DATA, OUT, dft_runs                          # noqa: E402

S.apply()
spec = importlib.util.spec_from_file_location('tr', REPO / 'code/te_rescreen.py')
tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
d = dft_runs()
row = d[(d.composition == 'CoTiSb') & (d.data_set == 'prediction')].iloc[0]
base = REPO / f'results/screening/{row.family}/novel_expA_ensemble'
ids = [l.strip() for l in open(base / 'ids.txt')]
j = [k for k, s in enumerate(ids) if s.lower().endswith(row.UUID.lower())][0]
mean = np.asarray(np.load(base / 'predictions.npy', mmap_mode='r')[j], float)
std = np.asarray(np.load(base / 'predictions_std.npy', mmap_mode='r')[j], float)
up, dn, su, sd = mean[:2001], mean[2001:], std[:2001], std[2001:]
g = up + dn
desc = tr.transport_descriptors(g)
assert abs(desc['A_seebeck'] - row.SCREEN_A_seebeck_uVK) < 0.5
mu = tr.MU_GRID
Sv = []
for m_ in mu:
    w = tr.fermi_window(tr.E_GRID - m_); gw = g * w
    Sv.append(-1e6 * ((gw * (tr.E_GRID - m_)).sum() / gw.sum()) / tr.T_OP)
Sv = np.array(Sv)
tj = json.load(open(REPO / 'CMS/anchor/runs_local' / row.run_dir / 'transport_v2.json'))
p900 = [p for p in tj['per_temperature'] if int(round(p['T_K'])) == 900][0]
S_int, mu_opt, S_opt = p900['intrinsic']['S_uVK'], p900['reopt_doping']['mu_eV'], p900['reopt_doping']['S_uVK']
pd.Series({'uuid': row.UUID, 'A_recomputed': desc['A_seebeck'], 'A_stored': row.SCREEN_A_seebeck_uVK, 'dft_S_at_EF_900K': S_int,
           'dft_mu_opt_900K': mu_opt, 'dft_S_at_mu_opt_900K': S_opt, 'qe_gap': row.gap}).to_csv(DATA / 'fig3_example_v10.csv')

fig = plt.figure(figsize=(89 * S.MM, 92 * S.MM))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.5, left=0.17, right=0.97, top=0.93, bottom=0.1)
E = tr.E_GRID; m = (E >= -3) & (E <= 3)
P_C = S.SET_COLOR['prediction']
# a  predicted d-DOS
ax = fig.add_subplot(gs[0])
ax.axvspan(-0.03, 0.03, color='#D9CFC1', lw=0, zorder=0)
ax.fill_between(E[m], up[m] - su[m], up[m] + su[m], color=P_C, alpha=0.25, lw=0)
ax.fill_between(E[m], -(dn[m] - sd[m]), -(dn[m] + sd[m]), color=P_C, alpha=0.25, lw=0)
ax.plot(E[m], up[m], color=P_C, lw=0.8); ax.plot(E[m], -dn[m], color=P_C, lw=0.8)
ax.axhline(0, color=S.INK2, lw=0.35)
ax.set_xlim(-3, 3); ax.set_xlabel(r'$E-E_F$ (eV)', labelpad=1); ax.set_ylabel('predicted d-DOS\n(states eV$^{-1}$ atom$^{-1}$)', labelpad=1)
ax.text(0.02, 0.96, 'spin up', transform=ax.transAxes, fontsize=5.2, color=S.INK2, va='top')
ax.text(0.02, 0.04, 'spin down', transform=ax.transAxes, fontsize=5.2, color=S.INK2, va='bottom')
ax.set_title('CoTiSb, prediction set: ensemble mean ± spread', loc='left', fontsize=5.8, fontweight='normal', pad=2)
S.panel(ax, 'a', x=-0.2, y=1.04)
# b  d-DOS Seebeck curve, indicator, and first-principles values of the same entry
ax = fig.add_subplot(gs[1])
ax.plot(mu, Sv, color=S.NAVY, lw=1.1, label=r'$S(\mu)$ from predicted d-DOS')
k_ = int(np.argmax(np.abs(Sv)))
ax.plot([mu[k_]], [Sv[k_]], marker='o', ms=3.4, color=S.NAVY, zorder=5)
ax.annotate(f'$S^{{\\mathrm{{d}}}}_{{\\max}}$ = {desc["A_seebeck"]:.0f} μV K$^{{-1}}$', (mu[k_], Sv[k_]), xytext=(0.76, 0.93),
            textcoords='axes fraction', fontsize=5.4, color=S.NAVY, ha='center', va='top',
            arrowprops=dict(arrowstyle='-', lw=0.4, color=S.NAVY))
ax.plot([0.0], [S_int], marker='s', ms=3.6, color=S.TEAL, ls='', zorder=6, label='QE-PBE + BoltzTraP2, at $E_F$')
ax.plot([mu_opt], [S_opt], marker='D', ms=3.4, color=S.TEAL, ls='', mfc='white', mew=0.9, zorder=6,
        label='QE-PBE + BoltzTraP2, μ-optimised PF')
ax.axhline(0, color=S.INK2, lw=0.35); ax.axvline(0, color=S.INK2, lw=0.35, ls=':')
ax.set_xlim(-1.05, 1.05); ax.set_ylim(-420, 440)
ax.set_xlabel(r'chemical potential $\mu-E_F$ (eV)', labelpad=1); ax.set_ylabel(r'Seebeck coefficient, 900 K ($\mu$V K$^{-1}$)', labelpad=1)
ax.legend(loc='lower left', fontsize=5.0, handlelength=1.4, borderaxespad=0.2)
ax.set_title(f'screen label: semiconductor (d-DOS gap {row.SCREEN_gap_eV:.2f} eV); QE-PBE gap {row.gap:.2f} eV',
             loc='left', fontsize=5.4, fontweight='normal', pad=2)
S.panel(ax, 'b', x=-0.2, y=1.04)
qa.check(fig, 'fig3_dos_to_indicator')
S.save(fig, OUT / 'fig3_dos_to_indicator', width_mm=89)
print('A', round(desc['A_seebeck'], 1), 'DFT S(EF)', round(S_int), 'mu_opt', round(mu_opt, 3), 'S_opt', round(S_opt))
