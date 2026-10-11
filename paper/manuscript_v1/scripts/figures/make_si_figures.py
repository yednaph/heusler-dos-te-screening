"""Supplementary figures S1-S6 (v2) (paper_rebuild/figures/si_*.pdf). Existing files only; sets kept separate."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import *

style()
TRACKS = ['prediction', 'training']
d_all = dft_runs()
d = d_all[d_all.data_set.isin(TRACKS)].copy()

# S1 per-family ECDF of per-material Pearson r (reference population, held-out folds)
ps = per_sample()
fig, axs = plt.subplots(2, 3, figsize=(W2, 3.2), sharex=True, sharey=True)
for ax, fam in zip(axs.flat, FAMILIES):
    g = ps[ps.family == fam]
    for col, cc, lab in [('pearson_r', INK, 'full spectrum'), ('fermi_pearson_2eV', POOL_COLOR['prediction'], r'$|E-E_F|\leq2$ eV')]:
        x = np.sort(g[col].dropna().values)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=cc, lw=0.9, label=lab)
    ax.set_title(f'{FAM_LABEL[fam]} (n = {len(g):,})', loc='left', fontsize=6.3)
    ax.set_xlim(0.2, 1.0)
for ax in axs[1]: ax.set_xlabel('per-material Pearson $r$')
for ax in axs[:, 0]: ax.set_ylabel('cumulative fraction')
axs[0, 0].legend(loc='upper left', fontsize=5.6)
save(fig, 'si_S1_pearson_by_family')

# S2 energy-resolved MAE (absolute; small near E_F partly because the DOS itself is small there)
fig, axs = plt.subplots(2, 3, figsize=(W2, 3.0), sharex=True, sharey=True)
for ax, fam in zip(axs.flat, FAMILIES):
    arr = np.mean([np.load(EXPORT / f'metadata/metrics/{fam}/fold{k}/per_energy_mae.npy') for k in range(5)], axis=0)
    ax.plot(E_GRID, arr[:2001], color=INK, lw=0.7, label=r'spin $\uparrow$')
    ax.plot(E_GRID, arr[2001:], color=POOL_COLOR['training'], lw=0.7, label=r'spin $\downarrow$')
    ax.axvline(0, color=INK2, lw=0.5, ls=':'); ax.set_title(FAM_LABEL[fam], loc='left', fontsize=6.3)
for ax in axs[1]: ax.set_xlabel(r'$E-E_F$ (eV)')
for ax in axs[:, 0]: ax.set_ylabel('MAE (states eV$^{-1}$ atom$^{-1}$)', fontsize=6)
axs[0, 0].legend(fontsize=5.6)
save(fig, 'si_S2_energy_resolved_mae')

# S3 selection for DFT: share of each set sent to DFT by d-DOS label (a,b) and by percentile of A (c,d)
pools = screen_pools()
done = {t_: set(d.loc[d.data_set == t_, 'UUID'].dropna()) for t_ in TRACKS}
pools = pools[pools.pool.isin(TRACKS)].copy()
pools['computed'] = [u in done[p] for u, p in zip(pools.UUID, pools.pool)]
LAB = ['metal', 'half-metal', '<0.10', '0.10–0.40', '0.40–1.00', '1.00–1.40', '>1.40']
TX = ['M', 'HM', '<0.1', '0.1–0.4', '0.4–1.0', '1.0–1.4', '>1.4']
def lab_(e, tr):
    if e == 'metal': return 'metal'
    if e == 'half_metal': return 'half-metal'
    return {'<0.10': '<0.10', '0.10-0.40': '0.10–0.40', '0.40-1.00': '0.40–1.00', '1.00-1.40': '1.00–1.40', '>1.40': '>1.40'}.get(tr, 'n/a')
pools['lab'] = [lab_(e, tr) for e, tr in zip(pools.etype, pools.gap_tier)]
pools['A_pct'] = pools.groupby('pool').A_seebeck_uVK.rank(pct=True) * 100
bins = [0, 50, 90, 99, 99.9, 100]
fig, axs = plt.subplots(2, 2, figsize=(W2 * 0.8, 3.6), sharey='row')
rows = []
for k, (t_, lt1, lt2) in enumerate([('training', 'a', 'c'), ('prediction', 'b', 'd')]):
    g = pools[pools.pool == t_]
    sh = g.groupby('lab').computed.agg(['size', 'mean']).reindex(LAB)
    ax = axs[0, k]
    ax.bar(range(len(LAB)), 100 * sh['mean'], color=POOL_COLOR[t_], width=0.68)
    for i, (n_, m_) in enumerate(zip(sh['size'], sh['mean'])):
        ax.text(i, 100 * m_ + 3, f'{int(n_):,}', ha='center', va='bottom', fontsize=5.0, rotation=90, color=INK2)
    ax.set_ylim(0, 140); ax.set_yticks([0, 50, 100]); ax.set_xticks(range(len(LAB))); ax.set_xticklabels(TX, fontsize=5.8)
    ax.set_xlabel('d-DOS screen label'); ax.set_title(POOL_LABEL[t_], loc='left', fontsize=6.2)
    panel(ax, lt1, x=-0.15 if k == 0 else -0.05)
    gb = g.assign(b=lambda x: pd.cut(x.A_pct, bins, include_lowest=True))
    fb = gb.groupby('b').computed.agg(['size', 'mean'])
    ax = axs[1, k]
    ax.bar(range(len(fb)), 100 * fb['mean'], color=POOL_COLOR[t_], width=0.68)
    ax.set_xticks(range(len(fb))); ax.set_xticklabels(['0–50', '50–90', '90–99', '99–99.9', '99.9–100'], fontsize=5.8)
    ax.set_xlabel(r'percentile of $S^{\mathrm{d}}_{\max}$ within the set')
    panel(ax, lt2, x=-0.15 if k == 0 else -0.05)
    rows += [(t_, str(i), int(r['size']), r['mean']) for i, r in fb.iterrows()]
axs[0, 0].set_ylabel('share sent to DFT (%)'); axs[1, 0].set_ylabel('share sent to DFT (%)')
fig.tight_layout()
pd.DataFrame(rows, columns=['set', 'A_percentile_bin', 'n', 'share_computed']).to_csv(DATA / 'si_share_by_A_percentile.csv', index=False)
save(fig, 'si_S3_selection')

# S5 mesh correction
mb = pd.read_csv(ANALYSIS / 'si/B15_mesh_bug_per_run.csv')
r = (mb.new_best_PF / mb.old_best_PF).replace([np.inf, -np.inf], np.nan).dropna()
fig, ax = plt.subplots(figsize=(W1, 1.7))
ax.hist(np.log10(r), bins=80, color=INK2, lw=0)
ax.axvline(0, color=INK, lw=0.5, ls=':')
ax.set_xlabel(r'log$_{10}$[best PF/$\tau$ (NSCF mesh) / best PF/$\tau$ (stale SCF mesh)]'); ax.set_ylabel('runs')
ax.set_title(f'scalar-relativistic runs with both files, n = {len(r):,}', loc='left', fontsize=6.2)
save(fig, 'si_S4_mesh_correction')

# S6 SOC / scalar-relativistic best PF ratio
s = d_all[(d_all.soc_status == 'done') & d_all.soc_best_PF.notna()]
ratio = s.soc_best_PF / s.best_PF
fig, ax = plt.subplots(figsize=(W1, 1.7))
ax.hist(np.log10(ratio.clip(lower=1e-3)), bins=70, color=INK2, lw=0)
ax.axvline(0, color=INK, lw=0.5, ls=':')
ax.set_xlabel(r'log$_{10}$[best PF/$\tau$ with SOC / scalar-relativistic]'); ax.set_ylabel('runs')
ax.set_title(f'completed SOC runs, n = {len(ratio)}; median ratio {ratio.median():.2f}', loc='left', fontsize=6.2)
pd.Series({'n': len(ratio), 'median_ratio': ratio.median()}).to_csv(DATA / 'si_soc_ratio.csv')
save(fig, 'si_S5_soc_ratio')

print('si figures done')
