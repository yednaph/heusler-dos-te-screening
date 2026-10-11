"""Fig. 5 (v9, compact version of the v4 funnel) -- screening and certification funnel, mirrored by set (deck slides 31 and 51).

Training set grows to the left, prediction set to the right, from a central label spine. Bar length is proportional to
log10(count), so every step stays visible; exact counts are printed on the bars. Two segments:
  SCREEN (v2): screened -> semiconductor-like d-DOS -> prime d-DOS gap window -> computed from that window
  DFT CERTIFICATION (v3): all computed runs -> dopable -> above PF bar -> stable (winners) -> DFT-gapped -> SOC-checked
They are not nested: the DFT queues also sampled screening groups outside the prime window (stated on the figure).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
SKILL = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SKILL))
from figkit import style as S, glyphs as G, qa          # noqa: E402
from common import dft_runs, screen_pools, DATA, OUT     # noqa: E402

S.apply()
d = dft_runs()
pools = screen_pools()
C = {}
for t in ['training', 'prediction']:
    g = pools[pools.pool == t]
    sc = g[g.etype == 'semiconductor']
    prime = sc[sc.gap_tier.isin(['0.10-0.40', '0.40-1.00', '1.00-1.40'])]
    r = d[d.data_set == t]
    done = set(r.UUID)
    m1 = r.dopable == True; m2 = m1 & (r.above_bar == True); m3 = r.winner == True
    m4 = m3 & (r.dft_class == 'gapped'); m5 = r.certified == True
    C[t] = dict(screened=len(g), semi=len(sc), prime=len(prime), prime_done=int(prime.UUID.isin(done).sum()),
                runs=len(r), dopable=int(m1.sum()), bar=int(m2.sum()), winners=int(m3.sum()), gapped=int(m4.sum()),
                cert=int(m5.sum()), comp=int(r[m5].composition.nunique()))
pd.DataFrame(C).to_csv(DATA / 'fig3_funnel_counts.csv')
# guards against silent data drift (values verified 2026-10-09 against deck slide 31 and the audit)
assert (C['prediction']['semi'], C['training']['semi'], C['prediction']['prime'], C['training']['prime']) == (2394, 1550, 603, 671)
assert (C['training']['cert'], C['prediction']['cert'], C['training']['winners'], C['prediction']['winners']) == (76, 79, 163, 146)

W, H = 183, 98
fig, ax = G.canvas(W, H)
XL, XR = 67.5, 115.5                     # inner edges of the two half-funnels; labels sit between
LMAX = 63.0                              # bar length for the largest count


def L(n):
    """Bar length ∝ log10(n) between 10 (length 4 mm) and 1e5 (LMAX)."""
    return 4 + (LMAX - 4) * (np.log10(max(n, 10)) - 1) / 4


def row(y, key, label, sub=None, badges=(), role=S.NAVY, h=4.4):
    for t, side, xa in [('training', 'left', XL), ('prediction', 'right', XR)]:
        n = C[t][key]
        G.funnel_bar(ax, xa, y, L(n), h, f'{n:,}', '', S.SET_COLOR[t], side=side, fs_count=6.8)
    cx = (XL + XR) / 2
    G.text(ax, cx, y + h / 2 + (1.05 if sub else 0), label, fs=5.7, w='bold', c=S.INK)
    if sub:
        G.text(ax, cx, y + h / 2 - 1.65, sub, fs=4.4, c=S.INK2)
    for i, (b, col) in enumerate(badges):
        G.badge(ax, XL + 3.2 + i * 4.4, y + h / 2, b, col, r=1.75, fs=5.6) if False else None
    # decision badges sit just inside the label spine, right side
    pass                                 # v11: revision tags removed (revision history lives in the SI)


def band(y, title, role, note=None):
    import matplotlib.patches as mp
    ax.add_patch(mp.FancyBboxPatch((2, y), W - 4, 4.6, boxstyle='round,pad=0,rounding_size=0.8', fc=S.TINT[role], ec='none', zorder=1))
    G.text(ax, 4, y + 2.3, title, fs=6.4, w='bold', c=role, ha='left')
    if note:
        G.text(ax, W - 4, y + 2.3, note, fs=5.0, c=S.INK2, ha='right')


# ---------------------------------------------------------------- headers
for t, x, ha in [('training', XL - 1, 'right'), ('prediction', XR + 1, 'left')]:
    G.text(ax, x, H - 3.2, f'{S.SET_LABEL[t]}  ·  screened on its {S.SET_SOURCE[t]}', fs=6.6, w='bold', c=S.SET_COLOR[t], ha=ha)
G.text(ax, (XL + XR) / 2, H - 3.2, 'stage', fs=6.0, w='bold', c=S.INK2)

# ---------------------------------------------------------------- SCREEN segment (v2)
band(H - 11.5, 'SCREENING: Cutler–Mott d-DOS indicators, d-DOS gap label and priority filters', S.BLUE,
     note='bar length $\\propto$ log$_{10}$(count)')
y = H - 19.0
row(y, 'screened', 'screened', sub='DXMag entries in the set'); y -= 5.75
row(y, 'semi', 'semiconductor d-DOS label', sub='both spin channels gapped at $E_F$'); y -= 5.75
row(y, 'prime', 'd-DOS gap 0.10–1.40 eV', sub='target gap range'); y -= 5.75
row(y, 'prime_done', 'computed from that range', sub='score-ordered selection rounds',
    badges=[(1, S.NAVY), (2, S.NAVY)]); y -= 5.3
G.text(ax, (XL + XR) / 2, y + 1.0, 'the selection rounds also sampled entries outside this range (Supplementary Fig. S3)',
       fs=4.9, c=S.PURPLE)

# ---------------------------------------------------------------- DFT CERTIFICATION segment (v3)
y -= 5.6
band(y, 'FIRST-PRINCIPLES ASSESSMENT: QE-PBE + BoltzTraP2 (criteria set during the campaign; Methods)', S.TEAL)
y -= 6.5
row(y, 'runs', 'all calculations', sub='QE-PBE · BoltzTraP2 (CRTA)'); y -= 5.75
row(y, 'dopable', 'Seebeck criterion', sub='max$_T$ |S($E_F$)| > 60 μV K$^{-1}$', badges=[(4, S.GREEN)]); y -= 5.75
row(y, 'bar', 'power-factor criterion', sub='best PF/τ > 1.09×10$^{12}$ W m$^{-1}$K$^{-2}$s$^{-1}$', badges=[(4, S.GREEN)]); y -= 5.75
row(y, 'winners', 'stability criterion', sub='DXMag hull < 0.10 eV atom$^{-1}$', badges=[(2, S.NAVY)]); y -= 5.75
row(y, 'gapped', 'QE-PBE gap', sub='gap ≥ 0.136 eV', badges=[(3, S.GREEN), (5, S.GREEN)]); y -= 5.75
row(y, 'cert', 'certified entries', sub='SOC check, atomic number ≥ 72', badges=[(6, S.GREEN)]); y -= 5.75
row(y, 'comp', 'certified compositions', sub='distinct per set')
# ---------------------------------------------------------------- convergence on the shared outcome
ybot = y
import matplotlib.patches as mp
G.arrow(ax, XL - L(C['training']['comp']) / 2, ybot, 76, ybot - 4.6, color=S.SET_COLOR['training'], lw=1.0, rad=0.25)
G.arrow(ax, XR + L(C['prediction']['comp']) / 2, ybot, 107, ybot - 4.6, color=S.SET_COLOR['prediction'], lw=1.0, rad=-0.25)
ax.add_patch(mp.FancyBboxPatch((76, ybot - 9.2), 31, 6.6, boxstyle='round,pad=0,rounding_size=1.0', fc=S.GREEN, ec='none', zorder=3))
G.text(ax, 91.5, ybot - 4.6, '80 compositions', fs=7.0, w='bold', c='white')
G.text(ax, 91.5, ybot - 7.6, '74 in both sets · 5 P · 1 T', fs=4.9, c='white')
G.text(ax, 109.5, ybot - 7.0, 'ranked on computed |S| and PF/τ at 900 K,\n|n| ≤ 10$^{21}$ cm$^{-3}$: 8 compositions on the first Pareto layer', fs=5.0, ha='left')

qa.check(fig, 'fig5_funnel')
S.save(fig, OUT / 'fig5_funnel', width_mm=183)
print(pd.DataFrame(C))
