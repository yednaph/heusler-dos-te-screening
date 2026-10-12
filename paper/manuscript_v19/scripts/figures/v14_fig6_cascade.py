"""v14 Fig. 6 -- certification cascade, single column (89 x 72 mm).

Question: how many calculated runs of each population pass each first-principles criterion, and where do the sets meet?
5-second takeaway: the criteria cut 1,562 training-set and 2,021 prediction-set runs to 76 and 79 certified entries, which
converge on 80 compositions (74 in both sets, 5 prediction only, 1 training only).

Mirrored paired bars (training left, orange; prediction right, blue), length proportional to log10(runs), count inside the
bar, runs removed by the criterion outside it in grey. The screening stages are in Fig. 1a.
Stages (applied in sequence, final protocol, paper_dataset.pkl):
  all runs -> (i) Seebeck |S| > 60 uV/K  -> (ii) best PF/tau > PF0 -> (iii) DXMag hull < 0.10 eV/atom (qualified)
  -> (iv) QE-PBE gap >= 0.136 eV -> (v) SOC PF/tau for Z >= 72 (certified)
v16 Pass-6: beside each bar, under the runs removed, the share KEPT by that criterion = runs passing / runs entering it (the previous
row), so the log-scaled bars no longer hide the size of a cut (e.g. the stability criterion keeps 26% of the training-set runs);
decade labels printed as 100 / 1,000 and criterion text at 5.5 pt (sub/superscripts >= 4.5 pt).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

SKILL = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SKILL))
from figkit import style as S, glyphs as G, qa                       # noqa: E402
from common import DATA, OUT, RECORDS, dft_runs                      # noqa: E402  (read-only use)

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
T_C, P_C = S.SET_COLOR['training'], S.SET_COLOR['prediction']

# ================================================================ counts
d = dft_runs()
STAGES = ['runs', 'seebeck', 'pf', 'stability', 'qe_gap', 'certified']
C, X = {}, {}
for t in ['training', 'prediction']:
    r = d[d.data_set == t]
    m_s = r.dopable == True                                          # (i) Seebeck criterion
    m_pf = m_s & (r.above_bar == True)                               # (ii) power-factor criterion
    m_st = r.winner == True                                          # (iii) stability -> qualified
    m_g = m_st & (r.dft_class == 'gapped')                           # (iv) QE-PBE gap
    m_c = r.certified == True                                        # (v) SOC power factor -> certified
    assert not (m_st & ~m_pf).any() and not (m_c & ~m_g).any()       # the stages are nested as drawn
    heavy_unres = m_g & (r.heavy == True) & (r.soc_status != 'done')  # heavy-element runs without an SOC result
    heavy_fail = m_g & (r.heavy == True) & (r.soc_status == 'done') & (r.soc_above_bar != True)
    assert (m_g & ~m_c).sum() == heavy_unres.sum() + heavy_fail.sum()
    C[t] = dict(runs=len(r), seebeck=int(m_s.sum()), pf=int(m_pf.sum()), stability=int(m_st.sum()), qe_gap=int(m_g.sum()),
                certified=int(m_c.sum()))
    X[t] = dict(heavy_gapped=int((m_g & (r.heavy == True)).sum()), soc_fail=int(heavy_fail.sum()), soc_unresolved=int(heavy_unres.sum()),
                compositions=int(r[m_c].composition.nunique()))
cert = d[d.certified == True]
CT = set(cert[cert.data_set == 'training'].composition); CP = set(cert[cert.data_set == 'prediction'].composition)
N_COMP, N_BOTH, N_PONLY, N_TONLY = len(CT | CP), len(CT & CP), len(CP - CT), len(CT - CP)
# guards (values from the brief and the v13 funnel record)
assert [C['training'][k] for k in STAGES] == [1562, 718, 631, 163, 102, 76]
assert [C['prediction'][k] for k in STAGES] == [2021, 436, 382, 146, 99, 79]
assert (X['training']['soc_unresolved'], X['prediction']['soc_unresolved']) == (5, 1)
assert (X['training']['soc_fail'], X['prediction']['soc_fail']) == (21, 19)
assert (X['training']['compositions'], X['prediction']['compositions']) == (75, 79)
assert (N_COMP, N_BOTH, N_PONLY, N_TONLY) == (80, 74, 5, 1)
assert len(d) == 3625 and int(d[d.data_set == 'ambiguous'].certified.sum()) == 0
fc = pd.read_csv(RECORDS / 'fig3_funnel_counts.csv', index_col=0)
for k_new, k_old in zip(STAGES, ['runs', 'dopable', 'bar', 'winners', 'gapped', 'cert']):
    assert all(C[t][k_new] == fc.loc[k_old, t] for t in ['training', 'prediction']), k_new
assert all(X[t]['compositions'] == fc.loc['comp', t] for t in ['training', 'prediction'])
out = pd.DataFrame({t: [C[t][k] for k in STAGES] for t in ['training', 'prediction']}, index=STAGES)
out['training_removed'] = -out.training.diff().fillna(0).astype(int); out['prediction_removed'] = -out.prediction.diff().fillna(0).astype(int)
# v16 Pass-6: share of the runs entering each criterion that pass it (percent of the previous row)
KEPT = {t: [100 * C[t][STAGES[i]] / C[t][STAGES[i - 1]] for i in range(1, len(STAGES))] for t in ['training', 'prediction']}
assert [round(v) for v in KEPT['training']] == [46, 88, 26, 63, 75]      # 718/1562, 631/718, 163/631, 102/163, 76/102
assert [round(v) for v in KEPT['prediction']] == [22, 88, 38, 68, 80]    # 436/2021, 382/436, 146/382, 99/146, 79/99
out['training_kept_pct'] = [np.nan] + KEPT['training']; out['prediction_kept_pct'] = [np.nan] + KEPT['prediction']
out.to_csv(DATA / 'v14_fig6_cascade.csv')
pd.DataFrame({t: X[t] for t in X}).T.assign(union=N_COMP, both=N_BOTH, prediction_only=N_PONLY, training_only=N_TONLY).to_csv(
    DATA / 'v14_fig6_soc_and_compositions.csv')

# ================================================================ drawing (canvas in mm)
W, H = 89, 66
fig, ax = G.canvas(W, H)
XL, XR = 31.0, 58.0                                                  # inner edges of the mirrored bars; labels between
LMAX, NLO, NHI = 22.0, 10, 3000                                      # bar length spans log10(10)..log10(3000)


def L(n):
    return LMAX * (np.log10(n) - np.log10(NLO)) / (np.log10(NHI) - np.log10(NLO))


def bar(x0, y, w, h, col):
    ax.add_patch(FancyBboxPatch((x0, y), w, h, boxstyle='round,pad=0,rounding_size=0.5', fc=col, ec='none', zorder=3))


LABELS = [('all calculated runs', 'QE-PBE + BoltzTraP2'), ('Seebeck criterion', '$|S|>60$ μV K$^{-1}$'),
          ('power-factor criterion', 'best PF/τ > PF$_0$'), ('stability: qualified', 'hull < 0.10 eV atom$^{-1}$'),
          ('QE-PBE gap', '≥ 0.136 eV'), ('SOC: certified entries', 'PF/τ with SOC, $Z\\geq72$')]
Y0, PITCH, BH = 55.1, 6.5, 4.3
YROW = [Y0 - i * PITCH for i in range(len(STAGES))]
# faint decade guides with their labels below the last row
yb = YROW[-1] - 1.6
for n in [100, 1000]:
    for xa, sg in [(XL, -1), (XR, 1)]:
        xg = xa + sg * L(n)
        ax.plot([xg, xg], [yb + 0.6, YROW[0] + BH + 0.4], color=S.RULE, lw=0.45, zorder=1)
        G.text(ax, xg, yb - 0.6, f'{n:,}', fs=5.0, c=S.INK2)                  # v16 Pass-6: plain numbers, no superscript
G.text(ax, (XL + XR) / 2, yb - 0.6, 'runs, log scale', fs=5.0, c=S.INK2)
prev = None
for i, (k, (lab, sub)) in enumerate(zip(STAGES, LABELS)):
    y = YROW[i]
    for t, sg, xa, ha_out in [('training', -1, XL, 'right'), ('prediction', 1, XR, 'left')]:
        n = C[t][k]; ln = L(n)
        bar(xa - ln if sg < 0 else xa, y, ln, BH, S.SET_COLOR[t])
        G.text(ax, xa + sg * (ln - 1.0), y + BH / 2, f'{n:,}', fs=6.0, c='white', ha='left' if sg < 0 else 'right', z=6)
        if i > 0:                                                    # runs removed by this criterion, and the share it keeps
            rm = C[t][STAGES[i - 1]] - n
            G.text(ax, xa + sg * (ln + 1.0), y + BH / 2 + 0.95, f'−{rm:,}', fs=5.0, c=S.INK2, ha=ha_out,
                   bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='none'))
            G.text(ax, xa + sg * (ln + 1.0), y + BH / 2 - 1.05, f'{KEPT[t][i - 1]:.0f}% kept', fs=5.0, c=S.INK, ha=ha_out,
                   bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='none'))
    G.text(ax, (XL + XR) / 2, y + BH / 2 + 1.05, lab, fs=5.4, c=S.INK)
    G.text(ax, (XL + XR) / 2, y + BH / 2 - 1.3, sub, fs=5.5, c=S.INK2)          # v16 Pass-6: 5.5 pt -> scripts >= 4.5 pt
# set headers
yh = YROW[0] + BH + 3.6
ax.plot([XL - 15.5], [yh], marker='^', ms=3.2, color=T_C, ls=''); G.text(ax, XL - 13.8, yh, 'training set', fs=6.2, c=T_C, ha='left')
ax.plot([XR + 1.6], [yh], marker='o', ms=2.9, color=P_C, ls=''); G.text(ax, XR + 3.3, yh, 'prediction set', fs=6.2, c=P_C, ha='left')

# ---------------------------------------------------------------- convergence on the 80 compositions
TX0, TX1, TY0, TY1 = 20.0, 69.0, 1.4, 10.6
xT = XL - L(C['training']['certified']) / 2; xP = XR + L(C['prediction']['certified']) / 2
G.arrow(ax, xT, YROW[-1] - 0.2, TX0 + 10.5, TY1 + 0.3, color=T_C, lw=0.9, rad=0.0, head=5)
G.arrow(ax, xP, YROW[-1] - 0.2, TX1 - 10.5, TY1 + 0.3, color=P_C, lw=0.9, rad=0.0, head=5)
G.text(ax, 26.4, 15.6, f'{X["training"]["compositions"]} compositions', fs=5.0, c=T_C, ha='right')
G.text(ax, 62.6, 15.6, f'{X["prediction"]["compositions"]} compositions', fs=5.0, c=P_C, ha='left')
G.text(ax, (XL + XR) / 2, TY1 + 2.2, f'{N_COMP} certified compositions', fs=6.2, c=S.GREEN)
cells = [(N_TONLY, 'training\nset only', T_C, 10.5), (N_BOTH, 'in both sets', S.GREEN, 28.0), (N_PONLY, 'prediction\nset only', P_C, 10.5)]
x = TX0
for n, lab, col, w in cells:
    ax.add_patch(FancyBboxPatch((x, TY0), w, TY1 - TY0, boxstyle='round,pad=0,rounding_size=0.8', fc=col, ec='white', lw=0.8, zorder=3))
    G.text(ax, x + w / 2, TY0 + 6.6, f'{n}', fs=7.5, c='white')
    G.text(ax, x + w / 2, TY0 + 2.5, lab, fs=5.0, c='white')
    x += w

assert_scripts(fig)                                                   # v16 Pass-6: scripts >= 4.5 pt
probs = qa.check(fig, 'v14_fig6_cascade')
S.save(fig, OUT / 'v14_fig6_cascade', width_mm=89)
assert not probs, probs
print(out); print(X)
