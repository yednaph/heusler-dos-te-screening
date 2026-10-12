"""v14 Fig. 3 -- predicted spectra reproduce shape but not gaps (paper-figures skill, data figure).

Question: how faithful are the predicted d-DOS on held-out training-set spectra, and where do they fail?
5-second takeaway: typical spectra (median r 0.911) follow the DFT d-DOS closely and beat composition baselines in every family, yet
the model flags only 778 of the 5,130 gapped spectra, and spectra with a missed gap score as high as correct ones.

Inputs (existing results only; no inference):
  DOS_EXPORT/runtime_assets/heldout_dumps/<family>/fold<k>/test_{predictions,targets}.npy + test_ids.txt (best-validation checkpoints)
  DOS_EXPORT/metadata/metrics/<family>/fold<k>/per_sample_metrics.csv (pearson_r, fermi_pearson_2eV, true_gapped, pred_gapped)
  code/te_rescreen.py:classify  (the screen's own label for panels a,b)
  baselines 0.488 / 0.501 / 0.605: Table 1 caption (si/supplementary.tex), from paper_rebuild/scripts/audit_20261009/ml_audit
Panels: a ZrNiSn, b Co2MnSi (selection as in fig2_representation.py); c entries nearest the 10/30/50/70/90th percentile of r;
d per-family r with baselines; e held-out gap-flag confusion matrix; f ECDF of r within +-2 eV by confusion cell.
v16 (V16_R2 sec. 4, colour roles): a,b shade the d-DOS gap the label code itself measures (contiguous range around E_F where the
SPIN-SUMMED DFT d-DOS < 0.15, code/te_rescreen.py:classify walk), so the shaded width equals the printed gap (0.72 eV for ZrNiSn), in a
neutral tint instead of the gapped-class teal; d baselines dark grey instead of the training-set orange; e,f confusion cells in a neutral
navy/grey scheme instead of the gapped teal and half-metal purple.
"""
import importlib.util
import re
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
from common import REPO, EXPORT, DATA, OUT, ROOT, E_GRID, FAMILIES, per_sample   # noqa: E402  (read-only use)

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
TER = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(TER)

# ------------------------------------------------------------------ data and asserts
ps = per_sample()
assert len(ps) == 51257 and ps.material_id.nunique() == 51257
assert round(ps.pearson_r.mean(), 3) == 0.895 and round(ps.pearson_r.median(), 3) == 0.911
# per-family mean r of the fidelity table must match the per-sample file; the table is written by
# paper/manuscript_v1/scripts/figures/make_tables.py (release layout) or sits in the manuscript sources
_tab = next((q for q in (ROOT / 'tables/tab1_data_ml.tex', REPO / 'paper/manuscript_v1/tables/tab1_data_ml.tex') if q.exists()), None)
if _tab is None:                                                       # neither present: say so instead of failing
    print('note: tab1_data_ml.tex not found (run paper/manuscript_v1/scripts/figures/make_tables.py); table check skipped')
else:
    tab = _tab.read_text()
    for fam in FAMILIES:
        row = re.search(rf'^{fam.replace("_", " ")} &[^&]*& ([0-9.]+) &', tab, re.M)
        assert row and abs(float(row.group(1)) - ps.loc[ps.family == fam, 'pearson_r'].mean()) < 5e-4, fam
BASE = {'family-mean d-DOS': 0.488, 'nearest composition': 0.501, 'mean of five nearest': 0.605}
_si = ROOT / 'si/supplementary.tex'                                   # manuscript source, not shipped with the code
if _si.exists():                                                      # cross-check the baselines against the SI text
    res_txt = _si.read_text()
    assert all(f'{v:.3f}' in res_txt for v in BASE.values())
else:                                                                 # release copy: the baselines are taken as stated
    print('note: si/supplementary.tex not present; baseline values not cross-checked against the SI text')

CELL = {(True, True): 'gapped, detected', (True, False): 'gapped, missed', (False, True): 'not gapped, flagged',
        (False, False): 'not gapped, not flagged'}
ps['cell'] = [CELL[(bool(a), bool(b))] for a, b in zip(ps.true_gapped, ps.pred_gapped)]
cnt = ps.cell.value_counts()
assert (cnt['gapped, detected'], cnt['gapped, missed'], cnt['not gapped, flagged'], cnt['not gapped, not flagged']) == (778, 4352, 651, 45476)
assert ps.fermi_pearson_2eV.notna().all()
CELL_C = {'gapped, detected': S.NAVY, 'gapped, missed': '#3A4652',         # v16: no class or set colour (line / outline colour)
          'not gapped, flagged': '#8A96A3', 'not gapped, not flagged': '#8A96A3'}
CELL_FILL = {'gapped, detected': S.TINT[S.NAVY], 'gapped, missed': '#DDE3EA', 'not gapped, flagged': '#FFFFFF',
             'not gapped, not flagged': '#EDF1F6'}
CELL_TXT = {'gapped, detected': S.NAVY, 'gapped, missed': '#3A4652', 'not gapped, flagged': S.INK2, 'not gapped, not flagged': S.INK2}
pd.DataFrame({'cell': list(CELL.values()), 'n': [cnt[c] for c in CELL.values()],
              'median_r': [ps.loc[ps.cell == c, 'pearson_r'].median() for c in CELL.values()],
              'median_r_pm2eV': [ps.loc[ps.cell == c, 'fermi_pearson_2eV'].median() for c in CELL.values()]}).to_csv(
    DATA / 'v14_fig3_confusion.csv', index=False)


def ordinal(v):
    n = int(round(v))
    return f'{n}' + ('th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th'))


def spectrum(fam, fold, mid):
    """Real held-out pair (prediction of the best-validation checkpoint of the fold in which the entry was a test entry)."""
    dd = EXPORT / 'runtime_assets/heldout_dumps' / fam / fold
    j = [l.strip() for l in open(dd / 'test_ids.txt')].index(mid)
    P = np.asarray(np.load(dd / 'test_predictions.npy', mmap_mode='r')[j]); T = np.asarray(np.load(dd / 'test_targets.npy', mmap_mode='r')[j])
    r = float(ps.loc[ps.material_id == mid, 'pearson_r'].iloc[0])
    assert abs(np.corrcoef(P, T)[0, 1] - r) < 1e-4, mid                # dump row and metrics row are the same spectrum
    return T, P, r


def tex_formula(f):
    return re.sub(r'(\d+)', r'$_{\1}$', f)


# a, b: the two examples kept from the current Fig. 2 (same entries as fig2_representation.py)
EX = [('half_cubic', 'fold4', 'NiZrSn_0eeb5376-6970-44bf-af09-0f50d7670343', 'ZrNiSn'),
      ('full_cubic', 'fold4', 'Co2MnSi_de46805a-bd68-446f-963c-c23c60fbf357', 'Co$_2$MnSi')]
ex_rows, examples = [], []
for fam, fold, mid, name in EX:
    T, P, r = spectrum(fam, fold, mid)
    pct = 100 * float((ps.pearson_r < r).mean())
    lab = {}
    for which, X in [('DFT', T), ('model', P)]:
        et, gw, _, _, pol = TER.classify(np.asarray(X[:2001]), np.asarray(X[2001:]))
        lab[which] = (et, gw, pol)
        ex_rows.append((name, mid, which, et, gw, pol, r, pct))
    examples.append(dict(name=name, T=T, P=P, r=r, pct=pct, lab=lab))
assert examples[0]['lab']['DFT'][0] == 'semiconductor' and examples[0]['lab']['model'][0] == 'semiconductor'
assert examples[1]['lab']['DFT'][0] == 'half_metal' and examples[1]['lab']['model'][0] == 'metal'
pd.DataFrame(ex_rows, columns=['example', 'material_id', 'spectrum', 'screen_label', 'dDOS_gap_eV', 'spin_pol', 'pearson_r',
                               'percentile']).to_csv(DATA / 'v14_fig3_examples.csv', index=False)

# c: entries nearest each percentile of per-spectrum r (declared rule; ties resolved by first occurrence)
QS = [10, 30, 50, 70, 90]
quant = []
for q in QS:
    v = np.percentile(ps.pearson_r, q); row = ps.loc[(ps.pearson_r - v).abs().idxmin()]
    T, P, r = spectrum(row.family, row.fold, row.material_id)
    quant.append(dict(q=q, target=v, row=row, T=T, P=P, r=r))
assert [x['row'].formula for x in quant] == ['CdRuPb', 'V2PtSn', 'Pt2VGa', 'Nb2OsGa', 'Mn2In']
pd.DataFrame([dict(percentile=x['q'], r_percentile=x['target'], material_id=x['row'].material_id, formula=x['row'].formula,
                   family=x['row'].family, fold=x['row'].fold, pearson_r=x['r'], true_gapped=x['row'].true_gapped,
                   pred_gapped=x['row'].pred_gapped) for x in quant]).to_csv(DATA / 'v14_fig3_quantiles.csv', index=False)

fam_stats = ps.groupby('family').pearson_r.agg(n='size', mean='mean', median='median', below_0p4=lambda v: int((v < 0.4).sum()))
fam_stats.to_csv(DATA / 'v14_fig3_family_r.csv')
pd.Series(BASE).to_csv(DATA / 'v14_fig3_baselines.csv')

# ------------------------------------------------------------------ figure (axes placed on a millimetre grid)
WMM, HMM = 183, 100
fig = plt.figure(figsize=(WMM * S.MM, HMM * S.MM))


def axmm(x, y, w, h):
    return fig.add_axes([x / WMM, y / HMM, w / WMM, h / HMM])


def letter(x, y, s):
    fig.text(x / WMM, y / HMM, s, fontsize=S.FS['panel'], fontweight='bold', ha='left', va='center')


DFT_C, PRED_C = '#C9D2DC', S.SET_COLOR['prediction']
LABEL_STRIP, GAP_TINT = '#D9CFC1', '#E6EBF0'                     # v16: neutral gap tint (teal = gapped class)
up, dn = slice(0, 2001), slice(2001, 4002)
EG = (E_GRID >= -3) & (E_GRID <= 3)


def label_gap_span(T, gap_thr=0.15):
    """The d-DOS gap of the d-DOS gap label: the contiguous run of bins around E_F where the SPIN-SUMMED d-DOS < gap_thr, found
    exactly as code/te_rescreen.py:classify walks it (start at E_F, or at the nearest qualifying bin within +-3 bins).
    Returns (E_lo, E_hi) of the bin edges, so E_hi - E_lo = number of bins x DE = the printed gap; None if there is no run."""
    g = T[up] + T[dn]; below = g < gap_thr; ef = TER.EF_IDX; start = ef
    if not below[ef]:
        win = np.arange(ef - 3, ef + 4); cand = win[below[win]]
        if len(cand):
            start = int(cand[np.argmin(np.abs(cand - ef))])
    if not below[start]:
        return None
    lo = hi = start
    while lo > 0 and below[lo - 1]: lo -= 1
    while hi < TER.NE - 1 and below[hi + 1]: hi += 1
    return E_GRID[lo] - TER.DE / 2, E_GRID[hi] + TER.DE / 2


def draw_pair(ax, T, P, lw=0.8, shade=None):
    """Mirrored spin-resolved d-DOS: DFT fill, model line; shade = (E_lo, E_hi) of the label's d-DOS gap (both spins), None (no gap,
    label window only) or False (no annotation at all)."""
    if shade is not None and shade is not False:                      # False = quantile strip (no label annotations)
        ax.axvspan(*shade, color=GAP_TINT, lw=0, zorder=0)
    if shade is not False:
        ax.axvspan(-0.03, 0.03, color=LABEL_STRIP, lw=0, zorder=0.5)
    for ch, sgn in [(up, 1), (dn, -1)]:
        ax.fill_between(E_GRID[EG], 0, sgn * T[ch][EG], color=DFT_C, lw=0, zorder=1)
        ax.plot(E_GRID[EG], sgn * P[ch][EG], color=PRED_C, lw=lw, zorder=3)
    ax.axhline(0, color=S.INK, lw=0.4); ax.axvline(0, color=S.INK2, lw=0.5, ls='--', zorder=2)
    ym = 1.08 * max(np.abs(T[up][EG]).max(), np.abs(T[dn][EG]).max(), P[up][EG].max(), P[dn][EG].max())
    ax.set_xlim(-3, 3); ax.set_ylim(-ym, ym)


def label_txt(lab):
    et, gw, pol = lab
    return et.replace('_', '-') + (f' ({gw:.2f} eV)' if et == 'semiconductor' else f' (|P| {abs(pol):.2f})')


# a, b -------------------------------------------------------------
for k, x in enumerate(examples):
    x0 = 13.0 + k * 90.0
    ax = axmm(x0, 68.5, 78.0, 22.0)
    et, gw, _ = x['lab']['DFT']
    span = label_gap_span(x['T']) if et == 'semiconductor' else None    # the label reports a gap only for 'semiconductor'
    if span is not None:
        assert abs((span[1] - span[0]) - gw) < 1e-9, (x['name'], span, gw)   # shaded width = printed d-DOS gap
    x['span'] = span
    draw_pair(ax, x['T'], x['P'], shade=span)
    ax.set_xlabel(r'$E-E_\mathrm{F}$ (eV)', labelpad=0.5, fontsize=6.2)
    ax.tick_params(labelsize=5.6, pad=1.2)
    if k == 0:
        ax.set_ylabel('d-DOS (archive units)', fontsize=5.8, labelpad=1)
    ax.text(0.0, 1.17, x['name'], transform=ax.transAxes, va='bottom', ha='left', fontsize=6.4)
    ax.text(1.0, 1.17, f'held-out r {x["r"]:.3f} ({ordinal(x["pct"])} percentile)', transform=ax.transAxes, va='bottom', ha='right',
            fontsize=5.4, color=S.INK2)
    ax.text(0.0, 1.03, f'd-DOS gap label: DFT {label_txt(x["lab"]["DFT"])}; model {label_txt(x["lab"]["model"])}',   # v16 Pass-6
            transform=ax.transAxes, va='bottom', ha='left', fontsize=5.3, color=S.INK2)
    ax.text(0.012, 0.95, 'spin ↑', transform=ax.transAxes, ha='left', va='top', fontsize=5.2, color=S.INK2)
    ax.text(0.012, 0.05, 'spin ↓', transform=ax.transAxes, ha='left', va='bottom', fontsize=5.2, color=S.INK2)
    letter(x0 - (11.0 if k == 0 else 7.0), 96.6, 'ab'[k])
    if k == 0:
        h = [mp.Patch(color=DFT_C, label='DFT d-DOS'), Line2D([], [], color=PRED_C, lw=0.8, label='model, held out'),
             mp.Patch(color=GAP_TINT, label='d-DOS gap (spin sum < 0.15)'), mp.Patch(color=LABEL_STRIP, label='label window ±0.03 eV')]
        ax.legend(handles=h, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=5.0, frameon=True, facecolor='white',
                  edgecolor='none', framealpha=1, handlelength=1.1, borderaxespad=0.15, labelspacing=0.25, ncol=1)

# c: quantile strip -----------------------------------------------------
letter(2.0, 59.2, 'c')
QW, QG = 31.4, 3.0
for i, x in enumerate(quant):
    x0 = 13.0 + i * (QW + QG)
    ax = axmm(x0, 44.0, QW, 12.5)
    draw_pair(ax, x['T'], x['P'], lw=0.6, shade=False)
    ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_xticks([-2, 0, 2]); ax.tick_params(labelsize=5.2, pad=1.0, length=1.8)
    row = x['row']
    tail = ', gap missed' if (row.true_gapped and not row.pred_gapped) else ''
    ax.text(0.0, 1.04, f'{x["q"]}th: {tex_formula(row.formula)}, r {x["r"]:.3f}{tail}', transform=ax.transAxes, fontsize=5.5,   # v16 Pass-6: subscripts >= 4.5 pt
            va='bottom', ha='left', color=S.INK)
    if i == 0:
        ax.text(-0.06, 0.5, 'spin ↑ / ↓', transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=5.2, color=S.INK2)
fig.text((13.0 + 5 * QW + 4 * QG) / WMM, 38.6 / HMM, r'$E-E_\mathrm{F}$ (eV); each spectrum scaled to its own maximum', ha='right',
         va='center', fontsize=5.5, color=S.INK2)                  # v16 Pass-6: E_F subscript >= 4.5 pt

# d: per-family r with baselines ----------------------------------------
ax = axmm(13.0, 10.0, 52.0, 24.0)
letter(2.0, 36.6, 'd')
pos = np.arange(len(FAMILIES))
vals = [ps.loc[ps.family == f, 'pearson_r'].values for f in FAMILIES]
vp = ax.violinplot([v[v >= 0.4] for v in vals], positions=pos, widths=0.8, showextrema=False, bw_method=0.12)
for b in vp['bodies']:
    b.set_facecolor(S.TINT[S.NAVY]); b.set_edgecolor(PRED_C); b.set_linewidth(0.6); b.set_alpha(1)
for p_, v in zip(pos, vals):
    ax.plot([p_ - 0.22, p_ + 0.22], [np.median(v)] * 2, color=PRED_C, lw=1.0, zorder=4)
    ax.plot([p_], [v.mean()], marker='o', ms=2.2, color=S.INK, zorder=5)
    ax.text(p_ + 0.08, 0.418, f'{int((v < 0.4).sum())}', ha='left', va='bottom', fontsize=5.0, color=S.INK2)
BSTY = {'family-mean d-DOS': (0, (1.2, 1.2)), 'nearest composition': (0, (3, 1.5)), 'mean of five nearest': '-'}
BASE_C = '#5A6672'                                                     # v16: dark grey (orange is the training-set colour)
for key, v in BASE.items():
    ax.axhline(v, color=BASE_C, lw=0.7, ls=BSTY[key], zorder=2)
ax.set_xlim(-0.6, len(FAMILIES) - 0.4); ax.set_ylim(0.4, 1.0)
ax.set_xticks(pos); ax.set_xticklabels([f.replace('_', '\n') for f in FAMILIES], fontsize=5.4, linespacing=1.0)
ax.tick_params(axis='x', length=0, pad=1.5); ax.tick_params(axis='y', labelsize=5.6, pad=1.2)
ax.set_ylabel('per-spectrum r', fontsize=6.2, labelpad=1)
# baseline labels in a right-hand gutter (0.488 and 0.501 are 0.013 apart, so they are stacked with leaders)
GL = [('five nearest comp. 0.605', 0.605, 0.640), ('nearest comp. 0.501', 0.501, 0.535), ('family mean 0.488', 0.488, 0.462)]
for s_, yv, yt in GL:
    ax.annotate(s_, xy=(len(FAMILIES) - 0.4, yv), xytext=(len(FAMILIES) - 0.25, yt), textcoords='data', fontsize=5.0, color=BASE_C,
                va='center', ha='left', annotation_clip=False,
                arrowprops=dict(arrowstyle='-', color=BASE_C, lw=0.4, shrinkA=0, shrinkB=0))
ax.text(len(FAMILIES) - 0.25, 0.94, 'baselines,', fontsize=5.0, color=BASE_C, va='center', ha='left', transform=ax.transData,
        clip_on=False)
ax.text(len(FAMILIES) - 0.25, 0.895, 'all families', fontsize=5.0, color=BASE_C, va='center', ha='left', clip_on=False)
ax.text(len(FAMILIES) - 0.25, 0.80, '● mean, — median', fontsize=5.0, color=S.INK2, va='center', ha='left', clip_on=False)
ax.text(len(FAMILIES) - 0.25, 0.755, 'numbers: r < 0.4', fontsize=5.0, color=S.INK2, va='center', ha='left', clip_on=False)

# e: confusion matrix ----------------------------------------------------
ax = axmm(103.0, 10.0, 25.0, 24.0)
letter(91.0, 36.6, 'e')
M = [['gapped, detected', 'gapped, missed'], ['not gapped, flagged', 'not gapped, not flagged']]
TXT = [['detected', 'missed'], ['false flag', 'correct']]
for i in range(2):
    rs = sum(cnt[c] for c in M[i])
    for jj in range(2):
        c = M[i][jj]
        ax.add_patch(mp.FancyBboxPatch((jj + 0.05, 1 - i + 0.05), 0.9, 0.9, boxstyle='round,pad=0,rounding_size=0.06',
                                       fc=CELL_FILL[c], ec=CELL_C[c], lw=0.9))
        ax.text(jj + 0.5, 1 - i + 0.68, f'{cnt[c]:,}', ha='center', va='center', fontsize=6.6, color=S.INK)
        ax.text(jj + 0.5, 1 - i + 0.43, TXT[i][jj], ha='center', va='center', fontsize=5.0, color=CELL_TXT[c])
        ax.text(jj + 0.5, 1 - i + 0.22, f'{100 * cnt[c] / rs:.1f}% of row', ha='center', va='center', fontsize=5.0, color=S.INK2)
ax.set_xlim(0, 2); ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(['flagged', 'not flagged'], fontsize=5.4)
ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(['gapped', 'not\ngapped'], fontsize=5.4, linespacing=1.0)
ax.set_xlabel('model gap flag', fontsize=5.8, labelpad=1); ax.set_ylabel('DFT gap flag', fontsize=5.8, labelpad=1)
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.tick_params(length=0, pad=1.2)

# f: ECDF of r within +-2 eV by confusion cell ---------------------------
ax = axmm(140.0, 10.0, 41.0, 24.0)
letter(130.5, 36.6, 'f')
STY = {'not gapped, flagged': (0.9, (0, (3, 1.5))), 'gapped, detected': (0.9, '-'), 'not gapped, not flagged': (1.4, '-'),
       'gapped, missed': (1.4, '-')}
med2 = ps.groupby('cell').fermi_pearson_2eV.median()
for key in ['not gapped, flagged', 'gapped, detected', 'not gapped, not flagged', 'gapped, missed']:
    v = np.sort(ps.loc[ps.cell == key, 'fermi_pearson_2eV'].values); lw_, ls_ = STY[key]
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=CELL_C[key], lw=lw_, ls=ls_, label=f'{TXT[0][0] if False else key} ({med2[key]:.2f})')
ax.set_xlim(-0.7, 1.0); ax.set_ylim(0, 1.02)
ax.set_xticks([-0.5, 0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
ax.tick_params(labelsize=5.6, pad=1.2)
ax.set_xlabel(r'r within $\pm$2 eV of $E_\mathrm{F}$', fontsize=5.8, labelpad=1)
ax.set_ylabel('cumulative fraction', fontsize=5.8, labelpad=1)
ax.legend(loc='upper left', fontsize=5.0, handlelength=1.4, borderaxespad=0.1, labelspacing=0.25, title='cell (median r)',
          title_fontsize=5.0, alignment='left')
med2.to_csv(DATA / 'v14_fig3_ecdf_medians.csv')
assert abs(med2['gapped, missed'] - 0.889) < 5e-4 and abs(med2['not gapped, not flagged'] - 0.866) < 5e-4

assert_scripts(fig)                                                   # v16 Pass-6: scripts >= 4.5 pt
probs = qa.check(fig, 'v14_fig3_fidelity')
assert not probs, probs
S.save(fig, OUT / 'v14_fig3_fidelity', width_mm=183)
print(pd.read_csv(DATA / 'v14_fig3_examples.csv').to_string(index=False))
print(pd.read_csv(DATA / 'v14_fig3_quantiles.csv').to_string(index=False))
print(fam_stats.to_string()); print(med2.to_string())
