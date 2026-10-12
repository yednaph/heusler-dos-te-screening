"""Fig. 7 (v14) -- the 80 certified compositions in context (data figure, paper-figures skill).

Question: what did certification return, and what is already known about it?
5-second takeaway: the best computed p- and n-type power factors sit on a short Pareto layer next to the established half-Heusler
thermoelectrics; most certified compositions have only computational evidence, and an experimentally reported orthorhombic phase is
concentrated on the Si, Ge and As Z-sites (none on Sn, Sb or Bi).

Built from scripts/figures/fig7_candidates.py (v13, panels d,e) -- the rank-vs-PF panels moved to Fig. 4.
Inputs (existing results only):
  results/paper_analysis/ranking_fixed_n.xlsx  sheets n1e21_p_900K, n1e21_n_900K (one row per certified composition; SOC values for Z>=72)
  results/paper_analysis/paper_dataset.pkl     certified entries per set (76 T / 79 P -> 80 compositions = 74 + 5 + 1)
  literature/literature_status.csv              evidence class and orthorhombic flag of the 80 compositions
  figures/data/known_hh_check.csv               the 11 established half-Heusler thermoelectrics (10 certified)
Outputs: figures/v14_fig7_candidates.pdf/.png; figures/data/v14_fig7_maps.csv, v14_fig7_grid.csv, v14_fig7_zsite.csv
v16 (V16_R2 sec. 2d item 8 and sec. 4): labels must sit nearest their own square (VRhGe was placed beside another square); a predicted
orthorhombic phase is a dashed outline in a,b as in d; NbOsSb (DXMag OsNbSb; both certified entries, training and prediction set) is
marked with a dagger -- certified on a scalar-relativistic gap that closes with SOC (analysis/soc_paired.csv); tile text 5.5 pt.
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.lines import Line2D

SKILL = Path(__file__).resolve().parents[4]   # repository root, which holds figkit/
sys.path.insert(0, str(SKILL))
from figkit import style as S, qa                                                # noqa: E402
from common import dft_runs, ANALYSIS, DATA, OUT, PF_BAR, RECORDS                # noqa: E402  (read-only import)

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
ROOT = Path(__file__).resolve().parents[2]                                       # paper/manuscript_v19

# ------------------------------------------------------------------ evidence classes: one green sequential family (palette_check OK)
EV_ORDER = ['established', 'experimentally investigated', 'computationally investigated', 'underexplored']
EV_COLOR = {'established': '#16452A', 'experimentally investigated': '#3F8F55',
            'computationally investigated': '#A3CFAC', 'underexplored': '#FFFFFF'}
EV_LABEL = {'established': 'established', 'experimentally investigated': 'measured',
            'computationally investigated': 'computed only', 'underexplored': 'no study found'}
EV_EDGE = '#3F8F55'                       # outline of the white 'no study' marker so it stays visible
ORTH_EDGE = S.INK                         # orthorhombic flag = dark outline (solid: reported, dashed: predicted)
PRED_LS = (0, (1.6, 1.0))                 # v16: dash of a predicted-phase outline on the 3.9 pt map squares

# ------------------------------------------------------------------ certified entries -> compositions (assert the brief's counts)
D = dft_runs()
cert = D[D.certified == True]                                                     # noqa: E712
n_T, n_P = int((cert.data_set == 'training').sum()), int((cert.data_set == 'prediction').sum())
prov = cert.groupby('composition').data_set.apply(lambda x: frozenset(x))
n_both = int(sum(1 for s in prov if s == {'training', 'prediction'}))
n_Ponly = int(sum(1 for s in prov if s == {'prediction'})); n_Tonly = int(sum(1 for s in prov if s == {'training'}))
assert (n_T, n_P, len(cert)) == (76, 79, 155), (n_T, n_P)
assert (len(prov), n_both, n_Ponly, n_Tonly) == (80, 74, 5, 1)

# ------------------------------------------------------------------ literature status (80 rows) and composition parsing
LIT = pd.read_csv(ROOT / 'literature/literature_status.csv').set_index('composition')
LIT['orth'] = LIT.polymorph.fillna('').astype(str)                               # '', 'exp.', 'pred.'
assert len(LIT) == 80 and set(LIT.index) == set(prov.index)
assert LIT['class'].value_counts().to_dict() == {'computationally investigated': 60, 'established': 10,
                                                 'experimentally investigated': 9, 'underexplored': 1}
assert (LIT.orth == 'exp.').sum() == 17 and (LIT.orth == 'pred.').sum() == 6      # v17 citation audit: +NbRhGe, -ZrCoAs; +VIrGe, ScNiAs, ScPdAs, NbRuAs
VAL = {'Sc': 3, 'Y': 3, 'Ti': 4, 'Zr': 4, 'Hf': 4, 'V': 5, 'Nb': 5, 'Ta': 5, 'Fe': 8, 'Ru': 8, 'Os': 8, 'Co': 9, 'Rh': 9, 'Ir': 9,
       'Ni': 10, 'Pd': 10, 'Pt': 10, 'Au': 11, 'Al': 3, 'Ga': 3, 'In': 3, 'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 4, 'As': 5, 'Sb': 5, 'Bi': 5}
parts = {c: re.findall(r'([A-Z][a-z]?)(\d*)', c) for c in LIT.index}            # DXMag order X(2) Y Z, e.g. CoNbSn, Fe2TiSi
LIT['X'] = [parts[c][0][0] + parts[c][0][1] for c in LIT.index]
LIT['Y'] = [parts[c][1][0] for c in LIT.index]
LIT['Z'] = [parts[c][-1][0] for c in LIT.index]                                  # Z-site = last element
LIT['conv'] = LIT.conventional_name.str.replace(r'\s*\(.*\)', '', regex=True)       # conventional name, e.g. NbCoSn, Fe2TiSi
assert all(LIT.conv[c].endswith(LIT.Z[c]) for c in LIT.index)
LIT['vec'] = [sum(VAL[e] * (int(n) if n else 1) for e, n in parts[c]) for c in LIT.index]
assert LIT.vec.value_counts().to_dict() == {18: 75, 24: 5}

# Z-site pattern of orthorhombic competition (V14_R1 sec. 2.3, H8)
ZS = LIT.groupby('Z').agg(n=('orth', 'size'), exp=('orth', lambda s: int((s == 'exp.').sum())),
                          pred=('orth', lambda s: int((s == 'pred.').sum())))
assert tuple(ZS.loc['Si']) == (9, 6, 0) and tuple(ZS.loc['Ge']) == (12, 9, 2) and tuple(ZS.loc['As']) == (13, 2, 3)
assert ZS.loc[['Sn', 'Sb', 'Bi'], 'n'].sum() == 35 and ZS.loc[['Sn', 'Sb', 'Bi'], ['exp', 'pred']].values.sum() == 0
si_half = LIT[(LIT.Z == 'Si') & (LIT.orth == 'exp.')]
assert (si_half.vec == 18).all() and len(si_half) == 6                           # all six Si flags are half-Heuslers

# ------------------------------------------------------------------ NbOsSb: certified on a scalar gap that closes with SOC (V16_R2 2d.8)
SOCP = pd.read_csv(ROOT / 'analysis/soc_paired.csv')
DAG = LIT.index[LIT.conv == 'NbOsSb'][0]                                         # DXMag OsNbSb
nb = SOCP[SOCP.composition == DAG]
assert DAG == 'OsNbSb' and len(nb) == 2 and set(nb.data_set) == {'training', 'prediction'} and nb.certified.all()
assert (nb.dft_class == 'gapped').all() and (nb.soc_class == 'metal').all() and (nb.soc_gap == 0).all()
assert np.allclose(nb.gap, 0.259, atol=5e-4)                                     # scalar QE-PBE gap 0.26 eV -> 0 with SOC
# ------------------------------------------------------------------ established thermoelectrics (numbered on the maps)
known = pd.read_csv(RECORDS / 'known_hh_check.csv')
assert len(known) == 11
known = known[known.outcome == 'certified'].reset_index(drop=True)
assert len(known) == 10 and set(known.composition) == set(LIT.index[LIT['class'] == 'established'])
NUM_ORDER = ['ZrNiSn', 'TiNiSn', 'NbFeSb', 'TaFeSb', 'VFeSb', 'ZrCoBi', 'ZrCoSb', 'NbCoSn', 'HfCoSb', 'TiCoSb']   # v13 numbering
NUM = {known.set_index('name').loc[nm, 'composition']: (i + 1, nm) for i, nm in enumerate(NUM_ORDER)}

# ------------------------------------------------------------------ figure geometry: 183 x 95 mm
FIG_H = 96
fig = plt.figure(figsize=(S.W2, FIG_H * S.MM))
ax_p = fig.add_axes([0.062, 0.545, 0.425, 0.43])
ax_n = fig.add_axes([0.567, 0.545, 0.418, 0.43])
ax_g = fig.add_axes([0.255, 0.004, 0.745, 0.355])
ax_c = fig.add_axes([0.075, 0.245, 0.17, 0.085])


def obstacles_px(ax, xs, ys, size_pt):
    """Marker discs in display pixels (centre, radius) for the label placer."""
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    r = size_pt / 2 * fig.dpi / 72
    return [(x, y, r) for x, y in pts]


def seg_points_px(ax, xs, ys, n=60):
    """Dense sample of a polyline in display pixels (to keep labels off the Pareto and PF0 lines)."""
    out = []
    for (x0, y0), (x1, y1) in zip(zip(xs[:-1], ys[:-1]), zip(xs[1:], ys[1:])):
        t = np.linspace(0, 1, n)
        out.append(ax.transData.transform(np.column_stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0)])))
    return np.vstack(out) if out else np.zeros((0, 2))


def place_labels(ax, items, discs, linepts, fontsize, color=S.INK, pad=1.2, placed=None, leader_relaxed=None):
    """Greedy label placement: for each (text, x, y) try offsets around the point and keep the first whose text box clears every
    marker disc, every line sample, every placed label and the axes; offsets beyond 7 pt get a hairline leader that crosses no marker
    and no line (relaxed pass: it may cross the dashed PF0 reference line, given in leader_relaxed as the lines it must still avoid)."""
    r = fig.canvas.get_renderer(); placed = [] if placed is None else placed
    axbb = ax.get_window_extent(renderer=r)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1), (2, 1), (-2, 1), (2, -1), (-2, -1),
            (0.5, 1), (-0.5, 1), (0.5, -1), (-0.5, -1)]
    out = []
    for txt, x, y in items:
        best = None
        for strict, rad in [(True, r_) for r_ in (3.5, 5.5, 8, 11, 15, 20, 26)] + [(False, r_) for r_ in (8, 11, 15, 20, 26)]:
            for dx, dy in dirs:
                nrm = np.hypot(dx, dy); ox, oy = rad * dx / nrm, rad * dy / nrm
                ha = 'left' if dx > 0.3 else ('right' if dx < -0.3 else 'center')
                va = 'bottom' if dy > 0.3 else ('top' if dy < -0.3 else 'center')
                a = ax.annotate(txt, (x, y), xytext=(ox, oy), textcoords='offset points', fontsize=fontsize, ha=ha, va=va, color=color,
                                zorder=8)
                bb = a.get_window_extent(renderer=r).expanded(1.0, 1.0)
                x0, y0, x1, y1 = bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad
                ok = x0 > axbb.x0 + 1 and x1 < axbb.x1 - 1 and y0 > axbb.y0 + 1 and y1 < axbb.y1 - 1
                if ok:
                    for cx, cy, cr in discs:
                        qx, qy = min(max(cx, x0), x1), min(max(cy, y0), y1)
                        if (qx - cx) ** 2 + (qy - cy) ** 2 < cr ** 2:
                            ok = False; break
                if ok and len(linepts):
                    inside = (linepts[:, 0] > x0) & (linepts[:, 0] < x1) & (linepts[:, 1] > y0) & (linepts[:, 1] < y1)
                    ok = not inside.any()
                if ok:
                    for p in placed:
                        if not (x1 < p.x0 or x0 > p.x1 or y1 < p.y0 or y0 > p.y1):
                            ok = False; break
                if ok and rad <= 7:                                                  # v16: no leader -> the own square must be
                    px, py = ax.transData.transform((x, y))                            # clearly the nearest one to the label box
                    dist = lambda cx, cy: np.hypot(min(max(cx, x0), x1) - cx, min(max(cy, y0), y1) - cy)
                    d_own = dist(px, py)
                    d_oth = [dist(cx, cy) for cx, cy, _ in discs if abs(cx - px) > 0.5 or abs(cy - py) > 0.5]
                    ok = not d_oth or min(d_oth) > d_own + 4.0 * fig.dpi / 72
                if ok and rad > 7:                                                   # the leader must not cross a line or a marker
                    px, py = ax.transData.transform((x, y))
                    qx, qy = min(max(px, x0), x1), min(max(py, y0), y1)
                    t = np.linspace(0.15, 1, 40)[:, None]
                    seg = np.hstack([px + t * (qx - px), py + t * (qy - py)])
                    lpl = linepts if (strict or leader_relaxed is None) else leader_relaxed
                    if len(lpl):
                        dmin = np.min(np.hypot(seg[:, :1] - lpl[None, :, 0], seg[:, 1:] - lpl[None, :, 1]))
                        ok = dmin > 1.5
                    if ok:
                        for cx, cy, cr in discs:
                            if abs(cx - px) < 0.5 and abs(cy - py) < 0.5:
                                continue
                            if np.min(np.hypot(seg[:, 0] - cx, seg[:, 1] - cy)) < cr:
                                ok = False; break
                if ok:
                    best = (a, bb, rad, ox, oy); break
                a.remove()
            if best:
                break
        assert best is not None, f'no free position for label {txt}'
        a, bb, rad, ox, oy = best
        if rad > 7:                                                                # hairline leader for distant labels (own Line2D,
            px, py = ax.transData.transform((x, y))                                   # so the text extent stays the text only)
            qx, qy = min(max(px, bb.x0), bb.x1), min(max(py, bb.y0), bb.y1)          # nearest point of the label box
            d = np.hypot(qx - px, qy - py); sh = 3.2 * fig.dpi / 72                   # start just outside the marker
            if d > sh + 1:
                p0 = ax.transData.inverted().transform((px + (qx - px) * sh / d, py + (qy - py) * sh / d))
                p1 = ax.transData.inverted().transform((qx, qy))
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=S.INK2, lw=0.35, zorder=7)
        placed.append(bb); out.append(a)
    return out


# ------------------------------------------------------------------ a, b  |S|-PF/tau maps, 900 K, |n| <= 1e21 cm^-3
MS = 15                                                                            # marker area (pt^2), squares = one composition
map_rows, pareto = [], {}
for ax, sheet, lab, letter, xl in [(ax_p, 'n1e21_p_900K', 'p-type', 'a', (80, 342)), (ax_n, 'n1e21_n_900K', 'n-type', 'b', (70, 300))]:
    rk = pd.read_excel(ANALYSIS / 'ranking_fixed_n.xlsx', sheet_name=sheet)
    assert len(rk) == 80 and not rk.wrong_sign.astype(bool).any() and set(rk.composition) == set(LIT.index)
    rk['S'] = rk.S_uVK.abs(); rk['PF12'] = rk.PF / 1e12
    rk['ev'] = rk.composition.map(LIT['class']); rk['orth'] = rk.composition.map(LIT.orth)
    ax.set_xlim(*xl)
    ylim = (0.2, 1.95) if lab == 'p-type' else (0.25, 1.48)
    ax.set_ylim(*ylim)
    # PF0 bar (declared PF/tau threshold) and first Pareto layer
    ax.axhline(PF_BAR / 1e12, color=S.INK2, lw=0.6, ls=(0, (3.2, 1.6)), zorder=1)
    fr = rk[rk.tier == 1].sort_values('S')
    ax.plot(fr.S, fr.PF12, color=S.INK, lw=0.9, zorder=2, solid_capstyle='round')
    pareto[lab] = list(fr.composition)
    # markers: light classes first so the dark established squares sit on top; flagged ones outlined
    for ev in ['computationally investigated', 'underexplored', 'experimentally investigated', 'established']:
        for flag in ('', 'pred.', 'exp.'):
            h = rk[(rk.ev == ev) & (rk.orth == flag)]
            if not len(h):
                continue
            ec = ORTH_EDGE if flag else ('white' if ev != 'underexplored' else EV_EDGE)
            ax.scatter(h.S, h.PF12, s=MS, marker='s', facecolor=EV_COLOR[ev], edgecolor=ec, lw=0.85 if flag else 0.4,
                       linestyle=PRED_LS if flag == 'pred.' else '-',              # v16: predicted phase dashed, as in d
                       zorder=4 if ev in ('established', 'experimentally investigated') else 3)
    for _, r_ in rk.iterrows():
        map_rows.append((lab, r_.composition, r_.S, r_.PF12, int(r_.tier), r_.ev, r_.orth, NUM.get(r_.composition, ('', ''))[0]))
    ax.set_xlabel(f'computed |S|, {lab} (μV K$^{{-1}}$)', labelpad=1)
    ax.set_ylabel(r'computed PF/$\tau$ ($10^{12}$ W m$^{-1}$K$^{-2}$s$^{-1}$)', labelpad=1.5)
    ax.text(0.015, 0.985, f'900 K, |n| ≤ 10$^{{21}}$ cm$^{{-3}}$, {lab}', transform=ax.transAxes, fontsize=S.FS['small'], va='top',
            ha='left', color=S.INK2)
    ax.text(xl[1] - 0.006 * (xl[1] - xl[0]), PF_BAR / 1e12 - 0.012 * (ylim[1] - ylim[0]), 'PF$_0$', ha='right', va='top',
            fontsize=S.FS['small'], color=S.INK2)
    S.panel(ax, letter, x=-0.06, y=0.97)                                         # Pass-6: right of the longer y-label, above the ticks

    # labels: Pareto layer by name, established by number; nothing else
    fig.canvas.draw()
    discs = obstacles_px(ax, rk.S.values, rk.PF12.values, np.sqrt(MS) + 1.0)
    lp = np.vstack([seg_points_px(ax, fr.S.values, fr.PF12.values, n=200),
                    seg_points_px(ax, np.array(xl, float), np.array([PF_BAR / 1e12] * 2), n=1500)])
    r = fig.canvas.get_renderer()
    fixed = [t.get_window_extent(renderer=r) for t in ax.texts]                     # condition note, PF0 tag, panel letter
    placed = list(fixed)
    place_labels(ax, [(LIT.conv[c], x, y) for c, x, y in zip(fr.composition, fr.S, fr.PF12)], discs, lp, S.FS['small'] + 0.2,
                 placed=placed)
    kn = rk[rk.composition.isin(NUM) & (rk.tier != 1)].sort_values('PF12', ascending=False)   # a named Pareto member keeps its name
    place_labels(ax, [(str(NUM[c][0]), x, y) for c, x, y in zip(kn.composition, kn.S, kn.PF12)], discs, lp, S.FS['small'],
                 color=S.INK2, placed=placed, leader_relaxed=seg_points_px(ax, fr.S.values, fr.PF12.values, n=200))
    dg = rk[rk.composition == DAG]; assert len(dg) == 1                            # v16: NbOsSb, dagger (gap closes with SOC)
    place_labels(ax, [('NbOsSb†', dg.S.iloc[0], dg.PF12.iloc[0])], discs, lp, S.FS['small'], color=S.INK, placed=placed,
                 leader_relaxed=seg_points_px(ax, fr.S.values, fr.PF12.values, n=200))

assert len(pareto['p-type']) == 6 and len(pareto['n-type']) == 3
par_all = sorted(set(pareto['p-type']) | set(pareto['n-type']))
assert par_all == sorted(['CoNbGe', 'CoNbSn', 'CoTiAs', 'CoVSn', 'RhNbSn', 'RhVGe', 'CoHfAs', 'CoZrAs'])
par_flag = [c for c in par_all if LIT.loc[c, 'orth']]
assert sorted(par_flag) == ['CoHfAs', 'CoNbGe', 'RhVGe']                           # 3 of 8 Pareto compositions flagged (v17: ZrCoAs not counted)
pd.DataFrame(map_rows, columns=['carrier', 'composition', 'abs_S_uVK', 'PF_tau_1e12', 'pareto_tier', 'evidence', 'orthorhombic',
                                'established_no']).to_csv(DATA / 'v14_fig7_maps.csv', index=False)

# ------------------------------------------------------------------ d  composition grid: Z-site rows, one tile per composition
ROWS = [['Al', 'Ga', 'In'], ['Si', 'Ge', 'Sn', 'Pb'], ['As', 'Sb', 'Bi']]          # groups 13, 14, 15 of the periodic table
TW, TH, GAP = 1.0, 0.78, 0.45                                                       # tile width/height (data units), group gap
ycur, ypos = 0.0, {}
for grp in ROWS:
    for z in grp:
        ypos[z] = ycur; ycur += 1.0
    ycur += GAP
ymax = ycur - GAP
NCOL = int(ZS.n.max())
rank_ev = {k: i for i, k in enumerate(EV_ORDER)}
X_ORDER = ['Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir', 'Ni', 'Pd', 'Pt', 'Au', 'Fe2', 'Ru2']   # by group of the X element; full Heuslers last
grid_rows = []
for z, yy in ypos.items():
    sub = LIT[LIT.Z == z].copy()
    sub['k1'] = sub['class'].map(rank_ev); sub['k2'] = (sub.orth == '').astype(int); sub['k3'] = sub.X.map(X_ORDER.index)
    sub = sub.sort_values(['k1', 'k2', 'k3', 'Y'])
    for i, (comp, row) in enumerate(sub.iterrows()):
        x0 = i * TW
        flag = row.orth
        ec = ORTH_EDGE if flag else ('white' if row['class'] != 'underexplored' else EV_EDGE)
        ls = (0, (2.0, 1.2)) if flag == 'pred.' else '-'
        ax_g.add_patch(mp.Rectangle((x0 + 0.04, yy - TH / 2), TW - 0.08, TH, facecolor=EV_COLOR[row['class']], edgecolor=ec,
                                    lw=0.85 if flag else 0.4, ls=ls, zorder=2))
        xs = row.conv[:-len(z)].replace('2', '$_2$') + ('†' if comp == DAG else '')   # conventional name without Z; † = NbOsSb
        tc = 'white' if row['class'] in ('established', 'experimentally investigated') else S.INK
        ax_g.text(x0 + TW / 2, yy, xs, ha='center', va='center', fontsize=5.5, color=tc, zorder=3)       # v16: 5.5 pt
        grid_rows.append((comp, row.conv, row.X, row.Y, z, int(row.vec), row['class'], flag, i + 1))
    ax_g.text(-0.25, yy, z, ha='right', va='center', fontsize=S.FS['body'], color=S.INK)
    zr = ZS.loc[z]
    ax_g.text(NCOL + 0.35, yy, f"{zr.exp}{'+' + str(zr.pred) if zr.pred else ''} / {zr.n}", ha='left', va='center',
              fontsize=S.FS['small'], color=S.INK)
ax_g.text(NCOL + 0.35, -0.95, 'orthorhombic / all', ha='left', va='center', fontsize=S.FS['small'], color=S.INK2)
ax_g.text(-0.25, -0.95, 'Z', ha='right', va='center', fontsize=S.FS['small'], color=S.INK2)
ax_g.text(0.0, -0.95, 'one tile per certified composition, named without its Z element', ha='left', va='center',
          fontsize=S.FS['small'], color=S.INK2)
ax_g.set_xlim(-1.0, NCOL + 3.05); ax_g.set_ylim(ymax - 0.45, -1.35)
ax_g.axis('off'); ax_g.set_xticks([]); ax_g.set_yticks([])
fig.text(0.255, 0.355, 'd', fontsize=S.FS['panel'], fontweight='bold', va='top', ha='left', color=S.INK)
pd.DataFrame(grid_rows, columns=['composition', 'conventional_name', 'X', 'Y', 'Z', 'valence_electrons', 'evidence', 'orthorhombic', 'tile_position']) \
    .to_csv(DATA / 'v14_fig7_grid.csv', index=False)
ZS.assign(group=[next(i for i, g in enumerate(ROWS) if z in g) + 13 for z in ZS.index]).rename(
    columns={'n': 'certified_compositions', 'exp': 'orthorhombic_reported', 'pred': 'orthorhombic_predicted'}).to_csv(DATA / 'v14_fig7_zsite.csv')

# ------------------------------------------------------------------ shared key: one legend row under the maps, number key right of c
cnt = LIT['class'].value_counts()
h = [Line2D([], [], ls='', marker='s', ms=4.2, mfc=EV_COLOR[k], mec=EV_EDGE if k == 'underexplored' else 'white', mew=0.5,
            label=f'{EV_LABEL[k]} ({cnt[k]})') for k in EV_ORDER]
h += [Line2D([], [], ls='', marker='s', ms=4.2, mfc=EV_COLOR['computationally investigated'], mec=ORTH_EDGE, mew=0.85,
             label=f'orthorhombic phase reported ({int((LIT.orth == "exp.").sum())})'),
      mp.Patch(facecolor=EV_COLOR['computationally investigated'], edgecolor=ORTH_EDGE, lw=0.85, ls=(0, (2.0, 1.2)),
               label=f'predicted ({int((LIT.orth == "pred.").sum())})'),
      Line2D([], [], color=S.INK, lw=0.9, label='first Pareto layer'),
      Line2D([], [], color=S.INK2, lw=0.6, ls=(0, (3.2, 1.6)), label=r'PF$_0$')]
fig.legend(handles=h, loc='center', bbox_to_anchor=(0.5, 0.405), ncol=len(h), fontsize=S.FS['small'], handlelength=1.3,
           handletextpad=0.4, columnspacing=1.1, borderaxespad=0.0, frameon=False)

# ------------------------------------------------------------------ c  evidence classes: all 80 compositions and the 8 on the Pareto layer
lit_rows = []
for yy, (lab_, comps) in enumerate([(f'Pareto {len(par_all)}', par_all), (f'all {len(LIT)}', list(LIT.index))]):
    x0 = 0.0; N_ = len(comps)
    for k in EV_ORDER:
        sub = [c for c in comps if LIT.loc[c, 'class'] == k]
        if not sub:
            continue
        nf = sum(1 for c in sub if LIT.loc[c, 'orth'])                           # orthorhombic-flagged part of the segment
        w, wf = 100 * len(sub) / N_, 100 * nf / N_
        ax_c.barh(yy, w - wf, left=x0, height=0.62, color=EV_COLOR[k], edgecolor=EV_EDGE if k == 'underexplored' else 'white', lw=0.4)
        if nf:
            ax_c.barh(yy, wf, left=x0 + w - wf, height=0.62, color=EV_COLOR[k], edgecolor=ORTH_EDGE, lw=0.85)
        if w >= 9:
            ax_c.text(x0 + (w - wf) / 2 if (nf and w - wf >= 9) else x0 + w / 2, yy, str(len(sub)), ha='center', va='center',
                      fontsize=S.FS['small'], color='white' if k in ('established', 'experimentally investigated') else S.INK)
        lit_rows.append((lab_, k, len(sub), nf)); x0 += w
ax_c.set_xlim(0, 100); ax_c.set_ylim(-0.45, 1.45); ax_c.set_xticks([0, 50, 100])
ax_c.set_yticks([0, 1]); ax_c.set_yticklabels([r_[0] for r_ in lit_rows[:1]] + [f'all {len(LIT)}'], fontsize=S.FS['small'])
ax_c.tick_params(axis='y', length=0, pad=2)
ax_c.set_xlabel('share of compositions (%)', labelpad=1, fontsize=S.FS['small'])
for s_ in ('left', 'top', 'right'):
    ax_c.spines[s_].set_visible(False)
fig.text(0.003, 0.355, 'c', fontsize=S.FS['panel'], fontweight='bold', va='top', ha='left', color=S.INK)
lr = pd.DataFrame(lit_rows, columns=['group', 'evidence', 'n', 'n_orthorhombic_flag'])
assert lr[lr.group.str.startswith('Pareto')].set_index('evidence').n.to_dict() == {'established': 1, 'experimentally investigated': 1,
                                                                              'computationally investigated': 6}
lr.to_csv(DATA / 'v14_fig7_evidence.csv', index=False)
key = sorted(NUM.values())                                                      # established thermoelectrics, numbered in a, b
fig.text(0.012, 0.172, 'established, numbered in a, b', fontsize=S.FS['small'], color=S.INK2, va='top', ha='left')
fig.text(0.012, 0.145, '\n'.join(f'{n_}  {nm}' for n_, nm in key[:5]), fontsize=S.FS['small'], va='top', ha='left', linespacing=1.3)
fig.text(0.13, 0.145, '\n'.join(f'{n_}  {nm}' for n_, nm in key[5:]), fontsize=S.FS['small'], va='top', ha='left', linespacing=1.3)

assert_scripts(fig)                                                   # v16 Pass-6: scripts >= 4.5 pt
probs = qa.check(fig, 'v14_fig7_candidates')
assert not probs, probs
S.save(fig, OUT / 'v14_fig7_candidates', width_mm=183)
print('certified entries T/P', n_T, n_P, '-> compositions', len(prov), '(both/P only/T only)', n_both, n_Ponly, n_Tonly)
print('Z-site orthorhombic (exp, pred, n):'); print(ZS.to_string())
print('Pareto p:', pareto['p-type'], 'n:', pareto['n-type'], 'flagged:', par_flag)
