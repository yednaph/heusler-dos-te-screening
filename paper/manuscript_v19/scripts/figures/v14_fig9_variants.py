"""Fig. 9 (v14) -- higher-energy structural variants of calculated compositions (data figure, paper-figures skill; single column).

Question: does a higher-energy structural variant of a composition change its electronic class or its calculated power factor?
5-second takeaway: 814 of 824 variants keep the class of their composition's reference crystal and none is gapped where the reference
is not; power-factor ratios scatter widely but their median stays near 1 up to ~100 meV/atom.

Panel a is new (V14_R2 sec. 2.7: 4 x 4 class-transition matrix, reference -> variant, mirroring the SOC matrix of Fig. 8b).
Panel b is v13 fig8_soc_polymorph.py panel e, cleaned: no summary text over data, labels placed off markers, the one variant beyond
1000 meV/atom kept on the axis, class change shown by a heavy outline.
Inputs (existing results only): analysis/polymorph_variants.csv (polymorph_variants.py; reference = lowest-DXMag-energy calculated entry
of a composition, variant = another calculated entry >= 5 meV/atom above it).
Outputs: figures/v14_fig9_variants.pdf/.png; figures/data/v14_fig9_class_matrix.csv, v14_fig9_variant_median.csv, v14_fig9_nonmetal_pairs.csv
v16 (V16_R2 sec. 4): every composition label has a hairline leader to its own marker (VNiGa / VNiAl were ambiguous between amber and
teal markers), and the type is brought down to the other figures' size (~6 pt; it was 6.3-7 pt at 89 mm).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SKILL = Path(__file__).resolve().parents[4]   # repository root, which holds figkit/
sys.path.insert(0, str(SKILL))
from figkit import style as S, qa                                                # noqa: E402
from common import DATA, OUT                                                     # noqa: E402  (read-only import)

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
FS_N, FS_S = 6.0, 5.6                                                            # v16: numbers / axis labels 6 pt, keys and ticks 5.6 pt
mpl.rcParams.update({'axes.labelsize': FS_N, 'xtick.labelsize': FS_S, 'ytick.labelsize': FS_S, 'legend.fontsize': FS_S})
AN = Path(__file__).resolve().parents[2] / 'analysis'
v = pd.read_csv(AN / 'polymorph_variants.csv')
assert len(v) == 824 and v.composition.nunique() == 491
assert v.data_set.value_counts().to_dict() == {'prediction': 661, 'training': 163}
mm = (v.dft_class == 'metal') & (v.ref_dft_class == 'metal')                     # metallic in both crystals
nm = v[~mm].copy()                                                               # at least one crystal not metallic
assert int(mm.sum()) == 800 and len(nm) == 24

ORDER = ['gapped', 'narrow', 'half_metal', 'metal']
LAB = {'gapped': 'gapped', 'narrow': 'narrow', 'half_metal': 'half-metal', 'metal': 'metal'}
M = pd.crosstab(v.ref_dft_class, v.dft_class).reindex(index=ORDER, columns=ORDER, fill_value=0)
assert int(np.trace(M.values)) == 814 and round(np.trace(M.values) / 824, 3) == 0.988
assert M['gapped'].drop('gapped').sum() == 0                                     # no variant gapped where its reference is not
assert M.loc['gapped'].tolist() == [6, 1, 0, 2] and M.loc['metal'].tolist() == [0, 4, 1, 800]
M.to_csv(DATA / 'v14_fig9_class_matrix.csv')

fig = plt.figure(figsize=(S.W1, 96 * S.MM))
ax_m = fig.add_axes([0.235, 0.655, 0.36, 0.305])
ax_l = fig.add_axes([0.64, 0.64, 0.355, 0.33]); ax_l.axis('off'); ax_l.set_xticks([]); ax_l.set_yticks([])
ax = fig.add_axes([0.15, 0.1, 0.825, 0.42])

# ------------------------------------------------------------------ a  class of the reference crystal (rows) -> class of the variant
frac = M.div(M.sum(1), axis=0).values
ax_m.imshow(frac, cmap='Greys', vmin=0, vmax=1.6, aspect='auto')
for i in range(4):
    for j in range(4):
        ax_m.text(j, i, f'{M.values[i, j]}', ha='center', va='center', fontsize=FS_N,
                  color='white' if frac[i, j] > 0.55 else (S.INK if M.values[i, j] else S.INK2))
ax_m.set_xticks(range(4)); ax_m.set_xticklabels([LAB[k].replace('-', '-\n') for k in ORDER], fontsize=FS_S, linespacing=1.0)
ax_m.set_yticks(range(4)); ax_m.set_yticklabels([f'{LAB[k]} ({n})' for k, n in zip(ORDER, M.sum(1).values)], fontsize=FS_S)
ax_m.tick_params(length=0, pad=6.5)
for i, k in enumerate(ORDER):                                                  # class colour chip per row (= colours in b)
    ax_m.scatter([-0.62], [i], s=14, marker='o', color=S.CLASS_COLOR[k], edgecolor='white', lw=0.3, clip_on=False, zorder=5)
ax_m.set_xlabel(f'class, variant ({len(v)} variants)', labelpad=1.5)
ax_m.set_ylabel('class, reference crystal', labelpad=1)
for s_ in ax_m.spines.values():
    s_.set_visible(False)
fig.text(0.01, 0.975, 'a', fontsize=S.FS['panel'], fontweight='bold', va='top', ha='left', color=S.INK)

# ------------------------------------------------------------------ b  PF/tau ratio against energy above the reference crystal
XL, YL = (4.5, 4500), (1e-3, 1e3)
ax.scatter(v[mm].dE, v[mm].pf_ratio, s=2.2, marker='o', lw=0, color=S.MUTED, zorder=2, rasterized=False)
bins = np.array([5, 12, 25, 50, 100, 200, 1000]); mid, med, nb = [], [], []                 # v13 bins (manuscript medians)
for a_, b_ in zip(bins[:-1], bins[1:]):
    y = v[mm & (v.dE >= a_) & (v.dE < b_)].pf_ratio
    if len(y) >= 10:
        mid.append(np.sqrt(a_ * b_)); med.append(y.median()); nb.append(len(y))
ax.plot(mid, med, color=S.INK2, lw=0.9, marker='s', ms=2.0, zorder=3)
ax.axhline(1, color=S.INK2, lw=0.5, ls=':', zorder=1)
for t in ['prediction', 'training']:                                            # training triangles on top
    for _, r in nm[nm.data_set == t].iterrows():
        ch = r.dft_class != r.ref_dft_class
        ax.scatter(r.dE, r.pf_ratio, s=15 if ch else 12, marker=S.SET_MARKER[t], facecolor=S.CLASS_COLOR[r.ref_dft_class],
                   edgecolor=S.INK if ch else 'white', lw=0.9 if ch else 0.4, zorder=6 if t == 'training' else 5)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(*XL); ax.set_ylim(*YL)
ax.set_xlabel(r'energy above the reference crystal (meV atom$^{-1}$)', labelpad=1)
ax.set_ylabel(r'best PF/$\tau$, variant / reference', labelpad=1)
fig.text(0.01, 0.535, 'b', fontsize=S.FS['panel'], fontweight='bold', va='top', ha='left', color=S.INK)

# labels: the three certified compositions with a variant, and the three gapped references whose variant changes class
LABELS = ['IrTaSn', 'NiVGa', 'PdScAs', 'NiVAl', 'La2IrAs', 'LaCdSb']
LIT = pd.read_csv(Path(__file__).resolve().parents[2] / 'literature/literature_status.csv').set_index('composition')
CONV = LIT.conventional_name.str.replace(r'\s*\(.*\)', '', regex=True).to_dict()   # conventional names exist for the 80 certified only
CONV['NiVAl'] = 'VNiAl'                                   # conventional order (X Y Z, tetrahedral-site element second); not certified
cert_var = sorted(set(nm.composition[nm.ref_certified.astype(bool)]))
assert cert_var == ['IrTaSn', 'NiVGa', 'PdScAs'] and all(c in CONV for c in cert_var)
fig.canvas.draw(); r_ = fig.canvas.get_renderer()
allpts = ax.transData.transform(np.column_stack([v.dE, v.pf_ratio]))
is_nm = (~mm).values
discs = [(x, y, (4.0 if c else 1.9) * fig.dpi / 150) for (x, y), c in zip(allpts, is_nm)]
med_pts = ax.transData.transform(np.column_stack([np.repeat(mid, 1), med]))
linepts = []
for (x0, y0), (x1, y1) in zip(med_pts[:-1], med_pts[1:]):
    t_ = np.linspace(0, 1, 80)[:, None]; linepts.append(np.hstack([x0 + t_ * (x1 - x0), y0 + t_ * (y1 - y0)]))
x0_, x1_ = ax.transData.transform((XL[0], 1))[0], ax.transData.transform((XL[1], 1))[0]
y1p = ax.transData.transform((1, 1))[1]
linepts.append(np.column_stack([np.linspace(x0_, x1_, 1500), np.full(1500, y1p)]))
linepts = np.vstack(linepts)
axbb = ax.get_window_extent(renderer=r_)
placed = [t.get_window_extent(renderer=r_) for t in ax.texts]
lab_rows = nm[nm.composition.isin(LABELS)].drop_duplicates('composition')
assert sorted(lab_rows.composition) == sorted(LABELS)
dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1), (2, 1), (-2, 1), (2, -1), (-2, -1),
        (0.5, 1), (-0.5, 1), (0.5, -1), (-0.5, -1)]
leaders = []                                                                     # v16: drawn leaders (display px), never crossed


def crosses(a0, a1, b0, b1):
    """True if segment a0-a1 properly intersects segment b0-b1 (display coordinates)."""
    o = lambda p, q, r: np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return o(a0, a1, b0) * o(a0, a1, b1) < 0 and o(b0, b1, a0) * o(b0, b1, a1) < 0


for _, row in lab_rows.sort_values('pf_ratio').iterrows():
    txt = CONV.get(row.composition, row.composition).replace('La2', 'La$_2$')   # TaIrSn, VNiGa, ScPdAs; others DXMag order
    px, py = ax.transData.transform((row.dE, row.pf_ratio)); done = False
    for strict, rad in [(True, r) for r in (9, 11, 13, 15, 18, 22, 26, 33)] + [(False, r) for r in (9, 11, 15, 20, 26)]:   # v16: rad > 7 -> leader
        for dx, dy in dirs:
            nrm = np.hypot(dx, dy); ox, oy = rad * dx / nrm, rad * dy / nrm
            ha = 'left' if dx > 0.3 else ('right' if dx < -0.3 else 'center')
            va = 'bottom' if dy > 0.3 else ('top' if dy < -0.3 else 'center')
            a = ax.annotate(txt, (row.dE, row.pf_ratio), xytext=(ox, oy), textcoords='offset points', fontsize=FS_S,
                            ha=ha, va=va, color=S.INK, zorder=8)
            bb = a.get_window_extent(renderer=r_); X0, Y0, X1, Y1 = bb.x0 - 1.2, bb.y0 - 1.2, bb.x1 + 1.2, bb.y1 + 1.2
            ok = X0 > axbb.x0 + 1 and X1 < axbb.x1 - 1 and Y0 > axbb.y0 + 1 and Y1 < axbb.y1 - 1
            if ok:                                                               # text box clear of every point and line
                qx = np.clip(allpts[:, 0], X0, X1); qy = np.clip(allpts[:, 1], Y0, Y1)
                hit = np.hypot(qx - allpts[:, 0], qy - allpts[:, 1]) < np.array([d[2] for d in discs])
                ok = not (hit if strict else hit & is_nm).any()                  # relaxed pass: may sit on grey points (masked)
            if ok:
                ok = not ((linepts[:, 0] > X0) & (linepts[:, 0] < X1) & (linepts[:, 1] > Y0) & (linepts[:, 1] < Y1)).any()
            if ok:
                ok = all(X1 < p.x0 or X0 > p.x1 or Y1 < p.y0 or Y0 > p.y1 for p in placed)
            if ok and rad > 7:                                                   # leader clear of coloured markers
                qx, qy = min(max(px, X0), X1), min(max(py, Y0), Y1)
                tt = np.linspace(0.2, 1, 40)[:, None]; seg = np.hstack([px + tt * (qx - px), py + tt * (qy - py)])
                for (cx, cy, cr), c in zip(discs, is_nm):
                    if c and not (abs(cx - px) < 0.5 and abs(cy - py) < 0.5) and np.min(np.hypot(seg[:, 0] - cx, seg[:, 1] - cy)) < cr:
                        ok = False; break
                if ok:                                                           # v16: no crossing of an earlier leader or label
                    ok = not any(crosses((px, py), (qx, qy), l0, l1) for l0, l1 in leaders)
                if ok:
                    ok = not any((pb.x0 - 1 < seg[:, 0]).any() and ((pb.x0 - 1 < seg[:, 0]) & (seg[:, 0] < pb.x1 + 1) &
                                 (pb.y0 - 1 < seg[:, 1]) & (seg[:, 1] < pb.y1 + 1)).any() for pb in placed)
            if ok:
                if not strict:                                                   # white mask over the grey background points
                    a.set_bbox(dict(boxstyle='square,pad=0.12', fc='white', ec='none'))
                if rad > 7:
                    d_ = np.hypot(qx - px, qy - py); sh = 2.3 * fig.dpi / 72   # v16: leader touches the marker rim
                    p0 = ax.transData.inverted().transform((px + (qx - px) * sh / d_, py + (qy - py) * sh / d_))
                    p1 = ax.transData.inverted().transform((qx, qy))
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=S.INK2, lw=0.35, zorder=7)
                    leaders.append(((px, py), (qx, qy)))
                placed.append(bb); done = True; break
            a.remove()
        if done:
            break
    assert done, f'no free position for {txt}'

# ------------------------------------------------------------------ key for b (right of a)
h = [Line2D([], [], ls='', marker='o', mfc=S.CLASS_COLOR[k], mec='white', mew=0.3, ms=3.6, label=f'reference {LAB[k]}')
     for k in ORDER]
h += [Line2D([], [], ls='', marker='o', mfc=S.MUTED, mec='none', ms=2.0, label='metal in both'),
      Line2D([], [], color=S.INK2, lw=0.9, marker='s', ms=2.0, label='median, metal in both'),
      Line2D([], [], ls='', marker='o', mfc='white', mec=S.INK, mew=0.9, ms=3.8, label='class changes'),
      Line2D([], [], ls='', marker='^', mfc=S.INK2, mec='white', mew=0.3, ms=3.8, label=S.SET_LABEL['training']),
      Line2D([], [], ls='', marker='o', mfc=S.INK2, mec='white', mew=0.3, ms=3.6, label=S.SET_LABEL['prediction'])]
ax_l.legend(handles=h, loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=FS_S, handletextpad=0.35, borderaxespad=0.0,
            labelspacing=0.42, handlelength=1.4)

assert_scripts(fig)                                                   # v16 Pass-6: scripts >= 4.5 pt
probs = qa.check(fig, 'v14_fig9_variants')
assert not probs, probs
S.save(fig, OUT / 'v14_fig9_variants', width_mm=89)
pd.DataFrame({'dE_bin_mid_meV': mid, 'median_pf_ratio_metal_pairs': med, 'n': nb}).to_csv(DATA / 'v14_fig9_variant_median.csv', index=False)
nm[['composition', 'data_set', 'ref_dft_class', 'dft_class', 'dE', 'pf_ratio', 'ref_certified']].to_csv(DATA / 'v14_fig9_nonmetal_pairs.csv',
                                                                                                       index=False)
print('medians', [round(m, 2) for m in med], nb)
assert [round(m, 2) for m in med][-2:] == [0.54, 0.33]                           # Sec. r-variants: 0.54 and 0.33 in the two highest bins
print('beyond 1000 meV/atom (drawn, not binned):', v[v.dE >= 1000][['composition', 'dE', 'pf_ratio']].to_dict('records'))
