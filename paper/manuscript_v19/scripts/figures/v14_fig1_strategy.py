"""v14 Fig. 1 -- strategy and the two populations (183 x 100 mm).

Question: what is searched, and by which route does each population reach first-principles transport?
5-second takeaway: two d-DOS sources (DFT for the training set, ALIGNN for the prediction set) feed one screen; 3,625
QE-PBE + BoltzTraP2 runs judge it; the two sets converge on 80 certified compositions.

  a  one-row pipeline of real objects (half-Heusler cell -> real ZrNiSn d-DOS, DFT and held-out ALIGNN -> Cutler-Mott S(mu) of
     those spectra -> QE-PBE + BoltzTraP2 -> criteria -> 80 compositions), with paired training/prediction counts on two lanes that
     run under the stages (absorbs the screening half of the v13 funnel)
  b  the materials space: entries per family, mirrored by set and split by state
  c  energy of each prediction entry above the training entry of the same family and composition

Stage table (counts verified below against results files and asserted):
  DXMag entries          -> split by DOS availability        -> 51,257 T / 54,978 P entries   (fig2v9_entries.csv, screen CSVs)
  d-DOS                  -> DFT archive (T) / ALIGNN mean (P) -> same entries
  screen                 -> d-DOS gap label ('semiconductor') -> 1,550 / 2,394 entries         (results/screening/*.csv)
                         -> d-DOS gap 0.10-1.40 eV            -> 671 / 603 entries
  QE-PBE + BoltzTraP2    -> score-ordered selection rounds    -> 1,562 / 2,021 runs (+42 not attributable = 3,625)  (paper_dataset.pkl)
  criteria (i)-(v)       -> certification                     -> 76 / 79 entries
  union                  -> distinct compositions             -> 80 (74 both, 5 P only, 1 T only)
The calculated runs are not a subset of the gap window: the selection rounds also sampled other labels (stated on the figure).
No new DFT, training or inference: the ZrNiSn spectra are the stored held-out target and prediction of DOS_EXPORT.
v16 (V16_R2 sec. 2e/4): the purple footnote is ink grey (purple = half-metal class); the label station reads "d-DOS gap label" (the
text's term); panel c marks 1 meV/atom (15.1% of all prediction entries lie below it).
v16 Pass-6: the 1 meV/atom line is NOT the near-duplicate definition (code/paper/dedup_twins.py: same composition, family and space
group, |dE| < 5 meV/atom and |dV|/V < 1%), so it is labelled by its count only; the 74/5/1 split left the outcome tile (Fig. 5 carries
it); sub/superscripts in a and c >= 4.5 pt (text 5.5 pt, SHRINK_FACTOR 0.82; log ticks of c as plain numbers).
"""
import sys
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon

SKILL = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SKILL))
from figkit import style as S, glyphs as G, qa                       # noqa: E402
from common import REPO, EXPORT, DATA, OUT, RECORDS, dft_runs, screen_pools    # noqa: E402  (read-only use)

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
T_L, P_L = '#E9B877', '#9DBBDA'                                       # light tints of the set colours (state split in b)

# ================================================================ counts (every number drawn in a)
d = dft_runs()                                                        # 3,625 QE-PBE + BoltzTraP2 runs
pools = screen_pools()                                                # one row per DXMag entry, per set
WINDOW = ['0.10-0.40', '0.40-1.00', '1.00-1.40']                      # the 0.10-1.40 eV d-DOS gap range
C = {}
for t in ['training', 'prediction']:
    g = pools[pools.pool == t]                                        # screened entries of this set
    sc = g[g.etype == 'semiconductor']                                # semiconductor-like d-DOS label
    r = d[d.data_set == t]                                            # runs attributable to this set
    C[t] = dict(entries=len(g), label=len(sc), window=int(sc.gap_tier.isin(WINDOW).sum()), runs=len(r),
                certified=int((r.certified == True).sum()), cert_comp=int(r[r.certified == True].composition.nunique()))
cert = d[d.certified == True]
CT = set(cert[cert.data_set == 'training'].composition); CP = set(cert[cert.data_set == 'prediction'].composition)
N_COMP, N_BOTH, N_PONLY, N_TONLY = len(CT | CP), len(CT & CP), len(CP - CT), len(CT - CP)
N_AMB = int((d.data_set == 'ambiguous').sum())
# guards: a data change must fail the build
assert len(d) == 3625 == C['training']['runs'] + C['prediction']['runs'] + N_AMB and N_AMB == 42
assert (C['training']['entries'], C['prediction']['entries']) == (51257, 54978)
assert (C['training']['label'], C['prediction']['label']) == (1550, 2394)
assert (C['training']['window'], C['prediction']['window']) == (671, 603)
assert (C['training']['runs'], C['prediction']['runs']) == (1562, 2021)
assert (C['training']['certified'], C['prediction']['certified']) == (76, 79)
assert (N_COMP, N_BOTH, N_PONLY, N_TONLY) == (80, 74, 5, 1)
assert int(d[d.data_set == 'ambiguous'].certified.sum()) == 0
fc = pd.read_csv(RECORDS / 'fig3_funnel_counts.csv', index_col=0)      # cross-check against the v13 funnel record
for k_new, k_old in [('entries', 'screened'), ('label', 'semi'), ('window', 'prime'), ('runs', 'runs'), ('certified', 'cert')]:
    assert all(C[t][k_new] == fc.loc[k_old, t] for t in ['training', 'prediction']), k_new
rows = [(k, C['training'][k], C['prediction'][k]) for k in ['entries', 'label', 'window', 'runs', 'certified', 'cert_comp']]
rows += [('runs_not_attributable', N_AMB, N_AMB), ('compositions_union', N_COMP, N_COMP), ('compositions_both', N_BOTH, N_BOTH),
         ('compositions_prediction_only', N_PONLY, N_PONLY), ('compositions_training_only', N_TONLY, N_TONLY)]
pd.DataFrame(rows, columns=['stage', 'training', 'prediction']).to_csv(DATA / 'v14_fig1_pipeline.csv', index=False)

# ================================================================ ZrNiSn spectra (stored held-out target + prediction)
_sp = importlib.util.spec_from_file_location('ter', REPO / 'code/te_rescreen.py')   # the screen's own descriptor code
TER = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(TER)
dd = EXPORT / 'runtime_assets/heldout_dumps/half_cubic/fold4'
MID = 'NiZrSn_0eeb5376-6970-44bf-af09-0f50d7670343'
j = [l.strip() for l in open(dd / 'test_ids.txt')].index(MID)
SP_T = np.array(np.load(dd / 'test_targets.npy', mmap_mode='r')[j])   # DFT d-DOS, 2 x 2,001 values
SP_P = np.array(np.load(dd / 'test_predictions.npy', mmap_mode='r')[j])
EG = TER.E_GRID; up, dn = slice(0, 2001), slice(2001, 4002)


def s_of_mu(g):
    """Cutler-Mott S(mu) at 900 K of a total d-DOS, exactly as the screen computes it (te_rescreen.py)."""
    out = []
    for m_ in TER.MU_GRID:
        w = TER.fermi_window(EG - m_); gw = g * w
        out.append(-1e6 * ((gw * (EG - m_)).sum() / gw.sum()) / TER.T_OP)
    return np.array(out)


S_T, S_P = s_of_mu(SP_T[up] + SP_T[dn]), s_of_mu(SP_P[up] + SP_P[dn])
A_T = TER.transport_descriptors(SP_T[up] + SP_T[dn])['A_seebeck']
row_z = d[d.run_dir == 'NiZrSn__0eeb5376'].iloc[0]
assert abs(A_T - row_z.SCREEN_A_seebeck_uVK) < 5                     # thumbnail = the stored training-set indicator

# ================================================================ figure: one canvas in mm + data axes placed in mm
W, H = 183, 100
fig = plt.figure(figsize=(W * S.MM, H * S.MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')


def mm_axes(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def rbox(x, y, w, h, fc, ec='none', lw=0.8, r=1.0, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, zorder=z))


def pill(x, y, w, s, role, h=3.8, fs=5.3):
    """Calculation step (regular weight: no bold inside graphics)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={h / 2}', fc='white', ec=role, lw=0.7, zorder=4))
    G.text(ax, x + w / 2, y + h / 2, s, fs=fs, c=role)
    return x + w


def head(x, s, role):
    G.text(ax, x, 97.6, s, fs=6.0, c=role)


YO0, YO1 = 66.0, 95.0                                                 # object row
# ---------------------------------------------------------------- 0  database: DXMag count + a real half-Heusler cell (ZrNiSn type)
head(13.5, 'DXMag database', S.NAVY)
rbox(3, 85.5, 21, 9.5, S.NAVY)
G.text(ax, 13.5, 91.9, '106,235', fs=8.5, c='white'); G.text(ax, 13.5, 87.6, 'Heusler entries', fs=5.4, c='white')
# oblique projection of the conventional cubic cell; Z (Sn) fcc, X (Zr) octahedral, Y (Ni) half the tetrahedral sites
O, A, KX, KY = (3.4, 67.6), 10.4, 0.42 * np.cos(np.radians(35)), 0.42 * np.sin(np.radians(35))
P = lambda x, y, z: (O[0] + A * (x + KX * y), O[1] + A * (z + KY * y))
for (a0, a1) in [((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 1, 0)), ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 1, 0)), ((1, 0, 0), (1, 0, 1)),
                 ((0, 1, 0), (1, 1, 0)), ((0, 1, 0), (0, 1, 1)), ((0, 0, 1), (1, 0, 1)), ((0, 0, 1), (0, 1, 1)), ((1, 1, 0), (1, 1, 1)),
                 ((1, 0, 1), (1, 1, 1)), ((0, 1, 1), (1, 1, 1))]:
    (x0, y0), (x1, y1) = P(*a0), P(*a1)
    ax.plot([x0, x1], [y0, y1], color=S.INK2, lw=0.4, zorder=3)
fcc = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
       (.5, .5, 0), (.5, .5, 1), (.5, 0, .5), (.5, 1, .5), (0, .5, .5), (1, .5, .5)]
octa = [(.5, 0, 0), (.5, 1, 0), (.5, 0, 1), (.5, 1, 1), (0, .5, 0), (1, .5, 0), (0, .5, 1), (1, .5, 1),
        (0, 0, .5), (1, 0, .5), (0, 1, .5), (1, 1, .5), (.5, .5, .5)]
tetra = [(.25, .25, .25), (.75, .75, .25), (.75, .25, .75), (.25, .75, .75)]
atoms = [(p, S.NAVY, 0.95) for p in fcc] + [(p, '#8FB0D0', 0.85) for p in octa] + [(p, 'white', 0.75) for p in tetra]
for (p, col, rad) in sorted(atoms, key=lambda a: -a[0][1]):          # back to front
    ax.add_patch(Circle(P(*p), rad * (1 - 0.18 * p[1]), fc=col, ec=S.NAVY, lw=0.35, zorder=4 + 1 - p[1]))
G.text(ax, 21.4, 80.6, 'Sn', fs=5.0, c=S.NAVY, ha='left'); G.text(ax, 21.4, 77.3, 'Zr', fs=5.0, c='#5E86AE', ha='left')
G.text(ax, 21.4, 74.0, 'Ni', fs=5.0, c=S.INK2, ha='left')
ax.add_patch(Circle((20.2, 80.6), 0.7, fc=S.NAVY, ec=S.NAVY, lw=0.35, zorder=5))
ax.add_patch(Circle((20.2, 77.3), 0.7, fc='#8FB0D0', ec=S.NAVY, lw=0.35, zorder=5))
ax.add_patch(Circle((20.2, 74.0), 0.7, fc='white', ec=S.NAVY, lw=0.35, zorder=5))

# ---------------------------------------------------------------- 1  d-DOS: DFT archive (training set) and ALIGNN prediction
head(51.5, 'spin-resolved d-DOS', S.NAVY)
XT0, XT1 = 40.0, 63.0


def dos_thumb(y0, spec, col, label, ref=None):
    a = mm_axes(XT0, y0, XT1 - XT0, 11.0)
    m = (EG >= -3) & (EG <= 3)
    for ch, sg in [(up, 1), (dn, -1)]:
        if ref is not None:
            a.fill_between(EG[m], 0, sg * ref[ch][m], color='#DDE3EA', lw=0, zorder=1)   # opaque light grey: prediction line stays legible
        else:
            a.fill_between(EG[m], 0, sg * spec[ch][m], color='#F0D8B8', lw=0, zorder=1)   # opaque tint = orange at 0.28 over white (no transparency)
        a.plot(EG[m], sg * spec[ch][m], color=col, lw=0.7, zorder=3)
    a.axhline(0, color=S.INK2, lw=0.3); a.axvline(0, color=S.INK, lw=0.45, ls=':')
    ym = 1.05 * max(np.abs(SP_T[ch][m]).max() for ch in (up, dn))         # same scale in both thumbnails (DFT maximum)
    a.set_xlim(-3, 3); a.set_ylim(-ym, ym); a.set_xticks([]); a.set_yticks([])
    for s_ in a.spines.values(): s_.set_visible(False)
    a.patch.set_alpha(0)
    G.text(ax, XT0 + 0.3, y0 + 12.2, label, fs=5.0, c=col, ha='left', va='center')
    return a


dos_thumb(82.2, SP_T, T_C, 'DFT d-DOS')
dos_thumb(66.6, SP_P, P_C, 'predicted d-DOS', ref=SP_T)
# ALIGNN model node: trained on the training-set spectra, predicts the prediction-set spectra
rbox(27.0, 70.6, 10.0, 4.6, S.NAVY, r=0.8)
G.text(ax, 32.0, 72.9, 'ALIGNN', fs=5.3, c='white')
G.arrow(ax, 40.6, 82.6, 34.0, 75.6, color=S.NAVY, lw=0.7, ls=(0, (2.4, 1.2)), head=5)
G.text(ax, 34.2, 79.6, 'train', fs=5.0, c=S.NAVY, ha='right')
G.arrow(ax, 37.0, 72.9, 39.8, 72.9, color=P_C, lw=0.9, head=5)
# routes out of the database, one per set
G.arrow(ax, 24.0, 90.2, 39.8, 90.2, color=T_C, lw=0.9, head=5); G.text(ax, 31.6, 92.0, 'training set', fs=5.0, c=T_C)
G.arrow(ax, 24.0, 72.9, 27.0, 72.9, color=P_C, lw=0.9, head=5)

# ---------------------------------------------------------------- 2  screen: Cutler-Mott S(mu) of those spectra + gap label
head(84.0, 'screen (each set separately)', S.BLUE)
a2 = mm_axes(73.0, 78.0, 25.0, 16.0)
a2.plot(TER.MU_GRID, S_T, color=T_C, lw=0.9)
a2.plot(TER.MU_GRID, S_P, color=P_C, lw=0.9, ls=(0, (3, 1.3)))
for Sv, c_, mk in [(S_T, T_C, '^'), (S_P, P_C, 'o')]:
    k = int(np.argmax(np.abs(Sv))); a2.plot([TER.MU_GRID[k]], [Sv[k]], marker=mk, ms=2.6, color=c_, ls='', zorder=5)
a2.axhline(0, color=S.INK2, lw=0.3); a2.axvline(0, color=S.INK2, lw=0.3, ls=':')
a2.set_xlim(-1.0, 1.0); a2.set_xticks([-1, 0, 1]); a2.set_yticks([-300, 0, 300])
a2.tick_params(labelsize=5.0, length=1.6, pad=1)
a2.set_xlabel(r'$\mu-E_F$ (eV)', fontsize=5.5, labelpad=0.5)               # v16 Pass-6: 5.5 pt -> scripts >= 4.5 pt
a2.set_ylabel(r'$S^{\mathrm{d}}$ (μV K$^{-1}$)', fontsize=5.5, labelpad=0.5)
pill(67.0, 67.4, 15.4, r'$S^{\mathrm{d}}_{\max}$, $P^{\mathrm{d}}_{\max}$', S.BLUE, fs=5.5)
pill(83.4, 67.4, 15.6, 'd-DOS gap label', S.BLUE, fs=5.0)
G.chevron(ax, 65.0, 75.5, S.ARROW, size=1.0)

# ---------------------------------------------------------------- 3  first-principles judge
head(120.0, 'first-principles judge', S.TEAL)
rbox(104.5, YO0, 31.0, YO1 - YO0 - 1.0, S.TINT[S.TEAL], ec=S.TEAL, lw=0.6)
xr = pill(106.3, 88.2, 10.8, 'QE-PBE', S.TEAL)
G.chevron(ax, xr + 0.9, 88.2 + 1.9, S.TEAL, size=0.8)
pill(xr + 1.8, 88.2, 14.6, 'BoltzTraP2', S.TEAL)
pill(106.3, 83.0, 27.4, 'CRTA, 300–1200 K', S.TEAL, fs=5.1)
pill(106.3, 77.8, 27.4, '+ SOC for $Z\\geq72$', S.TEAL, fs=5.1)
G.text(ax, 120.0, 72.4, '3,625 runs', fs=8.0, c=S.TEAL)
G.text(ax, 120.0, 68.6, f'incl. {N_AMB} not attributable to a set', fs=5.0, c=S.INK2)
G.chevron(ax, 102.6, 80.5, S.ARROW, size=1.0)

# ---------------------------------------------------------------- 4  certification criteria (final protocol)
head(150.5, 'criteria', S.GREEN)
crit = ['$|S|>60$ μV K$^{-1}$', 'PF/τ > PF$_0$', 'hull < 0.10 eV atom$^{-1}$', 'QE-PBE gap ≥ 0.136 eV', 'SOC PF/τ, $Z\\geq72$']
for i, s_ in enumerate(crit):
    yy = 91.9 - i * 5.2
    G.text(ax, 139.6, yy, ['i', 'ii', 'iii', 'iv', 'v'][i], fs=5.5, c=S.GREEN, ha='center')
    G.text(ax, 141.6, yy, s_, fs=5.5, c=S.INK, ha='left')                  # v16 Pass-6: 5.5 pt -> scripts >= 4.5 pt
ax.plot([138.0, 138.0], [YO0 + 2.0, 94.0], color=S.GREEN, lw=0.8, solid_capstyle='butt')
G.chevron(ax, 136.9, 80.5, S.ARROW, size=1.0)
G.chevron(ax, 163.9, 80.5, S.ARROW, size=1.0)

# ---------------------------------------------------------------- 5  outcome tile; both lanes end here
XF0, XF1 = 165.5, 181.5
rbox(XF0, 52.0, XF1 - XF0, YO1 - 52.0, S.GREEN)
xc = (XF0 + XF1) / 2
G.text(ax, xc, 80.5, f'{N_COMP}', fs=11.0, c='white')                     # v16 Pass-6: the 74/5/1 split is in Fig. 5 only
G.text(ax, xc, 74.4, 'certified\ncompositions', fs=5.4, c='white')

# ---------------------------------------------------------------- the two count lanes (paired counts on the connectors)
YT, YP, YL = 58.8, 54.4, 62.6                                         # training lane, prediction lane, stage-label row
for yl, col in [(YT, T_C), (YP, P_C)]:
    ax.plot([3.0, XF0 - 1.4], [yl, yl], color=col, lw=1.0, zorder=3, solid_capstyle='butt')
    G.chevron(ax, XF0 - 1.0, yl, col, size=1.0)
ax.plot([3.4], [YT], marker='^', ms=3.4, color=T_C, zorder=6); ax.plot([3.4], [YP], marker='o', ms=3.0, color=P_C, zorder=6)
BB = dict(boxstyle='square,pad=0.18', fc='white', ec='none')
stations = [(13.5, 'entries', 'entries'), (82.0 - 7.6, 'd-DOS gap\nlabel', 'label'), (90.6 + 0.4, 'gap 0.10–1.40 eV', 'window'),
            (120.0, 'calculated runs', 'runs'), (150.5, 'certified entries', 'certified')]
for x, lab, k in stations:
    G.text(ax, x, YL + (0.9 if '\n' in lab else 0), lab, fs=5.0, c=S.INK2)
    G.text(ax, x, YT, f'{C["training"][k]:,}', fs=6.2, c=T_C, bbox=BB)
    G.text(ax, x, YP, f'{C["prediction"][k]:,}', fs=6.2, c=P_C, bbox=BB)
G.text(ax, 46.0, YT, 'training set · DFT d-DOS', fs=5.0, c=T_C, bbox=BB)
G.text(ax, 46.0, YP, 'prediction set · ALIGNN d-DOS', fs=5.0, c=P_C, bbox=BB)
G.text(ax, 105.0, 50.7, 'runs also cover entries outside the gap range', fs=5.0, c=S.INK2)   # v16: ink grey, not purple
G.text(ax, 0.6, 99.0, 'a', fs=9, w='bold', ha='left', va='top')

# ================================================================ b  materials space: entries per family by set and state
dx = pd.read_csv(RECORDS / 'fig2v9_entries.csv'); dx['type'] = dx.family.str.split('_').str[0]
dx['gs'] = False; dx.loc[dx.groupby(['type', 'composition'])['energy (eV/atom)'].idxmin(), 'gs'] = True
dx['fammin'] = False; dx.loc[dx.groupby(['family', 'composition'])['energy (eV/atom)'].idxmin(), 'fammin'] = True
assert len(dx) == 106235 and dx.gs.sum() == 27864 and (dx[dx.set == 'training'].gs).sum() == 27860
assert (dx.set == 'training').sum() == 51257 and (dx.set == 'prediction').sum() == 54978
FAMS = ['full_cubic', 'full_tetra', 'half_cubic', 'half_tetra', 'inverse_cubic', 'inverse_tetra']
axb = mm_axes(19.0, 6.5, 92.0, 34.5)
popr = []
for i, f in enumerate(FAMS):
    y = len(FAMS) - 1 - i; t = dx[(dx.family == f) & (dx.set == 'training')]; p = dx[(dx.family == f) & (dx.set == 'prediction')]
    tg, tm = int(t.gs.sum()), int((~t.gs).sum()); pf, pm = int(p.fammin.sum()), int((~p.fammin).sum())
    axb.barh(y, -tg, color=T_C, height=0.72, lw=0); axb.barh(y, -tm, left=-tg, color=T_L, height=0.72, lw=0)
    axb.barh(y, pf, color=P_L, height=0.72, lw=0); axb.barh(y, pm, left=pf, color=P_C, height=0.72, lw=0)
    axb.text(-(tg + tm) - 250, y, f'{tg + tm:,}', ha='right', va='center', fontsize=5.0, color=T_C)
    axb.text(pf + pm + 250, y, f'{pf + pm:,}', ha='left', va='center', fontsize=5.0, color=P_C)
    popr.append((f, tg, tm, pf, pm))
pop = pd.DataFrame(popr, columns=['family', 'train_ground', 'train_metastable', 'pred_family_min', 'pred_metastable'])
ref = pd.read_csv(RECORDS / 'fig1v13_population.csv')                    # the v13 record of the same split
assert (pop.values == ref[pop.columns].values).all()
assert pop[['train_ground', 'train_metastable']].values.sum() == 51257 and pop[['pred_family_min', 'pred_metastable']].values.sum() == 54978
pop.to_csv(DATA / 'v14_fig1_population.csv', index=False)
axb.axvline(0, color=S.INK, lw=0.6)
axb.set_yticks(range(len(FAMS))); axb.set_yticklabels([f.replace('_', ' ') for f in FAMS[::-1]], fontsize=5.6)
axb.set_ylim(-0.5, 5.5)
axb.set_xlim(-17000, 17000); axb.set_xticks([-15000, -10000, -5000, 0, 5000, 10000, 15000])
axb.set_xticklabels(['15k', '10k', '5k', '0', '5k', '10k', '15k'], fontsize=5.4)
axb.set_xlabel('DXMag entries', labelpad=1, fontsize=6.0)
# direct labels on the segments (no floating legend)
axb.text(-1349 - 8047 / 2, 5, 'cubic minimum above\ntetragonal ground state', ha='center', va='center', fontsize=5.0, color=S.INK, linespacing=1.0)
axb.text(-8047 / 2, 4, 'ground state', ha='center', va='center', fontsize=5.0, color='white')
axb.text(908 + 12571 / 2, 4, 'other metastable configuration', ha='center', va='center', fontsize=5.0, color='white')
axb.text(11600, 5.0, 'tetragonal minimum,\nno DFT d-DOS', fontsize=5.0, color=S.INK2, ha='center', va='center', linespacing=1.0)
axb.plot([8300, 500], [4.78, 4.36], color=S.INK2, lw=0.45)          # leader to the light-blue segment of full tetra
axb.text(0.25, 1.015, 'training set (51,257)', transform=axb.transAxes, ha='center', va='bottom', fontsize=5.8, color=T_C)
axb.text(0.75, 1.015, 'prediction set (54,978)', transform=axb.transAxes, ha='center', va='bottom', fontsize=5.8, color=P_C)
axb.text(-0.205, 1.0, 'b', transform=axb.transAxes, fontsize=9, fontweight='bold', va='bottom')

# ================================================================ c  energy above the same-family, same-composition training entry
tr_ = dx[dx.set == 'training'].set_index(['family', 'composition'])['energy (eV/atom)']
assert tr_.index.is_unique
pr_ = dx[dx.set == 'prediction'].copy()
pr_['dE'] = 1e3 * (pr_['energy (eV/atom)'] - [tr_.get((f, c_), np.nan) for f, c_ in zip(pr_.family, pr_.composition)])
sib = pr_.dropna(subset=['dE'])
LO, HI = 1e-2, 10 ** 3.3
n_lo, n_hi = int((sib.dE < LO).sum()), int((sib.dE > HI).sum())
f1 = (sib.dE < 1).sum() / len(pr_)
assert len(sib) == 51103 and abs(f1 - 0.1508) < 5e-4 and (sib.dE >= 0).all() and abs(sib.dE.median() - 32.26) < 0.05
assert (n_lo, n_hi) == (347, 25)
axc = mm_axes(131.0, 10.0, 49.5, 31.0)
bins = np.logspace(np.log10(LO), np.log10(HI), 40)
hc, _ = np.histogram(sib.dE, bins=bins)
axc.hist(sib.dE[(sib.dE >= LO) & (sib.dE <= HI)], bins=bins, color=P_C, lw=0)
axc.set_xscale('log'); axc.set_xlim(LO, HI)
axc.set_xticks([1e-2, 1e-1, 1, 1e1, 1e2, 1e3])
axc.set_xticklabels(['0.01', '0.1', '1', '10', '100', '1000'])   # v16 Pass-6: plain numbers, no superscripts
axc.tick_params(labelsize=5.4)
axc.set_xlabel('energy above training counterpart (meV atom$^{-1}$)', labelpad=1, fontsize=5.8)
axc.set_ylabel('prediction entries', labelpad=1, fontsize=5.8)
axc.set_ylim(0, 5200); axc.set_yticks([0, 1000, 2000, 3000, 4000, 5000])   # v16: headroom so the notes clear the 1 meV line
axc.text(0.98, 0.98, f'{len(sib):,} entries with a\nsame-family counterpart\nmedian {sib.dE.median():.0f} meV atom$^{{-1}}$',
         transform=axc.transAxes, va='top', ha='right', fontsize=5.5, color=S.INK2)
axc.text(0.03, 0.98, f'not drawn:\n{n_lo} below 0.01,\n{n_hi} above 2,000', transform=axc.transAxes, va='top', fontsize=5.0,
         color=S.INK2)
axc.text(-0.27, 1.02, 'c', transform=axc.transAxes, fontsize=9, fontweight='bold', va='bottom')
# v16: 1 meV/atom above the same-family, same-composition training entry (a reference line, not the near-duplicate definition)
assert round(100 * f1, 1) == 15.1
axc.axvline(1.0, ymin=0, ymax=1.0, color=S.INK, lw=0.6, ls=(0, (1.2, 1.2)), zorder=3)
axc.annotate(f'{100 * f1:.1f}% of all prediction entries\nlie below 1 meV atom$^{{-1}}$', (1.0, 1.0),
             xycoords=('data', 'axes fraction'), xytext=(0, 2.2), textcoords='offset points', ha='center', va='bottom', fontsize=5.5,
             color=S.INK, linespacing=1.05, annotation_clip=False)
pd.DataFrame({'bin_lo_meV': bins[:-1], 'bin_hi_meV': bins[1:], 'prediction_entries': hc}).to_csv(DATA / 'v14_fig1_dE_hist.csv', index=False)
pd.Series({'n_with_counterpart': len(sib), 'median_meV': sib.dE.median(), 'frac_all_pred_within_1meV': f1,
           'n_below_1e-2': n_lo, 'n_above_2000': n_hi, 'A_seebeck_ZrNiSn_dft': A_T}).to_csv(DATA / 'v14_fig1_dE_summary.csv')

assert_scripts(fig)                                                   # v16 Pass-6: scripts >= 4.5 pt
probs = qa.check(fig, 'v14_fig1_strategy')
S.save(fig, OUT / 'v14_fig1_strategy', width_mm=183)
assert not probs, probs
print(pd.DataFrame(C))
