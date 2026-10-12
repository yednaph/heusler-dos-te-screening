"""v14 Fig. 2 -- the d-DOS model as implemented and its Fermi-focused loss (paper-figures skill, diagram + data inset).

Question: what does the model see, and what does it optimise?
5-second takeaway: a stock ALIGNN encoder reads an atom graph and a bond-angle line graph; a task-specific head returns a non-negative
4,002-value spin-resolved d-DOS; the two largest loss terms look only at the region near E_F, the largest one at a ~0.3 eV strip.

Stage table (every entry verified in the released code / checkpoint, see SOURCES below):
  stage            input                     operation                                          output               source
  graphs           DXMag structure           atom_dgl_multigraph(cutoff 8 A, max_neighbors 12,  atom graph + line     code/dos_dataset.py _build_graph
                                             compute_line_graph=True)                           graph (bond angles)
  node features    Z of each atom            92 CGCNN + 5 Heusler (_get_heusler_node_features;  97 per atom          dos_dataset.py L145-185
                                             F10 == F12 = n_d/10, so 4 distinct)
  embeddings       97 / bond d / cos(theta)  Linear 97->512 ; RBF80->64->512 ; RBF40->64->512    512 each            checkpoint tensor shapes
  encoder          embeddings                8 ALIGNN (line graph then atom graph) + 8 GCN      512 per atom         per-fold YAML, checkpoint keys
  pool + head      512 per atom              mean over atoms; Linear 512-1024-2048-4002, GELU,   4,002 raw            train_hpc.py create_model
                                             dropout 0.2 after the first two layers
  output           4,002 raw                 softplus(beta=1), outside the module               g_up, g_down >= 0    train_hpc.py L579
  loss             softplus output vs DFT    five active terms (w_peak = w_tv = 0 in all 30)    scalar               train_hpc.py L82-295 + 30 YAMLs
Sequential, no loop. Panel b plots the weight vector exactly as coded (ThermoelectricDOSLoss.__init__), T from the YAMLs.
v16 Pass-6: every sub/superscript prints at >= 4.5 pt (SHRINK_FACTOR 0.82 and scripted text >= 5.5 pt; asserted): log ticks of b, the
held-out note, and the term descriptions (the second difference now written Delta$^2$ in mathtext instead of a Unicode superscript).
"""
import collections
import glob
import pickle
import re
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Arc
SKILL = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SKILL))
from figkit import style as S, glyphs as G, qa                        # noqa: E402
from common import per_sample, EXPORT, DATA, OUT, E_GRID               # noqa: E402  (read-only use)

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
RELEASE = Path(__file__).resolve().parents[4]                  # repository root: the released code is the source of truth for this figure

# ------------------------------------------------------------------ 1. verify the implementation in the code itself
src_train = (RELEASE / 'train_hpc.py').read_text()                     # loss + model construction
src_data = (RELEASE / 'code/dos_dataset.py').read_text()               # graph construction + Heusler features
for needle in ['fermi_w  = (beta / 4.0) / (torch.cosh(beta * ENERGIES / 2.0))**2',   # -df/dE, sech^2 form
               'fermi_w  = fermi_w / fermi_w.sum() * 2001',                           # mean weight 1 per channel
               'weight_4002 = torch.cat([fermi_w, fermi_w])',                         # same window for both spins
               'self.fermi_win_up   = slice(800,  1201)', 'self.fermi_win_down = slice(2801, 3202)',  # +-2 eV, both spins
               'self.slope_win = slice(700, 1301)',                                   # +-3 eV of the CONCATENATED vector
               'L_slope  = L_slope1 + 0.1 * L_slope2',                                # 1st + 0.1 x 2nd differences
               'L_wass = torch.mean(torch.abs(pred_sorted - true_sorted))',           # L1 of sorted 4,002 values
               'cosine_similarity(pred, true, dim=1, eps=1e-6)',                      # cosine over all 4,002 values
               'nn.Linear(hidden, hidden * 2)', 'nn.Linear(hidden * 4, 4002)',        # head 512-1024-2048-4002
               "dos_pred   = torch.nn.functional.softplus(output_raw, beta=1)",      # softplus outside the module
               "cutoff        = data_config.get('cutoff', 8.0)", "max_neighbors = data_config.get('max_neighbors', 12)"]:
    assert needle in src_train, needle
for needle in ['compute_line_graph=True', 'n_d / 10.0,               # F10', 'n_d / 10.0,               # F12']:
    assert needle in src_data, needle                                  # F10 and F12 are the same quantity -> 4 distinct

# the 30 per-fold training configs: active loss weights, temperature, model settings (no cutoff override anywhere)
cfgs = sorted(glob.glob(str(EXPORT / 'configs/per_fold/*/config_expA_fold*.yaml')))
assert len(cfgs) == 30
LW_KEYS = ['w_mse', 'w_fermi', 'w_cosine', 'w_slope', 'w_wass', 'w_peak', 'w_tv']
seen = set()
for f in cfgs:
    c = yaml.safe_load(open(f))
    assert 'cutoff' not in c['data'] and 'max_neighbors' not in c['data']
    seen.add((tuple(c['loss'][k] for k in LW_KEYS), c['training']['screening_temperature'],
              c['model']['alignn_layers'], c['model']['gcn_layers'], c['model']['hidden_dim'], c['model']['dropout']))
assert seen == {((0.35, 0.35, 0.15, 0.1, 0.05, 0.0, 0.0), 700, 8, 8, 512, 0.2)}, seen
T_K = 700


def ckpt_shapes(path):
    """Tensor shapes of a raw state_dict .pt (zip + pickle) without importing torch (torch is not installed here)."""
    class _T:
        def __init__(s, shape): s.shape = tuple(shape)
    class _U(pickle.Unpickler):
        def find_class(s, mod, name):
            if name.startswith('_rebuild_tensor'):
                return lambda storage, off, size, *a: _T(size)
            if mod == 'collections' and name == 'OrderedDict':
                return collections.OrderedDict
            if mod.startswith('torch'):
                return lambda *a, **k: None
            return super().find_class(mod, name)
        def persistent_load(s, pid): return None
    z = zipfile.ZipFile(path)
    return {k: v.shape for k, v in _U(z.open([n for n in z.namelist() if n.endswith('data.pkl')][0])).load().items()}


sh = ckpt_shapes(EXPORT / 'checkpoints/half_cubic/fold0/swa_model.pt')
assert sh['atom_embedding.layer.0.weight'] == (512, 97)
assert sh['edge_embedding.0.centers'] == (80,) and sh['edge_embedding.1.layer.0.weight'] == (64, 80) and sh['edge_embedding.2.layer.0.weight'] == (512, 64)
assert sh['angle_embedding.0.centers'] == (40,) and sh['angle_embedding.1.layer.0.weight'] == (64, 40) and sh['angle_embedding.2.layer.0.weight'] == (512, 64)
assert [sh[f'fc.{i}.weight'] for i in (0, 3, 6)] == [(1024, 512), (2048, 1024), (4002, 2048)]
assert sorted({int(k.split('.')[1]) for k in sh if k.startswith('alignn_layers.')}) == list(range(8))
assert sorted({int(k.split('.')[1]) for k in sh if k.startswith('gcn_layers.')}) == list(range(8))
n_par = sum(int(np.prod(v)) for k, v in sh.items() if 'running' not in k and 'num_batches' not in k and not k.endswith('centers'))
assert n_par == 42520354, n_par

# ------------------------------------------------------------------ 2. the weighting w(E) exactly as coded
kB = 8.617333e-5                                                      # eV/K, as in the code
beta = 1.0 / (kB * T_K)
w = (beta / 4.0) / np.cosh(beta * E_GRID / 2.0) ** 2                  # -df/dE
w = w / w.sum() * 2001                                                # mean 1 over the 2,001 points of one channel
Er = np.round(E_GRID, 2)
f03 = w[np.abs(Er) <= 0.3].sum() / w.sum()                            # share of the L_F weight within +-0.3 eV
f01 = w[np.abs(Er) <= 0.1].sum() / w.sum()
assert abs(w.max() - 82.93) < 0.01 and abs(f03 - 0.987) < 0.0015, (w.max(), f03)
pd.DataFrame({'E_minus_EF_eV': E_GRID, 'w_E': w}).to_csv(DATA / 'v14_fig2_weight.csv', index=False)

TERMS = [  # (symbol, weight, config key, window lo, hi, spins, what the code computes)
    (r'$\mathcal{L}_\mathrm{F}$', 0.35, 'w_mse', -10, 10, 'both', 'mean of w(E)(g-g_DFT)^2 over 4,002 values'),
    (r'$\mathcal{L}_{\pm2}$', 0.35, 'w_fermi', -2, 2, 'both', 'MSE on 401 points per channel, channel average'),
    (r'$\mathcal{L}_\mathrm{cos}$', 0.15, 'w_cosine', -10, 10, 'both (one vector)', '1 - cosine similarity of the 4,002-vectors'),
    (r'$\mathcal{L}_\mathrm{slope}$', 0.10, 'w_slope', -3, 3, 'spin up only', 'MSE of 1st + 0.1 x MSE of 2nd differences, indices 700:1301'),
    (r'$\mathcal{L}_\mathrm{sort}$', 0.05, 'w_wass', -10, 10, 'both (one vector)', 'mean |sort(g) - sort(g_DFT)| over 4,002 values')]
assert abs(sum(t[1] for t in TERMS) - 1.0) < 1e-12
pd.DataFrame(TERMS, columns=['term', 'weight', 'config_key', 'E_lo_eV', 'E_hi_eV', 'spins', 'definition']).to_csv(
    DATA / 'v14_fig2_loss_terms.csv', index=False)

# ------------------------------------------------------------------ 3. real held-out spectrum for the output thumbnail (median r entry)
ps = per_sample()
assert len(ps) == 51257 and abs(ps.pearson_r.median() - 0.911) < 5e-4
pick = ps.iloc[(ps.pearson_r - ps.pearson_r.median()).abs().argsort().iloc[0]]
dd = EXPORT / 'runtime_assets/heldout_dumps' / pick.family / pick.fold
j = [l.strip() for l in open(dd / 'test_ids.txt')].index(pick.material_id)
Pv = np.asarray(np.load(dd / 'test_predictions.npy', mmap_mode='r')[j]); Tv = np.asarray(np.load(dd / 'test_targets.npy', mmap_mode='r')[j])
r_chk = np.corrcoef(Pv, Tv)[0, 1]
assert abs(r_chk - pick.pearson_r) < 1e-4 and pick.formula == 'Pt2VGa'
SPEC = dict(cutoff_A=8.0, max_neighbors=12, node_features=97, cgcnn=92, heusler_cols=5, heusler_distinct=4, edge_rbf=80,
            angle_rbf=40, embedding=64, hidden=512, alignn_layers=8, gcn_layers=8, head='512-1024-2048-4002', dropout=0.2,
            outputs=4002, trainable_params=n_par, T_K=T_K, w_peak=float(w.max()), w_frac_pm0p3=float(f03), w_frac_pm0p1=float(f01),
            example=pick.material_id, example_r=float(pick.pearson_r), example_family=pick.family, example_fold=pick.fold)
pd.Series(SPEC).to_csv(DATA / 'v14_fig2_spec.csv')

# ------------------------------------------------------------------ 4. drawing (canvas in mm)
W, H = 183, 60
fig, ax = G.canvas(W, H)
STOCK, STOCK_FILL = S.ARROW, '#FFFFFF'                               # stock ALIGNN: white card, grey outline
TASK, TASK_FILL = S.NAVY, '#D2E3F3'                                  # specific to this model: navy outline, blue tint (dE 14.5 vs white)
LG = S.NAVY                                                          # line-graph objects (bonds as nodes)
FS_IN = 5.5                                                          # inner text size


def box(x, y, w_, h_, task=False, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w_, h_, boxstyle='round,pad=0,rounding_size=1.0', fc=TASK_FILL if task else STOCK_FILL,
                                ec=TASK if task else STOCK, lw=S.LW['card'], zorder=z))


def head(x, y, s, task=False):
    G.text(ax, x, y, s, fs=6.2, c=TASK if task else S.INK, ha='left')


S_Y0, S_Y1 = 3.0, 51.5                                               # stage cards span
ax.text(1.0, 56.2, 'a', fontsize=S.FS['panel'], fontweight='bold', va='center', ha='left')
# key: which parts are stock and which are specific to this model
box(6.5, 54.6, 4.2, 3.2); G.text(ax, 11.6, 56.2, 'stock ALIGNN', fs=5.4, ha='left', c=S.INK2)
box(31.5, 54.6, 4.2, 3.2, task=True); G.text(ax, 36.6, 56.2, 'specific to this d-DOS model', fs=5.4, ha='left', c=S.INK2)
G.text(ax, 119.0, 56.2, f'{n_par / 1e6:.1f} M trainable parameters', fs=5.4, ha='right', c=S.INK2)

# --- stage 1: crystal -> atom graph + line graph
x0, x1 = 1.5, 30.0
box(x0, S_Y0, x1 - x0, S_Y1 - S_Y0)
head(x0 + 1.8, 48.6, 'crystal → two graphs')
A = {'c1': (11.5, 35.5), 'c2': (20.5, 35.5), 'n1': (6.0, 42.0), 'n2': (6.0, 29.0), 'n3': (26.0, 42.0), 'n4': (26.0, 29.0)}
SPC = {'c1': '#5B6672', 'c2': '#8A96A3', 'n1': '#C9D2DC', 'n2': '#C9D2DC', 'n3': '#C9D2DC', 'n4': '#C9D2DC'}
BONDS = [('c1', 'c2'), ('c1', 'n1'), ('c1', 'n2'), ('c2', 'n3'), ('c2', 'n4')]
for a_, b_ in BONDS:                                                 # atom-graph edges = bonds
    ax.plot([A[a_][0], A[b_][0]], [A[a_][1], A[b_][1]], color=S.INK2, lw=0.8, zorder=3, solid_capstyle='round')
mid = {bd: ((A[bd[0]][0] + A[bd[1]][0]) / 2, (A[bd[0]][1] + A[bd[1]][1]) / 2) for bd in BONDS}
for i in range(len(BONDS)):                                          # line-graph edges = pairs of bonds sharing an atom (angles)
    for k in range(i + 1, len(BONDS)):
        if set(BONDS[i]) & set(BONDS[k]):
            p, q = mid[BONDS[i]], mid[BONDS[k]]
            ax.plot([p[0], q[0]], [p[1], q[1]], color=LG, lw=0.55, ls=(0, (1.6, 1.0)), zorder=4)
for bd, (mx, my) in mid.items():                                     # line-graph nodes = bonds
    ax.add_patch(Polygon([(mx, my + 0.95), (mx + 0.95, my), (mx, my - 0.95), (mx - 0.95, my)], fc=LG, ec='white', lw=0.3, zorder=6))
for k_, (px, py) in A.items():                                       # atom-graph nodes = atoms
    ax.add_patch(Circle((px, py), 1.55, fc=SPC[k_], ec='white', lw=0.5, zorder=5))
ang1 = np.degrees(np.arctan2(A['n1'][1] - A['c1'][1], A['n1'][0] - A['c1'][0]))
ax.add_patch(Arc(A['c1'], 6.0, 6.0, theta1=0.0, theta2=ang1, color=LG, lw=0.7, zorder=4))   # bond angle theta at c1
G.text(ax, 15.6, 38.3, 'θ', fs=5.6, c=LG)
ax.add_patch(Circle((x0 + 2.6, 24.4), 0.9, fc='#8A96A3', ec='none', zorder=5))          # key: atom node
G.text(ax, x0 + 4.2, 24.4, 'atom: graph node', fs=5.2, ha='left', c=S.INK2)
mx, my = x0 + 2.6, 21.2                                                                       # key: bond node
ax.add_patch(Polygon([(mx, my + 0.95), (mx + 0.95, my), (mx, my - 0.95), (mx - 0.95, my)], fc=LG, ec='none', zorder=5))
G.text(ax, x0 + 4.2, 21.2, 'bond: line-graph node', fs=5.2, ha='left', c=LG)
G.text(ax, x0 + 1.8, 16.4, 'atom graph: bonds within 8 Å,', fs=FS_IN, ha='left')
G.text(ax, x0 + 1.8, 13.2, '12-nearest-neighbour rule', fs=FS_IN, ha='left')
G.text(ax, x0 + 1.8, 9.0, 'line graph: bond pairs that', fs=FS_IN, ha='left')
G.text(ax, x0 + 1.8, 5.8, 'share an atom (angle θ)', fs=FS_IN, ha='left')

# --- stage 2: input features and embeddings
x0, x1 = 32.0, 53.5
box(x0, S_Y0, x1 - x0, S_Y1 - S_Y0)
head(x0 + 1.6, 48.6, 'input features')
xi = x0 + 1.6
G.text(ax, xi, 43.6, 'atom', fs=5.8, ha='left', c=S.INK)
ax.add_patch(FancyBboxPatch((xi, 37.2), 9.6, 3.6, boxstyle='round,pad=0,rounding_size=0.7', fc='white', ec=STOCK, lw=0.6, zorder=4))
G.text(ax, xi + 4.8, 39.0, '92 CGCNN', fs=5.0, c=S.INK2)
ax.add_patch(FancyBboxPatch((xi + 10.1, 37.2), 8.5, 3.6, boxstyle='round,pad=0,rounding_size=0.7', fc=TASK_FILL, ec=TASK, lw=0.6, zorder=4))
G.text(ax, xi + 14.35, 39.0, '5 Heusler', fs=5.0, c=TASK)
G.text(ax, xi, 33.8, 'two columns identical', fs=5.2, ha='left', c=TASK)
G.text(ax, xi, 30.4, '97 → 512', fs=FS_IN, ha='left')
G.text(ax, xi, 24.6, 'bond length d', fs=5.8, ha='left')
G.text(ax, xi, 21.0, '80 RBF → 64 → 512', fs=FS_IN, ha='left')
G.text(ax, xi, 15.0, 'bond angle cos θ', fs=5.8, ha='left', c=LG)
G.text(ax, xi, 11.4, '40 RBF → 64 → 512', fs=FS_IN, ha='left')
G.text(ax, xi, 6.0, 'RBF: radial basis', fs=5.0, ha='left', c=S.INK2)

# --- stage 3: encoder (layer stack glyph)
x0, x1 = 55.5, 74.5
box(x0, S_Y0, x1 - x0, S_Y1 - S_Y0)
head(x0 + 1.6, 48.6, 'encoder, 512')
bx, bw, bh, pitch = x0 + 2.0, 15.0, 0.85, 1.32
G.text(ax, bx, 44.6, '8 ALIGNN layers', fs=FS_IN, ha='left')
G.text(ax, bx + bw / 4, 41.4, 'bonds', fs=5.0, c=LG)
G.text(ax, bx + bw * 3 / 4 + 0.3, 41.4, 'atoms', fs=5.0, c=S.INK2)
for i in range(8):                                                   # each ALIGNN layer: line-graph update, then atom-graph update
    yy = 38.6 - i * pitch
    ax.add_patch(FancyBboxPatch((bx, yy), bw / 2 - 0.3, bh, boxstyle='round,pad=0,rounding_size=0.25', fc=LG, ec='none', zorder=4))
    ax.add_patch(FancyBboxPatch((bx + bw / 2 + 0.3, yy), bw / 2 - 0.3, bh, boxstyle='round,pad=0,rounding_size=0.25', fc=STOCK,
                                ec='none', zorder=4))
G.chevron(ax, bx + bw * 3 / 4 + 0.3, 27.4, STOCK, size=0.8, direction='down')
G.text(ax, bx, 24.6, '8 GCN layers', fs=FS_IN, ha='left')
for i in range(8):                                                   # each GCN layer: edge-gated update of the atom graph
    yy = 21.2 - i * pitch
    ax.add_patch(FancyBboxPatch((bx + bw / 2 + 0.3, yy), bw / 2 - 0.3, bh, boxstyle='round,pad=0,rounding_size=0.25', fc=STOCK,
                                ec='none', zorder=4))
G.text(ax, bx + bw / 4 - 0.4, 18.6, 'atoms', fs=5.0, c=S.INK2)
G.text(ax, bx + bw / 4 - 0.4, 15.6, 'only', fs=5.0, c=S.INK2)
G.text(ax, bx, 6.0, 'edge-gated conv.', fs=5.0, ha='left', c=S.INK2)

# --- stage 4: pooling and the task-specific head (bar length proportional to layer width)
x0, x1 = 76.5, 94.0
box(x0, S_Y0, x1 - x0, S_Y1 - S_Y0, task=True)
head(x0 + 1.6, 48.6, 'pool + head', task=True)
G.text(ax, x0 + 1.6, 44.4, 'mean over atoms', fs=FS_IN, ha='left')
hx, Lmax = x0 + 1.6, 14.2
for i, n in enumerate([512, 1024, 2048, 4002]):
    yy = 38.0 - i * 4.4; L = Lmax * n / 4002
    ax.add_patch(FancyBboxPatch((hx, yy), L, 2.4, boxstyle='round,pad=0,rounding_size=0.4', fc=TASK, ec='none', zorder=4))
    if L > 8:
        G.text(ax, hx + L - 0.6, yy + 1.2, f'{n:,}', fs=5.4, c='white', ha='right')
    else:
        G.text(ax, hx + L + 0.7, yy + 1.2, f'{n:,}', fs=5.4, c=TASK, ha='left')
G.text(ax, x0 + 1.6, 21.0, 'GELU, dropout 0.2', fs=5.2, ha='left', c=S.INK2)
ax.add_patch(FancyBboxPatch((x0 + 1.6, 11.0), 14.2, 4.6, boxstyle='round,pad=0,rounding_size=0.8', fc='white', ec=TASK, lw=0.7, zorder=4))
G.text(ax, x0 + 8.7, 13.3, 'softplus', fs=5.6, c=TASK)
G.text(ax, x0 + 1.6, 6.6, 'output ≥ 0', fs=5.4, ha='left', c=TASK)

# --- stage 5: the spin-resolved d-DOS (real held-out spectrum)
x0, x1 = 96.0, 119.5
box(x0, S_Y0, x1 - x0, S_Y1 - S_Y0, task=True)
head(x0 + 1.6, 48.6, 'spin-resolved d-DOS', task=True)
rect = (x0 + 2.6, 16.0, x1 - x0 - 5.0, 27.0)
ia = fig.add_axes([rect[0] / W, rect[1] / H, rect[2] / W, rect[3] / H])
up, dn = slice(0, 2001), slice(2001, 4002)
ia.fill_between(E_GRID, 0, Tv[up], color='#C9D2DC', lw=0); ia.fill_between(E_GRID, 0, -Tv[dn], color='#C9D2DC', lw=0)
ia.plot(E_GRID, Pv[up], color=S.SET_COLOR['prediction'], lw=0.5); ia.plot(E_GRID, -Pv[dn], color=S.SET_COLOR['prediction'], lw=0.5)
ia.axhline(0, color=S.INK2, lw=0.35); ia.axvline(0, color=S.INK2, lw=0.4, ls=':')
ym = 1.05 * max(np.abs(Tv).max(), np.abs(Pv).max()); ia.set_ylim(-ym, ym); ia.set_xlim(-10, 10)
ia.set_yticks([]); ia.set_xticks([-10, 0, 10]); ia.set_xticklabels(['−10', '0', '10'], fontsize=5.0)
ia.tick_params(axis='x', length=1.5, pad=1.0)
for s_ in ['left', 'top', 'right']:
    ia.spines[s_].set_visible(False)
ia.spines['bottom'].set_linewidth(0.4); ia.patch.set_alpha(0)
G.text(ax, x0 + 2.4, 44.3, 'g↑', fs=5.4, ha='left', c=S.INK2)
G.text(ax, x0 + 2.4, 18.3, 'g↓', fs=5.4, ha='left', c=S.INK2)
G.text(ax, x0 + 1.6, 9.6, '2 × 2,001 values', fs=FS_IN, ha='left')
G.text(ax, x0 + 1.6, 6.0, f'held-out Pt$_2$VGa, r {pick.pearson_r:.3f}', fs=5.5, ha='left', c=S.INK2)   # v16 Pass-6: 5.5 pt, fits the card

for xc in (31.0, 54.5, 75.5, 95.0):                                  # forward pass (chevrons coloured like the stage they leave)
    G.chevron(ax, xc, 27.0, TASK if xc > 94 else STOCK, size=0.95)

# ------------------------------------------------------------------ panel b: the loss -- real w(E) and the five active terms
ax.text(122.0, 56.2, 'b', fontsize=S.FS['panel'], fontweight='bold', va='center', ha='left')
XL = (-3.6, 3.6)
AX_X, AX_W = 149.5, 31.0
wa = fig.add_axes([AX_X / W, 33.5 / H, AX_W / W, 21.0 / H])
wa.fill_between(E_GRID, 1e-5, w, color=TASK_FILL, lw=0, zorder=1)
wa.plot(E_GRID, w, color=TASK, lw=0.9, zorder=3)
wa.axhline(1.0, color=S.INK2, lw=0.45, ls=':', zorder=2)
wa.set_yscale('log'); wa.set_ylim(1e-3, 4e2); wa.set_xlim(*XL)
wa.set_yticks([1e-3, 1e-1, 1e1]); wa.set_yticklabels(['10$^{-3}$', '10$^{-1}$', '10$^{1}$'], fontsize=5.5)   # v16 Pass-6: 5.5 pt
wa.minorticks_off(); wa.set_xticks([-3, -2, -1, 0, 1, 2, 3]); wa.tick_params(axis='x', labelbottom=False, length=1.8)
wa.tick_params(axis='y', length=1.8, pad=1.0)
wa.set_ylabel('$w(E)$', fontsize=5.8, labelpad=1.0)
wa.text(0.33, w.max(), f'{w.max():.1f}', fontsize=5.2, color=TASK, va='center', ha='left')
wa.text(-3.45, 1.45, 'mean 1', fontsize=5.0, color=S.INK2, va='bottom', ha='left')
wa.text(3.45, 1.2e-2, f'{100 * f03:.1f}% within', fontsize=5.0, color=S.INK2, va='center', ha='right')
wa.text(3.45, 3.6e-3, '±0.3 eV', fontsize=5.0, color=S.INK2, va='center', ha='right')

ta = fig.add_axes([AX_X / W, 9.0 / H, AX_W / W, 23.0 / H])                 # the five terms: where in energy each acts
ta.set_xlim(*XL); ta.set_ylim(0, 5)
ta.set_yticks([]); ta.spines['left'].set_visible(False)
ta.set_xticks([-3, -2, -1, 0, 1, 2, 3]); ta.set_xticklabels(['−3', '−2', '−1', '0', '1', '2', '3'], fontsize=5.4)
ta.tick_params(axis='x', length=1.8, pad=1.0)
ta.set_xlabel('$E-E_\\mathrm{F}$ (eV)', fontsize=5.8, labelpad=1.0)
ta.axvline(0, color=S.INK2, lw=0.4, ls=':', zorder=1)
DESC = ['weighted MSE', 'MSE', '1 − cosine', 'Δ, Δ$^2$ MSE', 'L1, sorted']          # v16 Pass-6: mathtext superscript (>= 4.5 pt)
NOTE = ['', 'both spins', 'all 4,002 values', 'spin ↑ only', 'all 4,002 values']
BH = 0.56
for i, (sym, wt, key, lo, hi, spins, _) in enumerate(TERMS):
    yc = 4.5 - i
    if key == 'w_mse':                                               # L_F: full grid, weight w(E); show the +-0.3 eV core
        ta.plot([XL[0], XL[1]], [yc, yc], color='#8FA6BC', lw=0.5, zorder=2, solid_capstyle='butt')   # opaque = navy at 0.5 over white
        ta.add_patch(FancyBboxPatch((-0.3, yc - BH / 2), 0.6, BH, boxstyle='round,pad=0,rounding_size=0.08', fc=TASK, ec='none', zorder=3,
                                    mutation_aspect=1 / 6))
        ta.text(0.5, yc + 0.02, '× w(E)', fontsize=5.0, color=TASK, va='center', ha='left',
                bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none'))
        ta.text(-0.5, yc + 0.02, 'both spins', fontsize=5.0, color=TASK, va='center', ha='right',
                bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none'))
    else:
        lo_, hi_ = max(lo, XL[0]), min(hi, XL[1])
        ta.add_patch(FancyBboxPatch((lo_, yc - BH / 2), hi_ - lo_, BH, boxstyle='round,pad=0,rounding_size=0.08', fc=TASK, ec='none',
                                    zorder=3, mutation_aspect=1 / 6))
        ta.text(0.0 if hi - lo < 15 else 0.0, yc + 0.02, NOTE[i], fontsize=5.0, color='white', va='center', ha='center', zorder=5,
                bbox=dict(boxstyle='square,pad=0.05', fc=TASK, ec='none'))
        if hi - lo > 15:                                             # global terms continue to -10 and +10 eV
            for xe, d_ in [(XL[0], -1), (XL[1], 1)]:
                ta.add_patch(Polygon([(xe, yc + 0.42), (xe + d_ * 0.32, yc), (xe, yc - 0.42)], closed=True, fc=TASK, ec='none', zorder=4,
                                     clip_on=False))
    yy = 9.0 + 23.0 * (yc / 5)                                        # same row in canvas mm
    G.tag(ax, 125.6, yy, f'{wt:.2f}', TASK, w=6.6, h=3.2, fs=5.2)
    G.text(ax, 129.4, yy + 0.05, sym, fs=5.8, ha='left', c=S.INK)
    G.text(ax, 136.6, yy, DESC[i], fs=5.5, ha='left', c=S.INK2)                     # v16 Pass-6: 5.5 pt
G.text(ax, 124.0, 5.2, 'arrowheads: to ±10 eV', fs=5.0, ha='left', c=S.INK2)
# the Fermi-window weight of L_F, as coded (text block beside the curve)
G.text(ax, 124.0, 50.6, 'weight of $\\mathcal{L}_\\mathrm{F}$', fs=5.6, ha='left', c=TASK)
G.text(ax, 124.0, 46.6, '$w(E)\\propto-\\partial f/\\partial E$', fs=5.6, ha='left', c=S.INK)
G.text(ax, 124.0, 42.6, f'T = {T_K} K', fs=5.4, ha='left', c=S.INK2)
G.text(ax, 124.0, 39.0, 'mean 1 per spin', fs=5.4, ha='left', c=S.INK2)
G.text(ax, 124.0, 35.4, 'channel', fs=5.4, ha='left', c=S.INK2)

assert_scripts(fig)                                                   # v16 Pass-6: scripts >= 4.5 pt
probs = qa.check(fig, 'v14_fig2_model')
assert not probs, probs
S.save(fig, OUT / 'v14_fig2_model', width_mm=183)
print(pd.Series(SPEC).to_string())
