"""Fig. 3 (v7) -- the ALIGNN d-DOS model and its task adaptation (architecture diagram, paper-figures skill).

Question: how does the network turn a crystal into a spin-resolved d-DOS, and which parts are specific to this task?
5-second takeaway: a stock ALIGNN encoder (configured deeper and wider) reads the crystal graph and its bond-angle line
graph; only the output head, the non-negative 4,002-value output and the loss are specific to the d-DOS task.

Stage table (verified in train_hpc.py, dos_dataset.py, predict_from_cif.py and checkpoint tensor shapes):
  crystal -> atom graph (8 A cutoff, <= 12 neighbours) + line graph (bond pairs, angles)            dos_dataset.py:234-239
  node features 92 CGCNN + 5 Heusler = 97 -> Linear embedding to 512                                atom_embedding [512, 97]
  bond length -> 80 RBF -> 64 -> 512 ; bond-angle cosine -> 40 RBF -> 64 -> 512                     edge/angle embedding shapes
  8 ALIGNN layers (line graph then atom graph) + 8 edge-gated GCN layers, width 512                  ALIGNNConfig(8, 8, 512)
  mean over atoms -> head 512 -> 1024 -> 2048 -> 4002 (GELU, dropout 0.2) -> softplus                train_hpc.py:363-371, 579
  output = 2 x 2,001 d-DOS ; training loss = five weighted terms                                    per-fold configs
The example spectrum is a REAL held-out pair (the entry with the overall-median Pearson r, as in Fig. 1).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.patches as mp
SKILL = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SKILL))
from figkit import style as S, glyphs as G, qa                       # noqa: E402
from common import per_sample, EXPORT, DATA, OUT, E_GRID              # noqa: E402

S.apply()
ps = per_sample()
med = ps.pearson_r.median()
pick = ps.iloc[(ps.pearson_r - med).abs().argsort().iloc[0]]          # same declared rule as Fig. 1
dd = EXPORT / 'runtime_assets/heldout_dumps' / pick.family / pick.fold
j = [l.strip() for l in open(dd / 'test_ids.txt')].index(pick.material_id)
Pv = np.asarray(np.load(dd / 'test_predictions.npy', mmap_mode='r')[j]); Tv = np.asarray(np.load(dd / 'test_targets.npy', mmap_mode='r')[j])
pd.Series({'material_id': pick.material_id, 'formula': pick.formula, 'pearson_r': pick.pearson_r}).to_csv(DATA / 'fig3_example.csv')
SPEC = dict(cutoff_A=8.0, max_neighbors=12, node_features=97, edge_rbf=80, angle_rbf=40, embedding=64, hidden=512,
            alignn_layers=8, gcn_layers=8, head='512-1024-2048-4002', dropout=0.2, outputs=4002, trainable_params=42520354)
pd.Series(SPEC).to_csv(DATA / 'fig3_spec.csv')

W, H = 183, 80
fig, ax = G.canvas(W, H)
STOCK, STOCK_FILL = S.ARROW, '#F1F4F7'                                # stock ALIGNN parts: neutral
TASK = S.NAVY                                                         # task-specific parts: navy (the model role)


def stock_card(x, y, w, h, head):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.0', fc=STOCK_FILL, ec=STOCK, lw=0.7, zorder=2))
    G.text(ax, x + 2.0, y + h - 3.0, head, fs=6.4, w='bold', c=S.INK, ha='left')


# ---------------------------------------------------------------- legend strip (top)
ax.add_patch(mp.FancyBboxPatch((3, 74.2), 6, 3.0, boxstyle='round,pad=0,rounding_size=0.6', fc=STOCK_FILL, ec=STOCK, lw=0.7))
G.text(ax, 10.5, 75.7, 'stock ALIGNN component (settings in navy)', fs=5.3, ha='left', c=S.INK2)
ax.add_patch(mp.FancyBboxPatch((66, 74.2), 6, 3.0, boxstyle='round,pad=0,rounding_size=0.6', fc=S.TINT[TASK], ec=TASK, lw=0.8))
G.text(ax, 73.5, 75.7, 'specific to the d-DOS task', fs=5.3, ha='left', c=S.INK2)
G.text(ax, 180, 75.7, '≈42.5 M trainable parameters', fs=5.6, ha='right', c=TASK, w='bold')

# ---------------------------------------------------------------- 1 crystal -> two graphs
stock_card(2, 20, 41, 51, 'Crystal → two graphs')
# atom graph sketch: a 2D section of a Heusler-like arrangement
A1, A2, A3 = '#4A5561', '#A9B4C0', '#D7DEE6'                         # three species, neutral (no data meaning)
pts = {(0, 0): A1, (1, 0): A2, (0, 1): A2, (1, 1): A1, (0.5, 0.5): A3, (2, 0): A1, (2, 1): A2, (1.5, 0.5): A3}
ox, oy, sc = 7.0, 50.5, 9.0
xy = {k: (ox + k[0] * sc, oy + k[1] * sc) for k in pts}
bonds = [((0.5, 0.5), (0, 0)), ((0.5, 0.5), (1, 0)), ((0.5, 0.5), (0, 1)), ((0.5, 0.5), (1, 1)), ((1.5, 0.5), (1, 0)),
         ((1.5, 0.5), (2, 0)), ((1.5, 0.5), (1, 1)), ((1.5, 0.5), (2, 1))]
for a, b in bonds:
    ax.plot([xy[a][0], xy[b][0]], [xy[a][1], xy[b][1]], color=S.INK2, lw=0.6, zorder=3)
for k, c in pts.items():
    ax.add_patch(mp.Circle(xy[k], 1.25, fc=c, ec=S.INK2, lw=0.4, zorder=4))
# highlight one angle (bond pair) -> line-graph node pair
ax.add_patch(mp.Arc(xy[(0.5, 0.5)], 6, 6, theta1=-45, theta2=45, color=S.NAVY, lw=0.9, zorder=5))
G.text(ax, 17.6, 55.0, 'θ', fs=6.5, c=S.NAVY, w='bold')
G.text(ax, 22.5, 45.0, 'atom graph: atoms = nodes, bonds = edges', fs=5.0, c=S.INK2)
G.text(ax, 22.5, 41.6, 'cutoff 8 Å, ≤ 12 neighbours per atom', fs=5.2, c=TASK, w='bold')
G.text(ax, 22.5, 37.2, 'line graph: bonds = nodes,\nbond pairs (angle θ) = edges', fs=5.0, c=S.INK2)
G.text(ax, 22.5, 30.0, 'training: structure stored with the DFT DOS\nprediction: DXMag cell, positions as listed', fs=4.7, c=S.INK2)
G.text(ax, 22.5, 23.3, 'graphs built without relaxation or symmetrisation', fs=4.6, c=S.PURPLE)

# ---------------------------------------------------------------- 2 features and embeddings
G.chevron(ax, 45.0, 45.5, STOCK, size=1.2)
stock_card(47, 20, 31, 51, 'Input features')
rows = [('atoms', '92 CGCNN\n+ 5 Heusler = 97', 'linear → 512'),
        ('bond length', '80 radial\nbasis fns', '→ 64 → 512'),
        ('bond angle', 'cos θ: 40\nradial basis fns', '→ 64 → 512')]
for i, (lab, feat, emb) in enumerate(rows):
    y0 = 58.5 - i * 12.6
    G.text(ax, 49.0, y0 + 3.2, lab, fs=5.6, w='bold', ha='left', c=S.INK)
    G.text(ax, 49.0, y0 - 1.3, feat, fs=5.0, ha='left', c=TASK if i == 0 else S.INK2)
    G.text(ax, 76.5, y0 - 1.3, emb, fs=5.0, ha='right', c=S.INK2)
G.text(ax, 62.5, 24.6, 'Heusler features: d count, oxidation\nstate, d filling, d-shell row', fs=4.6, c=TASK)

# ---------------------------------------------------------------- 3 encoder stack
G.chevron(ax, 80.0, 45.5, STOCK, size=1.2)
stock_card(82, 20, 34, 51, 'Encoder')
for i, (lab, sub) in enumerate([('8 × ALIGNN layer', 'gated update on line graph,\nthen on atom graph'),
                                ('8 × graph-conv layer', 'edge-gated, atom graph')]):
    y0 = 56.0 - i * 15.0
    ax.add_patch(mp.FancyBboxPatch((84.5, y0 - 4.6), 29, 10.6, boxstyle='round,pad=0,rounding_size=0.8', fc='white',
                                   ec=STOCK, lw=0.6, zorder=3))
    G.text(ax, 99.0, y0 + 3.0, lab, fs=5.9, w='bold', c=TASK, z=6)
    G.text(ax, 99.0, y0 - 1.5, sub, fs=4.8, c=S.INK2, z=6)
    if i == 0:
        G.chevron(ax, 99.0, y0 - 6.3, STOCK, size=0.9, direction='down')
G.text(ax, 99.0, 30.0, 'width 512; batch normalisation', fs=4.8, c=S.INK2)
G.text(ax, 99.0, 25.0, 'package default: 4 + 4 layers,\nwidth 256', fs=4.6, c=S.INK2)

# ---------------------------------------------------------------- 4 readout + task head
G.chevron(ax, 118.0, 45.5, STOCK, size=1.2)
ax.add_patch(mp.FancyBboxPatch((120, 57), 26, 14, boxstyle='round,pad=0,rounding_size=1.0', fc=STOCK_FILL, ec=STOCK, lw=0.7, zorder=2))
G.text(ax, 133, 66.6, 'mean over atoms', fs=6.0, w='bold')
G.text(ax, 133, 61.2, 'one 512-value vector\nper crystal', fs=4.8, c=S.INK2)
G.chevron(ax, 133, 55.4, STOCK, size=0.9, direction='down')
G.card(ax, 120, 20, 26, 33.5, TASK, heading='Output head')
x = 122.3
for k, lab in enumerate(['512', '1,024', '2,048', '4,002']):
    yk = 45.0 - k * 4.8
    G.node(ax, 133, yk, 17, 3.2, lab, TASK, fs=5.3)
G.text(ax, 133, 26.6, 'GELU + dropout 0.2\nafter the first two layers', fs=4.5, c=S.INK2)
G.text(ax, 133, 22.4, 'softplus → d-DOS ≥ 0', fs=5.2, c=TASK, w='bold')

# ---------------------------------------------------------------- 5 output spectrum (real held-out pair, both spins)
G.chevron(ax, 148.0, 45.5, TASK, size=1.2)
G.card(ax, 150, 20, 31, 51, TASK, heading='2 × 2,001 values')
G.text(ax, 152.2, 63.0, 'd-DOS, −10…10 eV, 0.01 eV', fs=4.8, c=S.INK2, ha='left')
m = (E_GRID >= -4) & (E_GRID <= 4)
ins = fig.add_axes([152.0 / W, 27.0 / H, 27.0 / W, 33.0 / H])
ins.fill_between(E_GRID[m], 0, Tv[:2001][m], color='#C9D2DC', lw=0)
ins.fill_between(E_GRID[m], 0, -Tv[2001:][m], color='#C9D2DC', lw=0)
ins.plot(E_GRID[m], Pv[:2001][m], color=S.SET_COLOR['prediction'], lw=0.6)
ins.plot(E_GRID[m], -Pv[2001:][m], color=S.SET_COLOR['prediction'], lw=0.6)
ins.axvline(0, color=S.INK2, lw=0.4, ls=':'); ins.axhline(0, color=S.INK2, lw=0.3)
ins.set_xticks([]); ins.set_yticks([])
for s_ in ins.spines.values(): s_.set_visible(False)
ins.patch.set_alpha(0)
G.text(ax, 153.0, 58.6, '↑ spin up', fs=4.6, c=S.INK2, ha='left')
G.text(ax, 153.0, 28.6, '↓ spin down', fs=4.6, c=S.INK2, ha='left')
G.text(ax, 165.5, 23.6, f'held-out {pick.formula}: DFT (grey),\nprediction (blue), r = {pick.pearson_r:.2f}', fs=4.4, c=S.INK2)

# ---------------------------------------------------------------- loss strip (training only)
G.card(ax, 2, 2, 179, 14.5, TASK, heading='Training loss: five weighted terms on the softplus output vs the DFT target')
terms = [('0.35', 'Fermi-weighted MSE', '−∂f/∂E, 700 K'), ('0.35', 'MSE within ±2 eV', 'both spins'),
         ('0.15', 'cosine distance', 'whole vector'), ('0.10', 'slope of the DOS', '±3 eV, spin up'),
         ('0.05', 'L1 of sorted values', 'shape histogram')]
for i, (wgt, lab, sub) in enumerate(terms):
    x0 = 5 + i * 29.5
    G.badge(ax, x0 + 2.4, 7.0, wgt, TASK, r=2.4, fs=4.6)
    G.text(ax, x0 + 5.6, 8.4, lab, fs=5.0, ha='left', c=S.INK, w='bold')
    G.text(ax, x0 + 5.6, 5.2, sub, fs=4.5, ha='left', c=S.INK2)
G.text(ax, 166.5, 7.0, 'two further terms in the\ncode had zero weight', fs=4.6, c=S.PURPLE)

qa.check(fig, 'fig3_model')
S.save(fig, OUT / 'fig3_model', width_mm=183)
print('fig3 example', pick.formula, round(pick.pearson_r, 3))
