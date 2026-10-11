"""Fig. 4 (v7) -- transfer learning, per-family training and prediction-set inference (diagram + one real curve).

Question: how were the 30 deployed models obtained, which checkpoint does what, and how did the prediction set get its d-DOS?
5-second takeaway: a foundation model trained on every training-set spectrum seeds the family models; each family is trained
in five folds; the best-validation checkpoint gives the reported metrics, the weight-averaged (SWA) checkpoint gives the
predictions; the prediction set is routed to its family's fold models and their mean is the predicted d-DOS.

Evidence (verification status shown on the figure):
  foundation: configs/pretrain.yaml (100 epochs, SWA from 85), create_pretrain_dataset.py (all 51,257, no exclusions)
  foundation -> full cubic, encoder lr x0.1: output/full_cubic/expA/fold1/run_metadata.json + optimiser state of folds 0-1
  other parents: training plan only (presentation slide 20, submit_transfer.sh); four sources disagree (SI Table)
  curve: TensorBoard log of full cubic fold 1 (read_tb_scalars.py -> figures/data/tb_full_cubic_fold1.csv)
  inference: generate_prediction_cifs_from_dxmag.py, predict_from_cif.py, scripts/screen_<family>.sh (5 SWA members),
             screening_summary.txt (5 checkpoints listed) for full cubic and inverse cubic only
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.patches as mp
SKILL = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SKILL))
from figkit import style as S, glyphs as G, qa                       # noqa: E402
from common import DATA, OUT                                          # noqa: E402

S.apply()
tb = pd.read_csv(DATA / 'tb_full_cubic_fold1.csv').pivot(index='step', columns='tag', values='value')
best_epoch = int(tb['Pearson/val_global'].iloc[:120].idxmax())        # best.pt is chosen before SWA starts
swa_start = int(tb.index[tb['SWA/active'] > 0.5].min())
assert best_epoch == 2 and swa_start == 120 and len(tb) == 150
pd.Series({'best_epoch': best_epoch, 'best_val_r': tb['Pearson/val_global'].iloc[best_epoch], 'swa_start': swa_start,
           'last_val_r': tb['Pearson/val_global'].iloc[-1]}).to_csv(DATA / 'fig4_curve_summary.csv')

W, H = 183, 89
fig, ax = G.canvas(W, H)
T_C, P_C = S.SET_COLOR['training'], S.SET_COLOR['prediction']

# ================================================================ a  transfer-learning lineage
G.card(ax, 2, 4, 58, 79, S.NAVY, heading='Transfer-learning lineage')
G.stat_tile(ax, 9, 62.5, 44, 8.5, '51,257 spectra · 6 families', 'all training-set entries', T_C, fs_count=6.6, fs_label=4.9)
G.node(ax, 31, 56.0, 34, 4.2, 'foundation model · 100 epochs', S.NAVY, fs=5.4)
G.text(ax, 31, 51.6, 'includes every later test fold', fs=4.7, c=S.PURPLE, w='bold')
ax.plot([31, 31], [49.6, 46.8], color=S.NAVY, lw=1.1); G.chevron(ax, 31, 46.6, S.NAVY, size=0.9, direction='down')
G.node(ax, 31, 43.4, 18, 4.0, 'full cubic', S.TEAL, fs=5.4)
G.text(ax, 51.5, 46.5, 'verified\n(folds 0, 1):\nencoder lr ×0.1', fs=4.4, c=S.NAVY, ha='center')
# planned edges (dashed): full cubic -> {full tetra, inverse cubic, half cubic}; inverse cubic -> inverse tetra; half cubic -> half tetra
DASH = (0, (2.2, 1.4))
kids = [(13.5, 31.5, 'full tetra', False), (31, 31.5, 'inv. cubic', True), (48.5, 31.5, 'half cubic', True)]
ax.plot([31, 31], [41.4, 37.0], color=S.ARROW, lw=0.7, ls=DASH)
ax.plot([13.5, 48.5], [37.0, 37.0], color=S.ARROW, lw=0.7, ls=DASH)
for x, y, lab, cub in kids:
    ax.plot([x, x], [37.0, y + 2.3], color=S.ARROW, lw=0.7, ls=DASH); G.chevron(ax, x, y + 2.3, S.ARROW, size=0.7, direction='down')
    if cub:
        G.node(ax, x, y, 15, 3.6, lab, S.TEAL, fs=5.0)
    else:
        ax.add_patch(mp.FancyBboxPatch((x - 7.5, y - 1.8), 15, 3.6, boxstyle='round,pad=0,rounding_size=0.8', fc='white', ec=S.TEAL, lw=0.7, zorder=5))
        G.text(ax, x, y, lab, fs=5.0, c=S.TEAL, w='bold')
for x, lab in [(31, 'inv. tetra'), (48.5, 'half tetra')]:
    ax.plot([x, x], [29.7, 25.9], color=S.ARROW, lw=0.7, ls=DASH); G.chevron(ax, x, 25.8, S.ARROW, size=0.7, direction='down')
    ax.add_patch(mp.FancyBboxPatch((x - 7.5, 22.0), 15, 3.6, boxstyle='round,pad=0,rounding_size=0.8', fc='white', ec=S.TEAL, lw=0.7, zorder=5))
    G.text(ax, x, 23.8, lab, fs=5.0, c=S.TEAL, w='bold')
ax.plot([5, 10], [16.4, 16.4], color=S.NAVY, lw=1.1); G.text(ax, 11.2, 16.4, 'recorded in run files', fs=4.7, ha='left', c=S.INK2)
ax.plot([5, 10], [12.9, 12.9], color=S.ARROW, lw=0.7, ls=DASH); G.text(ax, 11.2, 12.9, 'training plan only; sources disagree', fs=4.7, ha='left', c=S.INK2)
G.text(ax, 31, 8.0, 'filled: cubic · open: tetragonal\nweights loaded in full, head included', fs=4.6, c=S.INK2)

# ================================================================ b  per-family training and the two checkpoints
G.card(ax, 63, 4, 57, 79, S.NAVY, heading='Training each family (×6)')
for x_, lab_, w_ in [(66.0, 'test 20%', 13.5), (81.5, 'validation 8%', 17.5), (101.0, 'train 72%', 15.5)]:   # partitions, not steps
    G.pills(ax, x_, 70.0, [lab_], S.NAVY, h=3.8, widths=[w_], fs=4.7)
G.text(ax, 91.5, 75.6, 'per fold of a shuffled five-fold split (seed 42)', fs=4.6, c=S.INK2)
G.text(ax, 91.5, 64.9, 'AdamW · plateau schedule · 300 epochs\n(full cubic 150–200) · SWA over last 20%', fs=4.7, c=S.INK2)
# real training curve: full cubic fold 1
ins = fig.add_axes([71.0 / W, 30.5 / H, 45.0 / W, 26.0 / H])
ep = tb.index.values
ins.axvspan(swa_start, ep.max(), color=S.TINT[S.GREEN], lw=0)
ins.plot(ep, tb['Pearson/val_global'], color=S.NAVY, lw=0.8, label='global $r$')
ins.plot(ep, tb['Pearson/val_fermi'], color=S.TEAL, lw=0.8, label='±2 eV $r$')
ins.plot([best_epoch], [tb['Pearson/val_global'].iloc[best_epoch]], marker='o', ms=3.2, color=S.PURPLE, zorder=5)
ins.set_xlim(0, 150); ins.set_ylim(0.74, 0.88)
ins.set_xlabel('epoch (full cubic, fold 1)', fontsize=5.0, labelpad=1); ins.set_ylabel('validation $r$', fontsize=5.0, labelpad=1)
ins.tick_params(labelsize=5.0, length=1.6, pad=1)
ins.legend(loc='lower left', fontsize=5.0, handlelength=1.2, borderaxespad=0.2, ncol=2)
ins.text(best_epoch + 4, tb['Pearson/val_global'].iloc[best_epoch] + 0.004, f'best-validation (epoch {best_epoch})', fontsize=5.0, color=S.PURPLE)
ins.text(135, 0.873, 'SWA', fontsize=5.0, color=S.GREEN, ha='center')
# the two checkpoints and their uses
G.node(ax, 77.5, 19.5, 22, 4.0, 'best-validation', S.PURPLE, fs=5.2)
G.text(ax, 77.5, 13.6, 'held-out metrics\n(Fig. 4)', fs=4.7, c=S.INK2)
G.node(ax, 105.5, 19.5, 22, 4.0, 'SWA weights', S.GREEN, fs=5.4)
G.text(ax, 105.5, 13.6, 'deployed: 30 models\n(6 families × 5 folds)', fs=4.7, c=S.INK2)
G.text(ax, 91.5, 7.2, 'reported metrics and predictions come from different checkpoints', fs=4.5, c=S.PURPLE)

# ================================================================ c  prediction-set inference
G.card(ax, 123, 4, 58, 79, S.BLUE, heading='Prediction-set inference')
steps = [('54,978 DXMag entries without DOS', 'stat'),
         ('CIF from the listed cell and positions', 'step'),
         ('family from prototype + space group', 'step'),
         ('graph: same settings as training', 'step'),
         ('5 SWA fold models of that family', 'models'),
         ('softplus, then mean and spread', 'step')]
y = 70.0
for k, (lab, kind) in enumerate(steps):
    if kind == 'stat':
        G.stat_tile(ax, 127, y - 3.6, 50, 7.2, '54,978', 'entries without DOS', P_C, fs_count=6.8, fs_label=4.6)
    elif kind == 'models':
        for i in range(5):
            ax.add_patch(mp.FancyBboxPatch((128.5 + i * 9.6, y - 1.9), 8.0, 3.8, boxstyle='round,pad=0,rounding_size=0.8',
                                           fc=S.GREEN, ec='none', zorder=5))
            G.text(ax, 132.5 + i * 9.6, y, f'fold {i}', fs=4.5, c='white', w='bold')
        G.text(ax, 152, y - 4.3, lab, fs=4.8, c=S.INK)
    else:
        ax.add_patch(mp.FancyBboxPatch((127, y - 2.1), 50, 4.2, boxstyle='round,pad=0,rounding_size=2.1', fc='white', ec=S.BLUE, lw=0.7, zorder=4))
        G.text(ax, 152, y, lab, fs=4.9, c=S.BLUE, w='bold')
    if k < len(steps) - 1:
        G.chevron(ax, 152, y - (5.6 if kind == 'stat' else (6.6 if kind == 'models' else 3.7)), S.ARROW, size=0.75, direction='down')
    y -= 8.0 if kind != 'models' else 10.0
# outputs
G.chevron(ax, 152, 23.3, S.ARROW, size=0.75, direction='down')
G.node(ax, 139.5, 18.0, 23, 4.2, 'mean d-DOS $\\hat g(E)$', S.BLUE, fs=5.0)
G.node(ax, 166.0, 18.0, 23, 4.2, 'spread $u_s(E)$, $u_F$', S.BLUE, fs=5.0)
G.text(ax, 152, 12.4, '4,002 values per entry, files per family\n→ screening descriptors', fs=4.7, c=S.INK2)
G.text(ax, 152, 7.0, 'five members in all six scripts; logs kept for two families', fs=4.5, c=S.PURPLE)

# panel letters outside the cards
for x, l in [(2, 'a'), (63, 'b'), (123, 'c')]:
    G.text(ax, x + 0.2, 86.4, l, fs=9, w='bold', ha='left', c=S.INK)

qa.check(fig, 'fig4_training_inference')
S.save(fig, OUT / 'fig4_training_inference', width_mm=183)
print('best epoch', best_epoch, 'swa start', swa_start)
