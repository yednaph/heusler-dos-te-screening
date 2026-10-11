"""v7 Supplementary items built from stored records only:
  Fig. S6  training record of cubic full-Heusler fold 1 (TensorBoard scalars, read_tb_scalars.py output)
  Fig. S7  DOS-normalisation audit (dos_normalisation_check.py output)
  Table S8 per-family characterisation of the two sets (dataset_characterization.py output)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
SKILL = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SKILL))
from figkit import style as S, qa                                     # noqa: E402
from common import DATA, OUT, TABLES, FAMILIES                        # noqa: E402

S.apply()
# ---------------------------------------------------------------- Fig. S6: training record
tb = pd.read_csv(DATA / 'tb_full_cubic_fold1.csv').pivot(index='step', columns='tag', values='value')
ep = tb.index.values; swa0 = int(tb.index[tb['SWA/active'] > 0.5].min()); best = int(tb['Pearson/val_global'].iloc[:swa0].idxmax())
fig, axs = plt.subplots(1, 3, figsize=(183 * S.MM, 52 * S.MM), gridspec_kw=dict(wspace=0.42, left=0.075, right=0.985, bottom=0.21, top=0.9))
for ax in axs:
    ax.axvspan(swa0, ep.max(), color=S.TINT[S.GREEN], lw=0)
    ax.axvline(best, color=S.PURPLE, lw=0.7, ls=':')
    ax.set_xlabel('epoch')
ax = axs[0]
ax.plot(ep, tb['Loss/train'], color=S.NAVY, lw=0.9, label='training'); ax.plot(ep, tb['Loss/val'], color=S.ORANGE, lw=0.9, label='validation')
ax.set_ylabel('loss'); ax.legend(loc='upper right'); S.panel(ax, 'a', x=-0.26)
ax = axs[1]
ax.plot(ep, tb['Pearson/val_global'], color=S.NAVY, lw=0.9, label='concatenated spectra')
ax.plot(ep, tb['Pearson/val_fermi'], color=S.TEAL, lw=0.9, label='±2 eV, per spectrum')
ax.set_ylabel('validation Pearson $r$'); ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.55)); S.panel(ax, 'b', x=-0.3)
ax.text(best + 3, 0.874, f'best-validation checkpoint (epoch {best})', color=S.PURPLE, fontsize=5.4)
ax.set_ylim(0.74, 0.885)
ax = axs[2]
ax.semilogy(ep, tb['LR/current'], color=S.NAVY, lw=0.9)
ax.set_ylabel('learning rate (encoder group)'); S.panel(ax, 'c', x=-0.3)
ax.text((swa0 + ep.max()) / 2, ax.get_ylim()[1] * 0.5, 'SWA', color=S.GREEN, ha='center', fontsize=5.6)
qa.check(fig, 'si_S6'); S.save(fig, OUT / 'si_S6_training_record', width_mm=183)
pd.Series({'best_epoch': best, 'swa_start': swa0, 'train_loss_first': tb['Loss/train'].iloc[0], 'train_loss_last': tb['Loss/train'].iloc[-1],
           'val_loss_first': tb['Loss/val'].iloc[0], 'val_loss_last': tb['Loss/val'].iloc[-1]}).to_csv(DATA / 'si_S6_summary.csv')

# ---------------------------------------------------------------- Fig. S7: normalisation audit
nc = pd.read_csv(DATA / 'dos_normalisation_check.csv')
fig, ax = plt.subplots(figsize=(89 * S.MM, 52 * S.MM), gridspec_kw=dict(left=0.25, right=0.97, bottom=0.2, top=0.95))
for i, f in enumerate(FAMILIES):
    v = nc[nc.family == f].ratio.values
    ax.scatter(v, np.full(len(v), i) + np.linspace(-0.18, 0.18, len(v)), s=5, color=S.NAVY, lw=0)
ax.axvline(1, color=S.INK2, lw=0.6, ls=':'); ax.axvline(2, color=S.ORANGE, lw=0.6, ls=':')
ax.set_yticks(range(len(FAMILIES))); ax.set_yticklabels([f.replace('_', ' ') for f in FAMILIES]); ax.invert_yaxis()
ax.set_xlabel('DOS-integrated moment / DXMag moment'); ax.set_xlim(0.8, 2.4)
qa.check(fig, 'si_S7'); S.save(fig, OUT / 'si_S7_dos_normalisation', width_mm=89)

# ---------------------------------------------------------------- Table S8: per-family characterisation
d = pd.read_csv(DATA / 'dataset_characterization_by_family.csv').query('family != "all" and set != "all"')
ov = pd.read_csv(DATA / 'dataset_characterization_overlap.csv')
print(ov.columns.tolist())
rows = []
for f in FAMILIES:
    for st in ['training', 'prediction']:
        r = d[(d.family == f) & (d.set == st)].iloc[0]
        rows.append(f"{f.replace('_', ' ') if st == 'training' else ''} & {'T' if st == 'training' else 'P'} & {int(r.entries):,} & "
                    f"{int(r.unique_composition_strings):,} & {int(r.atoms_cell_median)} & {100 * r['share_hull_lt_0.10']:.1f} & "
                    f"{r.hull_median_eV:.2f} & {100 * r['share_absM_gt_0.1_of_available']:.1f} & {100 * r.share_vec_18:.1f} & "
                    f"{100 * r.share_vec_24:.1f} \\\\")
(TABLES / 'si_tab_dataset_family.tex').write_text(
    '\\begin{tabular}{@{}llrrrrrrrr@{}}\n\\toprule\nfamily & set & entries & compositions & atoms/cell & hull$<$0.10 (\\%) & '
    'median hull & $|M|>0.1$ (\\%) & 18 e$^-$ (\\%) & 24 e$^-$ (\\%) \\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')
print('si done; best', best, 'swa', swa0)
