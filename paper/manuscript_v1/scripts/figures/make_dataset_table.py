"""Table (v7): characterisation of the training and prediction sets, from dataset_characterization.py outputs and the
Fig. 2 per-entry cache. Writes tables/tab_dataset.tex. Numbers are read from figures/data/*.csv, except the phonon counts, taken from the DXMag "phonon computed" column (8,198 True, all in the training set; checked 2026-10-10)."""
import pandas as pd
from common import DATA, TABLES

d = pd.read_csv(DATA / 'dataset_characterization_by_family.csv').query('family == "all"').set_index('set')
dE = pd.read_csv(DATA / 'fig2_dE_summary.csv', index_col=0).iloc[:, 0]
T, P = d.loc['training'], d.loc['prediction']
pct = lambda v: f'{100 * v:.1f}\\%'                                   # share -> percent string
rows = [
    ('entries (DXMag UUIDs)', f"{int(T.entries):,}", f"{int(P.entries):,}"),
    ('entries per family + composition', '1', f"1--{int(P.max_entries_per_family_str)}"),
    ('compositions: strings / formulas', f"{int(T.unique_composition_strings):,} / {int(T.unique_reduced_formulas):,}",
     f"{int(P.unique_composition_strings):,} / {int(P.unique_reduced_formulas):,}"),
    ('elements', f"{int(T.distinct_elements)}", f"{int(P.distinct_elements)}"),
    ('atoms in listed cell, median (range)', f"{int(T.atoms_cell_median)} ({int(T.atoms_cell_min)}--{int(T.atoms_cell_max)})",
     f"{int(P.atoms_cell_median)} ({int(P.atoms_cell_min)}--{int(P.atoms_cell_max)})"),
    ('structure for the graph', 'stored with DOS', 'listed DXMag cell'),
    ('spin-resolved DFT DOS', 'yes (archive)', 'no (predicted)'),
    ('$\\Delta E$ to training sibling$^{a}$', '0 (lowest entry)', f"median {dE['median_meV']:.0f} meV\\,atom$^{{-1}}$; {pct(dE['share_lt_1meV_all'])} $<1$ meV\\,atom$^{{-1}}$"),
    ('median hull distance (eV\\,atom$^{-1}$)', f"{T.hull_median_eV:.2f}", f"{P.hull_median_eV:.2f}"),
    ('hull distance $<0.10$ eV\\,atom$^{-1}$', pct(T['share_hull_lt_0.10']), pct(P['share_hull_lt_0.10'])),
    ('formation energy $<0$', pct(T.share_formation_lt_0), pct(P.share_formation_lt_0)),
    ('$|M|>0.1\\,\\mu_B$/f.u.$^{b}$', pct(T['share_absM_gt_0.1_of_available']), pct(P['share_absM_gt_0.1_of_available'])),
    ('moment missing (cubic only)', f"{int(T.mag_missing_n):,}", f"{int(P.mag_missing_n):,}"),
    ('phonon results', '8,198', '0'),
    ('18 / 24 valence electrons', f"{pct(T.share_vec_18)} / {pct(T.share_vec_24)}", f"{pct(P.share_vec_18)} / {pct(P.share_vec_24)}"),
]
body = '\n'.join(f'{a} & {b} & {c} \\\\' for a, b, c in rows)
(TABLES / 'tab_dataset.tex').write_text(
    '\\begin{tabular}{@{}>{\\raggedright}p{0.36\\columnwidth} >{\\raggedright}p{0.22\\columnwidth} >{\\raggedright\\arraybackslash}p{0.32\\columnwidth}@{}}\n'
    '\\toprule\n & training set & prediction set \\\\\n\\midrule\n' + body + '\n\\bottomrule\n\\end{tabular}\n')
print((TABLES / 'tab_dataset.tex').read_text())
