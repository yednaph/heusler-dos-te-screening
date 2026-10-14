# paper/manuscript_v19: figure, analysis and literature code of the current manuscript

Code behind *High-throughput screening of Heusler alloys for thermoelectrics combining machine-learned densities of states and
first-principles transport* (S. Randive, R. Dixit, H. Pandey; manuscript in preparation), copied from the manuscript
build. It supersedes `../manuscript_v1/`, which holds the code of an earlier draft; the scripts for the Supplementary figures and
tables did not change and are used from there (table below).

The manuscript text, captions, figures and review files are not included. Run every script from the repository root with
`python3 <path>`. Project paths resolve from `HEUSLER_TE_ROOT` (default: the repository root) and the model export from
`HEUSLER_TE_EXPORT` (default `$HEUSLER_TE_ROOT/DOS_EXPORT`); the absolute paths of the original machine were replaced by these
two settings, so adapt them, not the scripts, if your data live elsewhere. Outputs go to `figures/` (PDF and PNG),
`figures/data/` (every number a figure draws) and `tables/`. Inputs that are not shipped, and where to get them, are listed in
`../../DATA.md`.

| Folder | Content |
|---|---|
| `scripts/figures/` | `v14_fig*.py`, the nine main-text figures; `common.py`, paths and data loaders (the shared style and QA gate are in `../../figkit/`) |
| `scripts/assemble_v14.py`, `scripts/renumber_si.py` | Build helpers of the manuscript: assemble the section files from their sources and renumber the Supplementary Information in order of first citation. They need the manuscript sources and are included for completeness; the table captions that `assemble_v14.py` embedded in the build are not shipped |
| `analysis/` | `soc_paired.py` (607 scalar/SOC pairs → `soc_paired.csv`, `soc_summary.txt`; reads the run directories), `soc_table.py` (SOC by heavy element → `tables/si_tab_soc.tex`), `polymorph_variants.py` (824 structural variants → `polymorph_variants.csv`, `polymorph_summary.txt`), `phonon_status.py`; `humanizer_scan.py` is a phrasing diagnostic that needs the manuscript sources. The verification script that recomputes the numbers quoted in the text will be added when the manuscript is published, because it quotes the unpublished text |
| `literature/` | Literature audit of the 80 certified compositions after the citation audit of this manuscript: `lit_part1–3.csv` (sources with DOIs), merged by `consolidate.py` into `literature_status.csv` and `tables/si_tab_literature.tex` (orthorhombic phase reported for 17 and predicted for 6 compositions) |

Some scripts read records written for the earlier draft (`fig3_funnel_counts.csv`, `fig1v13_population.csv`,
`fig2v13_indicator.csv`, `fig2v9_entries.csv`, `known_hh_check.csv`) from `../manuscript_v1/figures/data/` (`RECORDS` in
`scripts/figures/common.py`); several assert that they reproduce them.

## Printed figures and tables → script → inputs

Script numbers follow the order in which the figures were built, not the printed order. "Shipped" means the script runs from
this repository alone; otherwise the external inputs named are also needed (`DATA.md`). Supplementary items are named by their
caption, because their numbering is not final.

| Printed item | Script | Inputs |
|---|---|---|
| Fig. 1, two d-DOS populations, one screen, first-principles judge | `scripts/figures/v14_fig1_strategy.py` | shipped results and records + both screening tables + held-out dump (ZrNiSn, half_cubic fold 4) of the model export |
| Fig. 2, the d-DOS model and its Fermi-focused loss | `scripts/figures/v14_fig2_model.py` | `train_hpc.py`, `code/dos_dataset.py` + the 30 per-fold configs and one SWA checkpoint (Zenodo) + per-sample metrics and held-out dumps of the export |
| Fig. 3, predicted spectra reproduce the d-DOS shape but miss most gaps | `scripts/figures/v14_fig3_fidelity.py` | per-sample metrics and held-out dumps of the export; `code/te_rescreen.py`; checks the per-family table written by `../manuscript_v1/scripts/figures/make_tables.py` if it exists |
| Fig. 4, gap label against QE-PBE; the indicator places the certified entries at the top | `scripts/figures/v14_fig5_screening.py` | `results/paper_analysis/` + both screening tables |
| Fig. 5, certification cascade | `scripts/figures/v14_fig6_cascade.py` | shipped |
| Fig. 6, what the Seebeck indicator measures | `scripts/figures/v14_fig4_indicator.py` | `paper_dataset.pkl`, `code/te_rescreen.py` + held-out dump (ZrNiSn) + `CMS/anchor/runs_local/NiZrSn__0eeb5376/transport_v2.json` |
| Fig. 7, spin–orbit coupling | `scripts/figures/v14_fig8_soc.py` | shipped |
| Fig. 8, structural variants | `scripts/figures/v14_fig9_variants.py` | shipped |
| Fig. 9, certified compositions in context | `scripts/figures/v14_fig7_candidates.py` | shipped |
| Tables 1–3 (populations and screen; first Pareto layer; what each result establishes) | none: typeset in the manuscript sources | numbers from `results/paper_analysis/` (Table 2: `ranking_fixed_n.xlsx`) and the screening tables |
| SI figures: held-out fidelity by family; energy-resolved error; selection of entries for DFT; effect of the mesh correction; effect of SOC | `../manuscript_v1/scripts/figures/make_si_figures.py` | export metrics, both screening tables, `results/paper_analysis/si/B15_mesh_bug_per_run.csv` (not shipped) |
| SI figures: training record of cubic full-Heusler fold 1; normalisation of the DOS archive (and the per-family characterisation table) | `../manuscript_v1/scripts/figures/make_si_v7.py` | shipped |
| SI figure: the two sets and the learning target | `../manuscript_v1/scripts/figures/fig2_data_target.py` | DXMag metadata + held-out dumps + export metrics |
| SI figure: the d-DOS model, stock encoder with task-specific head | `../manuscript_v1/scripts/figures/fig3_model.py` | held-out dumps + export metrics |
| SI figure: how the 30 deployed models were obtained and applied | `../manuscript_v1/scripts/figures/fig4_training_inference.py` | shipped |
| SI figures: the screen's labels tested by QE-PBE; workflow revisions acting on the screen; what survives, and how robustly; held-out fidelity in detail | `../manuscript_v1/scripts/figures/figs_v4.py` | `results/paper_analysis/` + both screening tables + held-out dumps + export metrics |
| SI figure: the indicator for a prediction-set entry (CoTiSb) | `../manuscript_v1/scripts/figures/fig3_dos_to_indicator.py` | predicted d-DOS arrays (Zenodo) + `CMS/anchor/runs_local` |
| SI tables: held-out fidelity by family; indicator triage metrics; paired-bootstrap ROC-AUC differences; sensitivity of the counts; all 80 certified compositions; provenance by selection round; earlier Pareto set | `../manuscript_v1/scripts/figures/make_tables.py` | `results/paper_analysis/` + export metrics + both screening tables; the selection-round table needs `results/paper_analysis/si/B15_queue_provenance.csv` (not shipped) |
| SI table: spin–orbit coupling by heavy element | `analysis/soc_table.py` | shipped |
| SI table: prior thermoelectric evidence for the 80 compositions | `literature/consolidate.py` | shipped |
| SI table: established half-Heusler thermoelectrics in the campaign | `../manuscript_v1/scripts/figures/known_hh_check.py` | shipped |
| SI tables of settings, parent models, normalisation consequences and workflow development | none: typeset in the manuscript sources | — |

The manuscript's SI tables label the rows with the indicator names of the text (S^d_max, P^d_max, ξ, L₀*, "initial composite",
"electron count"), where `make_tables.py` writes the code names (A, B, C, D, D34, VEC); the numbers are the same.

**Checked for this release.** With the inputs above, the nine `v14_fig*.py` scripts pass their assertions and the `figkit`
QA gate and reproduce all 39 `figures/data/v14_fig*.csv` records of the manuscript build byte for byte. `soc_paired.py`
reproduces `soc_paired.csv` and `soc_summary.txt` from the run directories, and `soc_table.py`, `polymorph_variants.py`,
`phonon_status.py` and `consolidate.py` reproduce their outputs from the shipped files alone, as do the scripts of Figs. 5, 7, 8
and 9.
