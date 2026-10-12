# paper/manuscript_v1: earlier draft (superseded)

This folder holds the analyses, literature audit and figure scripts of an earlier draft of the manuscript (v13). It is
**superseded by `../manuscript_v19/`**, which holds the code of the current manuscript, *High-throughput screening of Heusler
alloys for thermoelectrics combining machine-learned densities of states and first-principles transport*. Two parts remain in
use, unchanged:

- **Supplementary figures and tables of the current manuscript** are still produced by scripts here: `make_si_figures.py`,
  `make_si_v7.py`, `fig2_data_target.py`, `fig3_model.py`, `fig4_training_inference.py`, `figs_v4.py`,
  `fig3_dos_to_indicator.py`, `make_tables.py` and `known_hh_check.py` in `scripts/figures/` (the manuscript build uses the same
  code). Which script draws which item is listed in `../manuscript_v19/README.md`.
- **Records in `figures/data/`** are read by the current scripts (`RECORDS` in `../manuscript_v19/scripts/figures/common.py`),
  several of which assert that they reproduce them.

Not current: the main-figure scripts of the earlier draft (`fig1_workflow.py`, `fig2_representation.py`, `fig5_funnel.py`,
`fig6_screening_vs_dft.py`, `fig7_candidates.py`, `fig8_soc_polymorph.py`), the literature table in `literature/` (before the
citation audit of the current manuscript; use `../manuscript_v19/literature/`), and `analysis/verify_v12.py` (checks the numbers
of the earlier text and needs its sources). The other `analysis/` scripts and their outputs are the same as in
`../manuscript_v19/analysis/` (there `polymorph_variants.py` reads its DXMag record from this folder).
`make_dataset_table.py`, `dataset_characterization.py`, `dos_normalisation_check.py`, `read_tb_scalars.py`, `palette_check.py`
and `v7_claim_checks.py` are helpers that wrote records in `figures/data/`.

Run every script from the repository root with `python3 <path>`; outputs go to `figures/`, `figures/data/` and `tables/`
here. Inputs that are not shipped are listed in `../../DATA.md`.
