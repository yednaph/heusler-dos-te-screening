# Data needed beyond this repository

| Data | Used by | Where to obtain | Place under `$HEUSLER_TE_ROOT` |
|---|---|---|---|
| DXMag HeuslerDB metadata, release 2025-09-09 (CC BY 4.0) | dataset scripts, Figs. 1, 3, SOC/variant energies | NIMS spin-theory group database site (Xiao & Tadano) | `data/DXMag_HeuslerDB_2025-09-09.xlsx` |
| HeuslerDB electronic-DOS archive (pymatgen JSON, CC BY 4.0) | `code/dos_dataset.py`, training-set screen | doi:10.48505/nims.5794 (51,373 spectra documented; 51,257 used) | `data/<family>/dos_json/` |
| Trained model checkpoints and held-out predictions | inference, Fig. 2, ML metrics | to be deposited (Zenodo DOI to be added) | `$HEUSLER_TE_EXPORT` (default `DOS_EXPORT/`) |
| Prediction-set ensemble spectra | prediction-set screen, Fig. 2/SI | regenerate with `code/run_inference.py`, or deposit | `results/screening/<family>/novel_expA_ensemble/` |
| Screening tables | Fig. 3 | regenerate with `code/te_rescreen*.py` | `results/screening/` |
| QE/BoltzTraP2 run directories (`scf/nscf.out`, `transport*.json`) | `code/paper/build_paper_dataset.py`, SOC analysis | to be deposited; or recompute with `CMS/anchor/` | `CMS/anchor/runs_local/`, `CMS/anchor/soc_campaign/runs/` |

Shipped and sufficient for the analyses in `paper/manuscript_v1/analysis` and `literature`: `results/paper_analysis/paper_dataset.pkl`,
`paper/manuscript_v1/analysis/*.csv` and the small CSVs in `paper/manuscript_v1/figures/data/`.

Software: Quantum ESPRESSO 7.1 (PBE, SSSP efficiency; fully relativistic PSlibrary for SOC), BoltzTraP2 24.7.2 (with
`CMS/anchor/boltztrap2_patch`), ALIGNN 2026.4.1 with PyTorch 2.2.2 + CUDA 12.1, DGL 1.1.3+cu121, jarvis-tools 2026.4.2, NumPy 1.26.4 on
Python 3.11.15 (recorded in the export's `environment/key_versions.json`; pymatgen version not recorded).
