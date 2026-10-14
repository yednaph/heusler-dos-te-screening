# Data: what is shipped, what is deposited, what must be obtained

All paths below are relative to `$HEUSLER_TE_ROOT`, which defaults to the repository root. The model export is
read from `$HEUSLER_TE_EXPORT` (default `$HEUSLER_TE_ROOT/DOS_EXPORT`).

## 1. Shipped in this repository

| Path | Content | Written by |
|---|---|---|
| `results/paper_analysis/paper_dataset.pkl` | One row per first-principles run (3,625 runs, 407 columns): DXMag identifiers and metadata, screening inputs, QE-PBE class and gap, BoltzTraP2 transport summaries, SOC results, certification chain | `code/paper/build_paper_dataset.py` |
| `results/paper_analysis/paper_dataset.csv` | Plain-text copy of the pickle (3,625 × 382; 25 screening columns omitted, see `results/paper_analysis/README.md`) | `code/paper/export_csv.py` |
| `results/paper_analysis/certified_entries.csv` | The 155 certified entries (80 compositions), reduced column set | `code/paper/export_csv.py` |
| `results/paper_analysis/*.xlsx`, `*.txt` | Screening evaluation, threshold sensitivity, fixed-carrier-concentration rankings, near-duplicates, ML metrics, build check | `code/paper/*.py` (see `results/paper_analysis/README.md`) |
| `paper/manuscript_v19/analysis/*.csv`, `*.txt` | 607 paired scalar/SOC runs, 824 structural variants, phonon status | `paper/manuscript_v19/analysis/*.py` |
| `paper/manuscript_v19/literature/*.csv` | Literature status of the 80 certified compositions after the citation audit of the current manuscript, with sources and DOIs | `paper/manuscript_v19/literature/consolidate.py` |
| `paper/manuscript_v1/` | Earlier draft (superseded): its analyses, literature table before the audit, figure scripts (those of the Supplementary figures and tables are still used) and the small records in `figures/data/` that the current scripts read | `paper/manuscript_v1/**/*.py` |

DXMag identifiers and metadata in these files (UUID, composition, space group, atoms per cell, energies, hull distance,
phonon flags) are copied from the DXMag database (E. Xiao and T. Tadano, CC BY 4.0); the DXMag database is the authoritative
source. The DFT reference spectra of the HeuslerDB electronic-DOS dataset are **not** redistributed here or in the data record.

## 2. The predicted d-DOS record (Zenodo)

*Predicted spin-resolved d-orbital densities of states for 54,978 DXMag Heusler entries*, version 1.0.0,
doi:[10.5281/zenodo.23329059](https://doi.org/10.5281/zenodo.23329059), CC BY 4.0. It holds the machine-learned d-DOS of the
prediction set (five-model mean and spread), the held-out predictions of the same models for the 51,257 training-set entries,
the prediction-set screening table, the input structures in space group P 1, an index of DXMag identifiers and metadata, and
the 30 SWA model weights with their per-fold configurations (`model_assets.zip`). It contains no DFT spectra.

The record holds the files as separate downloads; Zenodo's "Download all" button fetches them together. Put them in one
folder, here called `heusler_predicted_ddos_v1/`, and check them:

```bash
cd heusler_predicted_ddos_v1 && sha256sum -c SHA256SUMS.txt
```

### Record files and the paths the scripts expect

| Record file (in `heusler_predicted_ddos_v1/`) | Path used by the code | Identical to the original? |
|---|---|---|
| `<family>_ddos_mean.npy` | `results/screening/<family>/novel_expA_ensemble/predictions.npy` | byte-identical |
| `<family>_ddos_std.npy` | `results/screening/<family>/novel_expA_ensemble/predictions_std.npy` | byte-identical |
| `<family>_ids.txt` | `results/screening/<family>/novel_expA_ensemble/ids.txt` | byte-identical |
| `screen_prediction_set.csv` | `results/screening/rescreen_gapfix_ALL_annotated.csv` | byte-identical |
| `structures_P1_cif.tar.gz` (`structures/<family>/*.cif`) | `data/predictions/<family>/*.cif` | P 1 rewrite; identical to what `code/generate_prediction_cifs_from_dxmag.py` now writes |
| `model_<family>_fold<k>_swa.pt` | `$HEUSLER_TE_EXPORT/checkpoints/<family>/fold<k>/swa_model.pt` | byte-identical |
| `model_assets.zip` → `configs/per_fold/<family>/config_expA_fold<k>.yaml` | `$HEUSLER_TE_EXPORT/configs/per_fold/<family>/` | paths of the training cluster made relative |
| `heldout_<family>_ddos_pred.npy` + `heldout_<family>_ids.csv` | `$HEUSLER_TE_EXPORT/runtime_assets/heldout_dumps/<family>/fold<k>/test_predictions.npy`, `test_ids.txt` | split by fold (recipe below): byte-identical |
| `energy_grid_eV.npy` | (the code builds `linspace(-10, 10, 2001)` itself) | — |
| `index.csv` | (not read by the code; documentation and DXMag join) | — |

Recipe, run from the repository root (`Z` is the unzipped record). It was checked against the original files: every link
target, extracted CIF and split held-out file is identical to the file the code was run on.

```bash
Z=/path/to/heusler_predicted_ddos_v1
export HEUSLER_TE_EXPORT=${HEUSLER_TE_EXPORT:-$PWD/DOS_EXPORT}
for f in full_cubic full_tetra half_cubic half_tetra inverse_cubic inverse_tetra; do
  d=results/screening/$f/novel_expA_ensemble; mkdir -p "$d"
  ln -s "$Z/${f}_ddos_mean.npy" "$d/predictions.npy"
  ln -s "$Z/${f}_ddos_std.npy"  "$d/predictions_std.npy"
  ln -s "$Z/${f}_ids.txt"       "$d/ids.txt"
  for k in 0 1 2 3 4; do
    e="$HEUSLER_TE_EXPORT/checkpoints/$f/fold$k"; mkdir -p "$e"
    ln -s "$Z/model_${f}_fold${k}_swa.pt" "$e/swa_model.pt"
  done
done
ln -s "$Z/screen_prediction_set.csv" results/screening/rescreen_gapfix_ALL_annotated.csv
mkdir -p data/predictions && tar -xzf "$Z/structures_P1_cif.tar.gz" -C data/predictions --strip-components=1
unzip -q "$Z/model_assets.zip" 'configs/*' -d "$HEUSLER_TE_EXPORT"
Z=$Z python3 - <<'PY'
import os
import numpy as np, pandas as pd
Z, X = os.environ["Z"], os.environ["HEUSLER_TE_EXPORT"]
for f in ["full_cubic", "full_tetra", "half_cubic", "half_tetra", "inverse_cubic", "inverse_tetra"]:
    ids = pd.read_csv(f"{Z}/heldout_{f}_ids.csv")                      # row, fold, material_id, composition
    pred = np.load(f"{Z}/heldout_{f}_ddos_pred.npy", mmap_mode="r")
    for k, g in ids.groupby("fold"):                                    # one export folder per fold
        d = f"{X}/runtime_assets/heldout_dumps/{f}/fold{k}"; os.makedirs(d, exist_ok=True)
        np.save(f"{d}/test_predictions.npy", np.asarray(pred[g.row.values]))
        with open(f"{d}/test_ids.txt", "w") as fh:
            fh.write("".join(f"{c}_{u}\n" for c, u in zip(g.composition, g.material_id)))
PY
```

The held-out **DFT targets** (`test_targets.npy`) and the per-sample metrics (`metadata/metrics/<family>/fold<k>/
per_sample_metrics.csv`) of the export are not public, because they are computed from the NIMS DFT spectra. Rebuild them
from the HeuslerDB archive (Section 3): process each listed entry with `code/dos_dataset.py` exactly as in training (divide by
the number of atoms, align E_F, interpolate onto the grid, clip at zero, concatenate up and down) in the row order of
`test_ids.txt`, then run `code/compute_metrics.py` on each fold.

The prediction-set arrays were produced on 2026-05-26 (full_cubic, full_tetra), 2026-06-16 (inverse_cubic) and
2026-07-12 (half_cubic, half_tetra, inverse_tetra).

## 3. External inputs that are not shipped

| Input | Needed by | Where to obtain | Place under `$HEUSLER_TE_ROOT` |
|---|---|---|---|
| DXMag HeuslerDB metadata, release 2025-09-09 (CC BY 4.0) | CIF generation, both screens, `code/paper/build_index.py`, several figure scripts | NIMS Spin Theory Group, https://www.nims.go.jp/group/spintheory/database/ | `data/DXMag_HeuslerDB_2025-09-09.xlsx` |
| HeuslerDB electronic-DOS dataset, HeuslerDB_elecDOS_ver1.0, archives dated 241229 (pymatgen JSON, CC BY 4.0) | training (`code/dos_dataset.py`), training-set screen, held-out targets | doi:10.48505/nims.5794. The archives contain 51,257 spectra, all used (the record description quotes 51,373) | `data/<family>/dos_json/<composition>_<UUID>.json` (one file per entry, as named in the archives) |
| Run index workbook `ALL_DFT_3625_materials_DXMAG_style.xlsx` (sheet `Materials`: run directory ↔ DXMag UUID, DXMag columns, `SCREEN_*` inputs) | `code/paper/build_paper_dataset.py`, `build_index.py`, `dedup_twins.py` | not distributed (assembled during the campaign); `build_index.py` rebuilds the run ↔ UUID match from the `scf.in` cells | `results/screening/` |
| QE/BoltzTraP2 run directories: `scf.in`, `scf.out`, `nscf.out`, `transport.json`, `transport_v2.json`, `transport_w1.json`, `FAILED` | `code/paper/build_paper_dataset.py`, `build_index.py`, `sensitivity.py`, `paper/manuscript_v19/analysis/soc_paired.py`, `code/audit_dft_gaps.py`, the Fig. 6 script, the SI script for the CoTiSb example | not distributed (raw outputs); their per-run results are in `results/paper_analysis/`; recompute with `CMS/anchor/` | `CMS/anchor/runs_local/<run_dir>/`, `CMS/anchor/soc_campaign/runs/<run_dir>/` |
| First-principles inputs: SSSP efficiency library (`SSSP/`, `SSSP_efficiency.json`), fully relativistic PSlibrary PAW potentials (`pseudos_rel/` with `pseudo_index.json`), input CIFs (`cifs/`), queues (`materials_local.txt`, `soc_campaign/soc_queue.txt`, `widen/worklist_scalar.txt`, `widen/worklist_soc.txt`) | `CMS/anchor/` drivers | SSSP and PSlibrary from their distributors; CIFs from `data/predictions/` or the DXMag cells; queues are plain lists (see `CMS/anchor/README.md`) | `CMS/anchor/` |
| Training-set screening table `training_pool_screen_gapfix.csv` | figure scripts (`screen_pools()`), Figs. 1 and 4 | regenerate from the HeuslerDB archive (Section 4) | `results/screening/` |
| Prediction-set screening table `rescreen_gapfix_ALL_annotated.csv` | figure scripts (`screen_pools()`) | Zenodo `screen_prediction_set.csv` (Section 2), or regenerate (Section 4) | `results/screening/` |
| Model export: checkpoints, per-fold configs, held-out dumps, per-sample metrics, golden test pairs | `code/predict_from_cif.py`, `code/paper/ml_metrics.py`, ML figure scripts | public parts in the Zenodo record (Section 2); DFT targets and per-sample metrics must be rebuilt | `$HEUSLER_TE_EXPORT` |
| Cross-validation splits `expA_fold<k>_{train,val,test}.csv` and the pretraining list `pretrain_train.csv` | `train_hpc.py` | regenerate with `code/create_splits.py` (seed 42) and `code/create_pretrain_dataset.py` | `data/<family>/data_split/`, `data/pretrain/` |
| Campaign master table `master_dataset.csv` | `code/paper/sensitivity.py` (only to identify the first 1,401 completed runs for the corrected-basis bar) | not distributed | `results/dxmag_master/` |
| SI analysis output `B15_queue_provenance.csv` | `paper/manuscript_v1/scripts/figures/make_tables.py`, selection-round table (without it the script stops there, after writing the tables that precede it) | not distributed | `results/paper_analysis/si/` |
| Earlier screen versions `rescreen_unified_ranked_ALL_annotated.csv`, `train_rescreen_unified_ranked_ALL_annotated.csv`, `CMS/anchor/batch3_selection.csv` | comparison columns in `code/audit_dft_gaps.py` only | not distributed | `results/screening/`, `CMS/anchor/` |
| Manuscript sources (`sections/`, `scripts/v14_src/`, `review/captions/`, `tables/`, `si/`) | `paper/manuscript_v19/analysis/humanizer_scan.py` (and the verification script, added at publication), `scripts/assemble_v14.py`, `scripts/renumber_si.py`; optional cross-check in `v14_fig3_fidelity.py`; `paper/manuscript_v1/analysis/verify_v12.py` (earlier draft) | not distributed (the manuscript is not yet public) | `paper/manuscript_v19/` |

Do **not** join the DXMag CSV exports of 2026-03 (`DXMag_HeuslerDB_alld_2026-03-10.csv` and the quaternary file): their UUIDs
differ from the 2025-09-09 release used throughout. Join on the UUID, never on the formula: several entries share a formula.

## 4. Commands that produce the screening tables

Both commands were re-run for this release and reproduce the files used in the paper. Run them from the repository root.

```bash
# prediction set (54,978 entries, predicted d-DOS): writes rescreen_gapfix.csv (survivors of the priority
# filters) and rescreen_gapfix_ALL_annotated.csv (every entry) -- both byte-identical to the paper's files
python3 code/te_rescreen.py --screening_root results/screening \
    --stability data/DXMag_HeuslerDB_2025-09-09.xlsx --out results/screening/rescreen_gapfix.csv

# training set (51,257 entries, DFT d-DOS from data/<family>/dos_json/): writes training_pool_screen_gapfix.csv
rm -f results/screening/_ckpt_training_*.csv      # per-family resume files; stale ones would be reused
python3 code/screen_training_pool.py --data_root data \
    --stability data/DXMag_HeuslerDB_2025-09-09.xlsx --out results/screening/training_pool_screen_gapfix.csv
```

Defaults used: 900 K, μ scanned over E_F ± 1.0 eV in 0.02 eV steps, gap label threshold 0.15 states eV⁻¹ (archive units),
gap filter 0.10–1.00 eV. The prediction-set command reads `results/screening/<family>/novel_expA_ensemble/te_ranking.csv`,
when present, only to carry the earlier composite score (`pred_D34_TE_score`, written by `code/compute_te_descriptors.py`);
without it that one column is absent. `code/te_rescreen_training_local.py` is an earlier per-atom training-set screen and is
not used for the paper.

## 5. Conventions of the arrays and tables

- **Grid.** Every d-DOS row has 4,002 values: spin up in columns 0–2000 and spin down in 2001–4001, on E − E_F =
  `linspace(-10, 10, 2001)` eV. E_F is column 1000 (up) and 3001 (down).
- **Spin labels.** "Up" and "down" are the archive's spin channels `1` and `-1`, not necessarily the majority and minority channels.
- **Factor of two.** Integrating the archive's spin-resolved total DOS up to E_F gives twice the DXMag magnetisation, so archive
  spectra, targets and predictions carry a uniform factor of two relative to physical values (manuscript, Methods). Ratios are
  unaffected.
- **Per atom versus per cell.** Training targets and predictions are per atom (archive d-DOS divided by the number of atoms of the
  stored cell). The training-set screen (`code/screen_training_pool.py`) reads the archive d-DOS per stored cell, so P^d_max,
  L₀*, the d-DOS at E_F and the label threshold are on a 3–4 times larger scale there; S^d_max and ξ are scale-free. Compare raw
  indicator values only within one set.
- **Cells.** Cubic-family entries with 6 or 8 atoms are cubic minima stored by DXMag as two formula units in a body-centred
  tetragonal cell (c/a ≈ √2); strict symmetry finders may report I4/mmm or I-4m2 for them. The QE runs reduce every cell to its
  primitive cell with spglib.
- **Phonons.** DXMag phonon results exist for training-set entries only; the phonon columns are empty for the prediction set.
- **Composition strings** keep the DXMag site order (the half-Heusler ZrNiSn is `NiZrSn`).

## 6. Software

See `requirements.txt` for the two recorded environments: Python 3.8.10 for the analysis, screening, figures and the
first-principles driver (Quantum ESPRESSO 7.1, BoltzTraP2 24.7.2 with `CMS/anchor/boltztrap2_patch`, spglib 2.0.2), and Python
3.11.15 with CUDA 12.1 for the d-DOS model (PyTorch 2.2.2+cu121, DGL 1.1.3+cu121, ALIGNN 2026.4.1, jarvis-tools 2026.4.2,
pymatgen 2026.3.23, scikit-learn 1.8.0, NumPy 1.26.4), the latter recorded in the model export's `environment/` folder
(the `key_versions.json` there lists pymatgen as "unknown"; the pip freeze of the same environment records 2026.3.23).
