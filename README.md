# Density-of-states screening of Heusler thermoelectrics

Code and derived data for the manuscript *"Density-of-states screening of Heusler thermoelectrics across compositions and structural
variants, assessed by first-principles transport"* (S. Randive, H. Pandey, R. Dixit, A. Ghosh; in preparation).

The study uses the DXMag Heusler database, built for magnetic functionality, as the foundation for a thermoelectric search:

```
DXMag HeuslerDB (release 2025-09-09)         106,235 full-, inverse- and half-Heusler configurations
 ├─ training set    51,257 entries with a DFT spin-resolved d-orbital DOS (HeuslerDB electronic-DOS archive)
 │                  → ALIGNN d-DOS model (train_hpc.py, code/dos_dataset.py, configs/)
 └─ prediction set  54,978 entries without a DOS → predicted d-DOS (code/run_inference.py)
        │
        ├─ Cutler–Mott d-DOS indicators, gap label, priority filters    code/te_rescreen*.py, code/screen_training_pool.py
        ├─ 3,625 Quantum ESPRESSO (PBE) + BoltzTraP2 (CRTA) runs        CMS/anchor/
        ├─ 607 fully relativistic (SOC) recalculations                  CMS/anchor/soc_campaign/
        └─ certification, evaluation, SOC / structural-variant / literature analyses, figures
                                                                         code/paper/, paper/manuscript_v1/
```

Main results reproduced by the shipped scripts: the Seebeck indicator places 75% (DFT spectra) and 91% (predicted spectra) of the
compounds passing every first-principles criterion in the top fifth of the calculated entries; it does not rank power factors; SOC
suppresses p-type but not n-type power factors and removes a third of the heavy-element candidates; 80 compositions are certified, ten
of them established half-Heusler thermoelectrics. These are computational results under the constant-relaxation-time approximation,
not measured figures of merit.

## Repository layout

| Path | Contents |
|---|---|
| `train_hpc.py`, `config.yaml`, `configs/` | ALIGNN d-DOS model: training (pretraining, per-family fine-tuning, 5-fold CV, SWA) |
| `code/dos_dataset.py`, `code/create_splits.py`, `code/create_pretrain_dataset.py` | data pipeline: d-DOS target (2 × 2,001 points, −10…10 eV), splits |
| `code/run_inference.py`, `code/predict_from_cif.py`, `code/generate_prediction_cifs_from_dxmag.py`, `make_dos_export.py` | prediction-set structures, ensemble inference, model export |
| `code/te_rescreen.py`, `code/te_rescreen_training_local.py`, `code/screen_training_pool.py`, `code/compute_te_descriptors.py` | Cutler–Mott indicators $S^d_{max}$, $P^d_{max}$, ξ, $L_0^*$; gap label; priority filters |
| `CMS/anchor/` (GPL-3.0) | QE input generation, local driver, BoltzTraP2 transport extraction, SOC inputs and driver, ±1 eV re-extraction, BoltzTraP2 patch |
| `code/audit_dft_gaps.py` | electronic class (metal / half-metal / narrow / gapped) from NSCF eigenvalues |
| `code/paper/` | raw outputs → per-run dataset (`build_paper_dataset.py`), screening evaluation, sensitivity, near-duplicates, fixed-n ranking, ML metrics |
| `paper/manuscript_v1/analysis/` | paired SOC analysis, structural variants, phonon status; `verify_v12.py` checks manuscript numbers against the data (needs the manuscript sources, not included) |
| `paper/manuscript_v1/literature/` | literature-status classification of the 80 certified compositions (sources and DOIs in the CSVs) |
| `paper/manuscript_v1/scripts/figures/` | figure scripts for the manuscript (Figs. 1–6) and SI; `figkit/` is the shared figure style and QA gate |
| `results/paper_analysis/` | derived tables, including `paper_dataset.pkl` (one row per DFT run, 3,625 × 407) |

## Quick start

```bash
python3 -m pip install -r requirements.txt
export HEUSLER_TE_ROOT=$(pwd)            # all project paths resolve from here
python3 paper/manuscript_v1/analysis/soc_table.py            # SOC by heavy element and carrier type
python3 paper/manuscript_v1/analysis/polymorph_variants.py   # structural variants vs their reference crystal
python3 paper/manuscript_v1/analysis/phonon_status.py        # DXMag phonon status of the certified compositions
python3 paper/manuscript_v1/literature/consolidate.py        # literature-status table
```

Scripts that need data not shipped here (DXMag metadata, the HeuslerDB DOS archive, model checkpoints and held-out predictions, raw
QE/BoltzTraP2 run directories) read them from `HEUSLER_TE_ROOT` (and `HEUSLER_TE_EXPORT` for the model export); `DATA.md` lists what
each needs and where to obtain it. Run Python without `-I`; figure scripts expect matplotlib with the Liberation Sans font.

## Notes

* Composition strings keep the DXMag site order: the first element of a half-Heusler sits on the 4c site (DXMag `NiZrSn` is ZrNiSn).
* The two populations are never pooled; every rate is computed within one population.
* The archive DOS carries a factor of two relative to physical values; see the manuscript, Sec. 2.2.
* Paths and credentials of the original HPC environment have been replaced; SLURM scripts are not included.

## Licence and citation

MIT, except `CMS/anchor/` (GPL-3.0-or-later, because it derives from or imports BoltzTraP2). Please cite the manuscript (see
`CITATION.cff`) and the DXMag database: Xiao & Tadano, *Acta Mater.* **297**, 121312 (2025); *npj Comput. Mater.* **12**, 133 (2026);
HeuslerDB electronic-DOS dataset, doi:10.48505/nims.5794.
