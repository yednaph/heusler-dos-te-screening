# High-throughput screening of Heusler alloys for thermoelectrics

Code and first-principles results for the manuscript *High-throughput screening of Heusler alloys for thermoelectrics combining
machine-learned densities of states and first-principles transport* (S. Randive, R. Dixit, H. Pandey; manuscript in
preparation).

- **Code:** this repository, release v1.0.0 (https://github.com/yednaph/heusler-dos-te-screening/releases/tag/v1.0.0).
- **Data:** the machine-learned d-DOS of the prediction set, the held-out predictions, the prediction-set screening table, the
  input structures and the 30 model weights are deposited at Zenodo, doi:[10.5281/zenodo.23329059](https://doi.org/10.5281/zenodo.23329059)
  (CC BY 4.0), as individual files. `DATA.md` maps them onto the paths used here.

The study starts from the DXMag Heusler database, built for magnetic functionality, and tests a density-of-states (DOS) screen
for thermoelectrics against first-principles transport:

```
DXMag HeuslerDB, release 2025-09-09: 106,235 full-, inverse- and half-Heusler entries, six families
 |
 |-- training set    51,257 entries with a DFT spin-resolved d-DOS (HeuslerDB electronic-DOS dataset, NIMS)
 |                   -> trains the ALIGNN d-DOS model (train_hpc.py, code/dos_dataset.py, configs/)
 |                   -> screened on its own DFT d-DOS                       code/screen_training_pool.py
 |
 |-- prediction set  54,978 entries without a DFT DOS
 |                   -> d-DOS predicted by the 5 SWA fold models of its family   code/predict_from_cif.py
 |                   -> screened on the predicted d-DOS                     code/te_rescreen.py
 |
 |   Both sets, separately: Cutler-Mott d-DOS indicators (S^d_max, P^d_max, xi, L0*), d-DOS gap label, priority filters
 |
 '-- entries of BOTH sets selected for first-principles calculation, in nine rounds ordered by the screen
       3,625 QE-PBE + BoltzTraP2 (CRTA) runs: 1,562 training set, 2,021 prediction set, 42 not attributable   CMS/anchor/
       607 paired fully relativistic (SOC) recalculations of heavy-element runs                                CMS/anchor/soc_campaign/
       certification (criteria below): 76 training-set + 79 prediction-set entries = 155 entries, 80 compositions
       evaluation, sensitivity, rankings, SOC / structural-variant / literature analyses, figures               code/paper/, paper/manuscript_v19/
```

**Main results** (reproduced by the shipped scripts and records): keeping the fifth of the calculated entries with the highest
d-DOS Seebeck indicator S^d_max retains 75% (training set, DFT spectra) and 91% (prediction set, predicted spectra) of the
certified entries; the indicator does not rank power factors; the d-DOS gap label passes mostly metals; spin–orbit coupling
lowers the p-type but not the n-type power factors of heavy-element candidates and moves 40 of 115 qualified heavy-element runs
below the power-factor bar; 80 compositions are certified, ten of them among the eleven established half-Heusler
thermoelectrics that were calculated; predicted spectra reproduce the d-DOS shape (held-out mean Pearson r 0.895) but miss most
gaps (recall 0.15). These are computational results under the constant-relaxation-time approximation, not measured figures of
merit.

**Certified.** A run, and the DXMag entry it belongs to, is *certified* when it passes, in sequence: (i) |S| > 60 µV K⁻¹ at the
undoped point (the grid point nearest the QE Fermi level) at some temperature between 300 and 1200 K; (ii) a best PF/τ above
PF₀ = 1.09 × 10¹² W m⁻¹ K⁻² s⁻¹ (the median 900 K value of the first 1,401 completed runs, fixed during the campaign);
(iii) a DXMag hull distance below 0.10 eV atom⁻¹; (iv) a scalar-relativistic QE-PBE gap of at least 0.136 eV (the smearing
width); and (v) for compounds containing an element with Z ≥ 72, criterion (ii) again with SOC. Runs passing (i)–(iii) are
*qualified* (`winner` in the tables). Certification is a computational status, not a figure of merit: τ and the lattice thermal
conductivity are not computed.

## Repository layout

| Path | Contents |
|---|---|
| `train_hpc.py`, `config.yaml`, `configs/` | ALIGNN d-DOS model: pretraining and per-family fine-tuning, shuffled five-fold cross-validation (seed 42), stochastic weight averaging (SWA) |
| `code/dos_dataset.py`, `code/create_splits.py`, `code/create_pretrain_dataset.py` | data pipeline: d-DOS target (2 × 2,001 points, −10…10 eV, per atom), graphs, splits |
| `code/generate_prediction_cifs_from_dxmag.py` | prediction-set structures from DXMag, written in space group P 1 |
| `code/predict_from_cif.py` | **prediction-set inference**: each structure through the five SWA fold models of its family; mean and spread |
| `code/run_inference.py`, `code/compute_metrics.py`, `make_dos_export.py` | held-out test predictions (best-validation checkpoints), accuracy metrics, model export |
| `code/te_rescreen.py`, `code/screen_training_pool.py` | the screen: Cutler–Mott indicators S^d_max, P^d_max, ξ, L₀*, d-DOS gap label, priority filters (prediction and training set) |
| `code/compute_te_descriptors.py` | descriptor table of the prediction set, including the earlier composite score (D34) that the paper evaluates as a reference |
| `code/te_rescreen_training_local.py` | an earlier per-atom training-set screen; superseded, not used for the paper |
| `code/audit_dft_gaps.py` | electronic class (metal / half-metal / narrow / gapped) and gap from the NSCF eigenvalues |
| `CMS/anchor/` (GPL-3.0-or-later) | QE input generation and driver, BoltzTraP2 transport extraction, mesh correction and ±1 eV re-extraction, SOC inputs and driver, BoltzTraP2 SOC patch; see `CMS/anchor/README.md` |
| `code/paper/` | raw outputs → per-run table (`build_paper_dataset.py`, `build_index.py`), screening evaluation, sensitivity, near-duplicates, fixed-carrier-concentration ranking, ML metrics, CSV export |
| `results/paper_analysis/` | per-run results (`paper_dataset.pkl`, `paper_dataset.csv`, `certified_entries.csv`) and evaluation tables; column dictionary in its `README.md` |
| `paper/manuscript_v19/` | figure, analysis and literature code of the current manuscript, with the figure → script → inputs table |
| `paper/manuscript_v1/` | earlier draft (superseded); its Supplementary figure and table scripts and records are still used |
| `figkit/` | shared figure style and QA gate |

## Quick start

These commands run from the shipped files alone (Python 3.8-era pins; see `requirements.txt`):

```bash
python3 -m pip install -r requirements.txt
python3 paper/manuscript_v19/analysis/soc_table.py               # SOC by heavy element and carrier type
python3 paper/manuscript_v19/analysis/polymorph_variants.py      # structural variants against their reference crystal
python3 paper/manuscript_v19/analysis/phonon_status.py           # DXMag phonon status of the certified compositions
python3 paper/manuscript_v19/literature/consolidate.py           # literature status of the 80 certified compositions
python3 paper/manuscript_v19/scripts/figures/v14_fig6_cascade.py # Fig. 5, certification cascade (also Figs. 7-9: v14_fig8/9/7)
python3 code/paper/screen_eval.py                                # screening evaluation (ROC-AUC, PR-AUC, recovered fractions)
python3 code/paper/rank_certified.py                             # fixed-carrier-concentration rankings of the 80 compositions
```

```python
import pandas as pd                                              # per-run results without unpickling
runs = pd.read_csv("results/paper_analysis/paper_dataset.csv", low_memory=False)
cert = pd.read_csv("results/paper_analysis/certified_entries.csv")
print(runs.groupby("data_set").certified.sum(), cert.composition.nunique())   # 76 / 79 entries, 80 compositions
```

Paths resolve from `HEUSLER_TE_ROOT` (default: the repository root) and the model export from `HEUSLER_TE_EXPORT` (default
`$HEUSLER_TE_ROOT/DOS_EXPORT`). Scripts that need data not shipped here (DXMag metadata, the HeuslerDB DOS archive, the Zenodo
record, raw QE/BoltzTraP2 run directories) read them from there; `DATA.md` lists each input, where to get it, a tested recipe
that places the Zenodo files where the code expects them, and the exact commands that rebuild the screening tables. Figure
scripts need matplotlib with the Liberation Sans font. Run Python without `-I`.

## Notes

* The two sets are never pooled: every rate, rank and indicator comparison is made within one set.
* **Per cell versus per atom.** The model's targets and predictions are per atom (archive d-DOS divided by the number of atoms of
  the stored cell). The training-set screen (`code/screen_training_pool.py`) reads the archive d-DOS per stored cell of three or
  four atoms. The same code thus sees a 3–4 times larger DOS in the training set: P^d_max, L₀*, the d-DOS at E_F and the label
  threshold differ in scale between the sets, while S^d_max and ξ do not (manuscript, Methods).
* The archive DOS, and with it the targets and predictions, carry a uniform factor of two relative to physical values
  (manuscript, Methods).
* Composition strings keep the DXMag site order: the first element of a half-Heusler sits on the tetrahedral site (DXMag
  `NiZrSn` is ZrNiSn). Join tables on the DXMag UUID, never on the formula.
* The QE calculations use the DXMag cells reduced to their primitive cells, without relaxation.
* Paths and credentials of the original computing environment have been replaced; SLURM scripts are not included.

## Model training as run

- Family models: 300 epochs, batch size 128, except cubic full Heusler (folds 0–1: 150 epochs, folds 2–4: 200; batch size 64).
  SWA averages the weights over the last 20% of each run's epochs (`swa_start_fraction: 0.8`); prediction-set spectra come from
  the SWA weights, all held-out metrics from each fold's best-validation checkpoint.
- The family models were meant to be fine-tuned from an all-family pretrained model (`configs/pretrain.yaml`); the run records
  confirm this initialisation for 2 of the 30 models and leave the parent of the others unconfirmed. Because pretraining saw every
  training spectrum, the held-out metrics are upper bounds.
- Settings that appear in the configuration files but were not executed: data augmentation, early stopping and learning-rate
  warm-up (the per-fold configurations use a reduce-on-plateau schedule).

## Limitations

From the manuscript (Discussion):

- The selection for first-principles calculation was ordered by the screen, with criteria set during the campaign, so recall is
  not measured and the certified count depends on the thresholds; the 80 compositions are a prioritised sample, not a census.
  The list is biased against wide gaps, because the ±0.5 eV window of the certification, measured from a Fermi level placed by
  the smearing, cannot reach the band edges of many entries with gaps above about 1.2 eV.
- PBE underestimates gaps, while the unshifted NSCF meshes, which miss the X point in many runs, overestimate them where the
  conduction-band minimum lies there. The QE runs started from ferromagnetic moments or without spin polarisation, so they do not
  reproduce the magnetic configuration of each DXMag entry.
- Transport is CRTA at the PBE level, without τ, lattice thermal conductivity, convergence studies or relaxation in QE; optima at
  the edges of the temperature and chemical-potential scans make the reported power factors bounds within the scanned conditions.
- The model takes no magnetic input: entries with the same geometry receive the same predicted spectrum. Every certified
  prediction-set entry duplicates a training crystal; 15.1% of the prediction entries lie within 1 meV atom⁻¹ of a training entry.
  Predicted spectra miss most gaps (held-out gap recall 0.15). The model evaluation is favourable by construction: pretraining
  overlapped the test spectra, the splits were random within families, and the deployed five-model mean was not evaluated on
  held-out spectra.
- Nineteen prediction-set entries have unphysical DXMag energies (positive, up to about 1,000 eV atom⁻¹, or a hull distance
  above 3 eV atom⁻¹); they are flagged in the Zenodo `index.csv`.

## Known implementation details

These describe the code as it was run; the trained models are not changed.

- **Spin-down windows one point early.** Three windows on the spin-down half of the 4,002-value vector start one grid point
  (0.01 eV) below the intended energy: `slice(2900, 3102)` instead of `2901` in the peak-height term of
  `train_hpc.py::ThermoelectricDOSLoss` (weight `w_peak = 0` in all 30 production configurations, so it did not act), and
  `2950:3052` instead of `2951` in the ±0.5 eV transport-window Pearson r of `code/compute_metrics.py` (not reported in the
  paper) and in the valley proxy D38 of `code/compute_te_descriptors.py` (not used by the screen).
- **Slope term on spin up only.** The slope loss (first and, with weight 0.1, second finite differences within ±3 eV) uses
  indices 700–1300, the spin-up channel.
- **"Wasserstein" term.** The term `L_wass` is the L1 distance between the sorted values of the predicted and DFT spectra,
  sorted over all 4,002 values, so both spin channels and all energies are pooled; it ignores energy positions.
- **Active loss.** With the production weights the loss is 0.35 (Fermi-window-weighted MSE at 700 K) + 0.35 (MSE within ±2 eV)
  + 0.15 (cosine) + 0.10 (slope) + 0.05 (sorted L1); `w_peak` and `w_tv` are 0. The manuscript describes the loss this way,
  including the spin-up slope term and the pooled sorted-value term.
- **SWA details.** `SWALR` sets the learning rate of every parameter group to 5 × 10⁻⁵, so a fine-tuning learning-rate ratio
  between groups does not persist through the SWA phase. Batch-normalisation statistics of the SWA model are re-estimated with one
  pass over the training data using the layers' default momentum (an exponential moving average) rather than the cumulative
  average of `torch.optim.swa_utils.update_bn`.

## Licence and citation

MIT, except everything under `CMS/anchor/`, which is GPL-3.0-or-later because it derives from or imports BoltzTraP2 (full text
in `CMS/anchor/COPYING`). Please cite the article (`CITATION.cff`; manuscript in preparation), the data record
(doi:10.5281/zenodo.23329059) and the DXMag sources:

- E. Xiao and T. Tadano, High-throughput computational screening of Heusler compounds with phonon considerations for enhanced
  material discovery, *Acta Mater.* **297**, 121312 (2025), doi:10.1016/j.actamat.2025.121312;
- E. Xiao and T. Tadano, Accurate screening of functional materials with machine-learning potential and transfer-learned
  regressions: Heusler alloy benchmark, *npj Comput. Mater.* **12**, 133 (2026), doi:10.1038/s41524-026-02013-0;
- T. Tadano, HeuslerDB_elecDOS_ver1.0 (archives dated 241229), NIMS (2025), doi:10.48505/nims.5794 (CC BY 4.0). Values computed from these NIMS spectra (the training-set screening columns of
  `results/paper_analysis/paper_dataset.pkl`, and two DFT spectra and some example values in `paper/manuscript_v1/figures/data/`)
  are third-party-derived data, used with attribution; see the README files in those folders;
- DXMag Computational HeuslerDB, release 2025-09-09, https://www.nims.go.jp/group/spintheory/database/ (CC BY 4.0).
