# Figure data of the earlier manuscript draft: third-party content

Some files here contain data from, or values computed from, the **HeuslerDB electronic-DOS dataset** (T. Tadano, NIMS,
HeuslerDB_elecDOS_ver1.0, archives dated 241229, doi:10.48505/nims.5794, CC BY 4.0). These DFT spectra were produced by NIMS,
not by the authors of this repository. They are reproduced or used here under the CC BY 4.0 licence, with attribution:

- **DFT spectra (NIMS data):** `fig1_dos_spectra.csv`, columns `dft_up` and `dft_down`. These are the DFT d-DOS of ZrNiSn and
  Co2MnSi from the dataset above, as processed into the model target (per atom, on the 0.01 eV grid). The `pred_up` and
  `pred_down` columns are the authors' model predictions.
- **Per-entry values computed by the authors from the NIMS spectra:** `dos_normalisation_check.csv` (integral ratio for 150
  entries).
- **Example values and aggregate counts computed by the authors from the NIMS spectra** (training-set panels only):
  - `fig1_example.csv`, `fig1_dos_examples.csv`;
  - `fig2_example.csv`, `fig2_examples.csv`;
  - `fig2v10_labels.csv`, `fig2v10_landscape.csv`;
  - `fig2v13_indicator.csv`, `fig2v13_labels.csv`, `fig2v13_medians.csv`;
  - `fig3_example*.csv`, `fig3_spec.csv`, `fig3v13_calc_share_training.csv`;
  - `dft_class_by_track_and_ddos_label.csv`, `ddos_semiconductors_dft_metal.csv`.

All other values (QE/BoltzTraP2 results, predictions, prediction-set screening values, literature status) were produced by the
authors. DXMag identifiers and metadata (Xiao & Tadano, CC BY 4.0) are attributed in the main README.
