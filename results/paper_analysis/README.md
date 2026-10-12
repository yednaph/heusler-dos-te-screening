# results/paper_analysis

Per-run first-principles results and the evaluation tables of the manuscript. Everything here is derived from the raw
QE/BoltzTraP2 outputs and the run index by the scripts in `code/paper/`; the scripts and their inputs are listed in `DATA.md`.

| File | Content | Written by |
|---|---|---|
| `paper_dataset.pkl` | One row per first-principles run: 3,625 rows × 407 columns | `code/paper/build_paper_dataset.py` |
| `paper_dataset.csv` | The same table as plain text, without the 25 columns listed under "Omitted columns" (3,625 × 382). Use it if you prefer not to load a pickle: unpickling can execute code. Floats are written at full precision; `pd.read_csv(..., float_precision="round_trip")` recovers the pickled values exactly | `code/paper/export_csv.py` |
| `certified_entries.csv` | The 155 certified entries (76 training set, 79 prediction set; 80 compositions), 33 columns | `code/paper/export_csv.py` |
| `build_check.txt` | Agreement of the rebuilt table with the campaign workbook (all checks OK; 309 qualified runs, 155 certified entries, 80 compositions) | `build_paper_dataset.py` |
| `screen_eval.xlsx`, `.txt` | ROC-AUC, PR-AUC and recovered fractions of each screening score, per set and subset | `code/paper/screen_eval.py` |
| `sensitivity.xlsx`, `.txt` | Certified counts under changed thresholds | `code/paper/sensitivity.py` |
| `ranking_fixed_n.xlsx`, `ranking_summary.txt` | Rankings of the 80 compositions at fixed carrier concentration (\|n\| ≤ 10²⁰ and 10²¹ cm⁻³, p and n type, 900 K and best temperature) | `code/paper/rank_certified.py` |
| `twins.xlsx`, `.txt` | Near-duplicates between the two sets and distinct crystals among the certified entries | `code/paper/dedup_twins.py` |
| `ml_metrics.xlsx`, `.txt` | Held-out d-DOS accuracy per family (best-validation checkpoints) | `code/paper/ml_metrics.py` |

Rows are runs, not entries: 1,562 training-set runs, 2,021 prediction-set runs and 42 runs (`data_set = ambiguous`) computed
under a composition name that cannot be matched to one DXMag UUID; five prediction-set UUIDs were computed twice. The two sets
are evaluated separately and never pooled.

## Column dictionary

Units: S in µV K⁻¹; PF/τ in W m⁻¹ K⁻² s⁻¹; σ/τ in S m⁻¹ s⁻¹; κₑ/τ in W m⁻¹ K⁻¹ s⁻¹; μ in eV relative to the QE Fermi level;
carrier concentration n in cm⁻³ (positive: holes, p type; negative: electrons, n type); T in K; energies in eV atom⁻¹.
All transport quantities are constant-relaxation-time (CRTA) values per relaxation time τ; none is a measured or complete
figure of merit (τ and the lattice thermal conductivity are not computed).

**Identifiers and DXMag metadata** (copied from DXMag release 2025-09-09, CC BY 4.0, Xiao & Tadano)

| Column | Meaning |
|---|---|
| `UUID` | DXMag UUID of the entry (2025-09-09 release); blank for the 42 ambiguous runs |
| `composition` | DXMag composition string, DXMag site order (`NiZrSn` is the half-Heusler ZrNiSn) |
| `run_dir` | QE/BoltzTraP2 run directory name (`<composition>`, `<composition>__<first 8 hex of UUID>` or `<composition>_<n>`) |
| `data_set` | `training` (entry has a DFT d-DOS), `prediction` (predicted d-DOS) or `ambiguous` |
| `family` | `full_cubic`, `full_tetra`, `half_cubic`, `half_tetra`, `inverse_cubic`, `inverse_tetra` |
| `hull` | DXMag distance from the convex hull (eV atom⁻¹) |
| `phonon computed`, `phonon stable (commensurate)` | DXMag phonon flags (computed for training-set entries only) |
| `space group symbol`, `number of atoms in the cell` | DXMag space group and atoms in the stored cell |

**Screening columns kept** (`SCREEN_` prefix: values of the pre-DFT screen, copied from the run index)

| Column | Meaning |
|---|---|
| `SCREEN_ML_formE`, `SCREEN_ML_hull` | DXMag formation energy and hull distance (eV atom⁻¹). The `ML_` prefix is historical; these are DFT values from DXMag |
| `SCREEN_hull_tier` | Hull distance tier (`<=0.05`, `0.05-0.10`, `0.10-0.20`, `>0.20` eV atom⁻¹) |
| `SCREEN_phonon_stable`, `SCREEN_min_phonon_freq` | DXMag phonon stability and lowest frequency (cm⁻¹); training set only |
| `SCREEN_gate_stable` | Stability priority filter: hull < 0.10 eV atom⁻¹, negative formation energy, no phonon frequency below −5 cm⁻¹ (missing values pass) |
| `SCREEN_kL_proxy_mass` | Mean atomic mass of the composition (u); not used for ranking |
| `SCREEN_dos_provenance` | Spectra the screen read: `ML_predicted` (prediction set) or `DFT_real` (training set) |
| `SCREEN_std_fermi`, `SCREEN_std_fermi_pctile` | Ensemble spread u_F of the predicted d-DOS near E_F, and its percentile; prediction set only |
| `SCREEN_gate_reliable` | u_F below its 75th percentile; prediction set only |
| `SCREEN_passes_all`, `SCREEN_reject_reason` | All three priority filters passed, or why not; prediction set only (the filters ordered the selection and excluded nothing) |
| `SCREEN_pred_D34_TE_score`, `SCREEN_rank_pred_D34_TE_score` | An earlier composite of four d-DOS features at E_F and its rank; prediction set only; superseded, kept for reference |

**First-principles results, scalar-relativistic run** (this study; QE-PBE + BoltzTraP2)

| Column | Meaning |
|---|---|
| `gap`, `E_F`, `dft_class` | QE-PBE band gap read from the NSCF eigenvalues (eV), QE Fermi level (eV), and electronic class: `metal`, `half_metal`, `narrow` (gap below the smearing width 0.136 eV) or `gapped` |
| `scalar_file` | Transport file used: `transport_v2.json` (recomputed from the NSCF eigenvalues after the mesh correction), else `transport.json` |
| `max_S_intr` | Largest \|S\| at the undoped point (grid point nearest the QE Fermi level) over 300–1200 K |
| `dopable` | `max_S_intr` > 60 µV K⁻¹: the Seebeck criterion (i) |
| `best_PF`, `best_T`, `best_conv` | Best PF/τ over 300–1200 K and the two μ conventions (`intrinsic`: undoped point; `reopt_doping`: μ re-optimised for PF within ±0.5 eV, allowed only if `dopable`), its temperature and convention |
| `S_best`, `sigma_best`, `kappa_e_best`, `mu_best` | S, σ/τ, κₑ/τ and μ at that best point |
| `zT_ceiling` | S²/L₀ at the best point: the zT that would follow with zero lattice conductivity and the Sommerfeld Lorenz number; an upper bound, not a figure of merit |
| `zT_e` | S²σT/κₑ at the best point: τ-free electronic upper bound |
| `L_over_L0` | Lorenz ratio κₑ/(σT L₀), a Wiedemann–Franz consistency check |
| `reopt900_PF` | PF/τ at 900 K with re-optimised μ, the quantity on which the power-factor bar PF₀ was defined |
| `railed` | Best μ at the edge of the ±0.5 eV scan (\|μ\| ≥ 0.49 eV) |
| `w1_cap<C>_<t>_<T>_<q>` | From `transport_w1.json` (μ scanned within ±1.0 eV): the PF/τ optimum with \|n\| ≤ C (`1e20` or `1e21` cm⁻³), carrier type t (`p`, `n` or `any`), at T = `900` K or the `best` temperature; q is `PF`, `S`, `n`, `mu`, `sigma` (σ/τ), `kappa_e` (κₑ/τ), `L_over_L0` or `at_cap` (\|n\| > 0.9 C, optimum pressed against the cap); `w1_cap<C>_<t>_best_T` is that temperature |

**Spin–orbit coupling run** (this study; fully relativistic QE-PBE + BoltzTraP2, same structure and meshes)

| Column | Meaning |
|---|---|
| `soc_file`, `soc_failed`, `soc_status` | SOC transport file used, failure marker, and status: `done` (607 runs), `failed` (86) or `not run` |
| `soc_<q>`, `soc_w1_<q>` | The quantities above, from the SOC run |

**Certification chain** (rules of the manuscript, Methods; `code/paper/build_paper_dataset.py`)

| Column | Meaning |
|---|---|
| `heavy` | Composition contains an element with Z ≥ 72 (Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi) |
| `stable` | `hull` < 0.10 eV atom⁻¹: stability criterion (iii) (false when the hull is unknown) |
| `above_bar` | `best_PF` > PF₀ = 1.0857 × 10¹² W m⁻¹ K⁻² s⁻¹: power-factor criterion (ii) |
| `winner` | `dopable` and `above_bar` and `stable`: a **qualified** run in the manuscript's terms |
| `gapped` | `dft_class == "gapped"`: QE-PBE gap of at least 0.136 eV, criterion (iv) |
| `soc_above_bar` | SOC `best_PF` > PF₀: criterion (v) for heavy compounds |
| `certified` | `winner` and `gapped` and (not `heavy`, or `soc_above_bar`). Heavy-element runs without a finished SOC calculation are not certified |
| `final_<q>`, `final_cap<C>_<t>_<T>_<q>` | Reported values: SOC values for heavy compounds, scalar-relativistic values otherwise |

**What "certified" means.** A run, and the DXMag entry it belongs to, is certified when it passes, in sequence: (i) |S| > 60 µV K⁻¹
at the undoped point at some temperature; (ii) a best PF/τ above PF₀ (the median 900 K value of the first 1,401 completed runs,
fixed during the campaign); (iii) a DXMag hull distance below 0.10 eV atom⁻¹; (iv) a scalar-relativistic QE-PBE gap of at least
0.136 eV; and (v) for compounds with Z ≥ 72, criterion (ii) again with spin–orbit coupling. It is a computational status under
the CRTA at the PBE level, not a measured or predicted figure of merit.

## Omitted columns

The CSV files contain only data produced in this study and DXMag identifiers and metadata. These 25 screening columns of the
pickle are omitted because their training-set values were computed from the DFT d-DOS of the HeuslerDB electronic-DOS dataset
(NIMS, doi:10.48505/nims.5794), which is not redistributed:

`SCREEN_etype`, `SCREEN_gap_eV`, `SCREEN_gap_tier`, `SCREEN_dos_ef`, `SCREEN_spin_pol`, `SCREEN_A_seebeck_uVK`, `SCREEN_B_pf`,
`SCREEN_C_sharp`, `SCREEN_D_dosmass`, `SCREEN_mu_star_eV`, `SCREEN_S_at_star_uVK`, `SCREEN_carrier_n`, `SCREEN_mu_railed`,
`SCREEN_gate_semiconductor`, `SCREEN_B_pf_percentile`, `SCREEN_screening_group`, `SCREEN_rank_A_seebeck_uVK`, `SCREEN_rank_B_pf`,
`SCREEN_rank_C_sharp`, `SCREEN_rank_D_dosmass`, `SCREEN_famrank_A_seebeck_uVK`, `SCREEN_famrank_B_pf`, `SCREEN_window_truncated`,
`SCREEN_dft_e_lo`, `SCREEN_dft_e_hi`.

For the prediction set, the same indicators (from the predicted d-DOS) are in `screen_prediction_set.csv` of the Zenodo record
(doi:10.5281/zenodo.23315204; join on `UUID`). For the training set they can be recomputed from the NIMS archive with
`code/screen_training_pool.py` (`DATA.md`, Section 4). Their meanings are given in the Zenodo record's README.
