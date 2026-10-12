#!/usr/bin/env python3
"""
export_csv.py -- plain-text (CSV) copies of the per-run results table, for readers who should not load a pickle.

A pickle can execute code when it is loaded, so results/paper_analysis/paper_dataset.pkl is also given as CSV.
Only data produced in this study, plus DXMag identifiers and metadata (CC BY 4.0, Xiao & Tadano), are written:

  * kept:    identifiers and DXMag metadata; every first-principles (QE-PBE, BoltzTraP2, SOC) result and the
             certification chain; the screening columns that are DXMag metadata or exist only for prediction-set
             rows (computed from the machine-learned d-DOS);
  * omitted: the screening indicators and labels whose training-set values were computed from the DFT d-DOS of
             the HeuslerDB electronic-DOS archive (NIMS, doi:10.48505/nims.5794). They are listed in OMITTED below.
             The prediction-set values of the same indicators are in the Zenodo record (screen_prediction_set.csv,
             doi:10.5281/zenodo.23315204); the training-set values can be recomputed from the NIMS archive with
             code/screen_training_pool.py.

INPUT   results/paper_analysis/paper_dataset.pkl     (code/paper/build_paper_dataset.py)
OUTPUT  results/paper_analysis/paper_dataset.csv      one row per DFT run (3,625), all kept columns
        results/paper_analysis/certified_entries.csv  the 155 certified entries, a reduced column set
Column meanings: results/paper_analysis/README.md.

Run:  python3 code/paper/export_csv.py      (with HEUSLER_TE_ROOT unset, paths resolve from the repository)
"""
import os
import sys

import pandas as pd

ROOT = os.environ.get("HEUSLER_TE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # repo root
ANALYSIS = os.path.join(ROOT, "results", "paper_analysis")                       # input and output folder

# Screening columns whose TRAINING-SET values derive from the NIMS DFT d-DOS (not redistributed here).
OMITTED = [
    "SCREEN_etype", "SCREEN_gap_eV", "SCREEN_gap_tier", "SCREEN_dos_ef", "SCREEN_spin_pol",
    "SCREEN_A_seebeck_uVK", "SCREEN_B_pf", "SCREEN_C_sharp", "SCREEN_D_dosmass",
    "SCREEN_mu_star_eV", "SCREEN_S_at_star_uVK", "SCREEN_carrier_n", "SCREEN_mu_railed",
    "SCREEN_gate_semiconductor", "SCREEN_B_pf_percentile", "SCREEN_screening_group",
    "SCREEN_rank_A_seebeck_uVK", "SCREEN_rank_B_pf", "SCREEN_rank_C_sharp", "SCREEN_rank_D_dosmass",
    "SCREEN_famrank_A_seebeck_uVK", "SCREEN_famrank_B_pf",
    "SCREEN_window_truncated", "SCREEN_dft_e_lo", "SCREEN_dft_e_hi",
]
# Screening columns that are kept: DXMag metadata or composition-only quantities (both sets), and columns that are
# filled for prediction-set rows only (from the machine-learned d-DOS). Checked below, so a new column cannot slip in.
KEPT_SCREEN = [
    "SCREEN_kL_proxy_mass", "SCREEN_ML_formE", "SCREEN_ML_hull", "SCREEN_hull_tier", "SCREEN_phonon_stable",
    "SCREEN_min_phonon_freq", "SCREEN_gate_stable", "SCREEN_dos_provenance",
    "SCREEN_pred_D34_TE_score", "SCREEN_rank_pred_D34_TE_score", "SCREEN_std_fermi", "SCREEN_std_fermi_pctile",
    "SCREEN_gate_reliable", "SCREEN_passes_all", "SCREEN_reject_reason",
]
PREDICTION_ONLY = ["SCREEN_pred_D34_TE_score", "SCREEN_rank_pred_D34_TE_score", "SCREEN_std_fermi",
                   "SCREEN_std_fermi_pctile", "SCREEN_gate_reliable", "SCREEN_passes_all", "SCREEN_reject_reason"]
# Reduced, readable column set for the certified entries (all values are this study's or DXMag's).
CERTIFIED_COLS = [
    "UUID", "composition", "run_dir", "data_set", "family", "space group symbol", "number of atoms in the cell",
    "hull", "phonon computed", "phonon stable (commensurate)", "dft_class", "gap", "heavy", "soc_status",
    "max_S_intr", "best_PF", "best_T", "best_conv", "S_best", "soc_best_PF",
    "final_best_PF", "final_S_best", "final_best_T",
    "final_cap1e21_p_900_PF", "final_cap1e21_p_900_S", "final_cap1e21_p_900_n",
    "final_cap1e21_n_900_PF", "final_cap1e21_n_900_S", "final_cap1e21_n_900_n",
    "final_cap1e20_p_900_PF", "final_cap1e20_p_900_S", "final_cap1e20_n_900_PF", "final_cap1e20_n_900_S",
]


def main():
    d = pd.read_pickle(os.path.join(ANALYSIS, "paper_dataset.pkl"))              # the shipped per-run table
    screen = [c for c in d.columns if c.startswith("SCREEN_")]                    # all pre-DFT screening columns
    unknown = sorted(set(screen) - set(OMITTED) - set(KEPT_SCREEN))               # unclassified screening columns
    if unknown:
        sys.exit(f"unclassified SCREEN_ columns, decide before exporting: {unknown}")
    train = d.data_set == "training"                                              # training-set rows
    leaked = [c for c in PREDICTION_ONLY if d.loc[train, c].notna().any()]        # must be empty for training rows
    if leaked:
        sys.exit(f"columns expected to be prediction-set only carry training-set values: {leaked}")
    keep = [c for c in d.columns if c not in OMITTED]                             # original column order
    d[keep].to_csv(os.path.join(ANALYSIS, "paper_dataset.csv"), index=False)      # full precision (repr of floats)
    cert = d[d.certified.astype(bool)][CERTIFIED_COLS]                            # the certified entries
    cert = cert.sort_values(["composition", "data_set", "family", "UUID"])        # stable, readable order
    cert.to_csv(os.path.join(ANALYSIS, "certified_entries.csv"), index=False)
    print(f"paper_dataset.csv: {len(d)} rows x {len(keep)} columns ({len(OMITTED)} omitted); "
          f"certified_entries.csv: {len(cert)} rows, {cert.composition.nunique()} compositions")


if __name__ == "__main__":
    main()
