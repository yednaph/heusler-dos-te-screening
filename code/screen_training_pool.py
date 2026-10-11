#!/usr/bin/env python3
# =============================================================================
# screen_training_pool.py -- Cutler-Mott screen of the TRAINING pool from the
# real DFT d-projected DOS in data/<family>/dos_json/.
#
# WHY A SEPARATE SCRIPT, AND WHY THE POOLS ARE NEVER MERGED
# ---------------------------------------------------------
# te_rescreen.py screens the NOVEL pool (54,978) from ML-PREDICTED DOS.
# This screens the TRAINING pool (51,257) from REAL DFT DOS.
# Measured on 9,396 materials that have both, predicted-vs-true medians are:
#     dos_ef 1.03   C_sharp 0.98   D_dosmass 1.02   A_seebeck 0.84   B_pf 0.73
# i.e. descriptors built from predicted DOS carry A_seebeck ~16% low and B_pf
# ~27% low.  Pooling the two sets into one ranking would therefore hand the
# training pool a systematic ~19% A_seebeck advantage that is an artefact of DOS
# provenance, not thermoelectric quality.  The two pools are ranked SEPARATELY
# and their scores must never be compared across the pool boundary.
#
# The transport physics is imported from te_rescreen so both pools go through
# byte-identical descriptor code -- only the DOS source differs.
# =============================================================================
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import te_rescreen as T                      # transport_descriptors, classify, NE, EF_IDX, DE

# The energy grid the whole project uses: -10..+10 eV, 2001 points, E_F at index 1000.
E_GRID = (np.arange(T.NE) - T.EF_IDX) * T.DE
FAMILIES = ["full_cubic", "full_tetra", "inverse_cubic",
            "inverse_tetra", "half_cubic", "half_tetra"]


def d_dos_from_json(path):
    """Read one pymatgen CompleteDos json -> (d_up, d_down) on E_GRID, Fermi-aligned.

    Returns (up, down, e_lo, e_hi) where e_lo/e_hi are the actual energy limits
    available relative to E_F.  Those are reported so that any material whose
    DFT window does not span the full +/-10 eV grid can be flagged: np.interp
    pads with zeros outside the data range, and zero-padding would otherwise be
    silently indistinguishable from a genuine region of no states.
    """
    d = json.load(open(path))
    E   = np.asarray(d["energies"], dtype=np.float64)
    ef  = float(d["efermi"])
    dens = d["spd_dos"]["d"]["densities"]        # d-projected, summed over atoms
    rel = E - ef                                  # shift so E_F sits at 0
    up = np.interp(E_GRID, rel, np.asarray(dens["1"],  dtype=np.float64), left=0.0, right=0.0)
    dn = np.interp(E_GRID, rel, np.asarray(dens["-1"], dtype=np.float64), left=0.0, right=0.0)
    return up, dn, float(rel.min()), float(rel.max())


def process_family(fam, data_root, limit=None):
    """Screen every dos_json in one family. Returns a DataFrame, one row per material."""
    files = sorted((data_root / fam / "dos_json").glob("*.json"))
    if limit:
        files = files[:limit]
    rows = []
    for k, f in enumerate(files):
        try:
            up, dn, e_lo, e_hi = d_dos_from_json(f)
        except Exception as e:                     # a malformed json must not kill the sweep
            rows.append({"family": fam, "material_id": f.stem, "parse_error": str(e)[:80]})
            continue
        g = up + dn
        desc = T.transport_descriptors(g)                    # A/B/C/D, mu*, railing
        etype, gap_w, up_ef, dn_ef, pol = T.classify(up, dn)  # identical classifier
        comp = f.stem.split("_")[0]
        rows.append({
            "family": fam, "material_id": f.stem, "composition": comp,
            "UUID": f.stem.split("_", 1)[1].lower() if "_" in f.stem else None,
            "etype": etype, "gap_eV": gap_w, "dos_ef": up_ef + dn_ef, "spin_pol": pol,
            "A_seebeck_uVK": desc["A_seebeck"], "B_pf": desc["B_pf"],
            "C_sharp": desc["C_sharp"], "D_dosmass": desc["D_dosmass"],
            "mu_star_eV": desc["mu_star"], "S_at_star_uVK": desc["S_at_star"],
            "carrier_n": desc["carrier_n"], "mu_railed": desc["mu_railed"],
            "kL_proxy_mass": T.mean_atomic_mass(comp),
            # DFT energy window actually available; grid needs +/-10 eV
            "dft_e_lo": e_lo, "dft_e_hi": e_hi,
            "window_truncated": bool(e_lo > -10.0 or e_hi < 10.0),
            "dos_provenance": "DFT_real",        # never mix with 'ML_predicted' rows
            "parse_error": None,
        })
        if (k + 1) % 2000 == 0:
            print(f"    {fam}: {k+1}/{len(files)}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, default=Path("data"))
    ap.add_argument("--stability", type=Path, default=Path("data/DXMag_HeuslerDB_2025-09-09.xlsx"))
    ap.add_argument("--out", type=Path, default=Path("results/screening/training_pool_screen.csv"))
    ap.add_argument("--limit", type=int, default=None, help="per-family cap, for smoke tests")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"mu-scan: +/-{T.MU_MAX} eV, {len(T.MU_GRID)} points  (same as the novel-pool screen)")
    frames = []
    for fam in FAMILIES:
        ckpt = args.out.parent / f"_ckpt_training_{fam}.csv"
        if ckpt.exists() and not args.limit:      # resume: this family already done
            print(f"  [resume] {fam} from {ckpt.name}", flush=True)
            frames.append(pd.read_csv(ckpt)); continue
        print(f"  screening {fam} ...", flush=True)
        df = process_family(fam, args.data_root, args.limit)
        if not args.limit:
            df.to_csv(ckpt, index=False)          # checkpoint so a crash costs one family
        n_err = int(df.parse_error.notna().sum()) if "parse_error" in df else 0
        print(f"  {fam:14s}: {len(df):6,} materials  ({n_err} parse errors)", flush=True)
        frames.append(df)

    allm = pd.concat(frames, ignore_index=True)
    allm = allm[allm.parse_error.isna()].copy() if "parse_error" in allm else allm

    # --- stability, joined by UUID (never by formula: duplicate formulae exist) ---
    dxm = T.load_stability(args.stability) if args.stability.exists() else None
    if dxm is not None:
        allm = allm.merge(dxm, on="UUID", how="left")

    # --- same graded annotations as the novel-pool screen, so the two tables are
    #     structurally comparable even though their SCORES are not ------------
    allm["gate_semiconductor"] = (allm.etype == "semiconductor") & allm.gap_eV.between(0.1, 1.0)
    if "ML_hull" in allm.columns:
        hull_ok = (allm.ML_hull < 0.10) | allm.ML_hull.isna()
        form_ok = (allm.ML_formE < 0) | allm.ML_formE.isna()
        ph_bad = pd.Series(False, index=allm.index)
        if "phonon_stable" in allm.columns:
            ph_bad |= allm.phonon_stable.astype(str).str.lower().isin(["false", "no", "0", "unstable"])
        if "min_phonon_freq" in allm.columns:
            ph_bad |= (allm.min_phonon_freq < -5)
        allm["gate_stable"] = hull_ok & form_ok & ~ph_bad
        allm["hull_tier"] = pd.cut(allm.ML_hull, [-np.inf, 0.05, 0.10, 0.20, np.inf],
                                   labels=["<=0.05", "0.05-0.10", "0.10-0.20", ">0.20"], right=False)
        allm["hull_tier"] = allm["hull_tier"].cat.add_categories("unknown").fillna("unknown")
    allm["gap_tier"] = pd.cut(allm.gap_eV, [-np.inf, 0.10, 0.40, 1.00, 1.40, np.inf],
                              labels=["<0.10", "0.10-0.40", "0.40-1.00", "1.00-1.40", ">1.40"],
                              right=False)

    # --- ranks are WITHIN THIS POOL ONLY, by construction --------------------
    for col in ["A_seebeck_uVK", "B_pf", "C_sharp", "D_dosmass"]:
        allm[f"rank_{col}"] = allm[col].rank(ascending=False, method="min")
        allm[f"famrank_{col}"] = allm.groupby("family")[col].rank(ascending=False, method="min")

    allm.to_csv(args.out, index=False)
    print(f"\n=== TRAINING POOL SCREEN COMPLETE ===")
    print(f"  materials      : {len(allm):,}")
    print(f"  mu railed      : {int(allm.mu_railed.sum()):,} ({allm.mu_railed.mean()*100:.1f}%)")
    print(f"  window truncated: {int(allm.window_truncated.sum()):,}")
    print(f"  etype:"); print(allm.etype.value_counts().to_string())
    print(f"  A_seebeck median={allm.A_seebeck_uVK.median():.1f}  p95={allm.A_seebeck_uVK.quantile(.95):.1f}")
    print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    main()
