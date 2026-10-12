#!/usr/bin/env python3
# =============================================================================
# te_rescreen_training_local.py -- Cutler-Mott rescreen of the TRAINING set
#
# SUPERSEDED, NOT USED FOR THE PAPER. The training-set screen of the study is
# code/screen_training_pool.py (archive d-DOS per stored cell, output
# results/screening/training_pool_screen_gapfix.csv). This earlier, per-atom
# version (July 2026) is kept for reference only; its archive path below is the
# original local one and must be changed before use.
#
# (real DFT DOS), computed ENTIRELY LOCALLY from the raw pymatgen CompleteDos
# JSON archives (D:/Mattergen_project/HeuslerProject/DOS/PDOS_json_<family>_
# 241229.tar.xz), instead of the HPC's processed test_targets.npy files.
#
#   Reuses the exact, already-validated extraction logic from
#   code/dos_dataset.py (_process_dos_data: d-orbital extraction, per-atom
#   normalisation, Fermi alignment, interpolation onto the same 2001-pt grid,
#   non-negative clip) and the exact Cutler-Mott transport code from
#   code/te_rescreen.py (transport_descriptors, classify) -- so results are on
#   the identical scale as the predicted-set Stage-1.5 rescreen.
#
# Usage:
#   python code/te_rescreen_training_local.py --top 50
#   python code/te_rescreen_training_local.py --limit 50   # quick smoke test
# =============================================================================
import argparse, tarfile, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent))
from te_rescreen import transport_descriptors, classify, mean_atomic_mass, load_stability, NE, FAMILIES

ARCHIVE_DIR = Path(r"D:\Mattergen_project\HeuslerProject\DOS")
ARCHIVE_PATTERN = "PDOS_json_{fam}_241229.tar.xz"

# ---- extraction logic copied verbatim from code/dos_dataset.py's
# _process_dos_data (avoids importing that module, which pulls in torch at
# import time for its unrelated Dataset class). Same grid, same steps,
# same validated d-orbital source key -- see dos_dataset.py for the
# 200-material validation notes this was confirmed against.
_COMMON_GRID = np.linspace(-10.0, 10.0, 2001)
_EXPECTED_RAW_POINTS = 2401

def _process_dos_data(data, json_path=""):
    for field in ("efermi", "energies", "structure", "spd_dos"):
        if field not in data:
            raise KeyError(f"Missing field '{field}' in {json_path}")
    if "d" not in data["spd_dos"] or "densities" not in data["spd_dos"]["d"]:
        raise KeyError(f"spd_dos missing d/densities in {json_path}")
    for spin in ("1", "-1"):
        if spin not in data["spd_dos"]["d"]["densities"]:
            raise KeyError(f"spd_dos['d']['densities'] missing spin '{spin}' in {json_path}")

    energies = np.array(data["energies"], dtype=np.float64)
    efermi = float(data["efermi"])
    n_atoms = len(data["structure"]["sites"])
    if len(energies) != _EXPECTED_RAW_POINTS:
        raise ValueError(f"Expected {_EXPECTED_RAW_POINTS} energy points, got {len(energies)}")
    if n_atoms <= 0:
        raise ValueError(f"n_atoms={n_atoms} invalid")

    spd_d = data["spd_dos"]["d"]["densities"]
    dos_up = np.array(spd_d["1"], dtype=np.float64)
    dos_down = np.array(spd_d["-1"], dtype=np.float64)
    if len(dos_up) != _EXPECTED_RAW_POINTS or len(dos_down) != _EXPECTED_RAW_POINTS:
        raise ValueError("dos_up/dos_down length mismatch")

    dos_up = dos_up / n_atoms
    dos_down = dos_down / n_atoms
    e_shifted = energies - efermi

    f_up = interp1d(e_shifted, dos_up, kind="linear", fill_value=0.0, bounds_error=False)
    f_down = interp1d(e_shifted, dos_down, kind="linear", fill_value=0.0, bounds_error=False)
    dos_up_aligned = np.clip(f_up(_COMMON_GRID), 0.0, None)
    dos_down_aligned = np.clip(f_down(_COMMON_GRID), 0.0, None)

    target = np.concatenate([dos_up_aligned, dos_down_aligned])
    assert target.shape == (4002,)
    return target.astype(np.float32), efermi, n_atoms


def process_family_archive(fam, limit=None):
    arch_path = ARCHIVE_DIR / ARCHIVE_PATTERN.format(fam=fam)
    if not arch_path.exists():
        print(f"  [skip] {fam}: archive not found ({arch_path.name})")
        return None
    t0 = time.time()
    rows = []
    n_err = 0
    with tarfile.open(arch_path, mode="r:xz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".json")]
        if limit:
            members = members[:limit]
        for i, m in enumerate(members):
            fname = Path(m.name).name  # "{composition}_{UUID}.json"
            mid = fname[:-5]  # strip ".json"
            try:
                f = tf.extractfile(m)
                data = json.load(f)
                target, efermi, n_atoms = _process_dos_data(data, m.name)
            except Exception as e:
                n_err += 1
                continue
            up, dn = target[0:NE].astype(np.float64), target[NE:2 * NE].astype(np.float64)
            g = up + dn
            desc = transport_descriptors(g)
            etype, gap_w, up_ef, dn_ef, pol = classify(up, dn)
            comp = mid.split("_", 1)[0]
            rows.append({
                "family": fam, "material_id": mid, "composition": comp,
                "etype": etype, "gap_eV": gap_w, "dos_ef": up_ef + dn_ef, "spin_pol": pol,
                "A_seebeck_uVK": desc["A_seebeck"], "B_pf": desc["B_pf"], "C_sharp": desc["C_sharp"],
                "D_dosmass": desc["D_dosmass"], "mu_star_eV": desc["mu_star"],
                "S_at_star_uVK": desc["S_at_star"], "carrier_n": desc["carrier_n"],
                "kL_proxy_mass": mean_atomic_mass(comp), "n_atoms": n_atoms,
            })
            if (i + 1) % 1000 == 0:
                print(f"    {fam}: {i+1}/{len(members)} processed ({time.time()-t0:.0f}s elapsed)")
    df = pd.DataFrame(rows)
    dt = time.time() - t0
    print(f"  {fam:14s}: {len(df):6d} materials processed, {n_err} errors, {dt:.0f}s  "
          f"({(df.etype=='semiconductor').sum() if len(df) else 0} sc / "
          f"{(df.etype=='half_metal').sum() if len(df) else 0} hm / "
          f"{(df.etype=='metal').sum() if len(df) else 0} metal)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stability", type=Path, default=Path("data/DXMag_HeuslerDB_2025-09-09.xlsx"))
    ap.add_argument("--gap_window", type=float, nargs=2, default=[0.1, 1.0])
    ap.add_argument("--hull_max", type=float, default=0.10)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None, help="per-family cap, for smoke testing")
    ap.add_argument("--out", type=Path, default=Path("results/screening/train_rescreen_unified_ranked.csv"))
    args = ap.parse_args()

    dxmag = load_stability(args.stability) if args.stability and args.stability.exists() else None
    print("=== Training-set Cutler-Mott re-rank (900 K, ground-truth DOS, LOCAL archives) ===")
    frames = []
    for fam in FAMILIES:
        d = process_family_archive(fam, limit=args.limit)
        if d is not None and len(d):
            frames.append(d)
    if not frames:
        raise SystemExit("no family archive data processed")
    allm = pd.concat(frames, ignore_index=True)
    print(f"\nPooled: {len(allm):,} materials, {allm.family.nunique()} families")

    if dxmag is not None:
        allm["UUID"] = allm["material_id"].str.split("_", n=1).str[1].str.lower()
        allm = allm.merge(dxmag, on="UUID", how="left")

    gmin, gmax = args.gap_window
    allm["gate_semiconductor"] = (allm.etype == "semiconductor") & allm.gap_eV.between(gmin, gmax)
    allm["gate_reliable"] = True  # DFT DOS, no ensemble uncertainty
    if "ML_hull" in allm.columns:
        hull_ok = (allm.ML_hull < args.hull_max) | allm.ML_hull.isna()
        form_ok = (allm.ML_formE < 0) | allm.ML_formE.isna()
        ph_bad = pd.Series(False, index=allm.index)
        if "phonon_stable" in allm.columns:
            ph_bad |= allm.phonon_stable.astype(str).str.lower().isin(["false", "no", "0", "unstable"])
        if "min_phonon_freq" in allm.columns:
            ph_bad |= (allm.min_phonon_freq < -5)
        allm["gate_stable"] = hull_ok & form_ok & ~ph_bad
    else:
        allm["gate_stable"] = True
    allm["passes_all"] = allm.gate_semiconductor & allm.gate_reliable & allm.gate_stable

    surv = allm[allm.passes_all].copy().sort_values("B_pf", ascending=False)
    print("\n=== gate funnel ===")
    print(f"  semiconductor+gap : {allm.gate_semiconductor.sum():,}")
    print(f"  stable            : {allm.gate_stable.sum():,}")
    print(f"  PASS all          : {allm.passes_all.sum():,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    allm.to_csv(str(args.out).replace(".csv", "_ALL_annotated.csv"), index=False)
    cols = ["family", "composition", "material_id", "etype", "gap_eV", "dos_ef",
            "A_seebeck_uVK", "B_pf", "S_at_star_uVK", "mu_star_eV", "carrier_n",
            "C_sharp", "D_dosmass", "kL_proxy_mass", "ML_hull", "ML_formE"]
    cols = [c for c in cols if c in surv.columns]
    surv[cols].to_csv(args.out, index=False)
    print(f"\nwrote: {args.out}")
    print(f"wrote: {str(args.out).replace('.csv', '_ALL_annotated.csv')}")

    if len(surv):
        print(f"\n=== Top {args.top} training-set survivors by Power-Factor potential (900 K) ===")
        show = ["family", "composition", "etype", "gap_eV", "A_seebeck_uVK", "B_pf"]
        with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.3g}"):
            print(surv[show].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
