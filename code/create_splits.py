#!/usr/bin/env python3
"""
create_splits.py — Generate train/val/test split CSV files for ALIGNN DOS experiments.

Run ONCE before any training begins.

Usage:
    python code/create_splits.py --dataset full_cubic
    python code/create_splits.py --all
    python code/create_splits.py --validate-only --dataset full_cubic
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.model_selection import GroupKFold, KFold

# ── Constants ──────────────────────────────────────────────────────────────────

DATASETS = [
    "full_cubic",
    "full_tetra",
    "inverse_cubic",
    "inverse_tetra",
    "half_cubic",
    "half_tetra",
]

# ── Experiment D: Element Holdout Split ───────────────────────────────────────
# Hold out ALL compounds containing the holdout element from ALL datasets.
# This is the OOD-Element generalisation test (cf. OMat24, MatFold 2025).
# Rh chosen: 4d TM, well-represented, distinct d-band physics from majority 3d.
EXPD_HOLDOUT_ELEMENT = "Rh"       # X-site element to hold out globally
EXPD_VAL_FRACTION    = 0.10       # fraction of train pool used for val
EXPD_RANDOM_SEED     = 42         # same seed as Exp A for reproducibility

# Known-correct Z-site mappings for validation (chemically verified)
VALIDATION_EXAMPLES = {
    "Co2MnSi": "Si",
    "Co2MnGe": "Ge",
    "Co2MnAl": "Al",
    "Fe2VAl":  "Al",
    "NiMnSb":  "Sb",
    "Co2FeSi": "Si",
    "Co2CrGa": "Ga",
    "Ni2MnSn": "Sn",
    "Co2MnGa": "Ga",
}


# ── Z-site identification ──────────────────────────────────────────────────────

def get_z_site(formula_str: str) -> tuple:
    """
    Identify the Z-site element in a Heusler compound formula.

    Z-site is the p-block (main-group) element:
      - Full/Inverse Heusler X2YZ: X,Y are d-block transition metals; Z is p-block
      - Half Heusler XYZ: same convention

    Strategy:
      1. Primary: find the unique p-block element (el.block == 'p')
      2. If multiple p-block elements: pick the most electronegative
      3. If no p-block element (unusual): fall back to most electronegative overall

    Returns: (symbol: str, atomic_number: int)
    """
    comp = Composition(formula_str)
    elements = list(comp.elements)

    p_block = [el for el in elements if el.block == "p"]

    if len(p_block) == 1:
        z = p_block[0]
    elif len(p_block) > 1:
        # Multiple p-block: pick most electronegative, use atomic number as tiebreaker
        z = max(p_block, key=lambda el: (el.X, el.Z))
    else:
        # No p-block element — unusual for Heuslers; fall back to electronegativity
        z = max(elements, key=lambda el: (el.X, el.Z))

    return z.symbol, z.Z


def validate_z_site_parser() -> None:
    """
    Validate Z-site parsing against known-correct examples before any file writing.
    Raises ValueError if any expected mapping fails.
    """
    print("\n── Z-site parser validation ──────────────────────────────")
    print(f"  {'Formula':<20} {'Parsed Z':<12} {'Expected':<12} Status")
    print(f"  {'-'*20} {'-'*12} {'-'*12} ------")

    failures = []
    for formula, expected_z in VALIDATION_EXAMPLES.items():
        try:
            z_sym, z_an = get_z_site(formula)
        except Exception as e:
            z_sym = f"ERROR: {e}"
            z_an = 0
        ok = z_sym == expected_z
        status = "✓" if ok else "✗ FAIL"
        print(f"  {formula:<20} {z_sym:<12} {expected_z:<12} {status}")
        if not ok:
            failures.append((formula, z_sym, expected_z))

    if failures:
        msgs = [
            f"  {formula} → {got} (expected {exp})"
            for formula, got, exp in failures
        ]
        raise ValueError(
            "Z-site parsing produced unexpected results:\n"
            + "\n".join(msgs)
            + "\nCheck the get_z_site() function logic."
        )
    print("  All validation examples PASSED ✓\n")


# ── CIF directory parsing ──────────────────────────────────────────────────────

def parse_cif_directory(cif_dir: Path) -> pd.DataFrame:
    """
    Scan all *.cif files in cif_dir.
    Returns DataFrame: material_id, formula, numeric_id, z_site_element, z_atomic_number
    """
    cif_files = sorted(cif_dir.glob("*.cif"))
    if not cif_files:
        raise FileNotFoundError(f"No CIF files found in {cif_dir}")

    rows = []
    skipped = 0
    for cif_path in cif_files:
        stem = cif_path.stem  # e.g., "Co2MnSi_03248"
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            print(f"  WARNING: unexpected filename format: {cif_path.name} — skipping")
            skipped += 1
            continue

        formula_str, numeric_id = parts
        material_id = stem

        try:
            z_sym, z_an = get_z_site(formula_str)
        except Exception as e:
            print(f"  WARNING: cannot parse Z-site for '{formula_str}': {e} — skipping")
            skipped += 1
            continue

        rows.append({
            "material_id":    material_id,
            "formula":        formula_str,
            "numeric_id":     numeric_id,
            "z_site_element": z_sym,
            "z_atomic_number": int(z_an),
        })

    if skipped:
        print(f"  Skipped {skipped} files due to parse errors.")

    return pd.DataFrame(rows)


# ── CSV I/O ────────────────────────────────────────────────────────────────────

def write_csv(ids: list, filepath: Path) -> None:
    """Write a list of material_ids as a CSV with a 'material_id' header."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write("material_id\n")
        for mid in ids:
            f.write(f"{mid}\n")


# ── Integrity checks ───────────────────────────────────────────────────────────

def integrity_checks(train_ids, val_ids, test_ids, all_ids, fold_k: int, exp: str) -> None:
    """Assert no overlap between splits and full population coverage."""
    train_s = set(train_ids)
    val_s   = set(val_ids)
    test_s  = set(test_ids)

    overlap_tt = train_s & test_s
    overlap_vt = val_s   & test_s
    overlap_tv = train_s & val_s
    total      = len(train_ids) + len(val_ids) + len(test_ids)

    assert not overlap_tt, (
        f"FAIL [{exp} fold {fold_k}]: train-test overlap "
        f"({len(overlap_tt)} samples): {list(overlap_tt)[:5]}"
    )
    assert not overlap_vt, (
        f"FAIL [{exp} fold {fold_k}]: val-test overlap "
        f"({len(overlap_vt)} samples): {list(overlap_vt)[:5]}"
    )
    assert not overlap_tv, (
        f"FAIL [{exp} fold {fold_k}]: train-val overlap "
        f"({len(overlap_tv)} samples): {list(overlap_tv)[:5]}"
    )
    assert total == len(all_ids), (
        f"FAIL [{exp} fold {fold_k}]: missing IDs "
        f"(got {total}, expected {len(all_ids)})"
    )


def check_formula_leakage(train_ids: list, test_ids: list) -> set:
    """
    Return formulas (reduced, no numeric suffix) that appear in BOTH train and test.
    Same reduced formula = potential data leakage in Experiment A random splits.
    """
    def strip(mid):
        return mid.rsplit("_", 1)[0]

    train_formulas = {strip(mid) for mid in train_ids}
    test_formulas  = {strip(mid) for mid in test_ids}
    return train_formulas & test_formulas


# ── Experiment A: 5-fold random KFold ─────────────────────────────────────────

def create_experiment_a(all_ids: list, split_dir: Path) -> tuple:
    """
    Create 5-fold random CV splits.
    Val is carved from the non-test portion (10%). Test is never touched during val carving.

    Returns: (fold_stats list, leakage_warnings list)
    """
    print("  Creating Experiment A — 5-fold random KFold...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    arr = np.array(sorted(all_ids))

    fold_stats       = []
    leakage_warnings = []

    for fold_k, (remaining_idx, test_idx) in enumerate(kf.split(arr)):
        test_ids      = arr[test_idx].tolist()
        remaining_ids = arr[remaining_idx].tolist()

        rng   = np.random.default_rng(seed=42 + fold_k)
        val_n = int(round(0.10 * len(remaining_ids)))
        val_idx_local = rng.choice(len(remaining_ids), size=val_n, replace=False)
        val_set_local = set(val_idx_local.tolist())

        val_ids   = [remaining_ids[i] for i in val_idx_local]
        train_ids = [remaining_ids[i] for i in range(len(remaining_ids))
                     if i not in val_set_local]

        integrity_checks(train_ids, val_ids, test_ids, all_ids, fold_k, "ExpA")

        overlap = check_formula_leakage(train_ids, test_ids)
        if overlap:
            leakage_warnings.append((fold_k, sorted(overlap)))

        write_csv(train_ids, split_dir / f"expA_fold{fold_k}_train.csv")
        write_csv(val_ids,   split_dir / f"expA_fold{fold_k}_val.csv")
        write_csv(test_ids,  split_dir / f"expA_fold{fold_k}_test.csv")

        fold_stats.append({
            "fold":     fold_k,
            "train":    len(train_ids),
            "val":      len(val_ids),
            "test":     len(test_ids),
            "test_pct": round(len(test_ids) / len(all_ids) * 100, 1),
        })

    return fold_stats, leakage_warnings


# ── Experiment B: 5-fold composition GroupKFold ────────────────────────────────

def create_experiment_b(all_ids: list, z_site_map: dict, split_dir: Path) -> list:
    """
    Create 5-fold composition-aware GroupKFold splits (groups = Z-site element).
    Val is carved randomly from the non-test portion (10%).

    Returns: fold_stats list
    """
    print("  Creating Experiment B — 5-fold GroupKFold by Z-site element...")
    gkf      = GroupKFold(n_splits=5)
    arr      = np.array(sorted(all_ids))
    groups   = np.array([z_site_map[mid] for mid in arr])

    fold_stats = []

    for fold_k, (remaining_idx, test_idx) in enumerate(gkf.split(arr, groups=groups)):
        test_ids      = arr[test_idx].tolist()
        remaining_ids = arr[remaining_idx].tolist()

        rng   = np.random.default_rng(seed=42 + fold_k)
        val_n = int(round(0.10 * len(remaining_ids)))
        val_idx_local = rng.choice(len(remaining_ids), size=val_n, replace=False)
        val_set_local = set(val_idx_local.tolist())

        val_ids   = [remaining_ids[i] for i in val_idx_local]
        train_ids = [remaining_ids[i] for i in range(len(remaining_ids))
                     if i not in val_set_local]

        integrity_checks(train_ids, val_ids, test_ids, all_ids, fold_k, "ExpB")

        test_z_groups = sorted({z_site_map[mid] for mid in test_ids})

        write_csv(train_ids, split_dir / f"expB_fold{fold_k}_train.csv")
        write_csv(val_ids,   split_dir / f"expB_fold{fold_k}_val.csv")
        write_csv(test_ids,  split_dir / f"expB_fold{fold_k}_test.csv")

        fold_stats.append({
            "fold":          fold_k,
            "train":         len(train_ids),
            "val":           len(val_ids),
            "test":          len(test_ids),
            "test_pct":      round(len(test_ids) / len(all_ids) * 100, 1),
            "test_z_groups": test_z_groups,
        })

    return fold_stats


# ── Experiment C: Full-data Deployment Model ──────────────────────────────────

def create_experiment_c(all_ids: list, split_dir: Path) -> dict:
    """
    Experiment C: Full-data deployment model (no held-out test set).

    All materials are used for training.  A 10% random subset is carved out
    as a val set for LR scheduling / early-stopping only — it is NOT a true
    test set and those metrics must not be compared against Exp A/B results.

    The test CSV is set equal to the val CSV so the standard 4-step pipeline
    (train → infer → metrics) runs without modification; results files are
    tagged 'expC' so users can identify them.

    Files written:
      expC_fold0_train.csv  — 90% of all materials
      expC_fold0_val.csv    — 10% of all materials (LR scheduling only)
      expC_fold0_test.csv   — same as val (deployment model; no true test)
      expC_stats.json       — counts and deployment note

    Returns dict with keys: n_train, n_val, n_total, note
    """
    print("  Creating Experiment C — Full-data deployment model...")

    rng   = np.random.default_rng(seed=42)
    arr   = sorted(all_ids)
    val_n = max(1, int(round(0.10 * len(arr))))
    val_idx_arr = rng.choice(len(arr), size=val_n, replace=False)
    val_set     = set(val_idx_arr.tolist())

    val_ids   = [arr[i] for i in sorted(val_idx_arr)]
    train_ids = [arr[i] for i in range(len(arr)) if i not in val_set]

    write_csv(train_ids, split_dir / "expC_fold0_train.csv")
    write_csv(val_ids,   split_dir / "expC_fold0_val.csv")
    # test = val (deployment model; no true holdout — metrics are on val set)
    write_csv(val_ids,   split_dir / "expC_fold0_test.csv")

    stats = {
        'n_train':  len(train_ids),
        'n_val':    len(val_ids),
        'n_test':   len(val_ids),   # same as val
        'n_total':  len(arr),
        'val_frac': round(val_n / len(arr), 4),
        'note':     'Exp C is a full-data deployment model. '
                    'test.csv == val.csv — Exp C test metrics are NOT comparable '
                    'to Exp A/B held-out results.',
    }

    import json
    with open(split_dir / "expC_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"    train={len(train_ids)}, val={len(val_ids)}, "
          f"test=val ({val_n} samples — deployment model, no true holdout)")
    return stats


# ── Experiment D: Element Holdout Split ───────────────────────────────────────

def create_experiment_d(all_ids: list, df: pd.DataFrame, split_dir: Path) -> dict:
    """
    Experiment D: Element Holdout Split.

    Hold out all compounds containing EXPD_HOLDOUT_ELEMENT (Rh by default)
    from this dataset. The remaining compounds form the train+val pool.

    Split structure:
      expD_fold0_train.csv  — non-Rh compounds (90% of pool, after val carve-out)
      expD_fold0_val.csv    — non-Rh compounds (10% of pool, for LR scheduling)
      expD_fold0_test.csv   — ALL Rh-containing compounds (held out entirely)
      expD_stats.json       — counts, holdout element, leakage check result

    Returns dict with keys: n_train, n_val, n_test, n_total, holdout_element
    """
    print(f"  Creating Experiment D — Element Holdout ({EXPD_HOLDOUT_ELEMENT})...")

    HOLDOUT = EXPD_HOLDOUT_ELEMENT

    # ── Step 1: Build formula lookup from pre-parsed df ───────────────────────
    # df already has material_id → formula from parse_cif_directory()
    formula_map = dict(zip(df["material_id"], df["formula"]))

    # ── Step 2: Classify each material as holdout-element-containing or not ───
    rh_ids       = []
    non_rh_ids   = []
    parse_errors = []

    for mid in all_ids:
        formula_str = formula_map.get(mid, "")
        if not formula_str:
            parse_errors.append(mid)
            non_rh_ids.append(mid)   # conservative: include in train if missing
            continue
        try:
            comp     = Composition(formula_str)
            elements = [str(el) for el in comp.elements]
            if HOLDOUT in elements:
                rh_ids.append(mid)
            else:
                non_rh_ids.append(mid)
        except Exception as e:
            parse_errors.append(mid)
            non_rh_ids.append(mid)   # conservative fallback

    if parse_errors:
        print(f"  [ExpD] {len(parse_errors)} formula parse errors "
              f"(conservatively included in train)")

    # ── Step 3: Carve val set from non-holdout pool ───────────────────────────
    rng         = np.random.default_rng(EXPD_RANDOM_SEED)
    non_rh_arr  = np.array(sorted(non_rh_ids))
    rng.shuffle(non_rh_arr)

    n_val   = max(1, int(round(len(non_rh_arr) * EXPD_VAL_FRACTION)))
    n_train = len(non_rh_arr) - n_val

    train_ids = non_rh_arr[:n_train].tolist()
    val_ids   = non_rh_arr[n_train:].tolist()
    test_ids  = sorted(rh_ids)    # ALL holdout compounds go to test — never split

    # ── Step 4: Leakage checks ────────────────────────────────────────────────
    train_set = set(train_ids)
    val_set   = set(val_ids)
    test_set  = set(test_ids)

    assert len(train_set & test_set) == 0, (
        f"LEAKAGE [ExpD]: {len(train_set & test_set)} {HOLDOUT} IDs found in train!"
    )
    assert len(val_set & test_set) == 0, (
        f"LEAKAGE [ExpD]: {len(val_set & test_set)} {HOLDOUT} IDs found in val!"
    )
    assert len(train_set & val_set) == 0, (
        f"OVERLAP [ExpD]: {len(train_set & val_set)} IDs in both train and val!"
    )
    assert len(train_set) + len(val_set) + len(test_set) == len(all_ids), (
        f"MISSING [ExpD]: {len(train_set)+len(val_set)+len(test_set)} != {len(all_ids)}"
    )

    # ── Step 5: Write CSV files ───────────────────────────────────────────────
    write_csv(train_ids, split_dir / "expD_fold0_train.csv")
    write_csv(val_ids,   split_dir / "expD_fold0_val.csv")
    write_csv(test_ids,  split_dir / "expD_fold0_test.csv")

    # ── Step 6: Write stats JSON ──────────────────────────────────────────────
    stats = {
        "experiment":      "D",
        "split_type":      "element_holdout",
        "holdout_element": HOLDOUT,
        "n_total":         len(all_ids),
        "n_train":         len(train_ids),
        "n_val":           len(val_ids),
        "n_test":          len(test_ids),
        "test_fraction":   round(len(test_ids) / max(len(all_ids), 1), 4),
        "val_fraction":    EXPD_VAL_FRACTION,
        "random_seed":     EXPD_RANDOM_SEED,
        "parse_errors":    len(parse_errors),
        "leakage_check":   "PASSED",
    }
    with open(split_dir / "expD_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  [ExpD] train={len(train_ids):,}, val={len(val_ids):,}, "
          f"test={len(test_ids):,} ({HOLDOUT}-containing), leakage=NONE")

    if len(test_ids) == 0:
        print(f"  [ExpD] WARNING: No {HOLDOUT}-containing compounds found. "
              f"expD_fold0_test.csv is empty — this dataset will be skipped in ExpD evaluation.")

    return stats


# ── Statistics printing and saving ────────────────────────────────────────────

def print_fold_stats(label: str, fold_stats: list, show_z_groups: bool = False) -> None:
    print(f"\n  {label}:")
    header = f"  {'Fold':<6} {'Train':<8} {'Val':<6} {'Test':<8} {'Test%':<7}"
    if show_z_groups:
        header += " Test Z-groups"
    print(header)
    for s in fold_stats:
        line = f"  {s['fold']:<6} {s['train']:<8} {s['val']:<6} {s['test']:<8} {s['test_pct']:<7.1f}%"
        if show_z_groups:
            line += f" {s['test_z_groups']}"
        print(line)

    trains = [s["train"] for s in fold_stats]
    vals   = [s["val"]   for s in fold_stats]
    tests  = [s["test"]  for s in fold_stats]
    print(
        f"  Mean: train={np.mean(trains):.0f}±{np.std(trains):.0f}  "
        f"val={np.mean(vals):.0f}±{np.std(vals):.0f}  "
        f"test={np.mean(tests):.0f}±{np.std(tests):.0f}"
    )


def print_z_distribution(group_counts: pd.DataFrame, n_total: int) -> None:
    print(f"\n  Z-site group distribution (sorted by count):")
    print(f"  {'Z-element':<12} {'Atomic#':<10} {'Count':<8} {'%':<7}")
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*7}")
    for _, row in group_counts.iterrows():
        cnt  = int(row["count"])
        warn = ""
        if cnt < 5:
            warn = "  ← warn: <5 samples (consider merging)"
        elif cnt < 10:
            warn = "  ← warn: <10 samples"
        print(f"  {row.z_site_element:<12} {int(row.z_atomic_number):<10} {cnt:<8} {row.pct:.1f}%{warn}")
    print(f"  Total groups: {len(group_counts)}")


# ── Main dataset processor ─────────────────────────────────────────────────────

def process_dataset(dataset_name: str, data_root: Path, validate_only: bool = False) -> None:
    dataset_dir = data_root / dataset_name
    cif_dir     = dataset_dir / "cif"
    split_dir   = dataset_dir / "data_split"

    print(f"\n{'═'*54}")
    print(f"DATASET: {dataset_name}")
    print(f"{'═'*54}")

    if not cif_dir.exists():
        print(f"  ERROR: CIF directory not found: {cif_dir}")
        return

    # Parse CIF files
    print(f"  Parsing CIF files from {cif_dir} ...")
    df      = parse_cif_directory(cif_dir)
    n_total = len(df)
    print(f"  Total materials: {n_total}")

    if n_total == 0:
        print("  ERROR: No valid CIF files found.")
        return

    # Print random sample for visual inspection
    sample_n  = min(20, n_total)
    sample_df = df.sample(n=sample_n, random_state=42)
    print(f"\n  Sample formula → Z-site mappings ({sample_n} random):")
    print(f"  {'material_id':<32} {'formula':<16} {'Z-site':<8} Z(atomic#)")
    for _, row in sample_df.iterrows():
        print(
            f"  {row.material_id:<32} {row.formula:<16} "
            f"{row.z_site_element:<8} {row.z_atomic_number}"
        )

    # Z-group distribution
    group_counts = (
        df.groupby(["z_site_element", "z_atomic_number"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    group_counts["pct"] = group_counts["count"] / n_total * 100
    print_z_distribution(group_counts, n_total)

    small_5  = group_counts[group_counts["count"] < 5]["z_site_element"].tolist()
    small_10 = group_counts[group_counts["count"] < 10]["z_site_element"].tolist()
    if small_10:
        print(f"  Groups with < 10 samples: {small_10}")
    if small_5:
        print(f"  Groups with <  5 samples: {small_5}")

    if validate_only:
        print("\n  [--validate-only: no files written]")
        return

    # Build lookup maps
    all_ids    = sorted(df["material_id"].tolist())
    z_site_map = dict(zip(df["material_id"], df["z_site_element"]))

    # Write group_assignments.csv
    split_dir.mkdir(parents=True, exist_ok=True)
    group_csv = split_dir / "group_assignments.csv"
    df[["material_id", "formula", "z_site_element", "z_atomic_number", "numeric_id"]].to_csv(
        group_csv, index=False
    )
    print(f"\n  Written: {group_csv}")

    # Create splits
    expA_stats, leakage_warnings = create_experiment_a(all_ids, split_dir)
    expB_stats = create_experiment_b(all_ids, z_site_map, split_dir)
    expC_stats = create_experiment_c(all_ids, split_dir)
    expD_stats = create_experiment_d(all_ids, df, split_dir)

    # Print fold stats
    print_fold_stats("Experiment A — Random 5-Fold CV", expA_stats, show_z_groups=False)
    print_fold_stats("Experiment B — Composition GroupKFold CV", expB_stats, show_z_groups=True)

    # Leakage reports
    print(f"\n  ── Data leakage check (Exp A) ──────────────────────")
    if not leakage_warnings:
        print("  [PASS — no formula overlap detected]")
    else:
        total_overlap = sum(len(o) for _, o in leakage_warnings)
        print(f"  [WARNING — {total_overlap} formula(s) found in both train AND test:]")
        for fold_k, overlap in leakage_warnings:
            for formula in overlap[:5]:
                print(f"    '{formula}' appears in both train and test of fold {fold_k}")
            if len(overlap) > 5:
                print(f"    ... and {len(overlap)-5} more")

    print(f"\n  ── Data leakage check (Exp B) ──────────────────────")
    print("  Z-element groups are mutually exclusive by construction: PASS ✓")

    # Save JSON stats
    def to_jsonable(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        return x

    stats = {
        "dataset":        dataset_name,
        "total_materials": n_total,
        "n_z_groups":     int(len(group_counts)),
        "z_group_distribution": [
            {
                "z_element":     str(row.z_site_element),
                "atomic_number": int(row.z_atomic_number),
                "count":         int(row["count"]),
                "pct":           float(round(row["pct"], 2)),
            }
            for _, row in group_counts.iterrows()
        ],
        "leakage_warnings_expA": [
            {"fold": k, "overlapping_formulas": ov}
            for k, ov in leakage_warnings
        ],
        "experiment_A": {
            "folds": expA_stats,
            "means": {
                "train": float(np.mean([s["train"] for s in expA_stats])),
                "val":   float(np.mean([s["val"]   for s in expA_stats])),
                "test":  float(np.mean([s["test"]  for s in expA_stats])),
            },
        },
        "experiment_B": {
            "folds": expB_stats,
            "means": {
                "train": float(np.mean([s["train"] for s in expB_stats])),
                "val":   float(np.mean([s["val"]   for s in expB_stats])),
                "test":  float(np.mean([s["test"]  for s in expB_stats])),
            },
        },
        "experiment_D": {
            "holdout_element": expD_stats["holdout_element"],
            "n_train":         expD_stats["n_train"],
            "n_val":           expD_stats["n_val"],
            "n_test":          expD_stats["n_test"],
            "test_fraction":   expD_stats["test_fraction"],
            "leakage_check":   expD_stats["leakage_check"],
        },
    }

    stats_path = split_dir / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Written: {stats_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate train/val/test splits for ALIGNN DOS experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset", choices=DATASETS,
        help="Process one specific dataset.",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Process all 6 datasets sequentially.",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Report statistics and check Z-site parsing only. No files written.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        metavar="DIR",
        help="Root data directory (default: data/).",
    )
    args = parser.parse_args()

    # Always validate Z-site parser first
    validate_z_site_parser()

    datasets = DATASETS if args.all else [args.dataset]
    for ds in datasets:
        process_dataset(ds, args.data_root, validate_only=args.validate_only)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
