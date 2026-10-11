#!/usr/bin/env python3
"""
create_pretrain_dataset.py — Build combined CSV for pre-training on all 6 Heusler datasets.

Foundation model approach: include ALL available JSON files from all 6 datasets.
No exclusions. Fine-tuning evaluates on its own held-out test split.

Exact counts (verified 2026-04-26):
    full_cubic      :  9,396
    inverse_cubic   :  9,072
    half_cubic      :  9,396
    full_tetra      :  8,047
    inverse_tetra   :  8,437
    half_tetra      :  6,909
    TOTAL           : 51,257

Output CSV columns:
    material_id  — unique material identifier (filename stem)
    dos_path     — absolute path to the DOS JSON file
    dataset      — source dataset name

Usage:
    python code/create_pretrain_dataset.py \
        --data_root /home/YOUR_USERNAME/heusler_dos_svnit/data \
        --output    /home/YOUR_USERNAME/heusler_dos_svnit/data/pretrain/pretrain_train.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

DATASETS = [
    'full_cubic', 'inverse_cubic', 'half_cubic',
    'full_tetra', 'inverse_tetra', 'half_tetra',
]

EXPECTED_COUNTS = {
    'full_cubic':    9396,
    'inverse_cubic': 9072,
    'half_cubic':    9396,
    'full_tetra':    8047,
    'inverse_tetra': 8437,
    'half_tetra':    6909,
}
EXPECTED_TOTAL = 51257


def main():
    parser = argparse.ArgumentParser(description="Build combined pre-training CSV (all 6 datasets).")
    parser.add_argument('--data_root', type=Path, required=True,
                        help='Root data directory (contains {dataset}/ subdirs)')
    parser.add_argument('--output',    type=Path, required=True,
                        help='Output CSV path')
    args = parser.parse_args()

    if not args.data_root.exists():
        print(f"ERROR: data_root not found: {args.data_root}")
        sys.exit(1)

    rows = []
    print("=" * 60)
    print("Building pre-training dataset (foundation model — all 6 datasets)")
    print("=" * 60)

    for ds in DATASETS:
        dos_dir = args.data_root / ds / 'dos_json'
        if not dos_dir.exists():
            print(f"  WARNING: {dos_dir} not found — skipping {ds}")
            continue

        json_files = sorted(dos_dir.glob('*.json'))
        n = len(json_files)
        expected = EXPECTED_COUNTS.get(ds, -1)
        flag = "" if expected < 0 else ("  OK" if n == expected else f"  WARNING: expected {expected:,}")
        print(f"  {ds:<20s}: {n:>6,} materials{flag}")

        for f in json_files:
            rows.append({
                'material_id': f.stem,
                'dos_path':    str(f.resolve()),
                'dataset':     ds,
            })

    if not rows:
        print("ERROR: No materials found. Check data_root path.")
        sys.exit(1)

    df_out = pd.DataFrame(rows, columns=['material_id', 'dos_path', 'dataset'])

    # Detect cross-dataset duplicates (should not occur with UUID-suffixed IDs)
    dupes = df_out.duplicated(subset='material_id').sum()
    if dupes > 0:
        print(f"  WARNING: {dupes} duplicate material_ids found — keeping first occurrence")
        df_out = df_out.drop_duplicates(subset='material_id', keep='first')

    total = len(df_out)
    print("=" * 60)
    print(f"Total: {total:,} materials (expected {EXPECTED_TOTAL:,})")
    if total != EXPECTED_TOTAL:
        print(f"  WARNING: count mismatch ({total:,} vs {EXPECTED_TOTAL:,})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
