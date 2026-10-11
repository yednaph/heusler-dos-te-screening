#!/usr/bin/env python3
"""
compute_te_descriptors.py — Standalone TE screening descriptor computation.

Calls compute_te_descriptors() from compute_metrics.py on saved inference outputs.

Usage:
    python code/compute_te_descriptors.py \\
        --predictions results/full_cubic/expA/fold0/test_predictions.npy \\
        --targets     results/full_cubic/expA/fold0/test_targets.npy \\
        --ids         results/full_cubic/expA/fold0/test_ids.txt \\
        --dataset     full_cubic \\
        --experiment  A \\
        --fold        0 \\
        --output_dir  results/full_cubic/expA/fold0/

Outputs:
    te_descriptors.csv          — N rows, pred/true/err per descriptor
    te_descriptor_summary.json  — MAE and Pearson r per descriptor
    te_ranking.csv              — materials ranked by D34_TE_score
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Compute 34 thermoelectric descriptors from d-projected DOS predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--predictions", required=True, type=Path,
                        help="test_predictions.npy (shape N x 4002)")
    parser.add_argument("--targets",     required=True, type=Path,
                        help="test_targets.npy (shape N x 4002)")
    parser.add_argument("--ids",         required=True, type=Path,
                        help="test_ids.txt (one material_id per line)")
    parser.add_argument("--dataset",     required=True,
                        help="Dataset name, e.g. full_cubic")
    parser.add_argument("--experiment",  required=True, choices=["A", "B", "C", "C_TL", "D"],
                        help="Experiment type (A=random CV, B=Z-site CV, C=full-data scratch, "
                             "C_TL=full-data TL, D=Rh holdout)")
    parser.add_argument("--fold",        required=True, type=int,
                        help="Fold index (0-4)")
    parser.add_argument("--output_dir",  required=True, type=Path,
                        help="Directory to write output files")
    parser.add_argument("--predictions_std", type=Path, default=None,
                        help="Optional: test_predictions_std.npy from ensemble_inference "
                             "(shape N x 4002). If provided, adds D35/D36/D37 uncertainty "
                             "columns to te_ranking.csv.")
    args = parser.parse_args()

    # Validate inputs
    for p in [args.predictions, args.targets, args.ids]:
        if not p.exists():
            print(f"ERROR: File not found: {p}")
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load arrays
    predictions = np.load(args.predictions)
    targets     = np.load(args.targets)
    with open(args.ids) as f:
        material_ids = [line.strip() for line in f if line.strip()]

    N = len(material_ids)
    assert predictions.shape == (N, 4002), \
        f"predictions shape {predictions.shape} != ({N}, 4002)"
    assert targets.shape == (N, 4002), \
        f"targets shape {targets.shape} != ({N}, 4002)"

    # Optional ensemble uncertainty
    std_pred = None
    if args.predictions_std is not None:
        if args.predictions_std.exists():
            std_pred = np.load(args.predictions_std)
            assert std_pred.shape == (N, 4002), \
                f"predictions_std shape {std_pred.shape} != ({N}, 4002)"
            print(f"  Ensemble std loaded: {args.predictions_std}")
        else:
            print(f"  WARNING: predictions_std not found: {args.predictions_std}. Skipping.")

    print(f"Loaded {N} samples from {args.dataset} Exp{args.experiment} fold{args.fold}")

    # Import and run
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compute_metrics import compute_te_descriptors
    import pandas as pd

    result, errors = compute_te_descriptors(
        predictions, targets, material_ids, args.output_dir
    )

    # Group 7: Add ensemble uncertainty columns (D35, D36, D37)
    if std_pred is not None:
        ranking_path = args.output_dir / "te_ranking.csv"
        if ranking_path.exists():
            ranking = pd.read_csv(ranking_path)
        else:
            ranking = result[['material_id', 'pred_D34_TE_score']].copy()

        # Fermi-window std (±2 eV) per spin channel
        std_up   = std_pred[:, 800:1201].mean(axis=1)    # (N,) up channel ±2 eV
        std_down = std_pred[:, 2801:3202].mean(axis=1)   # (N,) down channel ±2 eV

        ranking['D35_fermi_window_std_up']   = std_up
        ranking['D36_fermi_window_std_down'] = std_down
        # D37: True if both channels are in the low-uncertainty 75th-percentile
        q75_up   = np.quantile(std_up,   0.75)
        q75_down = np.quantile(std_down, 0.75)
        ranking['D37_reliability_flag'] = (
            (std_up   < q75_up) & (std_down < q75_down)
        )
        ranking.to_csv(ranking_path, index=False)
        n_reliable = ranking['D37_reliability_flag'].sum()
        print(f"\nUncertainty columns added to te_ranking.csv:")
        print(f"  D35 std_up mean:   {std_up.mean():.4f}")
        print(f"  D36 std_down mean: {std_down.mean():.4f}")
        print(f"  D37 reliable (75th pct): {n_reliable}/{N} materials")

    # ── D38: Valley degeneracy proxy ─────────────────────────────────────────
    # Narrow tall peak near E_F = single degenerate valley = high power factor
    # Window: ±0.5 eV = indices [950:1051] spin-up, [2950:3052] spin-down
    # Formula: peak_height / (half-max width in eV + epsilon)
    # High value → sharper peak → more valley-degenerate → better TE candidate
    # Physical basis: W-point valley degeneracy in half-Heusler TE materials
    #   (Snyder et al., Science Advances 2020; npj Comp. Mat. 2026 Heusler ML)
    import math as _math

    def _valley_proxy(dos_full, window_slice):
        """Peak height / half-max width for a DOS channel in a window."""
        win = dos_full[window_slice]
        peak = float(win.max())
        if peak < 1e-6:
            return 0.0
        half_max = 0.5 * peak
        above = (win > half_max).sum()
        width_eV = above * 0.01  # grid spacing is 0.01 eV
        return peak / (width_eV + 0.01)

    d38_vals = np.array([
        _math.sqrt(
            _valley_proxy(predictions[j], slice(950, 1051)) *
            _valley_proxy(predictions[j], slice(2950, 3052)) + 1e-12
        )
        for j in range(N)
    ], dtype=np.float32)

    # Append D38 to te_ranking.csv (always present after compute_te_descriptors())
    ranking_path_d38 = args.output_dir / "te_ranking.csv"
    ranking_d38 = pd.read_csv(ranking_path_d38)
    ranking_d38['D38_valley_degeneracy_proxy'] = d38_vals
    ranking_d38.to_csv(ranking_path_d38, index=False)
    print(f"\nD38 valley degeneracy proxy added to te_ranking.csv")
    print(f"  mean={d38_vals.mean():.4f}  max={d38_vals.max():.4f}  min={d38_vals.min():.4f}")

    print(f"\nDescriptor summary (top 5 by MAE):")
    sorted_descs = sorted(errors.items(), key=lambda x: -x[1]['mae'])[:5]
    for desc, vals in sorted_descs:
        print(f"  {desc:<35} MAE={vals['mae']:.4f}  Pearson={vals['pearson']:.4f}")

    print(f"\nOutputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
