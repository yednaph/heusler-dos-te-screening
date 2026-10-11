#!/usr/bin/env python3
"""
run_inference.py — Load trained checkpoint and produce test-set predictions.

Called by scripts/train_single_fold.sh immediately after training completes.
Reads the best.pt checkpoint, runs inference on the test split, and writes
numpy arrays that compute_metrics.py consumes.

Outputs (in --output_dir):
  test_predictions.npy   float32, shape (N, 4002)  — predicted DOS
  test_targets.npy       float32, shape (N, 4002)  — ground-truth DOS
  test_ids.txt           one material_id per line

Usage:
    python code/run_inference.py \
        --checkpoint output/full_cubic/expA/fold0/checkpoints/best.pt \
        --config     output/full_cubic/expA/fold0/config_expA_fold0.yaml \
        --test_file  data/full_cubic/data_split/expA_fold0_test.csv \
        --output_dir results/full_cubic/expA/fold0/ \
        --batch_size 256

MODIFY: Replace the DOSDataset and ALIGNN imports with your actual implementations.
"""

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# ALIGNN model — always from installed conda-env package.
from alignn.models.alignn import ALIGNN, ALIGNNConfig

# DOSDataset — custom project module in same code/ directory as this file.
_code_dir = str(Path(__file__).resolve().parent)
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)
try:
    from dos_dataset import DOSDataset
except ImportError as _e:
    raise ImportError(
        f"Cannot import DOSDataset from {_code_dir}/dos_dataset.py\n"
        f"Original error: {_e}"
    ) from _e


def load_model(checkpoint_path: Path, config: dict, device: torch.device):
    """
    Load model from checkpoint.
    MODIFY: Adjust model creation to match your actual ALIGNN variant.
    """
    model_config = config["model"]

    alignn_cfg = ALIGNNConfig(
        name="alignn",
        alignn_layers=model_config["alignn_layers"],
        gcn_layers=model_config["gcn_layers"],
        hidden_features=model_config["hidden_dim"],
        output_features=model_config["output_dim"],
        atom_input_features=model_config.get("atom_input_features", 97),
    )
    model = ALIGNN(alignn_cfg).to(device)

    # Apply deep output MLP head if enabled in config (must match train_hpc.py)
    if model_config.get('deep_output_head', False):
        hidden    = model_config['hidden_dim']
        dropout_p = model_config.get('dropout', 0.2)
        model.fc  = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden * 2, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden * 4, 4002),
        ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle three checkpoint formats:
    #   1. Full dict with 'model_state_dict' key  (save_checkpoint in train_hpc.py)
    #   2. Full dict with 'model_state' key        (legacy format)
    #   3. Raw state_dict                          (best_fermi.pt, swa_model.pt)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint   # raw state_dict saved as plain dict
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    print(f"  Loaded checkpoint: {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    if "metrics" in checkpoint:
        m = checkpoint["metrics"]
        print(f"  Val Pearson at checkpoint: {m.get('pearson', 'N/A'):.4f}")

    return model


def load_test_dataset(config: dict, test_file: Path):
    """
    Load test-split dataset.
    MODIFY: Adjust DOSDataset constructor arguments to match your implementation.
    """
    data_config = config["data"]

    # test_file overrides the config split — intentional for held-out test evaluation.
    # dos_dir comes from the merged per-fold config; cutoff/max_neighbors use
    # config values if present, otherwise match dos_dataset.py defaults (8.0 Å / 12).
    # Structure is read from data['structure'] in each JSON — no cif_dir needed.
    dataset = DOSDataset(
        dos_dir       = data_config["dos_dir"],
        ids_file      = str(test_file),
        cutoff        = data_config.get("cutoff", 8.0),
        max_neighbors = data_config.get("max_neighbors", 12),
    )
    return dataset


@torch.no_grad()
def ensemble_inference(
    checkpoint_paths: list,
    config: dict,
    dataset,
    batch_size: int,
    device: torch.device,
):
    """
    Run inference with an ensemble of fold models; return mean ± std predictions.

    Zero additional training cost — uses the 5 fold checkpoints that already
    exist after cross-validation. Materials with high cross-fold std in their
    D34_TE_score should be flagged for manual DFT validation.

    Args:
        checkpoint_paths: list of paths to fold checkpoint files (e.g. 5 folds)
        config:           merged config dict (model architecture)
        dataset:          DOSDataset for the test/screening set
        batch_size:       inference batch size
        device:           'cuda' or 'cpu'

    Returns:
        mean_pred: np.ndarray (N, 4002) — best estimate (ensemble mean)
        std_pred:  np.ndarray (N, 4002) — per-energy-point uncertainty
        ids:       list of str, length N
    """
    from torch.utils.data import DataLoader

    _nw = 0 if platform.system() == 'Windows' else 4
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_nw,
        pin_memory=False,   # DGL graphs are not pin_memory-safe across DGL versions
        prefetch_factor=2 if _nw > 0 else None,
        collate_fn=DOSDataset.collate,
    )

    all_fold_preds = []
    ids = None

    for fold_idx, ckpt_path in enumerate(checkpoint_paths):
        print(f"  Ensemble fold {fold_idx + 1}/{len(checkpoint_paths)}: {ckpt_path}")
        fold_model = load_model(Path(ckpt_path), config, device)

        fold_preds = []
        fold_ids   = []
        with torch.no_grad():
            for batch in loader:
                g, lg, lat = batch["graph"]
                graphs = (g.to(device), lg.to(device), lat.to(device))
                out_raw = fold_model(graphs)
                out     = torch.nn.functional.softplus(out_raw, beta=1)
                fold_preds.append(out.cpu().numpy().astype(np.float32))
                if ids is None:
                    fold_ids.extend(batch.get("id", []))

        all_fold_preds.append(np.concatenate(fold_preds, axis=0))   # (N, 4002)
        if ids is None:
            ids = fold_ids

    all_fold_preds = np.stack(all_fold_preds, axis=0)   # (n_folds, N, 4002)
    mean_pred = all_fold_preds.mean(axis=0)              # (N, 4002)
    std_pred  = all_fold_preds.std(axis=0)               # (N, 4002)

    return mean_pred, std_pred, ids


@torch.no_grad()
def run_inference(model, dataset, batch_size: int, device: torch.device):
    """
    Run model on entire dataset, collect predictions and targets.

    Returns:
        predictions: np.ndarray shape (N, 4002)
        targets:     np.ndarray shape (N, 4002)
        ids:         list of str, length N
    """
    from torch.utils.data import DataLoader

    _nw = 0 if platform.system() == 'Windows' else 4
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_nw,
        pin_memory=False,   # DGL graphs are not pin_memory-safe across DGL versions
        prefetch_factor=2 if _nw > 0 else None,
        collate_fn=DOSDataset.collate,   # required — DGL graphs need custom collation
    )

    all_preds   = []
    all_targets = []
    all_ids     = []
    n_batches   = len(loader)

    for batch_idx, batch in enumerate(loader):
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{n_batches}")

        # batch['graph'] is a 3-tuple (atom_graph, line_graph, lattice) — cannot .to() a tuple
        g, lg, lat = batch["graph"]
        g          = g.to(device)
        lg         = lg.to(device)
        lat        = lat.to(device)
        graphs     = (g, lg, lat)          # ALIGNN.forward() unpacks: g, lg, lat = g
        dos_target = batch["dos"].to(device)
        batch_ids  = batch.get("id", [f"sample_{i}" for i in range(len(dos_target))])

        dos_pred_raw = model(graphs)
        dos_pred     = torch.nn.functional.softplus(dos_pred_raw, beta=1)

        all_preds.append(dos_pred.cpu().numpy().astype(np.float32))
        all_targets.append(dos_target.cpu().numpy().astype(np.float32))
        all_ids.extend(batch_ids if not isinstance(batch_ids, torch.Tensor)
                       else [str(x) for x in batch_ids.tolist()])

    predictions = np.concatenate(all_preds,   axis=0)   # (N, 4002)
    targets     = np.concatenate(all_targets, axis=0)   # (N, 4002)

    return predictions, targets, all_ids


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with trained ALIGNN checkpoint on test split."
    )
    parser.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to best.pt checkpoint (output of train_hpc.py)."
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to the merged per-fold config YAML (generated by train_single_fold.sh)."
    )
    parser.add_argument(
        "--test_file", required=True, type=Path,
        help="CSV with material_ids for the test split (column: material_id)."
    )
    parser.add_argument(
        "--output_dir", required=True, type=Path,
        help="Directory where test_predictions.npy, test_targets.npy, test_ids.txt are written."
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Inference batch size (default: 256, larger than training is fine)."
    )
    parser.add_argument(
        "--ensemble_checkpoints", nargs="+", type=str, default=None,
        help="Paths to multiple fold checkpoints for ensemble inference "
             "(produces mean + std predictions). If provided, --checkpoint is ignored."
    )
    args = parser.parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)
    if not args.test_file.exists():
        print(f"ERROR: Test file not found: {args.test_file}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 56)
    print("ALIGNN Inference")
    print("=" * 56)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test file:  {args.test_file}")
    print(f"Output dir: {args.output_dir}")

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:     {device}")
    if device.type == "cuda":
        print(f"GPU:        {torch.cuda.get_device_name(0)}")

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("\nLoading test dataset...")
    dataset = load_test_dataset(config, args.test_file)
    print(f"  Test samples: {len(dataset)}")

    # ── Run inference: ensemble or single model ───────────────────────────────
    if args.ensemble_checkpoints:
        print(f"\nRunning ENSEMBLE inference ({len(args.ensemble_checkpoints)} folds)...")
        mean_pred, std_pred, ids = ensemble_inference(
            args.ensemble_checkpoints, config, dataset, args.batch_size, device
        )
        predictions = mean_pred
        print(f"\nEnsemble inference complete:")
        print(f"  mean_pred shape: {predictions.shape}")
        print(f"  std_pred  shape: {std_pred.shape}")
        print(f"  n_samples:       {len(ids)}")

        # Get targets from a single model pass for comparison
        model = load_model(Path(args.ensemble_checkpoints[0]), config, device)
        _, targets, _ = run_inference(model, dataset, args.batch_size, device)

        # Save ensemble std for uncertainty-aware TE ranking
        np.save(args.output_dir / "test_predictions_std.npy", std_pred)
        print(f"  test_predictions_std.npy saved (use in compute_te_descriptors.py)")
    else:
        print("\nLoading model...")
        model = load_model(args.checkpoint, config, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        print(f"\nRunning inference (batch_size={args.batch_size})...")
        predictions, targets, ids = run_inference(model, dataset, args.batch_size, device)
        std_pred = None

        print(f"\nInference complete:")
        print(f"  predictions shape: {predictions.shape}")
        print(f"  targets shape:     {targets.shape}")
        print(f"  n_samples:         {len(ids)}")

    # Quick sanity check
    if predictions.shape != targets.shape:
        print(f"ERROR: shape mismatch — predictions {predictions.shape} vs targets {targets.shape}")
        sys.exit(1)
    if predictions.shape[1] != 4002:
        print(f"WARNING: expected 4002 DOS points (2001×2 spin channels), got {predictions.shape[1]}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    pred_path = args.output_dir / "test_predictions.npy"
    tgt_path  = args.output_dir / "test_targets.npy"
    ids_path  = args.output_dir / "test_ids.txt"

    np.save(pred_path, predictions)
    np.save(tgt_path,  targets)
    with open(ids_path, "w") as f:
        for mid in ids:
            f.write(f"{mid}\n")

    print(f"\nSaved:")
    print(f"  {pred_path}  ({pred_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {tgt_path}   ({tgt_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {ids_path}   ({len(ids)} IDs)")
    print("\nInference DONE. Run compute_metrics.py next.")


if __name__ == "__main__":
    main()
