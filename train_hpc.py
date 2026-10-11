#!/usr/bin/env python3
"""
HPC Training Script for ALIGNN-DOS

This is a TEMPLATE script for distributed training on HPC clusters.
Adapt this to match your local pipeline's data loading and model architecture.
"""

import sys
from pathlib import Path

# ── Resolve project root and inject code/ into sys.path ──────────────────────
# MUST be first: SLURM batch jobs start in /var/spool/slurm/slurmd, NOT the
# project directory. os.getcwd() is unreliable. Path(__file__).resolve() always
# gives the absolute path to this script regardless of working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent
_code_dir = str(_PROJECT_ROOT / "code")
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)
print("DEBUG: sys.path[0] =", sys.path[0])

import argparse
import os
import platform
import time
import yaml
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast  # compatible with all PyTorch versions on this HPC
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.tensorboard import SummaryWriter

# ── DataLoader worker safety (Linux/HPC) ─────────────────────────────────────
# DGL graphs are many small tensors; with num_workers>0 the default
# 'file_descriptor' sharing strategy exhausts the process fd limit and crashes
# mid-run with "RuntimeError: received 0 items of ancdata". The 'file_system'
# strategy avoids it. No-op / harmless when num_workers=0.
import torch.multiprocessing as _torch_mp
try:
    _torch_mp.set_sharing_strategy("file_system")
except Exception:
    pass

# Distributed training kept for reference — disabled for SVNIT Aurora H100 NVL (single GPU per job)
try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False

import numpy as np
from scipy.stats import pearsonr as _scipy_pearsonr
from tqdm import tqdm

# ALIGNN model — always import from installed conda-env package.
# Do NOT wrap in try/except: if this fails, alignn is genuinely not installed.
from alignn.models.alignn import ALIGNN, ALIGNNConfig

# DOSDataset — custom project module at code/dos_dataset.py.
# sys.path already patched above; this import is now safe from any working dir.
try:
    from dos_dataset import DOSDataset
except ImportError as _e:
    raise ImportError(
        f"Cannot import DOSDataset from {_code_dir}/dos_dataset.py\n"
        f"Ensure code/dos_dataset.py exists in the project root.\n"
        f"Original error: {_e}"
    ) from _e


# OLD: EnhancedDOSLoss (MSE+cosine+Fermi-MSE) — replaced by ThermoelectricDOSLoss
# Reason: literature (EqDOS Chem.Mat.2025, Mat2Spec Nat.Commun.2022) shows MSE-only
# losses produce over-smooth predictions that miss sharp d-band peaks critical for
# thermoelectric screening via the Mott formula S ∝ (1/g_d) × dg_d/dE at E_F.


class ThermoelectricDOSLoss(nn.Module):
    """
    Physics-motivated loss function for thermoelectric d-projected DOS prediction.

    SCIENTIFIC BASIS:
    The thermoelectric Seebeck coefficient via Mott formula:
      S = -(pi^2 k_B^2 T / 3e) * d[ln g_d(E)]/dE |_{E=E_F}
        = -(pi^2 k_B^2 T / 3e) * (1/g_d(E_F)) * dg_d/dE |_{E=E_F}

    This requires the model to correctly predict:
      1. The d-band shape globally (BTE-weighted MSE + cosine)
      2. The distribution of spectral weight near E_F (Wasserstein)
      3. The slope dg_d/dE at E_F (finite-difference term)
      4. The Fermi-level region specifically (exact ±2 eV window MSE)

    LOSS COMPONENTS:
      L_total = w_mse*L_mse_bte + w_wass*L_wasserstein + w_cosine*L_cosine
              + w_slope*L_slope + w_fermi*L_fermi_window

    DEFAULT WEIGHTS (physics-motivated, sum to 1.0):
      w_mse=0.40   BTE-weighted MSE — global magnitude with transport emphasis
      w_wass=0.25  Sliced Wasserstein — peak positions and sharpness
      w_cosine=0.15 Cosine similarity — overall spectral shape
      w_slope=0.10  Finite-difference slope — Mott formula slope at E_F
      w_fermi=0.10  Fermi-window ±2eV MSE — near-E_F accuracy

    ENERGY CONSTANTS (confirmed from 51,257-file diagnostic):
      FERMI_IDX = 1000  (COMMON_GRID[1000] = 0.0 eV exactly)
      Fermi window ±2 eV: up=slice(800,1201), down=slice(2801,3202) (exact)

    LITERATURE:
      - Wasserstein DOS loss: EqDOS (Chem.Mat. 2025), Mat2Spec (Nat.Commun. 2022)
      - BTE thermal window: Boltzmann transport theory, Fermi-Dirac derivative
      - Slope loss: Mott formula, Fung et al. (Chem.Mat. 2022)
      - Heusler TE screening: standard d-band Mott formula application
    """

    def __init__(self, config: dict):
        super().__init__()

        # Physics constants
        k_B  = 8.617333e-5          # eV/K
        T_op = config.get('training', {}).get('screening_temperature', 700)  # K

        # BTE thermal window weight vector (Fermi-Dirac derivative)
        # Weights each energy point by its contribution to Boltzmann transport.
        # Peak at E_F, width ~ 3 k_B T; exact result from BTE theory.
        ENERGIES = torch.linspace(-10.0, 10.0, 2001)
        beta     = 1.0 / (k_B * T_op)
        fermi_w  = (beta / 4.0) / (torch.cosh(beta * ENERGIES / 2.0))**2
        fermi_w  = fermi_w / fermi_w.sum() * 2001   # normalise to ~1 mean

        # Full 4002-dim weight (same window for both spin channels)
        weight_4002 = torch.cat([fermi_w, fermi_w])    # (4002,)
        self.register_buffer('bte_weight', weight_4002)

        # Fermi window slices (±2 eV, EXACT — from COMMON_GRID = linspace(-10,10,2001))
        # Index of E=-2.0 eV: (-2.0 - (-10.0)) / 0.01 = 800
        # Index of E=+2.0 eV: (+2.0 - (-10.0)) / 0.01 = 1200, slice end=1201
        # Down channel offset: +2001 (spin-down starts at index 2001)
        self.fermi_win_up   = slice(800,  1201)   # spin-up  ±2 eV
        self.fermi_win_down = slice(2801, 3202)   # spin-down ±2 eV

        # Slope loss window: ±3 eV around E_F (indices 700:1301)
        # Wide enough to capture d-band peak region near E_F
        self.slope_win = slice(700, 1301)

        # Loss weights (from config, with physics-motivated defaults)
        # Updated 2026-03-27: added peak_height (5A) and tv_smooth (5B) components
        loss_cfg = config.get('loss', {})
        self.w_mse    = loss_cfg.get('w_mse',        0.35)  # was 0.40 — reduced for new components
        self.w_wass   = loss_cfg.get('w_wass',        0.19)  # was 0.25 — reduced for tv_smooth
        self.w_cosine = loss_cfg.get('w_cosine',      0.13)  # was 0.15 — reduced slightly
        self.w_slope  = loss_cfg.get('w_slope',       0.10)
        self.w_fermi  = loss_cfg.get('w_fermi',       0.10)
        self.w_peak   = loss_cfg.get('w_peak',        0.10)  # NEW 5A: asymmetric peak-height
        self.w_tv     = loss_cfg.get('w_tv',          0.03)  # NEW 5B: TV smoothness
        # Weights sum: 0.35+0.19+0.13+0.10+0.10+0.10+0.03 = 1.00

        # 5A: asymmetric penalty — underestimation of E_F peak costs gamma× more
        self.peak_gamma = loss_cfg.get('peak_height_gamma', 2.5)

    def _peak_height_loss(
        self, pred: torch.Tensor, true: torch.Tensor
    ) -> torch.Tensor:
        """
        Asymmetric penalty on Fermi-level peak height (±1 eV window).

        Underestimation (pred < true) penalised peak_gamma× more than
        overestimation. Directly counters MSE regression-to-mean bias that
        systematically produces peaks 30–50% shorter than DFT.

        Physics: d-band peak height near E_F determines Mott Seebeck proxy.
                 Mahan-Sofo optimal TE DOS is a sharp peak at E_F.
        """
        # ±1 eV window: up channel indices 900:1101, down 2900:3102
        total = torch.tensor(0.0, device=pred.device)
        for win in [slice(900, 1101), slice(2900, 3102)]:
            p = pred[:, win]
            t = true[:, win]
            residual = t - p   # positive = underestimation
            weight = torch.where(
                residual > 0,
                torch.full_like(residual, self.peak_gamma),
                torch.ones_like(residual),
            )
            total += (weight * residual ** 2).mean()
        return total / 2.0   # average over two spin windows

    def _tv_smoothness_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Total variation (L1 first-difference) smoothness penalty.

        Penalises unphysical high-frequency oscillations in the predicted DOS
        across the full energy range, excluding the ±2 eV Fermi window where
        genuine sharp peaks are physically expected.

        L1 (not L2) is sparse: penalises many small wiggles more than one
        sharp peak — appropriate for DOS with rare but real sharp features.
        """
        total = torch.tensor(0.0, device=pred.device)
        for ch_slice in [slice(0, 2001), slice(2001, 4002)]:
            dos  = pred[:, ch_slice]                         # (batch, 2001)
            diff = (dos[:, 1:] - dos[:, :-1]).abs()         # (batch, 2000)
            # Zero out diffs entirely within the Fermi window ±2 eV.
            # DOS window: indices 800–1200 (slice 800:1201, 401 points).
            # diff[k] = |DOS[k+1] - DOS[k]| — both endpoints inside window
            # when k in [800, 1199], i.e. slice 800:1200 (400 diffs).
            diff_masked = diff.clone()
            diff_masked[:, 800:1200] = 0.0
            total += diff_masked.mean()
        return total / 2.0

    def forward(self, pred: torch.Tensor,
                true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (batch, 4002) predicted d-projected DOS (after softplus)
            true: (batch, 4002) DFT d-projected DOS target
        Returns:
            scalar loss
        """
        # ── 1. BTE-weighted MSE ─────────────────────────────────────────────
        # Weights each energy point by its Boltzmann transport contribution.
        # Automatically emphasises E_F region without explicit window cutoff.
        w = self.bte_weight.unsqueeze(0)          # (1, 4002)
        L_mse = (w * (pred - true)**2).mean()

        # ── 2. Sliced Wasserstein Distance ──────────────────────────────────
        # Measures distance between spectral distributions.
        # Better than MSE alone for peak position and sharpness (EqDOS, 2025).
        # 1D Wasserstein = L1 between sorted values (exact, differentiable).
        pred_sorted = torch.sort(pred, dim=1)[0]
        true_sorted = torch.sort(true, dim=1)[0]
        L_wass = torch.mean(torch.abs(pred_sorted - true_sorted))

        # ── 3. Cosine Similarity Loss ───────────────────────────────────────
        # Scale-invariant shape matching — captures relative peak heights.
        # eps=1e-6: prevents NaN backward when pred/true are near-zero in bf16.
        # Default eps=1e-8 underflows in bf16 (min normal ~1.2e-38 but gradients
        # of norm computation lose precision) → NaN grads → GradScaler skips step.
        cos_sim  = torch.nn.functional.cosine_similarity(pred, true, dim=1, eps=1e-6)
        L_cosine = (1.0 - cos_sim).mean()

        # ── 4. Finite-Difference Slope Loss (Mott formula term) ─────────────
        # Directly penalises errors in dg_d/dE near E_F.
        # The logarithmic slope at E_F determines the Seebeck coefficient.
        # Applied in ±3 eV window (indices 700:1301) where transport occurs.
        pred_win = pred[:, self.slope_win]
        true_win = true[:, self.slope_win]

        d_pred   = torch.diff(pred_win, n=1, dim=1)   # 1st derivative (batch, 599)
        d_true   = torch.diff(true_win, n=1, dim=1)
        L_slope1 = torch.mean((d_pred - d_true)**2)

        d2_pred  = torch.diff(pred_win, n=2, dim=1)   # 2nd derivative (batch, 598)
        d2_true  = torch.diff(true_win, n=2, dim=1)
        L_slope2 = torch.mean((d2_pred - d2_true)**2)

        L_slope  = L_slope1 + 0.1 * L_slope2

        # ── 5. Fermi-Window MSE (exact ±2 eV) ──────────────────────────────
        # Direct penalty on near-E_F accuracy for both spin channels.
        # Up channel ±2 eV:   indices 800:1201 (E = -2.0 to +2.0 eV)
        # Down channel ±2 eV: indices 2801:3202 (offset by N_ENERGY_POINTS=2001)
        L_fermi = 0.5 * (
            torch.nn.functional.mse_loss(
                pred[:, self.fermi_win_up],
                true[:, self.fermi_win_up]
            ) +
            torch.nn.functional.mse_loss(
                pred[:, self.fermi_win_down],
                true[:, self.fermi_win_down]
            )
        )

        # ── 6. Peak-Height Penalty (Mahan-Sofo, asymmetric) ─────────────────
        # Penalises underestimation of d-band peaks near E_F (±1 eV).
        # gamma=2.5 makes underestimation 2.5× more costly than overestimation.
        L_peak = self._peak_height_loss(pred, true)

        # ── 7. Total Variation Smoothness (L1 first-difference) ─────────────
        # Suppresses unphysical high-frequency oscillations outside Fermi window.
        L_tv = self._tv_smoothness_loss(pred)

        # ── Total Loss (weights sum to 1.00) ─────────────────────────────────
        total = (self.w_mse    * L_mse    +
                 self.w_wass   * L_wass   +
                 self.w_cosine * L_cosine +
                 self.w_slope  * L_slope  +
                 self.w_fermi  * L_fermi  +
                 self.w_peak   * L_peak   +
                 self.w_tv     * L_tv)

        return total


def setup_distributed() -> tuple:
    """
    Single-GPU setup for SVNIT Aurora H100 NVL.
    One GPU per SLURM job — distributed training disabled.

    Returns:
        rank=0, world_size=1, local_rank=0
    """
    rank = 0
    world_size = 1
    local_rank = 0
    torch.cuda.set_device(0)
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training (no-op in single-GPU mode)."""
    if _DIST_AVAILABLE and dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Create ALIGNN model from configuration.

    MODIFY: Replace with your actual model creation logic.
    """
    model_config = config['model']

    # ALIGNNConfig takes kwargs matching its field names (pydantic BaseSettings).
    # ALIGNN.__init__ takes a single ALIGNNConfig object, NOT unpacked kwargs.
    alignn_cfg = ALIGNNConfig(
        name="alignn",
        alignn_layers=model_config['alignn_layers'],
        gcn_layers=model_config['gcn_layers'],
        hidden_features=model_config['hidden_dim'],
        output_features=model_config['output_dim'],
        atom_input_features=model_config.get('atom_input_features', 97),
        # 97 = 92 base CGCNN (one-hot over 92 elements) + 5 Heusler-specific features
        # appended in _build_graph() (d-electron count, VEC, d-filling, Hund's moment, d-row)
    )
    assert alignn_cfg.output_features == 4002, (
        f"output_features must be 4002, got {alignn_cfg.output_features}. "
        f"Check config.yaml model.output_dim."
    )
    assert alignn_cfg.atom_input_features == 97, (
        f"atom_input_features must be 97, got {alignn_cfg.atom_input_features}. "
        f"Check config.yaml model.atom_input_features and dos_dataset.ATOM_INPUT_FEATURES."
    )
    model = ALIGNN(alignn_cfg)

    # Replace single Linear output head with deep MLP for better DOS prediction.
    # Single Linear(hidden_dim→4002) cannot learn non-linear sharp Fermi features.
    # Deep MLP: hidden_dim → 2×hidden → 4×hidden → 4002
    hidden = model_config['hidden_dim']
    dropout_p = model_config.get('dropout', 0.2)
    if model_config.get('deep_output_head', True):
        model.fc = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden * 2, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden * 4, 4002),
        )
        print(f"[MODEL] Deep output MLP: {hidden}→{hidden*2}→{hidden*4}→4002")

    return model.to(device)


def create_dataloaders(
    config: Dict[str, Any],
    rank: int,
    world_size: int
) -> tuple:
    """
    Create training and validation dataloaders.

    MODIFY: Replace with your actual dataset loading logic.
    """
    data_config = config['data']
    training_config = config['training']

    # Config keys confirmed in merged per-fold YAML written by train_single_fold.sh:
    #   dos_dir    — path to DOS JSON directory (flat: {material_id}.json files directly inside)
    #               Structure is read from data['structure'] inside each JSON — no cif_dir needed.
    #   train_file — path to train IDs CSV (column: material_id)
    #   val_file   — path to val IDs CSV
    _common = dict(
        dos_dir       = data_config['dos_dir'],
        cutoff        = data_config.get('cutoff', 8.0),
        max_neighbors = data_config.get('max_neighbors', 12),
    )
    train_dataset = DOSDataset(ids_file=data_config['train_file'], **_common)
    val_dataset   = DOSDataset(ids_file=data_config['val_file'],   **_common)

    # On Windows, DGL graph multiprocessing fails with num_workers > 0.
    # On Linux (SVNIT Aurora) use the configured value (up to 16).
    _nw = 0 if platform.system() == 'Windows' else data_config.get('num_workers', 8)
    _loader_common = dict(
        batch_size         = training_config['batch_size'],
        num_workers        = _nw,
        pin_memory         = False,   # DGL graphs are not pin_memory-safe across DGL versions
        prefetch_factor    = data_config.get('prefetch_factor', 2) if _nw > 0 else None,
        persistent_workers = (_nw > 0),
        collate_fn         = DOSDataset.collate,  # required — DGL graphs need custom collation
    )
    train_loader = DataLoader(train_dataset, shuffle=True,  **_loader_common)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **_loader_common)

    return train_loader, val_loader


def create_optimizer(model: nn.Module, config: Dict[str, Any]):
    """Create optimizer from configuration."""
    opt_config = config['optimizer']
    training_config = config['training']

    opt_type = opt_config.get('type', 'AdamW')
    lr = training_config['learning_rate']
    weight_decay = opt_config.get('weight_decay', 0.0)

    if opt_type == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=opt_config.get('betas', (0.9, 0.999))
        )
    elif opt_type == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    return optimizer


def create_scheduler(optimizer, config: Dict[str, Any]):
    """
    Create LR scheduler based on config.

    Supports:
      - 'ReduceLROnPlateau': halves LR on val_loss plateau (best for small datasets)
      - 'CosineAnnealingWarmRestarts': cosine with linear warmup (default)
    """
    from torch.optim.lr_scheduler import (
        LambdaLR, CosineAnnealingWarmRestarts, SequentialLR, ReduceLROnPlateau
    )

    sched_config = config.get('scheduler', {})
    sched_type   = sched_config.get('type', 'CosineAnnealingWarmRestarts')
    warmup_epochs = sched_config.get('warmup_epochs', 10)

    if sched_type == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode    = sched_config.get('mode',    'min'),
            factor  = sched_config.get('factor',  0.5),
            patience= sched_config.get('patience', 15),
            min_lr  = sched_config.get('min_lr',  1e-7),
        )
        return scheduler

    # Default: linear warmup then CosineAnnealingWarmRestarts
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: max(epoch / max(warmup_epochs, 1), 1e-2)
    )

    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=sched_config.get('T_0', 50),
        T_mult=sched_config.get('T_mult', 1),
        eta_min=sched_config.get('eta_min', 1e-6),
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    return scheduler


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Returns:
        Dictionary with RMSE, MAE, and Pearson correlation
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    # RMSE
    rmse = np.sqrt(np.mean((pred_np - target_np) ** 2))

    # MAE
    mae = np.mean(np.abs(pred_np - target_np))

    # Pearson correlation — guard against zero-variance inputs (early training, constant pred)
    pred_flat   = pred_np.flatten()
    target_flat = target_np.flatten()
    if pred_flat.std() > 1e-8 and target_flat.std() > 1e-8:
        pearson = float(np.corrcoef(pred_flat, target_flat)[0, 1])
        if np.isnan(pearson):
            pearson = 0.0
    else:
        pearson = 0.0

    return {
        'rmse': float(rmse),
        'mae':  float(mae),
        'pearson': pearson,
    }


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer,
    criterion,
    scaler: Optional[GradScaler],
    device: torch.device,
    epoch: int,
    rank: int,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    num_batches = 0
    nan_count = 0   # NaN/Inf loss counter — abort after 5 consecutive bad batches

    # Determine autocast dtype from config (default fp16; prefer bf16 on H100).
    # config.yaml sets precision: 'bf16' — H100 NVL has native bf16 units that are
    # faster and more numerically stable than fp16 (bf16 range = fp32 range, no
    # overflow risk during SWA phase when loss landscape changes at epoch 240+).
    # autocast() without dtype= silently defaults to fp16 regardless of config.
    _precision      = config['training'].get('precision', 'fp16')
    _autocast_dtype = torch.bfloat16 if _precision == 'bf16' else torch.float16

    # Progress bar only on rank 0
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader


    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue   # entire batch skipped (disordered structures) — see collate()
        # batch['graph'] is a 3-tuple (atom_graph, line_graph, lattice) — cannot .to() a tuple
        g, lg, lat = batch['graph']
        g          = g.to(device)
        lg         = lg.to(device)
        lat        = lat.to(device)
        graphs     = (g, lg, lat)          # ALIGNN.forward() unpacks: g, lg, lat = g
        dos_target = batch['dos'].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast(dtype=_autocast_dtype):
                output_raw = model(graphs)
                dos_pred   = torch.nn.functional.softplus(output_raw, beta=1)
                loss = criterion(dos_pred, dos_target)

            # NaN/Inf detection — silent SLURM failures cost entire job allocation
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(f"[WARNING] NaN/Inf loss at epoch {epoch}, batch {batch_idx}. "
                      f"Skipping batch, halving LR (occurrence {nan_count}/5).")
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['lr'] * 0.5
                optimizer.zero_grad()
                if nan_count > 5:
                    raise RuntimeError(
                        f"[CRITICAL] {nan_count} NaN/Inf batches in epoch {epoch}. "
                        "Aborting training. Check data normalization or reduce base LR."
                    )
                continue

            scaler.scale(loss).backward()

            if config['training'].get('grad_clip', 0.0) > 0:
                scaler.unscale_(optimizer)

                # ── Gradient diagnostic: epoch 0, first 3 batches ─────────────
                # Detects GradScaler-silent NaN/Inf gradients (happen AFTER
                # unscaling — the loss itself can be finite while gradients
                # overflow in bf16 backward; scaler.step() silently skips).
                if rank == 0 and epoch == 0 and batch_idx < 3:
                    _nan_p = sum(
                        1 for p in model.parameters()
                        if p.grad is not None and not torch.isfinite(p.grad).all()
                    )
                    _tot_p = sum(1 for p in model.parameters() if p.grad is not None)
                    _gnorm = sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in model.parameters()
                        if p.grad is not None and torch.isfinite(p.grad).all()
                    ) ** 0.5
                    print(
                        f"[GRAD] e0b{batch_idx}: loss={loss.item():.4f}  "
                        f"nan_inf_params={_nan_p}/{_tot_p}  "
                        f"finite_grad_norm={_gnorm:.4e}  "
                        f"scaler_scale={scaler.get_scale():.2e}",
                        flush=True,
                    )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['grad_clip']
                )

            _pre_step_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Detect GradScaler step-skip: scale shrinks ↔ inf/nan in gradients
            if rank == 0 and epoch < 3 and scaler.get_scale() < _pre_step_scale:
                print(
                    f"[SCALER_SKIP] e{epoch}b{batch_idx}: optimizer.step SKIPPED  "
                    f"scale {_pre_step_scale:.0f}→{scaler.get_scale():.0f}",
                    flush=True,
                )
        else:
            # bf16 path: autocast WITHOUT GradScaler (bf16 has fp32 dynamic range).
            # scaler=None is set at startup when precision='bf16' to prevent
            # GradScaler silently skipping optimizer.step() on NaN gradients.
            _use_amp = config['training'].get('mixed_precision', False)
            if _use_amp:
                with autocast(dtype=_autocast_dtype):
                    output_raw = model(graphs)
                    dos_pred   = torch.nn.functional.softplus(output_raw, beta=1)
                    loss = criterion(dos_pred, dos_target)
            else:
                output_raw = model(graphs)
                dos_pred   = torch.nn.functional.softplus(output_raw, beta=1)
                loss = criterion(dos_pred, dos_target)

            # NaN/Inf detection
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(f"[WARNING] NaN/Inf loss at epoch {epoch}, batch {batch_idx}. "
                      f"Skipping batch, halving LR (occurrence {nan_count}/5).")
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['lr'] * 0.5
                optimizer.zero_grad()
                if nan_count > 5:
                    raise RuntimeError(
                        f"[CRITICAL] {nan_count} NaN/Inf batches in epoch {epoch}. "
                        "Aborting training. Check data normalization or reduce base LR."
                    )
                continue

            loss.backward()

            if config['training'].get('grad_clip', 0.0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['grad_clip']
                )

            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Print peak GPU memory for first 3 batches of epoch 0 — diagnose OOM kills
        if rank == 0 and epoch == 0 and batch_idx < 3 and torch.cuda.is_available():
            _peak = torch.cuda.max_memory_allocated(0) / 1e9
            _curr = torch.cuda.memory_allocated(0) / 1e9
            print(f"[VRAM] epoch0 batch{batch_idx}: peak={_peak:.2f}GB  current={_curr:.2f}GB")

        if rank == 0 and isinstance(pbar, tqdm):
            pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    return {'train_loss': avg_loss}


def compute_fermi_pearson(all_preds: torch.Tensor, all_targets: torch.Tensor) -> float:
    """
    Pearson r restricted to the ±2 eV Fermi window only.
    Indices: up channel 800:1201, down channel 2801:3202.
    Per-sample Pearson (flat over both windows), then averaged.
    scipy.stats.pearsonr imported at module level to avoid repeated NFS load cost.
    """
    p = all_preds.float().numpy()
    t = all_targets.float().numpy()
    p_fermi = np.concatenate([p[:, 800:1201], p[:, 2801:3202]], axis=1)
    t_fermi = np.concatenate([t[:, 800:1201], t[:, 2801:3202]], axis=1)
    r_vals = []
    for i in range(len(p_fermi)):
        if np.std(t_fermi[i]) > 1e-8 and np.std(p_fermi[i]) > 1e-8:
            r, _ = _scipy_pearsonr(p_fermi[i], t_fermi[i])
            if not np.isnan(r):
                r_vals.append(float(r))
    return float(np.mean(r_vals)) if r_vals else 0.0


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion,
    device: torch.device,
    rank: int
) -> Dict[str, float]:
    """Validate the model.

    torch.no_grad() is mandatory here:
      - model.eval() only disables dropout/BN training behaviour; it does NOT
        stop PyTorch from building computation graphs.
      - Without no_grad(), every val batch accumulates a full forward-pass graph.
        dos_pred.cpu() keeps all those graphs alive simultaneously, adding
        10–30 GB of GPU memory pressure for ~80 batches at batch_size=128.
        The test run used batch_size=8 (16× smaller) and masked this bug.
      - With no_grad(), no graph is built, GPU memory is ~2–3× lower, and
        validation runs 2–3× faster across all 300 epochs.
    """
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue   # entire batch skipped — see collate()
            g, lg, lat = batch['graph']
            g          = g.to(device)
            lg         = lg.to(device)
            lat        = lat.to(device)
            graphs     = (g, lg, lat)
            dos_target = batch['dos'].to(device)

            dos_pred_raw = model(graphs)
            dos_pred     = torch.nn.functional.softplus(dos_pred_raw, beta=1)
            loss = criterion(dos_pred, dos_target)

            total_loss += loss.item()
            num_batches += 1

            # .detach() is belt-and-suspenders inside no_grad() but makes the
            # intent explicit: never retain the computation graph in all_preds.
            all_preds.append(dos_pred.detach().cpu())
            all_targets.append(dos_target.detach().cpu())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Guard against all-None validation set (disordered-structure edge case)
    if not all_preds:
        if rank == 0:
            print("[VALIDATE] WARNING: all validation batches returned None — zero metrics returned")
        return {'val_loss': avg_loss, 'rmse': 0.0, 'mae': 0.0,
                'pearson': 0.0, 'fermi_pearson': 0.0}

    # Compute metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_preds, all_targets)

    metrics['val_loss'] = avg_loss
    # Group 1.2: Fermi-window Pearson (±2 eV) — used for best_fermi.pt checkpoint
    metrics['fermi_pearson'] = compute_fermi_pearson(all_preds, all_targets)

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    output_dir: Path,
    is_best: bool = False,
    swa_model=None,
    swa_scheduler=None,
):
    """Save training checkpoint.

    swa_model / swa_scheduler: pass during SWA epochs (240+) so that --requeue
    restarts can restore the averaged weights and LR schedule rather than
    restarting SWA from scratch (which would create an infinite 3-hour loop
    that never fits within the 2-hour SLURM time limit).
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics,
        'config': config,
    }
    if swa_model is not None:
        checkpoint['swa_model_state_dict'] = swa_model.state_dict()
    if swa_scheduler is not None:
        checkpoint['swa_scheduler_state_dict'] = swa_scheduler.state_dict()

    # Save last checkpoint
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, checkpoint_dir / 'last.pt')

    # Write epoch to last_epoch.txt — a 4-byte plain-text file read by the SLURM
    # SIGTERM handler in train_single_fold.sh.  Reading this avoids loading the full
    # checkpoint (300–400 MB + PyTorch import ~15 s) inside the 30–60 s KillWait window.
    (checkpoint_dir / 'last_epoch.txt').write_text(str(epoch))

    # Save best checkpoint
    if is_best:
        torch.save(checkpoint, checkpoint_dir / 'best.pt')

    # Save periodic epoch checkpoint (interval from config, default 50)
    save_period = config.get('training', {}).get('save_every_n_epochs', 50)
    if save_period > 0 and epoch % save_period == 0:
        torch.save(checkpoint, checkpoint_dir / f'epoch_{epoch}.pt')


def variance_aware_output_init(
    model: nn.Module,
    train_loader: DataLoader,
    n_samples: int = 500,
    device: str = 'cuda',
):
    """
    Re-scale output layer weights by sqrt(per-energy-point target variance).

    Computed from the first n_samples training examples. Focuses early training
    on the high-variance Fermi-level region rather than the low-variance deep-band
    region (Binned Spectral Power Loss, 2025 preprint). Does NOT change architecture.

    Args:
        model:        ALIGNN model (before training starts)
        train_loader: training DataLoader
        n_samples:    number of samples to estimate target variance
        device:       'cuda' or 'cpu'
    """
    model.eval()
    collected = []
    with torch.no_grad():
        for batch in train_loader:
            if batch is None:
                continue   # entire batch skipped — see collate()
            targets = batch['dos'].to(device)       # (batch, 4002)
            collected.append(targets.float().cpu())
            if sum(t.shape[0] for t in collected) >= n_samples:
                break

    if not collected:
        print("[INIT] No batches collected — skipping variance-aware init.")
        model.train()
        return

    all_targets = torch.cat(collected, dim=0)[:n_samples]   # (n_samples, 4002)
    target_var  = all_targets.var(dim=0)                    # (4002,) per-energy variance

    scale = target_var.sqrt()
    scale = (scale / scale.mean().clamp(min=1e-8)).to(device)

    # Find the output linear layer (out_features == 4002)
    output_layer = None
    output_name  = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == 4002:
            output_layer = module
            output_name  = name
            break

    if output_layer is None:
        print("[INIT] Could not find output layer with out_features=4002. "
              "Skipping variance-aware initialization.")
        model.train()
        return

    with torch.no_grad():
        # weight shape: (4002, hidden_dim) — scale each row by variance weight
        output_layer.weight.data *= scale.unsqueeze(1)
        if output_layer.bias is not None:
            output_layer.bias.data *= scale

    print(f"[INIT] Variance-aware output init applied to '{output_name}'. "
          f"Scale range: [{scale.min().item():.3f}, {scale.max().item():.3f}]")
    model.train()


def lock_all_seeds(seed: int = 42) -> None:
    """
    Lock all random seeds for full reproducibility across PyTorch, NumPy, Python,
    DGL (if available), and CUDA. Must be called before model creation, data
    loading, or any random operation.
    """
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import dgl
        dgl.seed(seed)
    except ImportError:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[SEED] All random seeds locked to {seed}")


def build_run_metadata(cfg: dict, args) -> dict:
    """
    Build a reproducibility metadata dict for this training run.
    Saved into every checkpoint and metrics_summary.json for paper Methods section.
    """
    import subprocess, datetime, platform
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = 'unknown'

    return {
        'git_commit':            git_hash,
        'timestamp_utc':         datetime.datetime.utcnow().isoformat(),
        'torch_version':         torch.__version__,
        'cuda_version':          getattr(torch.version, 'cuda', 'N/A'),
        'python_version':        platform.python_version(),
        'dataset':               getattr(args, 'dataset',    'unknown'),
        'experiment':            getattr(args, 'experiment', 'unknown'),
        'fold':                  getattr(args, 'fold',       -1),
        'epochs':                cfg['training']['num_epochs'],
        'batch_size':            cfg['training']['batch_size'],
        'atom_input_features':   cfg['model']['atom_input_features'],
        'output_features':       cfg['model'].get('output_dim', cfg['model'].get('output_features', 4002)),
        'dropout':               cfg['model']['dropout'],
        'lr':                    cfg['training']['learning_rate'],
        'weight_decay':          cfg['optimizer']['weight_decay'],
        'pretrain_checkpoint':   getattr(args, 'pretrain_checkpoint', None),
        'finetune_lr_scale':     getattr(args, 'finetune_lr_scale',   1.0),
    }


def main():
    parser = argparse.ArgumentParser(description='Train ALIGNN-DOS on HPC')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--epochs', '--max_epochs', dest='max_epochs', type=int, default=None,
                        help='Override config training.num_epochs (use --epochs 3 for HPC diagnostic runs)')
    # Experiment / run identity (used for metadata + output path tagging)
    parser.add_argument('--dataset', type=str, default='unknown',
                        help='Dataset name, e.g. full_cubic (informational; does not change paths)')
    parser.add_argument('--experiment', type=str, default='A',
                        choices=['A', 'B', 'C', 'C_TL', 'D'],
                        help='Experiment type: A=random CV, B=Z-site CV, '
                             'C=full-data scratch, C_TL=full-data TL, D=Rh holdout')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold index (0–4; 0 for single-model experiments C/D)')
    # Group 6: transfer learning arguments
    parser.add_argument('--pretrain_checkpoint', type=str, default=None,
                        help='Path to pretrained model checkpoint for cross-family transfer')
    parser.add_argument('--finetune_lr_scale', type=float, default=1.0,
                        help='LR multiplier for fine-tuning (use 0.1 for transfer learning)')
    args = parser.parse_args()

    # Lock all seeds first — before any model/data operation
    lock_all_seeds(seed=42)

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()

    # GPU setup — soft fallback to CPU if CUDA not available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if rank == 0:
            print(f"GPU:    {torch.cuda.get_device_name(0)}")
            print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"Mode:   Single-GPU (SVNIT Aurora H100 NVL)")
    else:
        device = torch.device("cpu")
        if rank == 0:
            print("WARNING: CUDA not available, falling back to CPU")
            print("On SVNIT Aurora: check 'module load anaconda3-2024.2' and conda activate alignn_env_svnit")

    if rank == 0:
        print(f"Output: {args.output_dir}")

    # Load configuration
    config = load_config(args.config)

    # Apply CLI epoch override (used for diagnostic runs: --epochs 3)
    if args.max_epochs is not None:
        config['training']['num_epochs'] = args.max_epochs
        if rank == 0:
            print(f"[OVERRIDE] num_epochs overridden to {args.max_epochs} via --epochs flag")

    # Build reproducibility metadata (after config load, before model creation)
    run_metadata = build_run_metadata(config, args)
    if rank == 0:
        print(f"[META] git={run_metadata['git_commit']}  "
              f"dataset={run_metadata['dataset']}  "
              f"exp={run_metadata['experiment']}  fold={run_metadata['fold']}")

    # Create output directory
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'logs').mkdir(exist_ok=True)
        import json as _json
        with open(output_dir / 'run_metadata.json', 'w') as _f:
            _json.dump(run_metadata, _f, indent=2)

    # Create model
    model = create_model(config, device)

    # DDP wrapping disabled — single GPU mode for SVNIT Aurora H100 NVL
    # To re-enable multi-GPU: restore setup_distributed() and uncomment below
    # if world_size > 1:
    #     model = DDP(model, device_ids=[local_rank],
    #                 gradient_as_bucket_view=config.get('distributed', {}).get('gradient_as_bucket_view', True))

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, rank, world_size)

    if train_loader is None:
        print("ERROR: Could not create dataloaders. Check dataset paths and implementation.")
        cleanup_distributed()
        sys.exit(1)

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)

    # Group 6: Transfer learning — load pretrained weights and scale LR
    if args.pretrain_checkpoint and os.path.exists(args.pretrain_checkpoint):
        if rank == 0:
            print(f"[TRANSFER] Loading pretrained weights from {args.pretrain_checkpoint}")
        state = torch.load(args.pretrain_checkpoint, map_location=device)
        # Handle checkpoints saved as full dict vs raw state_dict
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        missing, unexpected = model.load_state_dict(state, strict=False)
        if rank == 0:
            print(f"[TRANSFER] Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
        if args.finetune_lr_scale != 1.0:
            # Two-group optimizer: encoder at reduced LR, output head at full LR.
            # model.fc (Linear 768→4002, ~3.1M params) must adapt from cubic DOS
            # patterns to tetragonal DOS patterns — uniform 0.1× LR under-adapts it.
            base_lr = config['training']['learning_rate']
            wd      = config['optimizer'].get('weight_decay', 0.0)
            betas   = config['optimizer'].get('betas', (0.9, 0.999))
            head_ids      = {id(p) for p in model.fc.parameters()}
            encoder_params = [p for p in model.parameters() if id(p) not in head_ids]
            head_params    = [p for p in model.parameters() if id(p) in head_ids]
            optimizer = torch.optim.AdamW([
                {'params': encoder_params,
                 'lr': base_lr * args.finetune_lr_scale,
                 'weight_decay': wd * args.finetune_lr_scale},
                {'params': head_params,
                 'lr': base_lr,
                 'weight_decay': wd},
            ], betas=betas)
            if rank == 0:
                n_enc = sum(p.numel() for p in encoder_params)
                n_hd  = sum(p.numel() for p in head_params)
                print(f"[TRANSFER] Two-group optimizer rebuilt: "
                      f"encoder {n_enc:,} params @ lr={base_lr * args.finetune_lr_scale:.2e}  "
                      f"output_head {n_hd:,} params @ lr={base_lr:.2e}")

    scheduler = create_scheduler(optimizer, config)

    # Create loss function
    criterion = ThermoelectricDOSLoss(config).to(device)

    # Mixed precision scaler
    # bf16 must NOT use GradScaler: bf16 has fp32 dynamic range so gradient
    # overflow is impossible. GradScaler with bf16 silently skips optimizer.step()
    # whenever cosine_similarity backward produces near-zero denominators,
    # causing loss to never decrease (confirmed: fold2 stuck at 4.688 for 121 epochs).
    scaler = None
    _mp_precision = config['training'].get('precision', 'fp16')
    if config['training'].get('mixed_precision', False) and _mp_precision != 'bf16':
        scaler = GradScaler()
        if rank == 0:
            print(f"[AMP] fp16 mode: GradScaler enabled")
    elif config['training'].get('mixed_precision', False):
        if rank == 0:
            print(f"[AMP] bf16 mode: autocast without GradScaler (bf16 has fp32 range)")

    # Group 3: Variance-aware output layer initialization.
    # Skip when using a pretrain checkpoint: the transferred weights already encode the
    # correct output scale. Multiplying them by a dataset-variance factor would corrupt
    # the pre-trained output head and undo the benefit of transfer learning.
    if rank == 0 and not args.pretrain_checkpoint:
        variance_aware_output_init(model, train_loader, n_samples=500, device=str(device))

    # Group 4: Stochastic Weight Averaging setup
    num_epochs   = config['training']['num_epochs']
    swa_frac     = config['training'].get('swa_start_fraction', 0.80)
    swa_start    = int(num_epochs * swa_frac)   # begin averaging at swa_start_fraction of training
    swa_model    = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-5, anneal_epochs=10)
    if rank == 0:
        print(f"[SWA] Weight averaging starts at epoch {swa_start} / {num_epochs}")

    # TensorBoard writer (rank 0 only)
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=output_dir / 'logs' / 'tensorboard')

    # Resume from checkpoint
    start_epoch = 0
    best_pearson       = -1.0
    best_val_fermi_pearson = -1.0   # Group 1.2: secondary checkpoint on Fermi-window Pearson

    # Auto-resume from last.pt if --resume not specified
    if args.resume is None:
        last_ckpt = output_dir / 'checkpoints' / 'last.pt'
        if last_ckpt.exists():
            args.resume = str(last_ckpt)
            if rank == 0:
                print(f"[RESUME] Found checkpoint: {args.resume}")

    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)

        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_pearson = checkpoint['metrics'].get('pearson', -1.0)
        best_val_fermi_pearson = checkpoint['metrics'].get('fermi_pearson', -1.0)  # FIX Bug-A: restore on resume

        # Restore SWA state when re-entering the SWA phase after --requeue.
        # Without this, swa_model = AveragedModel(model) starts fresh every restart,
        # losing all averaging work and creating a >2h infinite loop past epoch 240.
        if start_epoch >= swa_start:
            if 'swa_model_state_dict' in checkpoint:
                swa_model.load_state_dict(checkpoint['swa_model_state_dict'])
                if rank == 0:
                    print(f"[RESUME] SWA model state restored at epoch {start_epoch}")
            if 'swa_scheduler_state_dict' in checkpoint:
                swa_scheduler.load_state_dict(checkpoint['swa_scheduler_state_dict'])
                if rank == 0:
                    print(f"[RESUME] SWA scheduler state restored")

    # ── GPU memory snapshot before first epoch ───────────────────────────────
    # Printed once to log — tells us how much VRAM is consumed by model +
    # optimizer + dataset indices before any batch is loaded.
    # If this value is already > (shard_allocation - 4GB), reduce batch_size or
    # increase --gres=shard.
    if rank == 0 and torch.cuda.is_available():
        _alloc  = torch.cuda.memory_allocated(0)  / 1e9
        _reserv = torch.cuda.memory_reserved(0)   / 1e9
        _total  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[VRAM] Pre-train: allocated={_alloc:.2f}GB  "
              f"reserved={_reserv:.2f}GB  total={_total:.2f}GB")
        print(f"[VRAM] Available budget for activations: "
              f"{_total - _reserv:.2f}GB")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Set epoch for distributed sampler
        if world_size > 1 and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, epoch, rank, config
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, rank)

        # Group 4: SWA — update averaged model or step normal scheduler
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['val_loss'])
                else:
                    scheduler.step()

        # Log metrics
        if rank == 0:
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss:      {train_metrics['train_loss']:.6f}")
            print(f"  Val Loss:        {val_metrics['val_loss']:.6f}")
            print(f"  Val RMSE:        {val_metrics['rmse']:.6f}")
            print(f"  Val Pearson:     {val_metrics['pearson']:.6f}")
            print(f"  Val Fermi-r:     {val_metrics.get('fermi_pearson', 0.0):.6f}")
            print(f"  LR:              {optimizer.param_groups[0]['lr']:.2e}")
            if scaler is not None:
                print(f"  GradScaler scale:{scaler.get_scale():.2e}")
            if epoch >= swa_start:
                print(f"  [SWA active since epoch {swa_start}]")
            if torch.cuda.is_available():
                _res = torch.cuda.memory_reserved(0) / 1e9
                _peak = torch.cuda.max_memory_allocated(0) / 1e9
                print(f"  [VRAM] reserved={_res:.2f}GB  peak_alloc={_peak:.2f}GB")

            if writer:
                writer.add_scalar('Loss/train',        train_metrics['train_loss'],              epoch)
                writer.add_scalar('Loss/val',          val_metrics['val_loss'],                  epoch)
                writer.add_scalar('Metrics/RMSE',      val_metrics['rmse'],                      epoch)
                writer.add_scalar('Pearson/val_global', val_metrics['pearson'],                   epoch)
                writer.add_scalar('Pearson/val_fermi',  val_metrics.get('fermi_pearson', 0.0),   epoch)
                writer.add_scalar('LR/current',        optimizer.param_groups[0]['lr'],          epoch)
                writer.add_scalar('SWA/active',        int(epoch >= swa_start),                  epoch)

            # Save checkpoint every epoch — required for --requeue resume correctness.
            # Normal training (epoch < swa_start): update best.pt, best_fermi.pt, last.pt
            # SWA phase (epoch >= swa_start): save last.pt WITH swa_model + swa_scheduler
            #   state so that a requeue during the SWA phase restores averaged weights
            #   instead of restarting SWA from scratch (a 3-hour loop > 2h time limit).
            if epoch < swa_start:
                is_best = val_metrics['pearson'] > best_pearson
                if is_best:
                    best_pearson = val_metrics['pearson']
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    val_metrics, config, output_dir, is_best
                )

                # Group 1.2: Secondary checkpoint on Fermi-window Pearson (best_fermi.pt)
                val_fermi_r = val_metrics.get('fermi_pearson', 0.0)
                if val_fermi_r > best_val_fermi_pearson:
                    best_val_fermi_pearson = val_fermi_r
                    fermi_ckpt_path = output_dir / 'checkpoints' / 'best_fermi.pt'
                    raw = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    torch.save(raw, fermi_ckpt_path)
                    print(f"[CKPT] New best Fermi Pearson: {val_fermi_r:.4f} → saved to best_fermi.pt")
            else:
                # SWA phase: save last.pt with SWA model + scheduler state every epoch
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    val_metrics, config, output_dir, is_best=False,
                    swa_model=swa_model, swa_scheduler=swa_scheduler,
                )

    # Group 4: SWA finalisation — update BN stats and save SWA model
    # NOTE: torch.optim.swa_utils.update_bn() cannot be used here because it
    # calls batch.to(device) internally, expecting a plain tensor batch.
    # Our DataLoader returns dicts — that call raises AttributeError: 'dict'
    # object has no attribute 'to'. Custom loop below handles our format.
    if rank == 0:
        print("\n[SWA] Updating BatchNorm statistics for SWA model...")
    _bn_modules = [m for m in swa_model.modules()
                   if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not _bn_modules:
        if rank == 0:
            print("[SWA] No BatchNorm layers detected — BN update skipped "
                  "(expected: ALIGNN uses LayerNorm, not BatchNorm)")
    else:
        if rank == 0:
            print(f"[SWA] Found {len(_bn_modules)} BatchNorm layer(s), updating...")
        for _m in _bn_modules:
            _m.running_mean.zero_()
            _m.running_var.fill_(1)
        _was_training = swa_model.training
        swa_model.train()
        with torch.no_grad():
            for _batch in train_loader:
                if _batch is None:
                    continue
                _g, _lg, _lat = _batch['graph']
                _g   = _g.to(device)
                _lg  = _lg.to(device)
                _lat = _lat.to(device)
                swa_model((_g, _lg, _lat))
        if not _was_training:
            swa_model.eval()

    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save SWA model (primary deployment model)
    swa_state = swa_model.module.state_dict() if hasattr(swa_model, 'module') else swa_model.state_dict()
    torch.save(swa_state, checkpoint_dir / 'swa_model.pt')

    # Save raw model as well for comparison
    raw_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(raw_state, checkpoint_dir / 'best_model.pt')

    if rank == 0:
        print(f"[SWA] swa_model.pt saved to {checkpoint_dir}")

    # Cleanup
    if rank == 0 and writer:
        writer.close()

    cleanup_distributed()

    if rank == 0:
        print("\nTraining completed!")
        print(f"Best validation Pearson (pre-SWA): {best_pearson:.6f}")


if __name__ == '__main__':
    main()
