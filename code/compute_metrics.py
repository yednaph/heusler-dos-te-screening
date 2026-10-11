#!/usr/bin/env python3
"""
compute_metrics.py — Publication-grade metric computation for ALIGNN DOS predictions.

Called once per fold after training completes.

Usage:
    python code/compute_metrics.py \\
        --predictions results/full_cubic/expA/fold0/test_predictions.npy \\
        --targets     results/full_cubic/expA/fold0/test_targets.npy \\
        --ids         results/full_cubic/expA/fold0/test_ids.txt \\
        --dataset     full_cubic \\
        --experiment  A \\
        --fold        0 \\
        --output_dir  results/full_cubic/expA/fold0/
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
# scikit-learn is OPTIONAL here.  It is used only by the half-metal
# classification metrics (M13/M14), and both of those functions re-import it
# locally, so this module-level import is redundant.  Guarding it lets the TE
# descriptor path (compute_te_descriptors) run in environments without sklearn
# installed; the M13/M14 functions still raise normally if actually called.
try:
    from sklearn.metrics import precision_score, recall_score, f1_score
except ModuleNotFoundError:
    precision_score = recall_score = f1_score = None

# =============================================================================
# ENERGY AXIS -- do not modify these constants
# =============================================================================
# All metrics below use this exact Fermi-aligned common grid.
# Changing these values would make metrics incomparable
# between training runs and across datasets.
#
# ENERGIES[FERMI_IDX] = 0.0 eV exactly (E = E_F)
# All 51,257 materials confirmed to have 100% coverage at +/-10 eV
# after Fermi alignment (diagnostic March 2026).
# =============================================================================

# Fermi-aligned common grid -- exact constants from diagnostic
# ALL 51,257 materials confirmed to have 100% coverage at +/-10 eV
# TARGET: d-projected spin-resolved DOS (spd_dos['d']['densities']['1'/'-1'])
ENERGIES     = np.linspace(-10.0, 10.0, 2001)   # shape (2001,)
GRID_SPACING = 0.01          # eV per point (exact)
FERMI_IDX    = 1000          # index of E=0.0 eV (exact)
N_POINTS     = 2001          # energy points per spin channel
TARGET_DIM   = 4002          # N_POINTS * 2 spin channels

# BTE thermal window at 700K Heusler TE operating temperature
k_B      = 8.617333e-5      # eV/K
T_op     = 700              # K
k_BT_700 = k_B * T_op       # = 0.0603 eV

# Fermi window index arrays (derived from ENERGIES, self-documenting)
W1EV    = np.where(np.abs(ENERGIES) <= 1.0)[0]          # ±1 eV: indices 900:1101, 201 pts
W2EV    = np.where(np.abs(ENERGIES) <= 2.0)[0]          # ±2 eV: indices 800:1201, 401 pts
W3EV    = np.where(np.abs(ENERGIES) <= 3.0)[0]          # ±3 eV: indices 700:1301, 601 pts
W5EV    = np.where(np.abs(ENERGIES) <= 5.0)[0]          # ±5 eV: indices 500:1501, 1001 pts
kBT_WIN = np.where(np.abs(ENERGIES) <= k_BT_700)[0]     # thermal window ~±0.06 eV

# Down-channel offset in the 4002-dim concatenated target
DOWN_OFFSET = N_POINTS   # = 2001

# Validate constants are self-consistent
assert len(ENERGIES) == 2001
assert abs(ENERGIES[FERMI_IDX]) < 1e-10, \
    f"FERMI_IDX={FERMI_IDX} does not correspond to E=0"
assert abs(ENERGIES[W1EV[0]]  - (-1.0)) < 0.001
assert abs(ENERGIES[W1EV[-1]] - (+1.0)) < 0.001
assert abs(ENERGIES[W2EV[0]]  - (-2.0)) < 0.001
assert abs(ENERGIES[W2EV[-1]] - (+2.0)) < 0.001

# M14 band-gap classification thresholds
GAP_THRESHOLDS = [0.01, 0.05, 0.10]
GAP_THRESHOLD_PRIMARY = 0.05   # primary threshold for electronic type classification

# Spin polarization threshold for half-metallic classification
HALF_METALLIC_POL_THRESH = 0.9

EPS = 1e-10

# BTE thermal weight at 700 K — matches ThermoelectricDOSLoss exactly
# w(E) = (β/4)/cosh²(βE/2), normalised to sum=1 over 2001 points
_beta_700       = 1.0 / (k_B * T_op)                                  # 16.583 eV⁻¹
_BTE_WEIGHT     = (_beta_700 / 4.0) / np.cosh(_beta_700 * ENERGIES / 2.0)**2
_BTE_WEIGHT     = _BTE_WEIGHT / _BTE_WEIGHT.sum()                     # normalised, shape (2001,)
_BTE_WEIGHT_4002 = np.concatenate([_BTE_WEIGHT, _BTE_WEIGHT])         # both spin channels, shape (4002,)
_BTE_WEIGHT_4002 = _BTE_WEIGHT_4002 / _BTE_WEIGHT_4002.sum()          # FIX Bug-46: renorm to sum=1 (each _BTE_WEIGHT sums to 1 → concat sums to 2 without this)

# Minimum integrated DOS to compute d-band center reliably
# (guards against near-zero channels in half-metals and semiconductors)
MIN_DOS_INTEGRAL = 0.1  # states/atom·eV


# ── Helper: aggregate per-sample array ────────────────────────────────────────

def agg(arr: np.ndarray, label: str, independence: str) -> dict:
    """Compute summary statistics for a 1-D per-sample metric array."""
    valid = arr[~np.isnan(arr)]
    return {
        "mean":          float(np.mean(valid)),
        "std":           float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
        "median":        float(np.median(valid)),
        "p5":            float(np.percentile(valid, 5)),
        "p95":           float(np.percentile(valid, 95)),
        "n_valid":       int(len(valid)),
        "n_nan":         int(np.isnan(arr).sum()),
        "independence":  independence,
    }


# ── M1: Pearson r global [PARTIALLY DEPENDENT — MSE in loss] ──────────────────

def compute_m1_pearson(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    N = len(predictions)
    per_sample = np.zeros(N)
    for i in range(N):
        r, _ = scipy.stats.pearsonr(predictions[i], targets[i])
        per_sample[i] = r
    return per_sample


# ── M2: RMSE global [PARTIALLY DEPENDENT — MSE in loss] ───────────────────────

def compute_m2_rmse(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((predictions - targets) ** 2, axis=1))


# ── M3: MAE global [FULLY INDEPENDENT ✓] ──────────────────────────────────────

def compute_m3_mae(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(predictions - targets), axis=1)


# ── M4: R² global [FULLY INDEPENDENT ✓] ───────────────────────────────────────

def compute_m4_r2(predictions: np.ndarray, targets: np.ndarray) -> float:
    flat_pred = predictions.flatten()
    flat_true = targets.flatten()
    ss_res = np.sum((flat_pred - flat_true) ** 2)
    ss_tot = np.sum((flat_true - flat_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > EPS else float("nan")


# ── M5: Jensen-Shannon Divergence [FULLY INDEPENDENT ✓] ───────────────────────

def _jsd_pair(dos_p: np.ndarray, dos_t: np.ndarray) -> float:
    """JSD for one spin channel (in bits). Returns value in [0, 1]."""
    # Shift to non-negative and normalize
    p = dos_p - dos_p.min() + EPS
    q = dos_t - dos_t.min() + EPS
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    # scipy.special.kl_div(x, y) = x*log(x/y) - x + y (element-wise)
    # For normalized distributions: sum = KL divergence in nats
    kl_pm = scipy.special.kl_div(p, m).sum()
    kl_qm = scipy.special.kl_div(q, m).sum()
    jsd_nats = 0.5 * (kl_pm + kl_qm)
    return float(jsd_nats / np.log(2))   # convert to bits; range [0, 1]


def compute_m5_jsd(y_pred_up: np.ndarray, y_pred_down: np.ndarray,
                   y_true_up: np.ndarray, y_true_down: np.ndarray) -> np.ndarray:
    N = len(y_pred_up)
    per_sample = np.zeros(N)
    for i in range(N):
        jsd_up   = _jsd_pair(y_pred_up[i],   y_true_up[i])
        jsd_down = _jsd_pair(y_pred_down[i], y_true_down[i])
        per_sample[i] = np.mean([jsd_up, jsd_down])
    return per_sample


# ── M6: Cosine similarity [FULLY DEPENDENT — in loss, supplementary only] ─────

def compute_m6_cosine(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    num  = np.sum(predictions * targets, axis=1)
    denom = (np.linalg.norm(predictions, axis=1)
             * np.linalg.norm(targets, axis=1) + EPS)
    return num / denom


# ── M7: Fermi-window MAE ±1 eV [FULLY INDEPENDENT ✓] ─────────────────────────

def compute_m7_fermi_mae_1eV(y_pred_up: np.ndarray, y_pred_down: np.ndarray,
                              y_true_up: np.ndarray, y_true_down: np.ndarray) -> np.ndarray:
    N = len(y_pred_up)
    per_sample = np.zeros(N)
    for i in range(N):
        err_up   = np.mean(np.abs(y_pred_up[i,   W1EV] - y_true_up[i,   W1EV]))
        err_down = np.mean(np.abs(y_pred_down[i, W1EV] - y_true_down[i, W1EV]))
        per_sample[i] = np.mean([err_up, err_down])
    return per_sample


# ── M8: Fermi-window MAE ±2 eV [FULLY DEPENDENT — in loss, supplementary] ─────

def compute_m8_fermi_mae_2eV(y_pred_up: np.ndarray, y_pred_down: np.ndarray,
                              y_true_up: np.ndarray, y_true_down: np.ndarray) -> np.ndarray:
    N = len(y_pred_up)
    per_sample = np.zeros(N)
    for i in range(N):
        err_up   = np.mean(np.abs(y_pred_up[i,   W2EV] - y_true_up[i,   W2EV]))
        err_down = np.mean(np.abs(y_pred_down[i, W2EV] - y_true_down[i, W2EV]))
        per_sample[i] = np.mean([err_up, err_down])
    return per_sample


# ── M9: Fermi-window Pearson ±2 eV [FULLY INDEPENDENT ✓] ─────────────────────

def compute_m9_fermi_pearson(y_pred_up: np.ndarray, y_pred_down: np.ndarray,
                              y_true_up: np.ndarray, y_true_down: np.ndarray) -> np.ndarray:
    N = len(y_pred_up)
    per_sample = np.zeros(N)
    for i in range(N):
        pred_w = np.concatenate([y_pred_up[i, W2EV], y_pred_down[i, W2EV]])
        true_w = np.concatenate([y_true_up[i, W2EV], y_true_down[i, W2EV]])
        r, _ = scipy.stats.pearsonr(pred_w, true_w)
        per_sample[i] = r
    return per_sample


# ── M10: D-band center MAE [FULLY INDEPENDENT ✓] ──────────────────────────────

def _d_band_center(dos: np.ndarray) -> float:
    """First moment of DOS. Returns nan if integrated DOS < MIN_DOS_INTEGRAL.

    Guard threshold = 0.1 states/atom·eV — much higher than EPS to exclude
    near-zero gapped channels (half-metal minority spin) from giving numerically
    valid but physically meaningless d-band center values.
    """
    denom = np.trapz(dos, ENERGIES)
    if abs(denom) < MIN_DOS_INTEGRAL:
        return float("nan")
    return float(np.trapz(dos * ENERGIES, ENERGIES) / denom)


def compute_m10_dband(y_pred_up: np.ndarray, y_pred_down: np.ndarray,
                      y_true_up: np.ndarray, y_true_down: np.ndarray
                      ) -> tuple:
    """Returns (per_sample_error, dc_pred_up, dc_pred_down, dc_true_up, dc_true_down)."""
    N = len(y_pred_up)
    per_sample  = np.full(N, np.nan)
    dc_pred_up  = np.full(N, np.nan)
    dc_pred_dn  = np.full(N, np.nan)
    dc_true_up  = np.full(N, np.nan)
    dc_true_dn  = np.full(N, np.nan)

    for i in range(N):
        dcp_u = _d_band_center(y_pred_up[i])
        dcp_d = _d_band_center(y_pred_down[i])
        dct_u = _d_band_center(y_true_up[i])
        dct_d = _d_band_center(y_true_down[i])

        dc_pred_up[i] = dcp_u
        dc_pred_dn[i] = dcp_d
        dc_true_up[i] = dct_u
        dc_true_dn[i] = dct_d

        errors = []
        for p, t in [(dcp_u, dct_u), (dcp_d, dct_d)]:
            if not (np.isnan(p) or np.isnan(t)):
                errors.append(abs(p - t))
        per_sample[i] = float(np.mean(errors)) if errors else np.nan

    return per_sample, dc_pred_up, dc_pred_dn, dc_true_up, dc_true_dn


# ── M11: DOS at Fermi level [FULLY INDEPENDENT ✓] ─────────────────────────────

def compute_m11_nef(y_pred_up: np.ndarray, y_pred_down: np.ndarray,
                    y_true_up: np.ndarray, y_true_down: np.ndarray) -> np.ndarray:
    err_up   = np.abs(y_pred_up[:,   FERMI_IDX] - y_true_up[:,   FERMI_IDX])
    err_down = np.abs(y_pred_down[:, FERMI_IDX] - y_true_down[:, FERMI_IDX])
    return np.mean(np.stack([err_up, err_down], axis=1), axis=1)


# ── M12: Integrated DOS error [FULLY INDEPENDENT ✓] ──────────────────────────

def compute_m12_integrated(y_pred_up: np.ndarray, y_pred_down: np.ndarray,
                            y_true_up: np.ndarray, y_true_down: np.ndarray) -> np.ndarray:
    """Per-sample mean absolute integrated-DOS error across spin channels."""
    N = len(y_pred_up)
    per_sample = np.zeros(N)
    for i in range(N):
        err_up   = abs(np.trapz(y_pred_up[i],   ENERGIES) - np.trapz(y_true_up[i],   ENERGIES))
        err_down = abs(np.trapz(y_pred_down[i], ENERGIES) - np.trapz(y_true_down[i], ENERGIES))
        per_sample[i] = np.mean([err_up, err_down])
    return per_sample


def compute_m12_bias(y_pred_up: np.ndarray, y_pred_down: np.ndarray,
                     y_true_up: np.ndarray, y_true_down: np.ndarray) -> dict:
    """Charge-conservation diagnostic: signed bias and per-channel MAE.

    int(d_DOS, dE) = total d-electron count (states/atom).
    Positive bias → model over-predicts d-electron count.
    Negative bias → model under-predicts d-electron count.
    Systematic bias in either channel indicates a normalisation problem.
    """
    int_up_pred   = y_pred_up.sum(axis=1)  * GRID_SPACING   # (N,)
    int_up_true   = y_true_up.sum(axis=1)  * GRID_SPACING
    int_down_pred = y_pred_down.sum(axis=1) * GRID_SPACING
    int_down_true = y_true_down.sum(axis=1) * GRID_SPACING

    err_up   = int_up_pred   - int_up_true    # signed: positive = overestimate
    err_down = int_down_pred - int_down_true
    err_total = (int_up_pred + int_down_pred) - (int_up_true + int_down_true)

    return {
        "mae_up":    float(np.abs(err_up).mean()),
        "mae_down":  float(np.abs(err_down).mean()),
        "mae_total": float(np.abs(err_total).mean()),
        "bias_up":   float(err_up.mean()),     # signed; non-zero → systematic over/under
        "bias_down": float(err_down.mean()),
        "bias_total":float(err_total.mean()),
        "std_up":    float(err_up.std()),
        "std_down":  float(err_down.std()),
        "note": "bias > 0 = model over-predicts d-electron count; check per-atom normalisation",
    }


# ── M13: Spin polarization error at E_F [FULLY INDEPENDENT ✓] ────────────────

def compute_m13_spinpol(y_pred_up: np.ndarray, y_pred_down: np.ndarray,
                        y_true_up: np.ndarray, y_true_down: np.ndarray) -> tuple:
    """Returns (per_sample_error, P_pred, P_true).

    Uses ±5-point window (±0.05 eV) around FERMI_IDX for numerical stability.
    More robust than single-point for noisy predictions near E_F.
    Compounds where mean total DOS < SPINPOL_THRESHOLD are assigned NaN
    in per_sample_error but retain a P value computed from the window mean.
    """
    SPINPOL_THRESHOLD = 0.01    # states/eV/atom; below → semiconductor
    WINDOW = 5                  # ±5 points = ±0.05 eV window
    sl = slice(FERMI_IDX - WINDOW, FERMI_IDX + WINDOW + 1)

    up_p   = y_pred_up[:,   sl].mean(axis=1)
    down_p = y_pred_down[:, sl].mean(axis=1)
    up_t   = y_true_up[:,   sl].mean(axis=1)
    down_t = y_true_down[:, sl].mean(axis=1)

    P_pred = (up_p - down_p) / (up_p + down_p + EPS)
    P_true = (up_t - down_t) / (up_t + down_t + EPS)

    # NaN out error for near-zero DOS compounds (semiconductors)
    metallic_mask = (up_t + down_t) >= SPINPOL_THRESHOLD
    per_sample    = np.where(metallic_mask, np.abs(P_pred - P_true), np.nan)

    return per_sample, P_pred, P_true


def compute_m13_halfmetal_recall(P_pred: np.ndarray, P_true: np.ndarray,
                                  threshold: float = HALF_METALLIC_POL_THRESH) -> dict:
    """Half-metallic compound recall: fraction of true half-metals (|P_true|>0.9)
    correctly identified as half-metallic by the model (|P_pred|>0.9).

    Requires P_pred and P_true from compute_m13_spinpol.
    """
    valid = ~(np.isnan(P_pred) | np.isnan(P_true))
    pt    = P_true[valid]
    pp    = P_pred[valid]
    hm_true = np.abs(pt) > threshold
    hm_pred = np.abs(pp) > threshold
    n_hm = int(hm_true.sum())
    if n_hm == 0:
        return {"n_half_metallic": 0, "recall": float("nan"),
                "precision": float("nan"), "f1": float("nan")}
    prec = float(np.logical_and(hm_pred, hm_true).sum() / (hm_pred.sum() + EPS))
    rec  = float(np.logical_and(hm_pred, hm_true).sum() / n_hm)
    f1   = 2 * prec * rec / (prec + rec + EPS)
    return {"n_half_metallic": n_hm, "recall": rec, "precision": prec, "f1": f1}


# ── M14: Band-gap proxy [FULLY INDEPENDENT ✓] ─────────────────────────────────

def compute_m14_bandgap(y_pred_up: np.ndarray, y_pred_down: np.ndarray,
                        y_true_up: np.ndarray, y_true_down: np.ndarray,
                        threshold: float = GAP_THRESHOLD_PRIMARY) -> dict:
    total_true = y_true_up + y_true_down
    total_pred = y_pred_up + y_pred_down

    true_gapped = total_true[:, W1EV].min(axis=1) < threshold
    pred_gapped = total_pred[:, W1EV].min(axis=1) < threshold
    N = len(true_gapped)

    TP = int(((pred_gapped) & (true_gapped)).sum())
    TN = int(((~pred_gapped) & (~true_gapped)).sum())
    FP = int(((pred_gapped) & (~true_gapped)).sum())
    FN = int(((~pred_gapped) & (true_gapped)).sum())

    accuracy  = (TP + TN) / N
    precision = TP / (TP + FP + EPS)
    recall    = TP / (TP + FN + EPS)
    f1        = 2 * precision * recall / (precision + recall + EPS)

    return {
        "threshold":   threshold,
        "accuracy":    float(accuracy),
        "precision":   float(precision),
        "recall":      float(recall),
        "f1":          float(f1),
        "n_gapped":    int(true_gapped.sum()),
        "n_metallic":  int((~true_gapped).sum()),
        "confusion_matrix": {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
        "true_gapped_arr": true_gapped,   # internal only, not serialized
    }


def compute_m14_threshold_sensitivity(y_pred_up, y_pred_down, y_true_up, y_true_down) -> dict:
    sensitivity = {}
    for thresh in GAP_THRESHOLDS:
        r = compute_m14_bandgap(y_pred_up, y_pred_down, y_true_up, y_true_down, thresh)
        sensitivity[str(thresh)] = {
            k: v for k, v in r.items() if k != "true_gapped_arr"
        }
    return sensitivity


# ── M15: Exchange splitting error [FULLY INDEPENDENT ✓] ──────────────────────

def compute_m15_exchange(y_pred_up: np.ndarray, y_pred_down: np.ndarray,
                         y_true_up: np.ndarray, y_true_down: np.ndarray) -> tuple:
    """Returns (per_sample_error, exchange_pred, exchange_true)."""
    N = len(y_pred_up)
    per_sample    = np.full(N, np.nan)
    exchange_pred = np.full(N, np.nan)
    exchange_true = np.full(N, np.nan)

    for i in range(N):
        dcp_u = _d_band_center(y_pred_up[i])
        dcp_d = _d_band_center(y_pred_down[i])
        dct_u = _d_band_center(y_true_up[i])
        dct_d = _d_band_center(y_true_down[i])

        if any(np.isnan([dcp_u, dcp_d, dct_u, dct_d])):
            continue

        exchange_pred[i] = dcp_u - dcp_d
        exchange_true[i] = dct_u - dct_d
        per_sample[i]    = abs(exchange_pred[i] - exchange_true[i])

    return per_sample, exchange_pred, exchange_true


# ── M_WASS_EVAL: True 1D Wasserstein distance (evaluation, not loss) ──────────

def compute_m_wasserstein_eval(y_pred_up: np.ndarray, y_pred_down: np.ndarray,
                                y_true_up: np.ndarray, y_true_down: np.ndarray
                                ) -> np.ndarray:
    """True 1D Wasserstein (Earth Mover's) distance between predicted and true DOS.

    Treats each DOS curve as a probability distribution (normalised to sum=1).
    Measures spectral mass displacement in eV — interpretable, distribution-level metric.

    Distinct from the loss's sliced Wasserstein (sorted-array L1): this uses
    scipy.stats.wasserstein_distance which computes the exact 1D EMD.

    Returns per-sample mean Wasserstein distance (eV) averaged over both channels.
    Typical well-trained values: 0.1–0.3 eV (Mat2Spec, Nat. Commun. 2022).

    Reference: Mat2Spec showed Wasserstein is more discriminating than MAE
    for spectral property prediction because it penalises peak *position* errors.
    """
    N = len(y_pred_up)
    per_sample = np.zeros(N)
    for i in range(N):
        wvals = []
        for p_ch, t_ch in [(y_pred_up[i], y_true_up[i]),
                           (y_pred_down[i], y_true_down[i])]:
            p = p_ch.clip(0)
            t = t_ch.clip(0)
            p_sum = p.sum()
            t_sum = t.sum()
            if p_sum < 1e-8 or t_sum < 1e-8:
                wvals.append(0.0)
                continue
            p_norm = p / p_sum
            t_norm = t / t_sum
            wvals.append(scipy.stats.wasserstein_distance(ENERGIES, ENERGIES,
                                                          p_norm, t_norm))
        per_sample[i] = float(np.mean(wvals))
    return per_sample


# ── M_PEAK_POS: Peak position error in occupied states ────────────────────────

def compute_m_peak_pos(y_pred_up: np.ndarray, y_pred_down: np.ndarray,
                       y_true_up: np.ndarray, y_true_down: np.ndarray
                       ) -> tuple:
    """Peak position error (eV): |argmax(pred_occ) - argmax(true_occ)|.

    Restricted to occupied states (E < 0, indices 0:1000) to measure how
    accurately the model locates the dominant orbital hybridisation feature.

    Different from d-band center (first moment): peak position is the mode,
    not the mean. More directly related to photoemission spectroscopy features
    and the Mott formula denominator (orbital hybridisation energy).

    Returns per-sample arrays (up, down) of absolute peak position errors (eV).
    NaN if either channel has max DOS < 1e-6 (no detectable peak).
    """
    occ = slice(0, FERMI_IDX)   # E = -10 to 0 eV (occupied states only)
    e_occ = ENERGIES[occ]
    N = len(y_pred_up)
    errs_up   = np.full(N, np.nan)
    errs_down = np.full(N, np.nan)

    for i in range(N):
        p_up = y_pred_up[i, occ]
        t_up = y_true_up[i, occ]
        if p_up.max() > 1e-6 and t_up.max() > 1e-6:
            errs_up[i] = abs(e_occ[p_up.argmax()] - e_occ[t_up.argmax()])

        p_dn = y_pred_down[i, occ]
        t_dn = y_true_down[i, occ]
        if p_dn.max() > 1e-6 and t_dn.max() > 1e-6:
            errs_down[i] = abs(e_occ[p_dn.argmax()] - e_occ[t_dn.argmax()])

    return errs_up, errs_down


# ── M_BTE: BTE-weighted evaluation metrics ─────────────────────────────────────

def compute_m_bte_weighted(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """BTE-weighted MAE, RMSE, and Pearson r at 700 K.

    Uses the same thermal weight as ThermoelectricDOSLoss:
    w(E) = (β/4)/cosh²(βE/2), normalised to sum=1.

    This directly evaluates the model on the criterion it was trained to
    optimise — the transport-relevant energy window at 700 K.

    M_bte_pearson_r uses the ±0.5 eV effective BTE window (indices 950:1051
    for spin-up, 2950:3052 for spin-down), which captures ~95% of the BTE weight.

    Note: M_bte_weighted_mae/rmse are scalars (vectorised over all N×4002 values).
    """
    w = _BTE_WEIGHT_4002                    # shape (4002,)
    residuals = predictions - targets       # (N, 4002)
    abs_res   = np.abs(residuals)

    bte_mae  = float(np.mean(abs_res  @ w))
    bte_rmse = float(np.sqrt(np.mean((residuals**2) @ w)))

    # Pearson r in the ±0.5 eV BTE-effective window (~95% of BTE weight)
    bte_up_p = predictions[:, 950:1051].reshape(-1)
    bte_up_t = targets[:,    950:1051].reshape(-1)
    bte_dn_p = predictions[:, 2950:3052].reshape(-1)
    bte_dn_t = targets[:,    2950:3052].reshape(-1)
    r_bte, _ = scipy.stats.pearsonr(
        np.concatenate([bte_up_p, bte_dn_p]),
        np.concatenate([bte_up_t, bte_dn_t])
    )

    return {
        "bte_weighted_mae":  bte_mae,
        "bte_weighted_rmse": bte_rmse,
        "bte_pearson_r":     float(r_bte),
        "bte_window_eV":     "±0.5 eV (indices 950:1051 up, 2950:3052 down)",
        "independence": "partially_dependent",
        "note": "BTE weight matches training loss — evaluates exactly what was optimised",
    }


# ── Publication-standard private interfaces (underscore prefix) ───────────────
# These provide the canonical entry points expected by integration tests and
# external callers. They delegate to the vectorised implementations above.

def _dband_center_safe(pred: np.ndarray, true: np.ndarray) -> dict:
    """M10: D-band center MAE per spin channel, with MIN_DOS_INTEGRAL guard.

    Returns per-channel and mean MAE in eV. Near-zero channels (half-metal
    gapped spin) are excluded and counted in M10_n_valid_* fields.
    """
    e = ENERGIES
    errs_up, errs_down = [], []
    for i in range(len(pred)):
        for errs, p_ch, t_ch in [
            (errs_up,   pred[i, :2001], true[i, :2001]),
            (errs_down, pred[i, 2001:], true[i, 2001:]),
        ]:
            int_p = np.trapz(p_ch, e)
            int_t = np.trapz(t_ch, e)
            if int_p > MIN_DOS_INTEGRAL and int_t > MIN_DOS_INTEGRAL:
                c_p = np.trapz(p_ch * e, e) / int_p
                c_t = np.trapz(t_ch * e, e) / int_t
                errs.append(abs(c_p - c_t))
    return {
        'M10_dband_center_mae_up_eV':   float(np.mean(errs_up))   if errs_up   else float('nan'),
        'M10_dband_center_mae_down_eV': float(np.mean(errs_down)) if errs_down else float('nan'),
        'M10_dband_center_mae_mean_eV': float(np.mean(errs_up + errs_down))
                                        if (errs_up or errs_down) else float('nan'),
        'M10_n_valid_up':   len(errs_up),
        'M10_n_valid_down': len(errs_down),
    }


def _spin_polarisation_robust(pred: np.ndarray, true: np.ndarray,
                               window: int = 5) -> dict:
    """M13: Windowed spin polarisation error + half-metal F1.

    Uses ±window grid-point average (default ±0.05 eV) around FERMI_IDX.
    Semiconducting compounds (total DOS < 0.02 states/eV/atom at EF) skipped.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    sl_up   = slice(FERMI_IDX - window, FERMI_IDX + window + 1)
    sl_down = slice(FERMI_IDX + DOWN_OFFSET - window,
                    FERMI_IDX + DOWN_OFFSET + window + 1)
    TOTAL_THRESHOLD = 0.02
    errs, p_true_list, p_pred_list = [], [], []

    for i in range(len(pred)):
        up_p = float(pred[i, sl_up].mean());  up_t = float(true[i, sl_up].mean())
        dn_p = float(pred[i, sl_down].mean()); dn_t = float(true[i, sl_down].mean())
        total_t = up_t + dn_t;  total_p = up_p + dn_p
        if total_t < TOTAL_THRESHOLD or total_p < TOTAL_THRESHOLD:
            continue
        p_t = (up_t - dn_t) / total_t;  p_p = (up_p - dn_p) / total_p
        errs.append(abs(p_p - p_t));  p_true_list.append(p_t);  p_pred_list.append(p_p)

    result = {
        'M13_spin_pol_error_mae': float(np.mean(errs)) if errs else float('nan'),
        'M13_spin_pol_error_std': float(np.std(errs))  if errs else float('nan'),
        'M13_n_metallic_samples': len(errs),
        'M13_halfmetal_n_true': 0,
        'M13_halfmetal_precision': float('nan'),
        'M13_halfmetal_recall':    float('nan'),
        'M13_halfmetal_f1':        float('nan'),
    }
    if p_true_list:
        pt = np.array(p_true_list);  pp = np.array(p_pred_list)
        hm_true = (pt > 0.9).astype(int);  hm_pred = (pp > 0.9).astype(int)
        n_hm = int(hm_true.sum())
        result['M13_halfmetal_n_true']    = n_hm
        if n_hm > 0:
            result['M13_halfmetal_precision'] = float(precision_score(hm_true, hm_pred, zero_division=0))
            result['M13_halfmetal_recall']    = float(recall_score(hm_true, hm_pred, zero_division=0))
            result['M13_halfmetal_f1']        = float(f1_score(hm_true, hm_pred, zero_division=0))
    return result


def _halfmetal_classification(pred: np.ndarray, true: np.ndarray,
                               threshold: float = 0.05) -> dict:
    """M14: Precision/Recall/F1 for half-metallic compound screening.

    A compound is GAPPED (positive class) if either spin channel has DOS at
    E_F below threshold. Accuracy alone is misleading at 0-1.1% positive rate.
    threshold: states/eV/atom (default 0.05 = 5% of typical metallic value).
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    def is_gapped(dos_vec: np.ndarray) -> int:
        return int(
            (dos_vec[FERMI_IDX] < threshold) or
            (dos_vec[FERMI_IDX + DOWN_OFFSET] < threshold)
        )

    y_true = np.array([is_gapped(true[i]) for i in range(len(true))])
    y_pred = np.array([is_gapped(pred[i]) for i in range(len(pred))])
    n_pos  = int(y_true.sum())
    base = {
        'M14_halfmetal_n_true':    n_pos,
        'M14_halfmetal_n_pred':    int(y_pred.sum()),
        'M14_halfmetal_accuracy':  float(np.mean(y_true == y_pred)),
        'M14_halfmetal_threshold': threshold,
    }
    if n_pos > 0:
        base['M14_halfmetal_precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        base['M14_halfmetal_recall']    = float(recall_score(y_true, y_pred, zero_division=0))
        base['M14_halfmetal_f1']        = float(f1_score(y_true, y_pred, zero_division=0))
    else:
        base['M14_halfmetal_precision'] = float('nan')
        base['M14_halfmetal_recall']    = float('nan')
        base['M14_halfmetal_f1']        = float('nan')
    return base


def _wasserstein_eval(pred: np.ndarray, true: np.ndarray) -> dict:
    """M15: True 1D Wasserstein (Earth Mover's Distance) as evaluation metric.

    Distinct from the loss's sorted-L1 approximation. Uses scipy's exact
    wasserstein_distance treating each DOS curve as a probability distribution.
    Units: eV (spectral mass displacement). Typical good values: 0.1–0.4 eV.
    """
    e = ENERGIES
    wass_up, wass_down = [], []
    for i in range(len(pred)):
        for channel, wl in [(slice(None, 2001), wass_up),
                            (slice(2001, None), wass_down)]:
            p = pred[i, channel].clip(0);  t = true[i, channel].clip(0)
            ps = p.sum();  ts = t.sum()
            if ps < 1e-8 or ts < 1e-8:
                continue
            wl.append(scipy.stats.wasserstein_distance(e, e, p / ps, t / ts))
    all_w = wass_up + wass_down
    return {
        'M15_wasserstein_eV_mean':      float(np.mean(all_w))       if all_w      else float('nan'),
        'M15_wasserstein_eV_std':       float(np.std(all_w))        if all_w      else float('nan'),
        'M15_wasserstein_up_eV_mean':   float(np.mean(wass_up))     if wass_up    else float('nan'),
        'M15_wasserstein_down_eV_mean': float(np.mean(wass_down))   if wass_down  else float('nan'),
    }


def _peak_position_error(pred: np.ndarray, true: np.ndarray) -> dict:
    """M16: Peak position error in occupied states (E < 0 eV).

    argmax(DOS) in E = -10 to 0 eV region per spin channel. Measures where
    the d-band main peak is predicted vs true. Different from M10 (first moment).
    """
    occ = slice(0, FERMI_IDX)
    e_occ = ENERGIES[occ]
    MIN_PEAK = 1e-6
    errs_up, errs_down = [], []
    for i in range(len(pred)):
        for errs, p_sl, t_sl in [
            (errs_up,   pred[i, :2001][occ], true[i, :2001][occ]),
            (errs_down, pred[i, 2001:][occ], true[i, 2001:][occ]),
        ]:
            if p_sl.max() > MIN_PEAK and t_sl.max() > MIN_PEAK:
                errs.append(abs(e_occ[int(p_sl.argmax())] - e_occ[int(t_sl.argmax())]))
    all_e = errs_up + errs_down
    return {
        'M16_peak_pos_mae_up_eV':   float(np.mean(errs_up))   if errs_up   else float('nan'),
        'M16_peak_pos_mae_down_eV': float(np.mean(errs_down)) if errs_down else float('nan'),
        'M16_peak_pos_mae_mean_eV': float(np.mean(all_e))     if all_e     else float('nan'),
        'M16_n_valid':              len(all_e),
    }


def _bte_weighted_eval(pred: np.ndarray, true: np.ndarray) -> dict:
    """M17: BTE-weighted evaluation at 700 K — same weight as ThermoelectricDOSLoss.

    Evaluates the model on the exact criterion it was trained to optimise.
    M17_bte_pearson_r uses the ±0.5 eV BTE-effective window (indices 950:1051 up,
    2950:3052 down), which captures ~95% of the BTE weight.
    """
    w = _BTE_WEIGHT_4002
    resid = pred - true
    bte_mae  = float(np.mean(np.abs(resid) * w[np.newaxis, :]))
    bte_rmse = float(np.sqrt(np.mean((resid ** 2) * w[np.newaxis, :])))
    BTE_WIN = 50
    sl_up   = slice(FERMI_IDX - BTE_WIN, FERMI_IDX + BTE_WIN + 1)
    sl_down = slice(FERMI_IDX + DOWN_OFFSET - BTE_WIN,
                    FERMI_IDX + DOWN_OFFSET + BTE_WIN + 1)
    try:
        r_bte, _ = scipy.stats.pearsonr(
            np.concatenate([pred[:, sl_up].reshape(-1), pred[:, sl_down].reshape(-1)]),
            np.concatenate([true[:, sl_up].reshape(-1), true[:, sl_down].reshape(-1)]),
        )
    except Exception:
        r_bte = float('nan')
    return {
        'M17_bte_weighted_mae':  bte_mae,
        'M17_bte_weighted_rmse': bte_rmse,
        'M17_bte_pearson_r':     float(r_bte),
    }


def _per_channel_pearson(pred: np.ndarray, true: np.ndarray) -> dict:
    """M18: Pearson r disaggregated by spin channel.

    Global M1 hides asymmetry. For half-metals: dos_up is metallic (harder),
    dos_down is gapped (easier). M18_spin_channel_gap = |r_up - r_down|.
    """
    try:
        r_up, _   = scipy.stats.pearsonr(pred[:, :2001].reshape(-1),
                                          true[:, :2001].reshape(-1))
        r_down, _ = scipy.stats.pearsonr(pred[:, 2001:].reshape(-1),
                                          true[:, 2001:].reshape(-1))
    except Exception:
        r_up = r_down = float('nan')
    try:
        r_f_up,   _ = scipy.stats.pearsonr(pred[:, 800:1201].reshape(-1),
                                             true[:, 800:1201].reshape(-1))
        r_f_down, _ = scipy.stats.pearsonr(pred[:, 2801:3202].reshape(-1),
                                             true[:, 2801:3202].reshape(-1))
    except Exception:
        r_f_up = r_f_down = float('nan')
    gap = float(abs(r_up - r_down)) if not (np.isnan(r_up) or np.isnan(r_down)) else float('nan')
    return {
        'M18_pearson_r_up':         float(r_up),
        'M18_pearson_r_down':       float(r_down),
        'M18_fermi_pearson_r_up':   float(r_f_up),
        'M18_fermi_pearson_r_down': float(r_f_down),
        'M18_spin_channel_gap':     gap,
    }


def _integrated_dos_error(pred: np.ndarray, true: np.ndarray) -> dict:
    """M12: Integrated d-DOS error (charge conservation diagnostic).

    Positive bias → model over-predicts d-electron count.
    Negative bias → systematic normalisation error.
    Units: states/atom (integrated DOS over ±10 eV).
    """
    GS = GRID_SPACING
    err_up = pred[:, :2001].sum(axis=1) * GS - true[:, :2001].sum(axis=1) * GS
    err_dn = pred[:, 2001:].sum(axis=1) * GS - true[:, 2001:].sum(axis=1) * GS
    return {
        'M12_int_dos_mae_up':    float(np.abs(err_up).mean()),
        'M12_int_dos_mae_down':  float(np.abs(err_dn).mean()),
        'M12_int_dos_mae_total': float(np.abs(err_up + err_dn).mean()),
        'M12_int_dos_bias_up':   float(err_up.mean()),
        'M12_int_dos_bias_down': float(err_dn.mean()),
    }


def _seebeck_proxy(pred: np.ndarray, true: np.ndarray) -> dict:
    """
    Seebeck proxy via Mott formula: S ∝ (1/g_d) × dg_d/dE |_{E=E_F}.

    Uses a ±10-point (±0.10 eV) central-difference window around FERMI_IDX
    for both spin channels. Only metallic channels where g_d(E_F) > 1e-6
    states/eV/atom are included (gapped channels produce meaningless Seebeck).

    Returns:
        M_seebeck_proxy_mae   — mean |S_pred − S_true| over metallic channels
        M_seebeck_proxy_n     — number of channel-material pairs included
    """
    WINDOW = 10      # ±10 grid points = ±0.10 eV
    GS = GRID_SPACING   # 0.01 eV per point

    seebeck_errs = []
    for i in range(len(pred)):
        for sl_center in [FERMI_IDX, FERMI_IDX + DOWN_OFFSET]:
            sl = slice(sl_center - WINDOW, sl_center + WINDOW + 1)
            p_w = pred[i, sl]
            t_w = true[i, sl]
            g_p = float(p_w[WINDOW])    # DOS at E_F (centre of window)
            g_t = float(t_w[WINDOW])
            if g_p < 1e-6 or g_t < 1e-6:
                continue   # skip gapped channel (half-metal / semiconductor)
            # First derivative at E_F via central finite difference
            dg_p = (float(p_w[WINDOW + 1]) - float(p_w[WINDOW - 1])) / (2.0 * GS)
            dg_t = (float(t_w[WINDOW + 1]) - float(t_w[WINDOW - 1])) / (2.0 * GS)
            s_p = dg_p / g_p
            s_t = dg_t / g_t
            seebeck_errs.append(abs(s_p - s_t))

    return {
        'M_seebeck_proxy_mae': float(np.mean(seebeck_errs)) if seebeck_errs else float('nan'),
        'M_seebeck_proxy_n':   len(seebeck_errs),
    }


def compare_scratch_vs_tl(results_root: str, dataset: str) -> dict:
    """
    Compare Exp C (scratch) vs Exp C_TL for one dataset.

    Reads metrics_summary.json from both experiment directories and returns
    a flat dict with delta metrics and a boolean tl_wins flag.  Returns an
    empty dict if either metrics file is missing (i.e. models not yet run).

    Args:
        results_root:  root results directory, e.g. "results"
        dataset:       dataset name, e.g. "half_tetra"

    Returns:
        dict with keys:
            dataset, scratch_pearson_r, tl_pearson_r, delta_pearson_r,
            scratch_fermi_r, tl_fermi_r, delta_fermi_r, tl_wins
        or {} if either file is absent.
    """
    import os as _os, json as _json

    scratch_path = _os.path.join(results_root, dataset, "expC",    "fold0", "metrics_summary.json")
    tl_path      = _os.path.join(results_root, dataset, "expC_TL", "fold0", "metrics_summary.json")

    if not _os.path.exists(scratch_path) or not _os.path.exists(tl_path):
        return {}

    with open(scratch_path) as f:
        scratch = _json.load(f)
    with open(tl_path) as f:
        tl = _json.load(f)

    def _get(d, *keys):
        """Try multiple key variants; return float or NaN."""
        for k in keys:
            v = d.get(k)
            if v is not None:
                return float(v) if not isinstance(v, dict) else float(v.get("mean", float("nan")))
        return float("nan")

    # FIX Bug-44: metrics_summary.json has nested structure d['metrics']['key']['mean'];
    # flat keys like M1_pearson_r_mean exist ONLY in metrics_all.json, not here.
    _ms = scratch.get("metrics", {})
    _mt = tl.get("metrics", {})
    scratch_r  = _get(_ms, "pearson_r")
    tl_r       = _get(_mt, "pearson_r")
    scratch_fr = _get(_ms, "fermi_pearson_2eV")
    tl_fr      = _get(_mt, "fermi_pearson_2eV")

    return {
        "dataset":           dataset,
        "scratch_pearson_r": scratch_r,
        "tl_pearson_r":      tl_r,
        "delta_pearson_r":   tl_r - scratch_r,
        "scratch_fermi_r":   scratch_fr,
        "tl_fermi_r":        tl_fr,
        "delta_fermi_r":     tl_fr - scratch_fr,
        "tl_wins":           bool(tl_r > scratch_r),
    }


def compute_all_metrics(pred: np.ndarray, true: np.ndarray,
                         ids: list,
                         dataset: str, experiment: str, fold: int,
                         output_dir) -> dict:
    """
    Compute the complete publication-standard metric suite and return a flat dict.

    This is the high-level entry point used by integration tests and external
    callers. It runs all M1–M18 metrics and returns a single merged dict whose
    keys match the M-number prefix convention.

    Also writes metrics_summary.json to output_dir (same format as main()).

    Args:
        pred:        np.ndarray (N, 4002) — predicted d-projected DOS
        true:        np.ndarray (N, 4002) — DFT d-projected DOS
        ids:         list of N material_id strings
        dataset:     dataset name (e.g. 'full_cubic')
        experiment:  'A', 'B', 'C', or 'D'
        fold:        fold index
        output_dir:  directory path (str or Path); created if absent

    Returns:
        flat dict with all metric values (floats/ints); NaN for undefined metrics.
    """
    import os as _os
    _os.makedirs(str(output_dir), exist_ok=True)

    y_pred_up   = pred[:, :N_POINTS]
    y_pred_down = pred[:, N_POINTS:]
    y_true_up   = true[:, :N_POINTS]
    y_true_down = true[:, N_POINTS:]

    flat: dict = {}

    # M1–M4
    m1 = compute_m1_pearson(pred, true)
    flat['M1_pearson_r_mean'] = float(np.nanmean(m1))
    flat['M1_pearson_r_std']  = float(np.nanstd(m1, ddof=1)) if len(m1) > 1 else 0.0

    m2 = compute_m2_rmse(pred, true)
    flat['M2_rmse_mean'] = float(np.nanmean(m2))

    m3 = compute_m3_mae(pred, true)
    flat['M3_mae_mean'] = float(np.nanmean(m3))

    flat['M4_r2'] = compute_m4_r2(pred, true)

    # M9: Fermi-window Pearson
    m9 = compute_m9_fermi_pearson(y_pred_up, y_pred_down, y_true_up, y_true_down)
    flat['M9_fermi_pearson_mean'] = float(np.nanmean(m9))

    # M10: D-band center (safe)
    flat.update(_dband_center_safe(pred, true))

    # M12: Integrated DOS error
    flat.update(_integrated_dos_error(pred, true))

    # M13: Spin polarisation (robust, windowed)
    flat.update(_spin_polarisation_robust(pred, true))

    # M14: Half-metal classification
    flat.update(_halfmetal_classification(pred, true))

    # M15: Wasserstein evaluation
    flat.update(_wasserstein_eval(pred, true))

    # M16: Peak position error
    flat.update(_peak_position_error(pred, true))

    # M17: BTE-weighted
    flat.update(_bte_weighted_eval(pred, true))

    # M18: Per-channel Pearson
    flat.update(_per_channel_pearson(pred, true))

    # Seebeck proxy (Mott formula finite-difference at E_F)
    flat.update(_seebeck_proxy(pred, true))

    # Write JSON
    import json as _json
    out_path = _os.path.join(str(output_dir), 'metrics_all.json')
    with open(out_path, 'w') as fh:
        _json.dump(
            {k: (None if (isinstance(v, float) and v != v) else v)
             for k, v in flat.items()},
            fh, indent=2
        )

    return flat


# ── Per-channel metrics ────────────────────────────────────────────────────────

def compute_per_channel_metrics(y_pred_up, y_pred_down,
                                y_true_up, y_true_down) -> dict:
    """M1, M2, M3, M9 computed separately for dos_up and dos_down."""
    out = {}
    for label, pred, true in [("up", y_pred_up, y_true_up),
                               ("down", y_pred_down, y_true_down)]:
        pearson = np.array([scipy.stats.pearsonr(pred[i], true[i])[0]
                            for i in range(len(pred))])
        rmse    = np.sqrt(np.mean((pred - true) ** 2, axis=1))
        mae     = np.mean(np.abs(pred - true), axis=1)
        # Fermi-window Pearson (±2 eV) per channel
        fp = np.array([
            scipy.stats.pearsonr(pred[i, W2EV], true[i, W2EV])[0]
            for i in range(len(pred))
        ])
        out[f"dos_{label}"] = {
            "pearson_r":         agg(pearson, f"pearson_{label}", "partial"),
            "rmse":              agg(rmse,    f"rmse_{label}",    "partial"),
            "mae":               agg(mae,     f"mae_{label}",     "full"),
            "fermi_pearson_2eV": agg(fp,      f"fp_{label}",      "full"),
        }
    return out


# ── Electronic type classification ────────────────────────────────────────────

def classify_electronic_type(P_true: np.ndarray, true_gapped: np.ndarray) -> np.ndarray:
    """Returns array of strings: 'half_metallic' | 'semiconducting' | 'metallic'."""
    N = len(P_true)
    types = np.array(["metallic"] * N, dtype=object)
    types[np.abs(P_true) > HALF_METALLIC_POL_THRESH] = "half_metallic"
    types[(true_gapped) & (np.abs(P_true) <= HALF_METALLIC_POL_THRESH)] = "semiconducting"
    return types


def per_type_metrics(electronic_types: np.ndarray, per_sample_pearson: np.ndarray,
                     per_sample_fermi_pearson: np.ndarray,
                     per_sample_dband: np.ndarray) -> dict:
    out = {}
    for etype in ["half_metallic", "semiconducting", "metallic"]:
        mask = electronic_types == etype
        n    = int(mask.sum())
        if n == 0:
            out[etype] = {"n_samples": 0}
            continue
        out[etype] = {
            "n_samples":         n,
            "pearson_r_mean":    float(np.nanmean(per_sample_pearson[mask])),
            "pearson_r_std":     float(np.nanstd(per_sample_pearson[mask], ddof=1)) if n > 1 else 0.0,
            "fermi_pearson_mean": float(np.nanmean(per_sample_fermi_pearson[mask])),
            "fermi_pearson_std":  float(np.nanstd(per_sample_fermi_pearson[mask], ddof=1)) if n > 1 else 0.0,
            "d_band_mae_mean":   float(np.nanmean(per_sample_dband[mask])),
            "d_band_mae_std":    float(np.nanstd(per_sample_dband[mask], ddof=1)) if n > 1 else 0.0,
        }
    return out


# ── Per-Z-element breakdown ────────────────────────────────────────────────────

def per_z_element_metrics(material_ids: list, id_to_z: dict, id_to_zan: dict,
                           per_sample_pearson: np.ndarray,
                           per_sample_fermi_mae: np.ndarray) -> pd.DataFrame:
    rows = []
    z_elements = sorted({id_to_z[mid] for mid in material_ids if mid in id_to_z})

    for z_el in z_elements:
        mask = np.array([id_to_z.get(mid, None) == z_el for mid in material_ids])
        n = int(mask.sum())
        if n == 0:
            continue
        zan = id_to_zan.get(
            next(mid for mid in material_ids if id_to_z.get(mid) == z_el), 0
        )
        rows.append({
            "z_element":          z_el,
            "atomic_number":      int(zan),
            "n_samples":          n,
            "mean_pearson":       float(np.nanmean(per_sample_pearson[mask])),
            "std_pearson":        float(np.nanstd(per_sample_pearson[mask], ddof=1)) if n > 1 else 0.0,
            "mean_fermi_mae_1eV": float(np.nanmean(per_sample_fermi_mae[mask])),
            "std_fermi_mae_1eV":  float(np.nanstd(per_sample_fermi_mae[mask], ddof=1)) if n > 1 else 0.0,
        })

    df = pd.DataFrame(rows).sort_values("atomic_number")

    # SOC effect check: if RMSE/Pearson correlates with atomic number → flag
    if len(df) >= 4:
        corr, pval = scipy.stats.spearmanr(df["atomic_number"], -df["mean_pearson"])
        df["soc_flag"] = corr > 0.5 and pval < 0.05
    else:
        df["soc_flag"] = False

    return df


# ── Outlier identification ─────────────────────────────────────────────────────

def identify_outliers(material_ids: list, per_sample_pearson: np.ndarray,
                      per_sample_rmse: np.ndarray, per_sample_mae: np.ndarray,
                      per_sample_fermi_pearson: np.ndarray,
                      per_sample_fermi_mae: np.ndarray,
                      per_sample_dband: np.ndarray,
                      per_sample_spinpol: np.ndarray,
                      P_pred: np.ndarray, P_true: np.ndarray,
                      electronic_types: np.ndarray,
                      id_to_z: dict, group_df: pd.DataFrame) -> dict:

    N = len(material_ids)
    sorted_by_pearson = np.argsort(per_sample_pearson)
    median_pos        = N // 2
    median_idx        = int(sorted_by_pearson[median_pos])

    def make_record(idx):
        mid = material_ids[idx]
        formula = mid.rsplit("_", 1)[0]
        return {
            "material_id":      mid,
            "formula":          formula,
            "z_site_element":   id_to_z.get(mid, "?"),
            "pearson_r":        float(per_sample_pearson[idx]),
            "rmse":             float(per_sample_rmse[idx]),
            "mae":              float(per_sample_mae[idx]),
            "fermi_pearson_2eV": float(per_sample_fermi_pearson[idx]),
            "fermi_mae_1eV":    float(per_sample_fermi_mae[idx]),
            "d_band_error_eV":  float(per_sample_dband[idx]) if not np.isnan(per_sample_dband[idx]) else None,
            "spin_pol_error":   float(per_sample_spinpol[idx]),
            "P_pred":           float(P_pred[idx]),
            "P_true":           float(P_true[idx]),
            "electronic_type":  str(electronic_types[idx]),
            "row_index":        int(idx),
        }

    best_5_idx   = sorted_by_pearson[-5:][::-1].tolist()
    worst_5_idx  = sorted_by_pearson[:5].tolist()

    # Also identify best half-metallic sample
    hm_mask = electronic_types == "half_metallic"
    hm_pearson = per_sample_pearson.copy()
    hm_pearson[~hm_mask] = -np.inf
    best_hm_idx = int(np.argmax(hm_pearson)) if hm_mask.any() else None

    selected = []
    # Best metallic, best half-metallic, median, worst
    metallic_mask   = electronic_types == "metallic"
    metal_pearson   = per_sample_pearson.copy()
    metal_pearson[~metallic_mask] = -np.inf
    best_metal_idx  = int(np.argmax(metal_pearson)) if metallic_mask.any() else int(sorted_by_pearson[-1])

    for idx in [best_metal_idx, best_hm_idx, median_idx, int(sorted_by_pearson[0])]:
        if idx is None:
            continue
        rec = make_record(idx)
        rec["role"] = (
            "best_metallic"    if idx == best_metal_idx else
            "best_half_metallic" if idx == best_hm_idx else
            "median"           if idx == median_idx else
            "worst"
        )
        selected.append(rec)

    return {
        "selected_samples": selected,
        "best_5":           [make_record(i) for i in best_5_idx],
        "worst_5":          [make_record(i) for i in worst_5_idx],
    }


# ── Per-energy MAE (for Figure 4) ─────────────────────────────────────────────

def compute_per_energy_mae(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Mean absolute error at each of 4002 DOS positions (2001 per spin channel), averaged over N samples."""
    return np.mean(np.abs(predictions - targets), axis=0)


# ── Thermoelectric descriptor computation ─────────────────────────────────────

def compute_te_descriptors(predictions: np.ndarray, targets: np.ndarray,
                            ids: list, output_dir) -> tuple:
    """
    Compute thermoelectric screening descriptors from d-projected DOS predictions.

    Called after inference. Saves:
      - te_descriptors.csv          (N rows, pred/true/error per descriptor)
      - te_descriptor_summary.json  (MAE and Pearson per descriptor)
      - te_ranking.csv              (materials ranked by TE composite score)

    Physical basis:
      Descriptors derived from Mott formula and BTE transport theory.
      D1-D4:  DOS at E_F (determines conductivity vs Seebeck tradeoff)
      D5-D9:  DOS slope at E_F (direct Mott formula: S propto dg_d/dE)
      D10-D11: DOS asymmetry (n-type vs p-type tendency)
      D12:    Thermal window DOS integral (BTE transport weight at 700K)
      D13-D16: Band gap / pseudo-gap indicators
      D17-D23: D-band moments (center, width, filling, exchange splitting)
      D24-D29: Peak sharpness analysis (Mahan-Sofo proximity)
      D30-D31: Seebeck proxies (Mott formula order-of-magnitude estimates)
      D32-D33: Spin channel metrics (half-metallicity, bipolar risk)
      D34:    Combined TE composite score (ranking metric)

    Args:
        predictions: np.ndarray shape (N, 4002) — predicted d-projected DOS
        targets:     np.ndarray shape (N, 4002) — DFT d-projected DOS
        ids:         list of N material ID strings
        output_dir:  path to output directory (str or Path)

    Returns:
        (result_df, errors_dict) where result_df has pred/true/err columns per descriptor
    """
    from scipy.signal import find_peaks
    import pandas as pd
    import os
    import json as json_module

    output_dir = str(output_dir)
    mott_700   = (np.pi**2 * k_B**2 * 700) / 3.0   # eV^2 (Mott prefactor)
    mott_300   = (np.pi**2 * k_B**2 * 300) / 3.0

    records = []

    for i, mid in enumerate(ids):
        pred = predictions[i]
        true = targets[i]

        for suffix, arr_up, arr_down in [
            ('pred', pred[:N_POINTS], pred[N_POINTS:]),
            ('true', true[:N_POINTS], true[N_POINTS:]),
        ]:
            d_up   = arr_up.astype(np.float64)
            d_down = arr_down.astype(np.float64)
            d_tot  = d_up + d_down
            rec = {'material_id': mid, 'type': suffix}

            # ── D-band DOS at Fermi level ──────────────────────────────────
            rec['D1_dos_ef_total'] = float(d_tot[FERMI_IDX])
            rec['D2_dos_ef_up']    = float(d_up[FERMI_IDX])
            rec['D3_dos_ef_down']  = float(d_down[FERMI_IDX])
            denom_ef = d_up[FERMI_IDX] + d_down[FERMI_IDX]
            rec['D4_spin_pol_ef']  = float(
                (d_up[FERMI_IDX] - d_down[FERMI_IDX]) / (denom_ef + EPS))

            # ── DOS slope at E_F (Mott formula terms) ─────────────────────
            slope_up   = np.gradient(d_up,   ENERGIES)[FERMI_IDX]
            slope_down = np.gradient(d_down, ENERGIES)[FERMI_IDX]
            slope_tot  = np.gradient(d_tot,  ENERGIES)[FERMI_IDX]
            rec['D5_slope_up']         = float(slope_up)
            rec['D6_slope_down']       = float(slope_down)
            rec['D7_slope_total']      = float(slope_tot)
            rec['D8_log_slope_up']     = float(slope_up   / (d_up[FERMI_IDX]   + EPS))
            rec['D9_log_slope_down']   = float(slope_down / (d_down[FERMI_IDX] + EPS))

            # ── DOS asymmetry (n-type vs p-type tendency) ─────────────────
            above_1 = np.trapz(d_tot[FERMI_IDX:FERMI_IDX+101], ENERGIES[FERMI_IDX:FERMI_IDX+101])
            below_1 = np.trapz(d_tot[FERMI_IDX-100:FERMI_IDX], ENERGIES[FERMI_IDX-100:FERMI_IDX])
            rec['D10_asymmetry_1eV']   = float(above_1 - below_1)
            above_2 = np.trapz(d_tot[FERMI_IDX:FERMI_IDX+201], ENERGIES[FERMI_IDX:FERMI_IDX+201])
            below_2 = np.trapz(d_tot[FERMI_IDX-200:FERMI_IDX], ENERGIES[FERMI_IDX-200:FERMI_IDX])
            rec['D11_asymmetry_2eV']   = float(above_2 - below_2)

            # ── Thermal window DOS (BTE transport at 700K) ────────────────
            rec['D12_dos_kBT_700K']    = float(np.trapz(d_tot[kBT_WIN], ENERGIES[kBT_WIN]))

            # ── Band gap / pseudo-gap ──────────────────────────────────────
            w05 = np.where(np.abs(ENERGIES) <= 0.5)[0]
            rec['D13_gap_min_total']   = float(d_tot[w05].min())
            rec['D14_gap_energy']      = float(ENERGIES[w05[np.argmin(d_tot[w05])]])
            rec['D15_gap_min_up']      = float(d_up[W1EV].min())
            rec['D16_gap_min_down']    = float(d_down[W1EV].min())

            # ── D-band moments ─────────────────────────────────────────────
            integ_up   = np.trapz(d_up,   ENERGIES) + EPS
            integ_down = np.trapz(d_down, ENERGIES) + EPS
            d_c_up   = np.trapz(ENERGIES * d_up,   ENERGIES) / integ_up
            d_c_down = np.trapz(ENERGIES * d_down, ENERGIES) / integ_down
            rec['D17_d_center_up']     = float(d_c_up)
            rec['D18_d_center_down']   = float(d_c_down)
            rec['D19_exchange_split']  = float(d_c_up - d_c_down)
            d_w_up   = np.sqrt(np.trapz((ENERGIES - d_c_up)**2   * d_up,   ENERGIES) / integ_up)
            d_w_down = np.sqrt(np.trapz((ENERGIES - d_c_down)**2 * d_down, ENERGIES) / integ_down)
            rec['D20_d_width_up']      = float(d_w_up)
            rec['D21_d_width_down']    = float(d_w_down)
            mask_occ = ENERGIES <= 0.0
            rec['D22_d_filling_up']    = float(np.trapz(d_up[mask_occ],   ENERGIES[mask_occ]))
            rec['D23_d_filling_down']  = float(np.trapz(d_down[mask_occ], ENERGIES[mask_occ]))

            # ── Peak sharpness in ±2eV (Mahan-Sofo proximity) ─────────────
            rec['D24_peak_height_up']  = float(d_up[W2EV].max())
            rec['D25_peak_height_down']= float(d_down[W2EV].max())

            for ch_name, ch_arr, pk_key, fw_key, fl_key in [
                ('up',   d_up,   'D24', 'D26', 'D28'),
                ('down', d_down, 'D25', 'D27', 'D29'),
            ]:
                ch_win = ch_arr[W2EV]
                peaks, props = find_peaks(ch_win, height=0.01 * max(ch_win.max(), EPS))
                if len(peaks) > 0:
                    best_ph = np.argmax(props['peak_heights'])
                    h       = props['peak_heights'][best_ph]
                    half    = h / 2.0
                    fwhm    = float(len(np.where(ch_win >= half)[0]) * GRID_SPACING)
                else:
                    fwhm = 2.0   # broad — no sharp peak found
                pk_h = rec[f'{pk_key}_peak_height_{ch_name}']
                rec[f'{fw_key}_fwhm_{ch_name}']     = fwhm
                rec[f'{fl_key}_flatness_{ch_name}'] = float(pk_h / (fwhm + EPS))

            # ── Seebeck proxies (Mott formula estimates) ───────────────────
            # S ≈ -log_slope × pi^2 k_B^2 T / 3  (rough μV/K estimate)
            rec['D30_seebeck_proxy_up_700K']   = float(-rec['D8_log_slope_up']   * mott_700 * 1e6)
            rec['D31_seebeck_proxy_down_700K'] = float(-rec['D9_log_slope_down'] * mott_700 * 1e6)

            # ── Spin / bipolar ─────────────────────────────────────────────
            rec['D32_bipolar_risk']       = float(rec['D15_gap_min_up'] * rec['D16_gap_min_down'])
            rec['D33_is_half_metallic']   = int(
                rec['D15_gap_min_up'] < 0.05 or rec['D16_gap_min_down'] < 0.05)

            # ── Combined TE composite score ────────────────────────────────
            s_proxy    = max(abs(rec['D30_seebeck_proxy_up_700K']),
                             abs(rec['D31_seebeck_proxy_down_700K']))
            flatness   = max(rec['D28_flatness_up'], rec['D29_flatness_down'])
            dos_ef_mod = 1.0 / (1.0 + abs(rec['D1_dos_ef_total'] - 2.0))
            bp_supp    = float(np.exp(-rec['D32_bipolar_risk']))
            rec['D34_TE_score']  = float(s_proxy * flatness * dos_ef_mod * bp_supp)

            records.append(rec)

    df = pd.DataFrame(records)
    df_pred = df[df['type'] == 'pred'].reset_index(drop=True)
    df_true = df[df['type'] == 'true'].reset_index(drop=True)

    desc_cols = [c for c in df.columns if c.startswith('D') and c[1].isdigit()]
    errors = {}
    col_dict = {'material_id': df_pred['material_id'].values}
    for col in desc_cols:
        pred_vals = df_pred[col].values
        true_vals = df_true[col].values
        err_vals  = np.abs(pred_vals - true_vals)
        col_dict[f'pred_{col}'] = pred_vals
        col_dict[f'true_{col}'] = true_vals
        col_dict[f'err_{col}']  = err_vals
        errors[col] = {
            'mae':     float(err_vals.mean()),
            'pearson': float(np.corrcoef(pred_vals, true_vals)[0, 1])
                       if pd.Series(pred_vals).std() > 0 else 0.0,
        }
    result = pd.DataFrame(col_dict)

    os.makedirs(output_dir, exist_ok=True)
    result.to_csv(os.path.join(output_dir, 'te_descriptors.csv'), index=False)
    with open(os.path.join(output_dir, 'te_descriptor_summary.json'), 'w') as f:
        json_module.dump(errors, f, indent=2)

    ranking = result[['material_id',
                       'pred_D34_TE_score',
                       'pred_D30_seebeck_proxy_up_700K',
                       'pred_D31_seebeck_proxy_down_700K',
                       'pred_D17_d_center_up',
                       'pred_D19_exchange_split',
                       'pred_D33_is_half_metallic',
                       'pred_D1_dos_ef_total']].copy()
    ranking = ranking.sort_values('pred_D34_TE_score', ascending=False)
    ranking.to_csv(os.path.join(output_dir, 'te_ranking.csv'), index=False)

    print(f'TE descriptors computed for {len(ids)} materials')
    print(f'Top 5 TE candidates:')
    print(ranking.head(5)[['material_id', 'pred_D34_TE_score',
                            'pred_D30_seebeck_proxy_up_700K']].to_string())
    return result, errors


# ── JSON serialisation helper ──────────────────────────────────────────────────

def to_json(obj):
    """Recursively convert numpy scalars/arrays to Python types for JSON."""
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and (obj != obj):   # NaN
        return None
    return obj


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute publication-grade metrics for ALIGNN DOS predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--predictions", required=True, type=Path,
                        help="Path to test_predictions.npy (shape Nx4002).")
    parser.add_argument("--targets",     required=True, type=Path,
                        help="Path to test_targets.npy (shape Nx4002).")
    parser.add_argument("--ids",         required=True, type=Path,
                        help="Path to test_ids.txt (one material_id per line).")
    parser.add_argument("--dataset",     required=True,
                        help="Dataset name, e.g. full_cubic.")
    parser.add_argument("--experiment",  required=True, choices=["A", "B", "C", "C_TL", "D"],
                        help="Experiment type (A=random CV, B=composition CV, C=full-data scratch, C_TL=full-data TL, D=element holdout).")
    parser.add_argument("--fold",        required=True, type=int,
                        help="Fold index (0–4).")
    parser.add_argument("--output_dir",  required=True, type=Path,
                        help="Directory for all output files.")
    parser.add_argument("--data-root",   type=Path, default=Path("data"),
                        help="Root data directory (default: data/).")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load arrays ───────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"METRICS: dataset={args.dataset}  Exp={args.experiment}  fold={args.fold}")
    print(f"{'═'*60}")

    predictions = np.load(args.predictions)
    targets     = np.load(args.targets)

    with open(args.ids) as f:
        material_ids = [line.strip() for line in f if line.strip()]

    N = len(material_ids)
    assert predictions.shape == (N, 4002), \
        f"predictions shape {predictions.shape} != ({N}, 4002)"
    assert targets.shape == (N, 4002), \
        f"targets shape {targets.shape} != ({N}, 4002)"

    print(f"  Loaded {N} samples, shape {predictions.shape}")

    # NaN check
    n_nan_pred   = int(np.isnan(predictions).sum())
    n_nan_target = int(np.isnan(targets).sum())
    if n_nan_pred or n_nan_target:
        warnings.warn(f"NaN values detected: predictions={n_nan_pred}, targets={n_nan_target}")

    # ── Channel decomposition ─────────────────────────────────────────────────
    # Predictions are d-projected [d_spin_up(2001), d_spin_down(2001)] -> (N, 4002)
    # Source: spd_dos['d']['densities']['1']=up, ['-1']=down (per-atom, Fermi-aligned)
    y_pred_up   = predictions[:, :N_POINTS]    # d-band spin-up channel
    y_pred_down = predictions[:, N_POINTS:]    # d-band spin-down channel
    y_true_up   = targets[:,    :N_POINTS]
    y_true_down = targets[:,    N_POINTS:]

    # ── Load group assignments ────────────────────────────────────────────────
    group_csv = args.data_root / args.dataset / "data_split" / "group_assignments.csv"
    if group_csv.exists():
        group_df  = pd.read_csv(group_csv)
        id_to_z   = dict(zip(group_df.material_id, group_df.z_site_element))
        id_to_zan = dict(zip(group_df.material_id, group_df.z_atomic_number))
    else:
        warnings.warn(f"group_assignments.csv not found at {group_csv} — Z-group breakdown skipped")
        group_df  = pd.DataFrame()
        id_to_z   = {}
        id_to_zan = {}

    # ── Compute all metrics ───────────────────────────────────────────────────
    print("  Computing metrics...")

    m1_pearson = compute_m1_pearson(predictions, targets)
    m2_rmse    = compute_m2_rmse(predictions, targets)
    m3_mae     = compute_m3_mae(predictions, targets)
    m4_r2      = compute_m4_r2(predictions, targets)
    m5_jsd     = compute_m5_jsd(y_pred_up, y_pred_down, y_true_up, y_true_down)
    m6_cosine  = compute_m6_cosine(predictions, targets)
    m7_fmae1   = compute_m7_fermi_mae_1eV(y_pred_up, y_pred_down, y_true_up, y_true_down)
    m8_fmae2   = compute_m8_fermi_mae_2eV(y_pred_up, y_pred_down, y_true_up, y_true_down)
    m9_fp      = compute_m9_fermi_pearson(y_pred_up, y_pred_down, y_true_up, y_true_down)
    m10_dband, dc_pred_up, dc_pred_dn, dc_true_up, dc_true_dn = compute_m10_dband(
        y_pred_up, y_pred_down, y_true_up, y_true_down)
    m11_nef    = compute_m11_nef(y_pred_up, y_pred_down, y_true_up, y_true_down)
    m12_int    = compute_m12_integrated(y_pred_up, y_pred_down, y_true_up, y_true_down)
    m13_spinpol, P_pred, P_true = compute_m13_spinpol(
        y_pred_up, y_pred_down, y_true_up, y_true_down)
    m14_primary = compute_m14_bandgap(y_pred_up, y_pred_down, y_true_up, y_true_down)
    m14_sensitivity = compute_m14_threshold_sensitivity(
        y_pred_up, y_pred_down, y_true_up, y_true_down)
    m15_exchange, exchange_pred, exchange_true = compute_m15_exchange(
        y_pred_up, y_pred_down, y_true_up, y_true_down)

    # ── New metrics: Wasserstein eval, peak position, BTE-weighted ────────────
    print("  Computing extended metrics (Wasserstein, peak position, BTE)...")
    m_wass_eval = compute_m_wasserstein_eval(
        y_pred_up, y_pred_down, y_true_up, y_true_down)
    m_peak_up, m_peak_down = compute_m_peak_pos(
        y_pred_up, y_pred_down, y_true_up, y_true_down)
    m_bte_dict = compute_m_bte_weighted(predictions, targets)

    # ── M12 charge-conservation bias ─────────────────────────────────────────
    m12_bias = compute_m12_bias(y_pred_up, y_pred_down, y_true_up, y_true_down)

    # ── M13 half-metal recall (windowed P) ────────────────────────────────────
    m13_hm_recall = compute_m13_halfmetal_recall(P_pred, P_true)

    # ── Electronic type classification ────────────────────────────────────────
    true_gapped     = m14_primary["true_gapped_arr"]
    electronic_types = classify_electronic_type(P_true, true_gapped)

    # M13 per-type breakdown
    hm_mask   = electronic_types == "half_metallic"
    meta_mask = electronic_types == "metallic"
    m13_hm_mean   = float(np.nanmean(m13_spinpol[hm_mask]))   if hm_mask.any()   else None
    m13_hm_std    = float(np.nanstd(m13_spinpol[hm_mask], ddof=1)) if hm_mask.sum() > 1 else None
    m13_meta_mean = float(np.nanmean(m13_spinpol[meta_mask])) if meta_mask.any() else None
    m13_meta_std  = float(np.nanstd(m13_spinpol[meta_mask], ddof=1)) if meta_mask.sum() > 1 else None

    # ── Assemble metrics_summary.json ─────────────────────────────────────────
    m14_for_json = {k: v for k, v in m14_primary.items() if k != "true_gapped_arr"}
    m14_for_json["threshold_sensitivity"] = m14_sensitivity
    m14_for_json["independence"] = "full"

    metrics_summary = {
        "dataset":     args.dataset,
        "experiment":  args.experiment,
        "fold":        args.fold,
        "n_samples":   N,
        "n_nan_pred":  n_nan_pred,
        "n_nan_target": n_nan_target,
        "loss_function": "ThermoelectricDOSLoss (7-component): 0.35*BTE-MSE + 0.19*Wasserstein + 0.13*(1-cosine) + 0.10*slope + 0.10*MSE(±2eV) + 0.10*peak-height(γ=2.5) + 0.03*TV-smooth",
        "target_type": "d_projected_spin_resolved",
        "metrics": {
            "pearson_r":         agg(m1_pearson, "pearson_r",  "partial"),
            "rmse":              agg(m2_rmse,    "rmse",       "partial"),
            "mae":               agg(m3_mae,     "mae",        "full"),
            "r2": {
                "value":        m4_r2,
                "independence": "full",
                "note":         "Single global scalar (not per-sample)",
            },
            "jsd":               agg(m5_jsd,     "jsd",        "full"),
            "fermi_mae_1eV":     agg(m7_fmae1,   "fermi_mae_1eV",    "full"),
            "fermi_pearson_2eV": agg(m9_fp,      "fermi_pearson_2eV","full"),
            "d_band_mae_eV":     agg(m10_dband,  "d_band_mae_eV",    "full"),
            "n_ef_error":        agg(m11_nef,    "n_ef_error",       "full"),
            "integrated_dos_error": agg(m12_int, "integrated",       "full"),
            "spin_pol_error": {
                **agg(m13_spinpol, "spin_pol_error", "full"),
                "half_metallic_mean": m13_hm_mean,
                "half_metallic_std":  m13_hm_std,
                "half_metallic_n":    int(hm_mask.sum()),
                "metallic_mean":      m13_meta_mean,
                "metallic_std":       m13_meta_std,
                "metallic_n":         int(meta_mask.sum()),
            },
            "band_gap": m14_for_json,
            # ── New evaluation metrics ────────────────────────────────────────
            "wasserstein_eV": {
                **agg(m_wass_eval, "wasserstein_eV", "full"),
                "note": "True 1D Earth Mover's Distance (eV); avg both spin channels; "
                        "distinct from loss's sorted-L1 Wasserstein; typical range 0.1–0.3 eV",
            },
            "peak_pos_mae_up_eV":   agg(m_peak_up,   "peak_pos_up",   "full"),
            "peak_pos_mae_down_eV": agg(m_peak_down, "peak_pos_down", "full"),
            "bte_weighted":         m_bte_dict,
            "integrated_dos_bias":  m12_bias,
            "spin_pol_halfmetal_recall": m13_hm_recall,
        },
        "electronic_type_counts": {
            etype: int((electronic_types == etype).sum())
            for etype in ["half_metallic", "semiconducting", "metallic"]
        },
    }

    # ── Supplementary metrics (dependent on loss) ─────────────────────────────
    supplementary = {
        "dataset":    args.dataset,
        "experiment": args.experiment,
        "fold":       args.fold,
        "note": (
            "Metrics in this file are either fully dependent on training loss "
            "components or supplementary-only. Do NOT include in primary results table."
        ),
        "cosine_similarity": {
            **agg(m6_cosine, "cosine", "fully_dependent"),
            "dependent_on_loss": True,
            "note": "0.30×(1−cosine) IS a loss component. Circular to report as independent evaluation.",
        },
        "fermi_mae_2eV": {
            **agg(m8_fmae2, "fermi_mae_2eV", "fully_dependent"),
            "dependent_on_loss": True,
            "note": "0.10×MSE(±2eV) IS a loss component. Supplementary only.",
        },
        "exchange_splitting_error": {
            **agg(m15_exchange, "exchange", "full"),
            "dependent_on_loss": False,
            "note": "Difference of spin-up and spin-down d-band centers. Supplementary.",
        },
    }

    # ── Per-channel metrics ────────────────────────────────────────────────────
    per_channel = compute_per_channel_metrics(
        y_pred_up, y_pred_down, y_true_up, y_true_down)

    # ── Electronic type metrics ────────────────────────────────────────────────
    electronic_type_metrics = per_type_metrics(
        electronic_types, m1_pearson, m9_fp, m10_dband)

    # ── Per-Z-element metrics ─────────────────────────────────────────────────
    if id_to_z:
        z_df = per_z_element_metrics(
            material_ids, id_to_z, id_to_zan, m1_pearson, m7_fmae1)
        soc_suspected = bool(z_df["soc_flag"].any()) if len(z_df) > 0 else False
    else:
        z_df          = pd.DataFrame()
        soc_suspected = False

    # ── Per-sample metrics CSV ─────────────────────────────────────────────────
    def formula_of(mid):
        return mid.rsplit("_", 1)[0]

    per_sample_df = pd.DataFrame({
        "material_id":           material_ids,
        "formula":               [formula_of(mid) for mid in material_ids],
        "z_site_element":        [id_to_z.get(mid, "?") for mid in material_ids],
        "z_atomic_number":       [id_to_zan.get(mid, 0) for mid in material_ids],
        "pearson_r":             m1_pearson,
        "rmse":                  m2_rmse,
        "mae":                   m3_mae,
        "jsd":                   m5_jsd,
        "fermi_pearson_2eV":     m9_fp,
        "fermi_mae_1eV":         m7_fmae1,
        "d_band_error_eV":       m10_dband,
        "spin_pol_error":        m13_spinpol,
        "P_pred":                P_pred,
        "P_true":                P_true,
        "electronic_type":       electronic_types,
        "true_gapped":           true_gapped,
        "pred_gapped":           (y_pred_up + y_pred_down)[:, W1EV].min(axis=1) < GAP_THRESHOLD_PRIMARY,
        "wasserstein_eV":        m_wass_eval,
        "peak_pos_mae_up_eV":    m_peak_up,
        "peak_pos_mae_down_eV":  m_peak_down,
    })

    # ── Outlier identification ─────────────────────────────────────────────────
    outliers = identify_outliers(
        material_ids, m1_pearson, m2_rmse, m3_mae, m9_fp, m7_fmae1,
        m10_dband, m13_spinpol, P_pred, P_true, electronic_types,
        id_to_z, group_df)

    # ── Per-energy MAE for Figure 4 ───────────────────────────────────────────
    per_energy_mae = compute_per_energy_mae(predictions, targets)

    # ── Write all output files ─────────────────────────────────────────────────
    def save_json(obj, name):
        path = out_dir / name
        with open(path, "w") as f:
            json.dump(to_json(obj), f, indent=2)
        return path

    paths = {
        "metrics_summary":          save_json(metrics_summary,       "metrics_summary.json"),
        "supplementary_metrics":    save_json(supplementary,         "supplementary_metrics.json"),
        "per_channel_metrics":      save_json(per_channel,           "per_channel_metrics.json"),
        "electronic_type_metrics":  save_json(electronic_type_metrics, "electronic_type_metrics.json"),
        "outlier_samples":          save_json(outliers,              "outlier_samples.json"),
    }

    per_sample_df.to_csv(out_dir / "per_sample_metrics.csv", index=False)
    paths["per_sample_metrics"] = out_dir / "per_sample_metrics.csv"

    if not z_df.empty:
        z_df.drop(columns=["soc_flag"], errors="ignore").to_csv(
            out_dir / "per_z_element_metrics.csv", index=False)
        paths["per_z_element"] = out_dir / "per_z_element_metrics.csv"
        if soc_suspected:
            print("  [FLAG] SOC effect suspected — Pearson r decreases with Z atomic number")

    np.save(out_dir / "per_energy_mae.npy", per_energy_mae)
    paths["per_energy_mae"] = out_dir / "per_energy_mae.npy"

    # ── Print summary ─────────────────────────────────────────────────────────
    m = metrics_summary["metrics"]
    print(f"\n  ── Results (Exp {args.experiment}, fold {args.fold}) ──────────────────")
    print(f"  Pearson r    [partial]:  {m['pearson_r']['mean']:.4f} ± {m['pearson_r']['std']:.4f}")
    print(f"  RMSE         [partial]:  {m['rmse']['mean']:.4f} ± {m['rmse']['std']:.4f}")
    print(f"  MAE          [indep ✓]: {m['mae']['mean']:.4f} ± {m['mae']['std']:.4f}")
    print(f"  R²           [indep ✓]: {m['r2']['value']:.4f}")
    print(f"  JSD          [indep ✓]: {m['jsd']['mean']:.4f} ± {m['jsd']['std']:.4f}")
    print(f"  Fermi-MAE±1eV[indep ✓]: {m['fermi_mae_1eV']['mean']:.4f} ± {m['fermi_mae_1eV']['std']:.4f}")
    print(f"  Fermi-Pearson[indep ✓]: {m['fermi_pearson_2eV']['mean']:.4f} ± {m['fermi_pearson_2eV']['std']:.4f}")
    print(f"  D-band MAE   [indep ✓]: {m['d_band_mae_eV']['mean']:.4f} eV ± {m['d_band_mae_eV']['std']:.4f}")
    print(f"  Spin-pol err [indep ✓]: {m['spin_pol_error']['mean']:.4f} ± {m['spin_pol_error']['std']:.4f}")
    print(f"  Band gap acc [indep ✓]: {m['band_gap']['accuracy']*100:.1f}%  (F1={m['band_gap']['f1']:.3f})")
    ec = metrics_summary["electronic_type_counts"]
    print(f"  Electronic types: metallic={ec['metallic']}, half_metallic={ec['half_metallic']}, semiconducting={ec['semiconducting']}")
    print(f"  ── Extended metrics ──────────────────────────────────────────────")
    print(f"  Wasserstein  [indep ✓]: {m['wasserstein_eV']['mean']:.4f} ± {m['wasserstein_eV']['std']:.4f} eV")
    print(f"  Peak pos err [indep ✓]: up={m['peak_pos_mae_up_eV']['mean']:.4f} eV  down={m['peak_pos_mae_down_eV']['mean']:.4f} eV")
    bte = m['bte_weighted']
    print(f"  BTE-Pearson  [part. ∂]: {bte['bte_pearson_r']:.4f}  (BTE-MAE={bte['bte_weighted_mae']:.4f})")
    hm_rec = m['spin_pol_halfmetal_recall']
    if hm_rec['n_half_metallic'] > 0:
        print(f"  HM recall    [indep ✓]: {hm_rec['recall']:.3f}  (F1={hm_rec['f1']:.3f}, n={hm_rec['n_half_metallic']})")
    bias = m['integrated_dos_bias']
    print(f"  Int DOS bias [indep ✓]: up={bias['bias_up']:+.4f}  down={bias['bias_down']:+.4f}  total={bias['bias_total']:+.4f}")

    print(f"\n  Output files written to: {out_dir}")
    for k, p in paths.items():
        print(f"    ✓ {Path(p).name}")

    print()


if __name__ == "__main__":
    main()
