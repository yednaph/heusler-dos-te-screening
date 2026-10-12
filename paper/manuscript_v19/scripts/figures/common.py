"""Shared style and data access for the manuscript figures and tables.

Every prototype reads EXISTING result files only (no model inference, no DFT, no re-analysis).
Sizes follow npj Computational Materials guidance: <=180 mm wide, sans-serif (Arial-metric
Liberation Sans), 7 pt text, vector PDF output.
"""
import os as _os_htr  # release v1: project paths come from the environment
_HTR = _os_htr.environ.get('HEUSLER_TE_ROOT', _os_htr.path.abspath(_os_htr.path.join(_os_htr.path.dirname(__file__), '../../../..')))
_HTE = _os_htr.environ.get('HEUSLER_TE_EXPORT', _os_htr.path.join(_HTR, 'DOS_EXPORT'))
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- paths (read-only sources)
REPO = Path(_HTR)
EXPORT = Path(_HTE)
ANALYSIS = REPO / 'results/paper_analysis'
ROOT = Path(__file__).resolve().parents[2]          # paper/manuscript_v19/
OUT = ROOT / 'figures'
DATA = OUT / 'data'
TABLES = ROOT / 'tables'
# Records of the earlier draft that the current scripts read or cross-check against; shipped in paper/manuscript_v1.
RECORDS = REPO / 'paper/manuscript_v1/figures/data'
for _p in (OUT, DATA, TABLES): _p.mkdir(exist_ok=True)

FAMILIES = ['full_cubic', 'full_tetra', 'half_cubic', 'half_tetra', 'inverse_cubic', 'inverse_tetra']
FAM_LABEL = {f: f.replace('_', ' ') for f in FAMILIES}
PF_BAR = 1.0857214253e12            # declared PF/tau bar (W m^-1 K^-2 s^-1), build_paper_dataset.py
MM = 1 / 25.4                       # inches per mm
W1, W2 = 88 * MM, 180 * MM          # single- and double-column widths

# ---------------------------------------------------------------- colours (checked with palette_check.py)
# DFT electronic class: every pair passes CVD dE>=8 and normal-vision dE>=15.
CLASS_ORDER = ['metal', 'half_metal', 'narrow', 'gapped']
CLASS_COLOR = {'metal': '#9aa6b2', 'half_metal': '#6d28d9', 'narrow': '#f59e0b', 'gapped': '#c8102e'}
CLASS_LABEL = {'metal': 'metal', 'half_metal': 'half-metal', 'narrow': r'narrow ($E_g<0.136$ eV)', 'gapped': 'gapped'}
# Two-series identity (pools, model vs baseline): blue / orange, plus ink greys.
POOL_COLOR = {'prediction': '#1f4fbf', 'training': '#d9480f'}
POOL_LABEL = {'prediction': 'Prediction track · ML d-DOS', 'training': 'Reference track · DFT d-DOS'}
POOL_SHORT = {'prediction': 'prediction track', 'training': 'reference track'}
INK, INK2, GRID = '#111111', '#4b4f56', '#e3e5e8'
MUTED = '#c9ced4'


def style():
    """Apply one restrained style to every prototype."""
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Nimbus Sans', 'DejaVu Sans'],
        'font.size': 7, 'axes.titlesize': 7.2, 'axes.labelsize': 7, 'legend.fontsize': 6.3,
        'xtick.labelsize': 6.3, 'ytick.labelsize': 6.3, 'axes.titleweight': 'bold', 'mathtext.default': 'regular',
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
        'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.4,
        'lines.linewidth': 1.0, 'legend.frameon': False, 'legend.handlelength': 1.2,
        'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.dpi': 150,
    })


def panel(ax, letter, x=-0.16, y=1.04):
    """Bold lower-case panel label, Nature style."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')


def track_header(ax, t, y=1.12, x=0.0):
    """Coloured track tag above a panel (identity by colour + text, never colour alone)."""
    ax.text(x, y, POOL_LABEL[t], transform=ax.transAxes, fontsize=6.8, fontweight='bold', color='white', va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.15', fc=POOL_COLOR[t], ec='none'))


def save(fig, name):
    """Write vector PDF (manuscript) and 300-dpi PNG (inspection)."""
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- data loaders (existing files)
def dft_runs():
    """The 3,625 QE/BoltzTraP2 paper runs with screen columns (results/paper_analysis/paper_dataset.pkl)."""
    return pd.read_pickle(ANALYSIS / 'paper_dataset.pkl')


def per_sample():
    """Per-material held-out ML metrics, all 6 families x 5 folds (DOS_EXPORT, best.pt checkpoints)."""
    frames = []
    for f in sorted(glob.glob(str(EXPORT / 'metadata/metrics/*/fold*/per_sample_metrics.csv'))):
        df = pd.read_csv(f)
        df['family'] = Path(f).parts[-3]
        df['fold'] = Path(f).parts[-2]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def screen_pools():
    """Full screened pools: prediction (ML d-DOS) and training (DFT d-DOS), one row per DXMag entry."""
    p = pd.read_csv(REPO / 'results/screening/rescreen_gapfix_ALL_annotated.csv',
                    usecols=['UUID', 'family', 'etype', 'gap_eV', 'gap_tier', 'A_seebeck_uVK', 'ML_hull'])
    t = pd.read_csv(REPO / 'results/screening/training_pool_screen_gapfix.csv',
                    usecols=['UUID', 'family', 'etype', 'gap_eV', 'gap_tier', 'A_seebeck_uVK', 'ML_hull'])
    p['pool'], t['pool'] = 'prediction', 'training'
    return pd.concat([p, t], ignore_index=True)


E_GRID = np.linspace(-10, 10, 2001)  # eV relative to E_F, code/dos_dataset.py

# ---------------------------------------------------------------- v4: inherit the approved figkit tokens (Fig. 1 language)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from figkit import style as _S          # noqa: E402
CLASS_COLOR = dict(_S.CLASS_COLOR); CLASS_LABEL = dict(_S.CLASS_LABEL); CLASS_ORDER = ['metal', 'half_metal', 'narrow', 'gapped']
POOL_COLOR = dict(_S.SET_COLOR)
POOL_LABEL = {'prediction': 'prediction set · ML-predicted d-DOS', 'training': 'training set · DFT d-DOS'}
POOL_SHORT = {'prediction': 'prediction set', 'training': 'training set'}
INK, INK2, MUTED = _S.INK, _S.INK2, _S.MUTED
W1, W2 = _S.W1, _S.W2


def style():
    _S.apply()
