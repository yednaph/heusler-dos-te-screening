"""figkit.style -- design tokens as code for every figure of the Heusler d-DOS thermoelectric paper.

Tokens are taken from the group presentation (New_Hpc_instruction_Manual/build_ppt.py L36-44 and the v4 deck XML)
and translated to print: same colour semantics, Arial-metric font, 89/183 mm widths, flat shapes.
Import, never redefine:   from figkit import style as S;  S.apply()
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- geometry (mm -> inch)
MM = 1 / 25.4
W1, W15, W2 = 89 * MM, 136 * MM, 183 * MM          # single, 1.5, double column (npj / Nature)
ALLOWED_WIDTHS_MM = (89, 136, 183)

# ---------------------------------------------------------------- colour ROLES (deck semantics; learned once, used everywhere)
NAVY = '#1F4E79'      # structure, the model, "what we changed"
BLUE = '#2E6CA4'      # the screen
TEAL = '#1B7F79'      # first-principles (DFT) stage, plain-language callouts
ORANGE = '#E08200'    # a problem / "what we saw"
GREEN = '#2E7D32'     # an outcome / "why it is better" / certified
PURPLE = '#6A4C93'    # caveats, outlook
SUPERSEDED = '#8A96A3'  # superseded method (v1 July)
ARROW = '#5B6672'     # neutral connectors
# tints (card fills) -- one per role
TINT = {NAVY: '#DDEAF6', BLUE: '#DDEAF6', TEAL: '#D5ECEA', ORANGE: '#FCE8C8', GREEN: '#D3ECD4', PURPLE: '#E6E0EF',
        SUPERSEDED: '#EDF1F6'}
CARD = '#EDF1F6'      # neutral card fill
RULE = '#CDD7E3'      # hairlines, separators

# ---------------------------------------------------------------- DATA encodings (pass palette_check within each family)
SET_COLOR = {'training': '#C87200', 'prediction': '#2E6DA4'}          # same pair as code/paper/figstyle.py
SET_MARKER = {'training': '^', 'prediction': 'o'}
SET_LABEL = {'training': 'training set', 'prediction': 'prediction set'}
SET_SOURCE = {'training': 'DFT d-DOS', 'prediction': 'ML-predicted d-DOS'}
CLASS_COLOR = {'gapped': '#008B7A', 'narrow': '#E0A100', 'half_metal': '#6A4C93', 'metal': '#B8C2CC'}
CLASS_LABEL = {'gapped': 'gapped', 'narrow': r'narrow ($E_g<0.136$ eV)', 'half_metal': 'half-metal', 'metal': 'metal'}
CLASS_ORDER = ['gapped', 'narrow', 'half_metal', 'metal']
MUTED = '#C9D2DC'     # "below threshold", de-emphasised data

# ---------------------------------------------------------------- ink and type scale (pt)
INK, INK2 = '#1A1A1A', '#5B6672'
FONT = ['Liberation Sans', 'Nimbus Sans', 'DejaVu Sans']
FS = dict(body=6.5, small=5.8, label=7.0, heading=7.2, count=9.0, panel=9.0, badge=6.2)
FS_MIN = 5.0          # v10: nothing prints below 5 pt (figures are placed at their drawn 183/89 mm width)
LW = dict(axis=0.6, card=0.8, accent=1.2, arrow=0.9, hair=0.45)


def apply():
    """One restrained rcParams set for every figure."""
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': FONT, 'font.size': FS['body'],
        'axes.titlesize': FS['heading'], 'axes.titleweight': 'bold', 'axes.labelsize': FS['label'],
        'legend.fontsize': FS['small'] + 0.4, 'xtick.labelsize': FS['body'] - 0.2, 'ytick.labelsize': FS['body'] - 0.2,
        'mathtext.default': 'regular', 'axes.linewidth': LW['axis'], 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': LW['axis'], 'ytick.major.width': LW['axis'],
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'legend.frameon': False, 'text.color': INK, 'lines.linewidth': 1.0,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600, 'figure.dpi': 150})


def panel(ax, letter, x=-0.12, y=1.04):
    """Bold lower-case panel letter (Nature style); never a title inside the image."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=FS['panel'], fontweight='bold', va='bottom', ha='left', color=INK)


def save(fig, path, width_mm=None):
    """Vector PDF + 600 dpi PNG. Enforces a journal width when width_mm is given."""
    path = Path(path)
    if width_mm is not None:
        w = fig.get_figwidth() / MM
        assert any(abs(w - a) < 0.6 for a in ALLOWED_WIDTHS_MM), f'figure width {w:.1f} mm is not one of {ALLOWED_WIDTHS_MM}'
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=600)
    plt.close(fig)
