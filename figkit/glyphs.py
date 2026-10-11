"""figkit.glyphs -- the diagram vocabulary of the group presentation, for matplotlib (vector, editable source).

Every glyph draws in a 'canvas' axis whose data coordinates are millimetres (see canvas()), so layouts are written on a
print-size grid. Each object type has its own glyph -- do not draw everything as the same rectangle:
    dataset      -> stat_tile()      big count + short label
    model/stage  -> card()           tinted card, coloured outline, numbered badge, heading in role colour
    calculation  -> pills()          chained rounded pills (SCF -> NSCF -> ...)
    descriptor   -> chip()           small formula chip
    decision     -> badge()          numbered round badge in the role colour
    selection    -> funnel_bar()     solid bar, length = f(count), big white count
    loop         -> loop_arc()       one dashed arc, teal
    connector    -> chevron() / arrow()  light, colour-matched to the stage it leaves
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
from . import style as S


def canvas(width_mm, height_mm):
    """Figure + full-bleed axis with millimetre data coordinates (origin bottom-left)."""
    fig = plt.figure(figsize=(width_mm * S.MM, height_mm * S.MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, width_mm); ax.set_ylim(0, height_mm); ax.axis('off')
    return fig, ax


def text(ax, x, y, s, fs=None, w='normal', c=None, ha='center', va='center', z=6, **kw):
    return ax.text(x, y, s, fontsize=max(fs or S.FS['body'], S.FS_MIN), fontweight=w, color=c or S.INK, ha=ha, va=va, zorder=z,
                   linespacing=1.2, **kw)


def card(ax, x, y, w, h, role, heading=None, body=None, number=None, fill=None, lw=None, fs_head=None, fs_body=None,
         head_pad=2.2, z=2):
    """The deck's card: tinted fill, outline in the role colour, optional numbered badge and heading in role colour."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.0', fc=fill or S.TINT.get(role, S.CARD),
                                ec=role, lw=lw or S.LW['card'], zorder=z))
    xh = x + 2.2
    if number is not None:
        badge(ax, x + 3.2, y + h - 3.2, number, role, r=2.0)
        xh = x + 6.2
    if heading:
        text(ax, xh, y + h - head_pad - 1.0, heading, fs=fs_head or S.FS['heading'], w='bold', c=role, ha='left', va='center')
    if body:
        text(ax, x + 2.2, y + h - head_pad - 4.4, body, fs=fs_body or S.FS['small'] + 0.3, ha='left', va='top')


def badge(ax, x, y, s, role, r=2.0, fs=None, z=8):
    ax.add_patch(Circle((x, y), r, fc=role, ec='white', lw=0.5, zorder=z))
    text(ax, x, y - 0.05, str(s), fs=fs or S.FS['badge'], w='bold', c='white', z=z + 1)


def tag(ax, x, y, s, role, w=5.6, h=3.4, fs=5.2, z=8):
    """Rounded-square tag for workflow revisions (R1-R7), distinct from round stage badges (v10)."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.6', fc=role, ec='white', lw=0.5, zorder=z))
    text(ax, x, y - 0.05, s, fs=fs, w='bold', c='white', z=z + 1)


def stat_tile(ax, x, y, w, h, count, label, role, fs_count=None, fs_label=None, z=3):
    """Solid tile with a big white count over a short label (deck slides 1 and 66)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.0', fc=role, ec='none', zorder=z))
    text(ax, x + w / 2, y + h * 0.62, count, fs=fs_count or S.FS['count'], w='bold', c='white', z=z + 1)
    text(ax, x + w / 2, y + h * 0.25, label, fs=fs_label or S.FS['small'], c='white', z=z + 1)


def pills(ax, x, y, labels, role, h=4.4, gap=1.6, fs=None, widths=None, z=4):
    """Chained calculation steps: rounded pills joined by small chevrons. Returns right edge x."""
    cx = x
    for i, lab in enumerate(labels):
        w = widths[i] if widths else max(8.0, 1.35 * len(lab) + 3)
        ax.add_patch(FancyBboxPatch((cx, y), w, h, boxstyle=f'round,pad=0,rounding_size={h / 2}', fc='white', ec=role,
                                    lw=S.LW['card'], zorder=z))
        text(ax, cx + w / 2, y + h / 2, lab, fs=fs or S.FS['small'], c=role, w='bold', z=z + 1)
        cx += w
        if i < len(labels) - 1:
            chevron(ax, cx + gap / 2, y + h / 2, role, size=1.1, direction='right')
            cx += gap
    return cx


def chip(ax, x, y, s, role, w=None, h=4.2, fs=None, z=4):
    w = w or max(7.0, 1.25 * len(s) + 3)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.8', fc='white', ec=role, lw=0.6, zorder=z))
    text(ax, x + w / 2, y + h / 2, s, fs=fs or S.FS['small'], c=role, z=z + 1)
    return x + w


def chevron(ax, x, y, color, size=1.4, direction='right', z=5):
    """Small solid chevron connector (deck: colour of the stage it leaves)."""
    s = size
    pts = {'right': [(x - s, y + s), (x + s * 0.6, y), (x - s, y - s), (x - s * 0.3, y)],
           'down': [(x - s, y + s), (x, y - s * 0.6), (x + s, y + s), (x, y + s * 0.3)],
           'left': [(x + s, y + s), (x - s * 0.6, y), (x + s, y - s), (x + s * 0.3, y)]}[direction]
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec='none', zorder=z))


def arrow(ax, x0, y0, x1, y1, color=None, lw=None, ls='-', rad=0.0, z=4, head=6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=head, lw=lw or S.LW['arrow'],
                                 color=color or S.ARROW, ls=ls, connectionstyle=f'arc3,rad={rad}', zorder=z,
                                 shrinkA=0, shrinkB=0))


def loop_arc(ax, x0, y0, x1, y1, label=None, rad=-0.35, color=None):
    """The single feedback loop of a diagram: dashed, teal, labelled once."""
    arrow(ax, x0, y0, x1, y1, color=color or S.TEAL, lw=1.0, ls=(0, (3.2, 1.6)), rad=rad, head=7)
    if label:
        mx, my = (x0 + x1) / 2, max(y0, y1)
        text(ax, mx, my + 4.0, label, fs=S.FS['small'] + 0.3, c=color or S.TEAL, w='bold')


def bus_branch(ax, parent_xy, children_xy, color=None, lw=None):
    """Thin grey tree: parent down to a horizontal bus, bus down to each child (deck slide 20)."""
    color = color or S.ARROW; lw = lw or S.LW['hair'] + 0.2
    px, py = parent_xy
    ybus = (py + max(c[1] for c in children_xy)) / 2
    ax.plot([px, px], [py, ybus], color=color, lw=lw, zorder=3, solid_capstyle='butt')
    xs = [c[0] for c in children_xy] + [px]
    ax.plot([min(xs), max(xs)], [ybus, ybus], color=color, lw=lw, zorder=3)
    for cx, cy in children_xy:
        ax.plot([cx, cx], [ybus, cy + 0.9], color=color, lw=lw, zorder=3)
        chevron(ax, cx, cy + 0.9, color, size=0.8, direction='down')


def node(ax, x, y, w, h, label, role, fs=None, z=5):
    """Solid rounded node with white text (tree nodes, version tags)."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.8', fc=role, ec='none', zorder=z))
    text(ax, x, y, label, fs=fs or S.FS['small'], c='white', w='bold', z=z + 1)


def funnel_bar(ax, x_anchor, y, length, h, count, label, color, side='right', fs_count=None, fs_label=None, z=3):
    """One funnel step. side='right' grows rightwards from x_anchor; side='left' grows leftwards (mirrored funnel)."""
    x0 = x_anchor if side == 'right' else x_anchor - length
    ax.add_patch(FancyBboxPatch((x0, y), length, h, boxstyle='round,pad=0,rounding_size=0.6', fc=color, ec='none', zorder=z))
    inside = length > 15
    if side == 'right':
        tx, ha = (x0 + length - 1.4, 'right') if inside else (x0 + length + 1.2, 'left')
    else:
        tx, ha = (x0 + 1.4, 'left') if inside else (x0 - 1.2, 'right')
    text(ax, tx, y + h / 2, count, fs=fs_count or S.FS['label'] + 0.6, w='bold', c='white' if inside else color, ha=ha, z=z + 1)


def mini_spectrum(fig, rect_mm, e, dft, pred, total_mm, color_pred=None):
    """Inset of a REAL spectrum pair (DFT fill vs prediction line) inside a card. rect_mm = (x, y, w, h) in canvas mm."""
    W, H = total_mm
    x, y, w, h = rect_mm
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.fill_between(e, 0, dft, color='#C9D2DC', lw=0)
    ax.plot(e, pred, color=color_pred or S.SET_COLOR['prediction'], lw=0.7)
    ax.axvline(0, color=S.INK2, lw=0.4, ls=':')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.spines['bottom'].set_visible(True); ax.spines['bottom'].set_linewidth(0.4)
    ax.patch.set_alpha(0)
    return ax
