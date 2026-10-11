"""figkit.qa -- automatic checks a figure must pass before it is shown to anyone.

Usage (inside a figure script, before saving):
    from figkit import qa;  qa.check(fig, name='fig1')       # raises on overlaps / out-of-bounds text, reports widths
CLI (after saving):  python3 qa.py <figures_dir>          # prints width of every PDF and builds a print-size contact sheet
"""
import sys
from pathlib import Path


def _tick_label_filter(fig):
    """Tick labels outside an axis' view interval are never drawn; collect them so they are skipped."""
    hidden, shown = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                ids = {id(tick.label1), id(tick.label2)}
                (shown if lo - 1e-12 <= tick.get_loc() <= hi + 1e-12 else hidden).update(ids)
    return hidden - shown


def _text_boxes(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    hidden = _tick_label_filter(fig)
    out, seen = [], set()
    for t in fig.findobj(match=lambda a: hasattr(a, 'get_text') and hasattr(a, 'get_window_extent')):
        if id(t) in seen or id(t) in hidden:
            continue
        seen.add(id(t))
        try:
            s = t.get_text()
        except Exception:
            continue
        if not s or not s.strip() or not t.get_visible():
            continue
        bb = t.get_window_extent(renderer=r)
        if bb.width > 0 and bb.height > 0:
            out.append((s.replace('\n', ' ')[:40], bb))
    return out, r


MIN_PT = 5.0                                                          # smallest text allowed at print size (v10)


def check(fig, name='figure', tol_px=1.0, ignore=()):
    """Fail on text overlapping other text, or text leaving the figure. Returns list of problems (empty = pass)."""
    boxes, _ = _text_boxes(fig)
    W, H = fig.bbox.width, fig.bbox.height
    problems = []
    for t in fig.findobj(lambda o: hasattr(o, 'get_fontsize') and hasattr(o, 'get_text')):   # minimum print size (v10)
        try:
            if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_PT - 1e-6:
                problems.append(f'font {t.get_fontsize():.1f} pt < {MIN_PT} pt: "{t.get_text()[:40]}"')
        except Exception:
            pass
    for s, bb in boxes:
        if bb.x0 < -tol_px or bb.y0 < -tol_px or bb.x1 > W + tol_px or bb.y1 > H + tol_px:
            problems.append(f'out of bounds: "{s}"')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (s1, a), (s2, b) = boxes[i], boxes[j]
            if any(k in s1 or k in s2 for k in ignore):
                continue
            if s1 == s2 and abs(a.x0 - b.x0) < 1 and abs(a.y0 - b.y0) < 1:   # same text drawn twice (legend internals)
                continue
            ox = min(a.x1, b.x1) - max(a.x0, b.x0); oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > tol_px and oy > tol_px:
                problems.append(f'overlap: "{s1}" x "{s2}"')
    print(f'[qa] {name}: {len(boxes)} text items, {len(problems)} problems')
    for p in problems:
        print('   ', p)
    return problems


def sheet(fig_dir):
    """Report each PDF's width (mm) and write a print-size (100 dpi ~ reading size) contact PNG of all figure PNGs."""
    import subprocess
    from PIL import Image
    d = Path(fig_dir)
    for pdf in sorted(d.glob('*.pdf')):
        info = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True).stdout
        line = [l for l in info.splitlines() if l.startswith('Page size')]
        if line:
            w_pt = float(line[0].split()[2])
            print(f'{pdf.name:40s} width {w_pt / 72 * 25.4:6.1f} mm')
    pngs = sorted(p for p in d.glob('*.png') if not p.name.startswith('_qa'))
    if pngs:
        ims = []
        for p in pngs:
            im = Image.open(p); scale = 100 / 600
            ims.append(im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale)))))
        W = max(i.width for i in ims); H = sum(i.height + 10 for i in ims)
        out = Image.new('RGB', (W, H), 'white'); y = 0
        for i in ims:
            out.paste(i, (0, y)); y += i.height + 10
        out.save(d / '_qa_contact_100dpi.png'); print('contact sheet:', d / '_qa_contact_100dpi.png')


if __name__ == '__main__':
    sheet(sys.argv[1] if len(sys.argv) > 1 else '.')
