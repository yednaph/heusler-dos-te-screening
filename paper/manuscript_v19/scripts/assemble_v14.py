"""Assemble sections/*.tex for v14 from scripts/v14_src/*.tex (text with %%FLOAT:<label>%% placeholders).

Figure floats are built from the figure PDFs and the captions in review/captions/v14_fig<N>.tex (the \\caption{...} block is
extracted by brace matching; the label is assigned here so that numbering follows the order of first citation). Table floats
take theirs from the caption files named in TABLES below (release copy: captions are not shipped). Checks: every placeholder is defined, no float is used twice, every float is used.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC, OUT, CAP = HERE / 'v14_src', ROOT / 'sections', ROOT / 'review' / 'captions'

# label -> (caption file, figure pdf, single column?)
FIGURES = {
    'fig:workflow':   ('v14_fig1.tex', 'figures/v14_fig1_strategy.pdf', False),
    'fig:model':      ('v14_fig2.tex', 'figures/v14_fig2_model.pdf', False),
    'fig:fidelity':   ('v14_fig3.tex', 'figures/v14_fig3_fidelity.pdf', False),
    'fig:screening':  ('v14_fig5.tex', 'figures/v14_fig5_screening.pdf', False),
    'fig:cascade':    ('v14_fig6.tex', 'figures/v14_fig6_cascade.pdf', True),
    'fig:indicator':  ('v14_fig4.tex', 'figures/v14_fig4_indicator.pdf', False),
    'fig:soc':        ('v14_fig8.tex', 'figures/v14_fig8_soc.pdf', False),
    'fig:variants':   ('v14_fig9.tex', 'figures/v14_fig9_variants.pdf', None),   # None: decide from the PDF width
    'fig:candidates': ('v14_fig7.tex', 'figures/v14_fig7_candidates.pdf', False),
}

# label -> (wide?, table body file, caption file). The captions are part of the unpublished manuscript text and are not
# shipped with the code: they are read from review/captions/<caption file> of the manuscript sources.
TABLES = {
    'tab:populations': (True, 'v14_tab_populations.tex', 'v14_tab_populations_caption.tex'),
    'tab:pareto': (True, 'v14_tab_pareto.tex', 'v14_tab_pareto_caption.tex'),
    'tab:evidence': (True, 'v14_tab_evidence.tex', 'v14_tab_evidence_caption.tex'),
}


def caption_block(text: str) -> str:
    """Return the full \\caption{...} of a caption file (brace-matched)."""
    i = text.index(r'\caption{')
    depth, j = 0, i + len(r'\caption')
    while True:
        c = text[j]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
        j += 1


def pdf_width_mm(path: Path) -> float:
    m = re.search(rb'/MediaBox\s*\[\s*[-\d.]+\s+[-\d.]+\s+([-\d.]+)', path.read_bytes())
    return float(m.group(1)) * 25.4 / 72


def figure_float(label: str) -> str:
    cap_file, pdf, single = FIGURES[label]
    f = CAP / cap_file
    cap = caption_block(f.read_text()) if f.exists() else r'\caption{\textbf{[caption pending].}}'
    if not f.exists():
        print('WARNING: caption pending for', label)
    if single is None:
        single = pdf_width_mm(ROOT / pdf) < 120
    env, width = ('figure', r'\columnwidth') if single else ('figure*', r'\textwidth')
    return (f"\\begin{{{env}}}[t]\n\\centering\n\\includegraphics[width={width}]{{{pdf}}}\n{cap}\n"
            f"\\label{{{label}}}\n\\end{{{env}}}")


def table_float(label: str) -> str:
    wide, tex, cap_file = TABLES[label]
    f = CAP / cap_file                                                  # caption file of the manuscript sources
    cap = caption_block(f.read_text()) if f.exists() else r'\caption{\textbf{[caption pending].}}'
    env = 'table*' if wide else 'table'
    return (f"\\begin{{{env}}}[t]\n\\centering\n{cap}\n\\label{{{label}}}\n\\footnotesize\n"
            f"\\input{{tables/{tex}}}\n\\end{{{env}}}")


def main() -> None:
    used = []
    for old in OUT.glob('0*.tex'):
        old.unlink()
    for src in sorted(SRC.glob('0*.tex')):
        def repl(m):
            lab = m.group(1)
            used.append(lab)
            return figure_float(lab) if lab in FIGURES else table_float(lab)
        (OUT / src.name).write_text(re.sub(r'%%FLOAT:([^%]+)%%', repl, src.read_text()))
    defined = set(FIGURES) | set(TABLES)
    dup = sorted({u for u in used if used.count(u) > 1})
    missing = sorted(defined - set(used))
    assert not dup and not missing, (dup, missing)
    print('floats placed in order:', used)


if __name__ == '__main__':
    main()
