"""Renumber the SI in order of first citation in the main text (Nature-family requirement).
Splits si/supplementary.tex into its 6 notes, 16 tables and 15 figures, reassembles them in first-citation order, and remaps every
textual reference (Supplementary Note n, Table Sn, Fig. Sn) in the manuscript sources, captions, tables and SI in a single regex pass.
Float numbers themselves come from LaTeX counters, so they follow the new physical order automatically."""
import re
from pathlib import Path
SI = Path('si/supplementary.tex'); L = SI.read_text().split('\n')          # 1-based line n is L[n-1]
NOTES = [(29, 54), (55, 69), (70, 102), (103, 189), (190, 206), (207, 214)]
TABLES = [(217, 220), (222, 225), (227, 233), (235, 240), (242, 249), (251, 256), (259, 275), (277, 282), (285, 288), (290, 296),
          (298, 302), (304, 307), (311, 316), (318, 326), (329, 330), (332, 344)]
FIGS = [(348, 352), (354, 359), (362, 367), (369, 373), (375, 379), (382, 388), (390, 394), (396, 406), (408, 417), (419, 431),
        (433, 441), (443, 454), (456, 469), (472, 478), (480, 485)]
# first-citation order in the main text (computed from the assembled v19 sections; old numbers)
ORDER = {'Note': [2, 1, 4, 3, 5, 6], 'Table': [8, 9, 15, 7, 11, 3, 12, 14, 4, 13, 16, 1, 2, 5, 10, 6],
         'Fig': [8, 7, 6, 9, 10, 3, 4, 12, 1, 2, 14, 11, 15, 5, 13]}
NEW = {k: {old: i + 1 for i, old in enumerate(v)} for k, v in ORDER.items()}   # old number -> new number
for k, n in (('Note', 6), ('Table', 16), ('Fig', 15)):
    assert sorted(ORDER[k]) == list(range(1, n + 1)), k                     # every item exactly once

# ---- coverage: every non-blank line of the SI body belongs to a block or is a known separator
covered = set()
for blocks in (NOTES, TABLES, FIGS):
    for a, b in blocks: covered |= set(range(a, b + 1))
SEPARATORS = {215: r'\section*{Supplementary Tables}', 309: r'\clearpage', 346: r'\section*{Supplementary Figures}'}
for n in range(29, 487):
    if n in covered or not L[n - 1].strip(): continue
    assert n in SEPARATORS and L[n - 1].strip() == SEPARATORS[n], (n, L[n - 1][:80])
assert L[486].strip() == r'\end{document}' and L[28].startswith(r'\section*{Supplementary Note 1')
block = lambda ab: '\n'.join(L[ab[0] - 1:ab[1]])
body = ([block(NOTES[o - 1]) for o in ORDER['Note']] + [SEPARATORS[215]] + [block(TABLES[o - 1]) for o in ORDER['Table']]
        + [SEPARATORS[346]] + [block(FIGS[o - 1]) for o in ORDER['Fig']])
new_si = '\n'.join(L[:28]) + '\n' + '\n\n'.join(body) + '\n\n' + '\n'.join(L[486:])

# ---- remap textual references in one pass
KIND = r'(?P<kind>Supplementary\s*~?\s*Notes?|Notes?|Supplementary\s*~?\s*Tables?|Tables?|Supplementary\s*~?\s*Figs?\.|Figs?\.)'
NUMS = r'(?P<nums>(?:S?\d+)(?:\s*(?:,|and)\s*~?\s*S?\d+)*)'
PAT = re.compile(KIND + r'(?P<sep>\s*~?\s*)' + NUMS)
def remap(text: str) -> str:
    def sub(m):
        kind = m.group('kind'); k = 'Note' if 'Note' in kind else 'Table' if 'Table' in kind else 'Fig'
        nums = m.group('nums')
        if k in ('Table', 'Fig') and not re.match(r'S\d', nums): return m.group(0)   # main-text Table/Fig (no S prefix): untouched
        if k == 'Note' and re.match(r'S\d', nums): return m.group(0)
        def one(x):
            v = int(x.group(2)); return x.group(1) + str(NEW[k].get(v, v)) if v in NEW[k] else x.group(0)
        return kind + m.group('sep') + re.sub(r'(S?)(\d+)', one, nums)
    return PAT.sub(sub, text)
FILES = (sorted(Path('scripts/v14_src').glob('0*.tex')) + sorted(Path('review/captions').glob('v14_fig*.tex'))
         + sorted(Path('tables').glob('v14_tab_*.tex')) + sorted(Path('tables').glob('si_*.tex')) + [Path('tables/tab1_data_ml.tex'),
         Path('scripts/assemble_v14.py'), Path('literature/consolidate.py'), Path('analysis/verify_v14.py')])
changed = []
SI.write_text(remap(new_si)); changed.append(str(SI))
for f in FILES:
    t = f.read_text(); u = remap(t)
    if u != t: f.write_text(u); changed.append(str(f))
print('reordered SI and remapped references in:', *changed, sep='\n  ')
