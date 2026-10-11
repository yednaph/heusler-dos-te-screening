"""Assemble sections/*.tex from scripts/v9_src/*.tex (text with %%FLOAT:<label>%% placeholders) and scripts/v9_floats.json.
Build helper; the assembled sections/*.tex are what main.tex compiles."""
import json, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
fl = json.load(open(HERE / 'v9_floats.json')); used = []
out = HERE.parent / 'sections'
for old in out.glob('0*.tex'):
    old.unlink()                                                      # drop the v8 section files copied into this package
for src in sorted((HERE / 'v9_src').glob('0*.tex')):
    s = re.sub(r'%%FLOAT:([^%]+)%%', lambda m: (used.append(m.group(1)), fl[m.group(1)])[1], src.read_text())
    (out / src.name).write_text(s)
print('floats used', len(used), 'unused', sorted(set(fl) - set(used)), 'dup', sorted({u for u in used if used.count(u) > 1}))
