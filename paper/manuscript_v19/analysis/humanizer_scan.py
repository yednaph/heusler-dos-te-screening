"""Count formulaic phrasing patterns in the manuscript sources (read-only style diagnostic; needs scripts/v14_src, not shipped)."""
import re, sys
from pathlib import Path
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else 'scripts/v14_src')
PAT = {
 'rather than': r'\brather than\b', 'not X but Y': r'\bnot\b[^.;]{1,60}\bbut\b', 'not only': r'\bnot only\b',
 'robust': r'\brobust', 'crucial/pivotal/key role': r'\b(crucial|pivotal|vital|key role)\b', 'underscore/highlight': r'\b(underscor|highlight)',
 'therefore': r'\btherefore\b', 'thus': r'\bthus\b', 'This answers': r'This answers', 'qualifications apply': r'qualifications? apply',
 'serves as/stands as': r'\b(serves|stands|functions) as\b', 'leverage/delve/landscape': r'\b(leverag|delve|landscape|realm|multifaceted)',
 'notably/additionally': r'\b(notably|additionally|moreover|furthermore)\b', 'In brief': r'\bIn brief\b',
}
tot = {k: 0 for k in PAT}
for f in sorted(SRC.glob('0*.tex')):
    t = re.sub(r'%.*', '', f.read_text()); t = re.sub(r'\s+', ' ', t)
    row = {k: len(re.findall(p, t, re.I)) for k, p in PAT.items()}
    for k in tot: tot[k] += row[k]
    words = len(re.sub(r'\\[a-zA-Z]+|[{}$]', ' ', t).split())
    print(f'{f.name:22s} words {words:5d} ' + ' '.join(f'{k}={v}' for k, v in row.items() if v))
print('TOTAL', ' '.join(f'{k}={v}' for k, v in tot.items()))
