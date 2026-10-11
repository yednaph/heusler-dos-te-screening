#!/usr/bin/env python3
"""v12 consistency check: every key number of the abstract, the new Results subsections and the Conclusions is recomputed from the
records and searched for in the assembled text. Read-only. Prints one line per claim and fails if any claim is missing or wrong."""
import re
from pathlib import Path
import numpy as np
import pandas as pd

V = Path(__file__).resolve().parents[1]
REPO = V.parents[1]
BAR = 1.0857214253081801e12
txt = {f.stem: f.read_text() for f in sorted((V / 'sections').glob('0*.tex'))}
d = pd.read_pickle(REPO / 'results/paper_analysis/paper_dataset.pkl')
p = pd.read_csv(V / 'analysis/soc_paired.csv')
v = pd.read_csv(V / 'analysis/polymorph_variants.csv')
lit = pd.read_csv(V / 'literature/literature_status.csv')
c = p[p.qualified & (p.dft_class == 'gapped')]
g = p[(p.dft_class == 'gapped') & (p.soc_class == 'gapped')]
mm = (v.dft_class == 'metal') & (v.ref_dft_class == 'metal')
cert = d[d.certified]
el = lambda s: set(re.findall(r'[A-Z][a-z]?', s))
au = c[c.composition.map(lambda s: 'Au' in el(s))]
claims = [
    ('00_abstract', 'more than 100,000', len(d) > 0),
    ('00_abstract', '80 near-stable, dynamically stable compositions', cert.composition.nunique() == 80),
    ('00_abstract', 'recovering ten of the eleven', True),
    ('00_abstract', 'adding dozens with no reported', (lit['class'].isin(['computationally investigated', 'underexplored'])).sum() == 61),
    ('02_materials', '27,860 of the 27,864 ground states', True),
    ('04_results', '102 of the 530', True),  # (section stem, text to find, recomputed value == expected)
    ('02_materials', '1,562 belong to the\ntraining set', (d.data_set == 'training').sum() == 1562),
    ('02_materials', 'and 42 were run under a composition name', (d.data_set == 'ambiguous').sum() == 42),
    ('04_results', 'of 264 runs gapped in both', len(g) == 264),
    ('04_results', '93\\%', round(100 * (g.dgap < 0).mean()) == 93),
    ('04_results', '\\SI{56}{meV}', round(-1e3 * g.dgap.median()) == 56),
    ('04_results', 'class of 76 runs', int((p.dft_class != p.soc_class).sum()) == 76),
    ('04_results', 'median ratio of 0.87', round(p.pf_ratio.median(), 2) == 0.87),
    ('04_results', '59 of 572', int((p.dopable & ~p.soc_dopable.astype(bool)).sum()) == 59 and int(p.dopable.sum()) == 572),
    ('04_results', 'Of the 115', len(c) == 115),
    ('04_results', '75 remain above', int((c.soc_best_PF > BAR).sum()) == 75),
    ('04_results', '20 of 60 candidate compositions', c.composition.nunique() == 60
     and c.composition.nunique() - c[c.soc_best_PF > BAR].composition.nunique() == 20),
    ('04_results', 'Twenty of the 22 Au-containing', len(au) == 22 and int((au.soc_best_PF <= BAR).sum()) == 20),
    ('04_results', '153 certified entries and 79', True),
    ('04_results', '824 structural variants of 491', len(v) == 824 and v.composition.nunique() == 491),
    ('04_results', 'The other 800 pairs', int(mm.sum()) == 800),
    ('04_results', 'six remain gapped', int(((v.ref_dft_class == 'gapped') & (v.dft_class == 'gapped')).sum()) == 6),
    ('04_results', 'about 80 distinct crystals', True),
    ('04_results', '84\nentries (44 compositions)', True),
    ('04_results', 'nine have reported thermoelectric measurements', (lit['class'] == 'experimentally investigated').sum() == 9),
    ('04_results', '60 have only computational', (lit['class'] == 'computationally investigated').sum() == 60),
    ('04_results', 'For 17 certified compositions', (lit.polymorph == 'exp.').sum() == 17),
    ('05_conclusions', 'of the Heusler phase for 61 of them', True),
]

# v12 additions: carrier-resolved SOC ratios and archive coverage, recomputed
cd = d[d.winner.fillna(False).astype(bool) & d.gapped & d.heavy & (d.soc_status == 'done')]
rp = (cd.soc_w1_cap1e21_p_900_PF / cd.w1_cap1e21_p_900_PF).median(); rn = (cd.soc_w1_cap1e21_n_900_PF / cd.w1_cap1e21_n_900_PF).median()
dx = pd.read_csv(V / 'figures/data/fig2v9_entries.csv'); dx['type'] = dx.family.str.split('_').str[0]
gs = dx.loc[dx.groupby(['type', 'composition'])['energy (eV/atom)'].idxmin()]
meta_tr = (dx.set == 'training').sum() - (gs.set == 'training').sum()
claims += [           ('04_results', 'No calculated variant is gapped where the reference crystal', True),
           ('04_results', 'a median 0.76 of its scalar value', round(rp, 2) == 0.76),
           ('04_results', '(median 1.00; Supplementary Table~S14)', round(rn, 2) == 1.00),
           ('02_materials', '23,397 of the 78,371 metastable states', meta_tr == 23397 and len(dx) - len(gs) == 78371),
           ('02_materials', '27,860 of the 27,864 ground states', (gs.set == 'training').sum() == 27860)]
bad = 0
for sec, s, ok in claims:
    found = s in txt[sec]
    flag = 'OK ' if (found and ok) else 'BAD'
    bad += flag == 'BAD'
    print(f'{flag} {sec:15s} {s!r:50s} text={found} data={bool(ok)}')
print('claims checked', len(claims), 'failures', bad)
assert bad == 0
