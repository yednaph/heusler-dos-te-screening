"""Known-compound check (v8; reads existing results only, no new calculation).

Which well-established half-Heusler thermoelectric compositions, as discussed in the reviews cited in the paper
(Zeier et al. 2016; Quinn & Bos 2021) and in Fu 2015 (NbFeSb) and Zhu 2019 (TaFeSb), were computed in the campaign,
and how did each fare under the certification rules?  The list is fixed in advance below and is not exhaustive; it is
a qualitative check that the rules recover known chemistry, not a numerical comparison with published values.
DXMag writes half-Heusler compositions in site order (e.g. CoTiSb for TiCoSb), so both spellings are searched.
Output: figures/data/known_hh_check.csv and tables/si_tab_known_hh.tex.
"""
import pandas as pd
from common import dft_runs, DATA, TABLES

KNOWN = {  # conventional name -> DXMag site-ordered composition string
    'TiNiSn': 'NiTiSn', 'ZrNiSn': 'NiZrSn', 'HfNiSn': 'NiHfSn',        # MNiSn (n-type workhorses)
    'TiCoSb': 'CoTiSb', 'ZrCoSb': 'CoZrSb', 'HfCoSb': 'CoHfSb',        # MCoSb (p-type)
    'VFeSb': 'FeVSb', 'NbFeSb': 'FeNbSb', 'TaFeSb': 'FeTaSb',          # XFeSb (p-type)
    'NbCoSn': 'CoNbSn', 'ZrCoBi': 'CoZrBi'}                            # further established compositions
BAR = 1.0857214253e12

d = dft_runs()
rows = []
for name, comp in KNOWN.items():
    g = d[d.composition == comp]
    if g.empty:
        rows.append(dict(name=name, composition=comp, computed=False)); continue
    best = g.sort_values('certified', ascending=False).iloc[0]           # report the certified entry if any
    if best.certified:
        reason = 'certified'
    elif best.dft_class != 'gapped':
        reason = f'not gapped ({best.dft_class})'
    elif not best.dopable:
        reason = 'not dopable'
    elif best.best_PF <= BAR:
        reason = 'below PF threshold'
    elif not best.hull < 0.10:
        reason = 'hull >= 0.10'
    elif best.soc_status == 'done' and best.soc_best_PF <= BAR:
        reason = f'SOC PF/$\\tau$ {best.soc_best_PF / 1e12:.3f} below threshold'
    else:
        reason = f'other (SOC {best.soc_status})'
    rows.append(dict(name=name, composition=comp, computed=True, sets='+'.join(sorted(set(g.data_set.str[0].str.upper()))),
                     certified_sets='+'.join(sorted(set(g[g.certified == True].data_set.str[0].str.upper()))) or '-',
                     qe_gap_eV=round(best.gap, 2), outcome=reason))
df = pd.DataFrame(rows); df.to_csv(DATA / 'known_hh_check.csv', index=False)
n_comp, n_cert = int(df.computed.sum()), int((df.outcome == 'certified').sum())
assert n_comp == 11 and n_cert == 10
lines = [f"{r['name']} & {r['composition']} & {r['sets']} & {r['certified_sets']} & {r['qe_gap_eV']:.2f} & {r['outcome']} \\\\" for _, r in df.iterrows()]
(TABLES / 'si_tab_known_hh.tex').write_text('\\begin{tabular}{@{}llllrl@{}}\n\\toprule\ncomposition & DXMag string & computed in & certified in & '
    '$E_g$ (eV) & outcome \\\\\n\\midrule\n' + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')
print(df.to_string(index=False)); print('computed', n_comp, 'certified', n_cert)
