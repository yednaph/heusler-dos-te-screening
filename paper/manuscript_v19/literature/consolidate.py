#!/usr/bin/env python3
"""Consolidate the three literature-audit parts into one table of the 80 certified compositions (explicit counting rules).

Inputs : lit_part{1,2,3}.csv (agent audits; every listed DOI checked against Crossref), certified_compositions.csv,
         ../figures/data/known_hh_check.csv (the ten certified benchmark compositions of the known-compound check).
Rules  : one row per composition; for rows audited twice (FeVSb, IrNbGe, IrNbSn) part 2 is used (its brief covered them).
  established                 = certified benchmark of the known-compound check (reviews cited in the main text)
  experimentally investigated = any other composition with reported thermoelectric MEASUREMENTS of the same (half- or full-) Heusler
                                structure, pristine or doped (audit class ESTABLISHED, or PRIOR with measurements)
  computationally investigated= thermoelectric properties only predicted
  computational screens only  = audit class KNOWN_TE_UNCERTAIN (listed only in computational stability/property screens; no synthesis
                                or thermoelectric transport study found)
  underexplored               = audit class UNDEREXPLORED (targeted search; not proof of absence)
  polymorph flag              = experimental reports of the composition exist only in another (orthorhombic) structure ('exp.'),
                                or a competing orthorhombic ground state is predicted ('pred.')
Outputs: literature_status.csv and ../tables/si_tab_literature.tex
"""
import re
from pathlib import Path
import pandas as pd

H = Path(__file__).resolve().parent
parts = [pd.read_csv(H / f'lit_part{i}.csv').assign(part=i) for i in (1, 2, 3)]
lit = pd.concat(parts)
lit = lit[~((lit.part == 1) & lit.composition.isin(['FeVSb', 'IrNbGe', 'IrNbSn']))]       # overlap rows -> part 2
cert = pd.read_csv(H / 'certified_compositions.csv')
assert set(lit.composition) == set(cert.composition) and len(lit) == 80
known = pd.read_csv(H.parents[1] / 'manuscript_v1/figures/data/known_hh_check.csv').query('outcome == "certified"')   # record shipped with the earlier draft
BENCH = set(known.composition); assert len(BENCH) == 10
MEASURED = {'CoTaSn', 'NiScSb', 'PdScSb', 'PtScSb', 'NiYBi', 'NiYSb', 'CoVSn', 'Fe2TiSi',
            'IrNbSn'}                    # v11 review: NbIrSn half-Heusler made and S, rho measured (Hohl 1998, 10.1088/0953-8984/10/35/016)
# v11 literature review (Crossref-verified): Pak et al. 2025 (J. Mater. Chem. A 13, 19455; 10.1039/d5ta00397k) give half-Heusler
# transport/quality-factor predictions for these compositions -> computationally investigated (screen level);
# CoHfAs also has a dedicated DFT TE study (Yang 2022, J. Solid State Chem. 314, 123386; 10.1016/j.jssc.2022.123386)
SCREEN_ONLY = {'AuHfAl', 'CoTiAs', 'CoZrAs', 'FeNbAs', 'OsNbSb', 'OsTaSb', 'PtVGa', 'RhTiAs', 'IrTaSi', 'NiZrGe', 'NiZrSi'}
COMP_DEDICATED = {'CoHfAs'}
POLY_EXP = {'CoTaSi', 'CoVSi', 'IrNbGe', 'IrTaSi', 'NiHfGe', 'NiHfSi', 'NiTiGe', 'NiTiSi', 'NiZrGe', 'NiZrSi'}
# v11 review additions: Ge group orthorhombic (Jeitschko 1969, cited in Singh 2019, 10.1088/1361-648x/ab4e71); ZrCoAs/HfCoAs orthorhombic
# (Kleinke 1998, 10.1002/(sici)1521-3749(199801)624:1<51::aid-zaac51>3.0.co;2-q); TaRhGe TiNiSi-type (Dinges 2010, 10.1515/znb-2010-0117);
# TiRhAs orthorhombic (Weber 2017, 10.1103/physrevb.96.235145)
POLY_EXP |= {'CoNbGe', 'CoTaGe', 'CoVGe', 'CoHfAs', 'RhTaGe', 'RhTiAs'}   # v17 audit: CoZrAs withdrawn (structure not verifiable; see row note)
# v17 citation audit (sources read in full): NbRhGe isostructural with TiNiSi-type TaRhGe (Dinges 2010, p. 97, citing its ref. 1);
# Gautier 2015 Table 1 places VIrGe, ScNiAs and ScPdAs, with VRhGe, in the orthorhombic MgSrSi type; Bilinska 2023 (p. 3) names
# TiNiPb, NbRuAs and VRhGe as half-Heuslers within 50 meV of a lower-energy Pnma structure.
POLY_EXP |= {'RhNbGe'}
POLY_PRED = {'RhVGe', 'NiTiPb', 'IrVGe', 'NiScAs', 'PdScAs', 'RuNbAs'}


def klass(r):
    if r.composition in BENCH: return 'established'
    if r.composition in MEASURED: return 'experimentally investigated'
    if r.composition in SCREEN_ONLY | COMP_DEDICATED: return 'computationally investigated'
    return {'PRIOR_TE_INVESTIGATION': 'computationally investigated', 'KNOWN_TE_UNCERTAIN': 'computational screens only',
            'UNDEREXPLORED': 'underexplored', 'ESTABLISHED': 'experimentally investigated'}[r.category]


lit['class'] = lit.apply(klass, axis=1)
assert set(lit[lit.category == 'ESTABLISHED'].composition) <= BENCH | MEASURED     # every audit-ESTABLISHED row is placed
lit['screen_only'] = lit.composition.isin(SCREEN_ONLY)

lit['polymorph'] = ['exp.' if c in POLY_EXP else 'pred.' if c in POLY_PRED else '' for c in lit.composition]
POLY_SRC = {                                                       # source of each polymorph flag ('sec.' = secondary citation only)
    'CoTaSi': 'Landolt-Bornstein, via patent (sec.)', 'CoVSi': 'Landolt-Bornstein, via patent (sec.)',
    'CoNbGe': 'Jeitschko 1969, via Singh 2019 (sec.)', 'CoTaGe': 'Jeitschko 1969, via Singh 2019 (sec.)',
    'CoVGe': 'Jeitschko 1969, via Singh 2019 (sec.)',
    'CoHfAs': 'Kleinke 1998; ICSD 406953, Pnma (10.1002/(sici)1521-3749(199801)624:1<51::aid-zaac51>3.0.co;2-q)',
    'RhTaGe': 'Dinges 2010 (10.1515/znb-2010-0117)', 'RhNbGe': 'Dinges 2010, isostructural with TaRhGe (sec.)', 'RhTiAs': 'Weber 2017 (10.1103/physrevb.96.235145)',
    'IrNbGe': 'Mishra 2001 (Z. Naturforsch.)', 'IrTaSi': 'orthorhombic TaIrSi literature (row notes)',
    'NiHfGe': 'Mallick 2025 (row notes)', 'NiHfSi': 'Saurabh 2021 (10.1103/physrevmaterials.5.085406)',
    'NiTiGe': 'Downie 2023 (10.1002/zaac.202300055)', 'NiTiSi': 'Downie 2023 (10.1002/zaac.202300055)',
    'NiZrGe': 'Saurabh 2021 (10.1103/physrevmaterials.5.085406)', 'NiZrSi': 'Saurabh 2022 (row notes)',
    'RhVGe': 'Gautier 2015, predicted (10.1038/nchem.2207)', 'NiTiPb': 'Bilinska 2023, predicted (10.3390/cryst13091378)',
    'IrVGe': 'Gautier 2015, predicted (10.1038/nchem.2207)', 'NiScAs': 'Gautier 2015, predicted (10.1038/nchem.2207)',
    'PdScAs': 'Gautier 2015, predicted (10.1038/nchem.2207)', 'RuNbAs': 'Bilinska 2023, predicted (10.3390/cryst13091378)'}
lit['polymorph_source'] = lit.composition.map(POLY_SRC).fillna('')
assert set(lit[lit.polymorph != ''].composition) == set(POLY_SRC)
lit = lit.merge(cert[['composition', 'families', 'sets', 'heavy']], on='composition')
ORDER = ['established', 'experimentally investigated', 'computationally investigated', 'underexplored']   # v11 review: screens-only class empty
REV = {c: 'High-throughput half-Heusler transport/quality-factor screen only.' for c in SCREEN_ONLY}
REVSRC = {c: ('Pak 2025, J. Mater. Chem. A 13, 19455', '10.1039/d5ta00397k') for c in SCREEN_ONLY}
REVSRC['CoHfAs'] = ('Yang 2022, J. Solid State Chem. 314, 123386', '10.1016/j.jssc.2022.123386')
REVSRC['IrNbSn'] = ('Hohl 1998, J. Phys.: Condens. Matter 10, 7843', '10.1088/0953-8984/10/35/016')
REV['CoHfAs'] = 'Dedicated DFT thermoelectric study; also in high-throughput screens.'
REV['IrNbSn'] = 'Half-Heusler NbIrSn synthesised; room-temperature Seebeck coefficient and resistivity measured.'
for c, (src, doi) in REVSRC.items():
    m = lit.composition == c
    lit.loc[m, 'evidence'] = REV[c] + ' ' + lit.loc[m, 'evidence']
    lit.loc[m, 'source_2'], lit.loc[m, 'doi_2'] = lit.loc[m, 'source_1'], lit.loc[m, 'doi_1']
    lit.loc[m, 'source_1'], lit.loc[m, 'doi_1'] = src, doi
lit['k'] = lit['class'].map(ORDER.index)
lit = lit.sort_values(['k', 'composition'])
cols = ['composition', 'conventional_name', 'families', 'sets', 'class', 'screen_only', 'polymorph', 'polymorph_source', 'audit_category', 'evidence', 'source_1', 'doi_1',
        'crossref_verified_1', 'source_2', 'doi_2', 'crossref_verified_2', 'notes']
lit.rename(columns={'category': 'audit_category'})[cols].to_csv(H / 'literature_status.csv', index=False)
cnt = lit['class'].value_counts().reindex(ORDER)
print(cnt.to_string()); print('polymorph flags:', lit.polymorph.value_counts().to_dict())
print(pd.crosstab(lit['class'], lit.polymorph))


def brk(s):
    return re.sub(r'([/.(:;-])', r'\1\\allowbreak{}', s)


def tex(s):
    return (str(s).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#')) if isinstance(s, str) else ''


L = [r'\begin{longtable}{@{}l l l >{\raggedright\arraybackslash}p{2.4cm} >{\raggedright\arraybackslash}p{5.4cm} >{\raggedright\arraybackslash}p{3.2cm}@{}}',
     r'\caption{\textbf{Prior thermoelectric evidence for the 80 certified compositions} (targeted literature audit, Methods). DXMag site-ordered',
     r'composition and conventional name; class by the strongest evidence for the same composition in the same structure type; polymorph: the',
     r'composition is reported experimentally in an orthorhombic structure (exp.) or such a ground state is predicted (pred.); screen-level predictions count as computational. Absence of a',
     r'report is not evidence that none exists. Every DOI was checked against its Crossref record.}\label{tab:si-literature}\\',
     r'\toprule', r'composition & conventional & class & poly. & evidence & principal source (DOI) \\', r'\midrule', r'\endfirsthead',
     r'\toprule', r'composition & conventional & class & poly. & evidence & principal source (DOI) \\', r'\midrule', r'\endhead',
     r'\bottomrule', r'\endfoot']
SHORT = {'established': 'established', 'experimentally investigated': 'exp. investig.', 'computationally investigated': 'comp. investig.',
         'computational screens only': 'screens only', 'underexplored': 'underexplored'}
for _, r in lit.iterrows():
    src = tex(r.source_1) + (f' ({brk(tex(r.doi_1))})' if isinstance(r.doi_1, str) and r.doi_1.strip() else '')
    ps_ = re.sub(r' \(10\..*\)', '', r.polymorph_source)          # author-year only; DOI kept in the CSV
    pol = r.polymorph + (' (' + tex(ps_) + ')' if r.polymorph else '')
    L.append(f'{r.composition} & {tex(r.conventional_name).split(" ")[0]} & {SHORT[r["class"]]} & {pol} & {tex(r.evidence)} & {src} \\\\')
L.append(r'\end{longtable}')
(H.parent / 'tables/si_tab_literature.tex').write_text('% generated by literature/consolidate.py\n' + '\n'.join(L) + '\n')
