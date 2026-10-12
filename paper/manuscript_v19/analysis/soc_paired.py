#!/usr/bin/env python3
"""Paired scalar-relativistic vs spin-orbit (SOC) comparison from EXISTING calculations only.

Reads (read-only):
  results/paper_analysis/paper_dataset.pkl            one row per DFT run; scalar and soc_* transport columns
  CMS/anchor/runs_local/<run_dir>/{scf,nscf}.in/.out   scalar inputs and band energies
  CMS/anchor/soc_campaign/runs/<run_dir>/{scf,nscf}.in/.out  SOC inputs and band energies
No DFT, no transport, no model is run. The SOC band gap and electronic class are parsed from the
existing SOC nscf.out with the project's own gap auditor (code/audit_dft_gaps.py), the same code
that produced the scalar class, so both sides use one rule.

Writes analysis/soc_paired.csv (one row per paired run) and analysis/soc_summary.txt.
"""
import os as _os_htr  # release v1: project paths come from the environment
_HTR = _os_htr.environ.get('HEUSLER_TE_ROOT', _os_htr.path.abspath(_os_htr.path.join(_os_htr.path.dirname(__file__), '../../..')))
_HTE = _os_htr.environ.get('HEUSLER_TE_EXPORT', _os_htr.path.join(_HTR, 'DOS_EXPORT'))
import importlib.util, re, os
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(_HTR)
RL = REPO / 'CMS/anchor/runs_local'                    # scalar run directories
SR = REPO / 'CMS/anchor/soc_campaign/runs'             # SOC run directories
OUT = Path(__file__).resolve().parent                  # paper_rebuild/v11_final/analysis
BAR = 1.0857214253081801e12                            # power-factor threshold PF0 (W m^-1 K^-2 s^-1)

_spec = importlib.util.spec_from_file_location('adg', REPO / 'code/audit_dft_gaps.py')
adg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(adg)


def card(text, tag):
    """Lines of a QE card (CELL_PARAMETERS / ATOMIC_POSITIONS / K_POINTS) up to the next blank or card line."""
    cards = ('ATOMIC_SPECIES', 'ATOMIC_POSITIONS', 'CELL_PARAMETERS', 'K_POINTS')
    lines, out, on = text.splitlines(), [], False
    for ln in lines:
        if ln.strip().startswith(tag):
            on = True; out.append(ln.strip()); continue
        if on and (not ln.strip() or ln.strip().startswith(cards)):
            break
        if on:
            out.append(' '.join(ln.split()))
    return ' | '.join(out) if out else None


def inputs(path):
    """Structure, k-mesh, cutoffs, magnetism flags and pseudopotentials of one QE input file."""
    t = path.read_text(errors='replace')
    g = lambda k: (re.search(rf'{k}\s*=\s*([^\s,]+)', t) or [None, None])[1]
    pseudos = re.findall(r'^\s*[A-Z][a-z]?\d*\s+[\d.]+\s+(\S+\.UPF)', t, re.M | re.I)
    return dict(cell=card(t, 'CELL_PARAMETERS'), pos=card(t, 'ATOMIC_POSITIONS'), kpts=card(t, 'K_POINTS'),
                ecutwfc=g('ecutwfc'), nspin=g('nspin'), noncolin=g('noncolin'), pseudos=' '.join(sorted(pseudos)))


d = pd.read_pickle(REPO / 'results/paper_analysis/paper_dataset.pkl')
p = d[d.soc_status == 'done'].copy()                   # genuinely paired: both calculations finished
rows = []
for _, r in p.iterrows():                          # _ = row index in paper_dataset (UUIDs can repeat)
    o = {'idx': _}
    g = adg.dft_gap(str(SR / r.run_dir / 'nscf.out'))  # (gap, E_F, class) from SOC band energies
    o['soc_gap'], o['soc_EF'], o['soc_class'] = g if g else (np.nan, np.nan, None)
    si, ss = inputs(RL / r.run_dir / 'scf.in'), inputs(SR / r.run_dir / 'scf.in')
    ni, ns = inputs(RL / r.run_dir / 'nscf.in'), inputs(SR / r.run_dir / 'nscf.in')
    o['same_cell'] = si['cell'] == ss['cell']          # structure copied verbatim?
    o['same_positions'] = si['pos'] == ss['pos']
    o['same_scf_kmesh'] = si['kpts'] == ss['kpts']
    o['same_nscf_kmesh'] = ni['kpts'] == ns['kpts']
    o['scalar_spin_polarised'] = si['nspin'] == '2'    # scalar run collinear spin-polarised?
    o['soc_noncolin'] = ss['noncolin'] == '.true.'
    o['scalar_ecutwfc'], o['soc_ecutwfc'] = si['ecutwfc'], ss['ecutwfc']
    o['scalar_pseudos'], o['soc_pseudos'] = si['pseudos'], ss['pseudos']
    rows.append(o)
p = p.join(pd.DataFrame(rows).set_index('idx'))      # join on row index, never on UUID
for c in ['best_PF', 'soc_best_PF', 'reopt900_PF', 'soc_reopt900_PF']:
    p[c] = p[c].astype(float)
p['pf_ratio'] = p.soc_best_PF / p.best_PF              # best PF/tau over 300-1200 K, SOC / scalar
p['pf900_ratio'] = p.soc_reopt900_PF / p.reopt900_PF   # 900 K, mu re-optimised within 0.5 eV
p['dS_intr'] = p.soc_max_S_intr - p.max_S_intr         # change in peak undoped |S| (uV/K)
p['dgap'] = p.soc_gap - p.gap                          # SOC - scalar band gap (eV)
p['qualified'] = p.winner.astype(bool)                 # scalar Seebeck + PF + stability criteria
p['soc_qualified'] = p.soc_dopable.fillna(False).astype(bool) & (p.soc_best_PF > BAR) & p.stable.astype(bool)
keep = ['UUID', 'run_dir', 'composition', 'data_set', 'family', 'space group symbol', 'hull', 'dft_class', 'soc_class', 'gap',
        'soc_gap', 'dgap', 'max_S_intr', 'soc_max_S_intr', 'dS_intr', 'dopable', 'soc_dopable', 'best_PF',
        'soc_best_PF', 'pf_ratio', 'reopt900_PF', 'soc_reopt900_PF', 'pf900_ratio', 'S_best', 'soc_S_best',
        'sigma_best', 'soc_sigma_best', 'best_T', 'soc_best_T', 'qualified', 'soc_qualified', 'gapped', 'certified',
        'same_cell', 'same_positions', 'same_scf_kmesh', 'same_nscf_kmesh', 'scalar_spin_polarised', 'soc_noncolin',
        'scalar_ecutwfc', 'soc_ecutwfc', 'scalar_pseudos', 'soc_pseudos', 'soc_file']
p[keep].to_csv(OUT / 'soc_paired.csv', index=False)
print('paired runs', len(p))

# ---------------------------------------------------------------- converged magnetisation of scalar runs
def abs_mag(rd):
    """Converged absolute magnetisation (Bohr mag/cell) from the scalar scf.out; 0 if the run was not spin-polarised."""
    t = (RL / rd / 'scf.out').read_text(errors='replace')
    a = re.findall(r'absolute magnetization\s*=\s*([-\d.]+)', t)
    return float(a[-1]) if a else 0.0
p['scalar_abs_mag'] = [abs_mag(rd) if sp else 0.0 for rd, sp in zip(p.run_dir, p.scalar_spin_polarised)]
p[keep + ['scalar_abs_mag']].to_csv(OUT / 'soc_paired.csv', index=False)

# ---------------------------------------------------------------- summary (every number used in the paper)
el = lambda s: set(re.findall(r'[A-Z][a-z]?', s))
S = []
S.append(f'paired runs (scalar and SOC both finished): {len(p)}  by set {p.data_set.value_counts().to_dict()}')
S.append(f'identical cell/positions/k-meshes: {int(p.same_cell.sum())}/{int(p.same_positions.sum())}/'
         f'{int((p.same_scf_kmesh & p.same_nscf_kmesh).sum())}; scalar runs spin-polarised {int(p.scalar_spin_polarised.sum())},'
         f' max converged |m| {p.scalar_abs_mag.max():.3f} Bohr mag/cell (all non-magnetic)')
S.append('class transitions scalar -> SOC:\n' + pd.crosstab(p.dft_class, p.soc_class, margins=True).to_string())
g = p[(p.dft_class == 'gapped') & (p.soc_class == 'gapped')]
S.append(f'gapped in both: {len(g)}; gap decreased in {(g.dgap < 0).mean():.1%}; median dgap {g.dgap.median():.3f} eV,'
         f' IQR [{g.dgap.quantile(.25):.3f}, {g.dgap.quantile(.75):.3f}]; median relative {(g.dgap / g.gap).median():.1%}')
S.append(f'PF ratio (best PF/tau SOC/scalar), all pairs: median {p.pf_ratio.median():.3f}; <1 in {(p.pf_ratio < 1).mean():.1%};'
         f' <0.9 in {(p.pf_ratio < 0.9).mean():.1%}; >1.1 in {(p.pf_ratio > 1.1).mean():.1%}; 900 K median {p.pf900_ratio.median():.3f}')
S.append('Seebeck criterion scalar (rows) vs SOC (cols):\n' + pd.crosstab(p.dopable, p.soc_dopable).to_string())
c = p[p.qualified & (p.dft_class == 'gapped')].copy()           # scalar candidates awaiting criterion (v)
c['kept'] = c.soc_best_PF > BAR
c['Au'] = c.composition.map(lambda s: 'Au' in el(s))
c['S_ratio'] = c.soc_S_best.abs() / c.S_best.abs(); c['sig_ratio'] = c.soc_sigma_best / c.sigma_best
S.append(f'scalar candidates (qualified + gapped, SOC done): {len(c)} {c.data_set.value_counts().to_dict()}; kept '
         f'{c.groupby("data_set").kept.sum().to_dict()}, lost {(~c.kept).groupby(c.data_set).sum().to_dict()}')
S.append(f'  compositions: {c.composition.nunique()} candidates, {c[c.kept].composition.nunique()} kept, '
         f'{c.composition.nunique() - c[c.kept].composition.nunique()} lost: {sorted(set(c.composition) - set(c[c.kept].composition))}')
S.append(f'  candidate medians: PF ratio {c.pf_ratio.median():.3f}, |S| ratio {c.S_ratio.median():.3f}, sigma/tau ratio {c.sig_ratio.median():.3f}')
for k, x in c.groupby('Au'):
    S.append(f'  Au-containing={k}: n {len(x)}, lost {int((~x.kept).sum())}, median PF {x.pf_ratio.median():.3f},'
             f' |S| {x.S_ratio.median():.3f}, sigma {x.sig_ratio.median():.3f}')
for e in ['Hf', 'Ta', 'Os', 'Ir', 'Pt', 'Au', 'Tl', 'Pb', 'Bi']:
    x = c[c.composition.map(lambda s: e in el(s))]
    if len(x): S.append(f'  {e}: candidates {len(x)}, lost {int((~x.kept).sum())}, median PF ratio {x.pf_ratio.median():.2f}')
lost_runs = c[~c.kept]
S.append(f'  all lost runs remain gapped under SOC: {bool((lost_runs.soc_class == "gapped").all())}; lost runs median PF ratio '
         f'{lost_runs.pf_ratio.median():.3f}, kept {c[c.kept].pf_ratio.median():.3f}')
cert = p[p.certified]
S.append(f'certified heavy entries with SOC class: {cert.soc_class.value_counts().to_dict()}; not gapped under SOC: '
         f'{cert[cert.soc_class != "gapped"][["composition", "data_set", "gap", "soc_gap"]].to_dict("records")}')
open(OUT / 'soc_summary.txt', 'w').write('\n'.join(S) + '\n')
print('\n'.join(S))
# key numbers asserted so a data change fails loudly
assert len(p) == 607 and len(c) == 115 and int(c.kept.sum()) == 75
