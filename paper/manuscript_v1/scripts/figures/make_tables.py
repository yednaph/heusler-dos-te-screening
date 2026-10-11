"""LaTeX tables (paper_rebuild/tables/*.tex) built from existing result files only.

tab1_data_ml.tex        data and held-out ML d-DOS fidelity by family (ml_metrics.txt values via per-sample CSVs)
tab3_candidates.tex     Pareto tier-1 compositions at 900 K, |n| <= 1e21 cm^-3 (ranking_fixed_n.xlsx), with track provenance
si_tab_*.tex            full triage metrics, bootstrap dAUC, queue provenance, sensitivity, SOC failures, all 80 compositions
"""
import numpy as np
import pandas as pd
from common import *

TRACKS = ['prediction', 'training']
d_all = dft_runs()
TR = {'prediction': 'P', 'training': 'T'}


def fmt(x, nd=2):
    return '--' if pd.isna(x) else f'{x:.{nd}f}'


def chem(s):
    """Composition with LaTeX subscripts, e.g. Fe2VAl -> Fe$_2$VAl."""
    import re
    return re.sub(r'(\d+)', r'$_{\1}$', s)


# ---------------------------------------------------------------- Table 1
ps = per_sample()
ml = pd.read_fwf(ANALYSIS / 'ml_metrics.txt')
pools = screen_pools()
rows = []
for fam in FAMILIES + ['ALL']:
    m = ml[ml.family == fam].iloc[0]
    n_ref = int(m.n_test)
    n_pred = int((pools.pool == 'prediction').sum()) if fam == 'ALL' else int(((pools.pool == 'prediction') & (pools.family == fam)).sum())
    sub = d_all if fam == 'ALL' else d_all[d_all.family == fam]
    runs_p, runs_r = int((sub.data_set == 'prediction').sum()), int((sub.data_set == 'training').sum())
    cert = sub[sub.certified == True]
    cp = cert[cert.data_set == 'prediction'].composition.nunique(); cr = cert[cert.data_set == 'training'].composition.nunique()
    name = r'\textit{all}' if fam == 'ALL' else FAM_LABEL[fam]
    rows.append(f'{name} & {n_ref:,} & {m.pearson_mean:.3f} & {m.fermi2eV_pearson:.3f} & {m.rmse:.3f} & '
                f'{m.balanced_accuracy:.2f} & {m.recall_gapped:.2f} & {n_pred:,} & {runs_r:,} & {runs_p:,} & {cr} & {cp} \\\\')
body = '\n'.join(rows[:-1]) + '\n\\midrule\n' + rows[-1]
(TABLES / 'tab1_data_ml.tex').write_text(r"""\begin{tabular}{@{}l r ccc cc r rr rr@{}}
\toprule
 & \multicolumn{6}{c}{training set: held-out ML d-DOS fidelity} & & \multicolumn{2}{c}{DFT runs} & \multicolumn{2}{c}{certified comp.} \\
\cmidrule(lr){2-7}\cmidrule(lr){9-10}\cmidrule(l){11-12}
family & $N_\mathrm{train}$ & $\bar r$ & $\bar r_{\pm2\,\mathrm{eV}}$ & RMSE & BA & rec. & $N_\mathrm{pred}$ & T & P & T & P \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------- Table 3 candidates (tier 1 at 900 K, |n| <= 1e21)
cert = d_all[d_all.certified == True]
prov = cert.groupby('composition').data_set.apply(lambda x: '+'.join(sorted({TR[v] for v in x if v in TR}, reverse=True)) or '?')
rp = pd.read_excel(ANALYSIS / 'ranking_fixed_n.xlsx', sheet_name='n1e21_p_900K').set_index('composition')
rn = pd.read_excel(ANALYSIS / 'ranking_fixed_n.xlsx', sheet_name='n1e21_n_900K').set_index('composition')
sel = sorted(set(rp.index[rp.tier == 1]) | set(rn.index[rn.tier == 1]))
lines = []
for c in sorted(sel, key=lambda k: (rp.tier.get(k, 99), rn.tier.get(k, 99), k)):
    a, b = rp.loc[c], rn.loc[c]
    lines.append(f'{chem(c)} & {a.family.replace("_", " ")} & {prov.get(c, "?").replace("+", "{+}")} & {a.gap:.2f} & {a.hull:.3f} & '
                 f'{"yes" if a.heavy else "no"} & {a.S_uVK:.0f} & {a.PF / 1e12:.2f} & {int(a.tier) if not pd.isna(a.tier) else "--"} & '
                 f'{b.S_uVK:.0f} & {b.PF / 1e12:.2f} & {int(b.tier) if not pd.isna(b.tier) else "--"} \\\\')
(TABLES / 'tab3_candidates.tex').write_text(r"""\begin{tabular}{@{}l l c cc c rrc rrc@{}}
\toprule
 & & & & & & \multicolumn{3}{c}{p-type} & \multicolumn{3}{c}{n-type} \\
\cmidrule(lr){7-9}\cmidrule(l){10-12}
composition & family & set & $E_g$ & $E_\mathrm{hull}$ & SOC & $S$ & PF/$\tau$ & tier & $S$ & PF/$\tau$ & tier \\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
""")
pd.Series(sel).to_csv(DATA / 'tab3_selection.csv', index=False)

# ---------------------------------------------------------------- SI: full triage metrics
se = pd.concat([pd.read_excel(ANALYSIS / 'screen_eval.xlsx', sheet_name=s) for s in ('screen_eval', 'prime_window')])
SCL = {'A_seebeck': '$A$', 'B_pf': '$B$', 'C_sharp': '$C$', 'D_dosmass': '$D$', 'D34_july': 'D34', 'VEC_rule': 'VEC 18/24', 'random': 'random'}
SUBL = {'all computed': 'all computed', 'prime window 0.10-1.40 eV': 'd-DOS gap 0.10--1.40 eV', 'VEC 18/24 only': 'VEC 18 or 24', 'half_cubic only': 'half cubic'}
out = []
for t in TRACKS:
    for sub in SUBL:
        g = se[(se.data_set == t) & (se.subset == sub)]
        if not len(g): continue
        g0 = g.iloc[0]
        out.append(f'\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{POOL_SHORT[t]}, {SUBL[sub]}}}: $n$ = {int(g0.n):,}, positives = {int(g0.positives)} (base rate {g0.base_rate:.3f})}} \\\\')
        for _, r in g.iterrows():
            out.append(f'\\quad {SCL[r.score]} & {r.ROC_AUC:.3f} & {r.PR_AUC:.3f} & {r.captured_top10pct:.2f} & {r.captured_top20pct:.2f} & {r.captured_top30pct:.2f} \\\\')
(TABLES / 'si_tab_triage.tex').write_text(r"""\begin{longtable}{@{}l ccccc@{}}
\toprule
score & ROC-AUC & PR-AUC & top 10\% & top 20\% & top 30\% \\
\midrule\endhead
""" + '\n'.join(out) + r"""
\bottomrule
\end{longtable}
""")
bt = pd.read_excel(ANALYSIS / 'screen_eval.xlsx', sheet_name='bootstrap_dAUC')
bl = [f'{POOL_SHORT[r.data_set]} & {SUBL.get(r.subset, r.subset)} & {r.comparison.replace("_seebeck", "").replace("_pf", "").replace("_sharp", "").replace("_july", "").replace("_rule", "").replace(" - ", " $-$ ")} & '
      f'{int(r.n):,} & {int(r.positives)} & {r.delta_AUC:+.3f} & [{r.CI95_low:+.3f}, {r.CI95_high:+.3f}] \\\\' for _, r in bt.iterrows()]
(TABLES / 'si_tab_bootstrap.tex').write_text(r"""\begin{tabular}{@{}l l l r r r c@{}}
\toprule
set & subset & comparison & $n$ & pos. & $\Delta$AUC & 95\% CI \\
\midrule
""" + '\n'.join(bl) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------- SI: sensitivity
sv = pd.read_excel(ANALYSIS / 'sensitivity.xlsx')
def case_tex(c):
    """Human-readable LaTeX for a sensitivity-case label."""
    return c.replace('x', r'$\times$').replace('<', '$<$').replace('>', '$>$').replace('|S|', '$|S|$')


sl = [f'{case_tex(r.case)} & {int(r.winners)} & {int(r.gapped_winners)} & {int(r.certified_entries)} & {int(r.certified_compounds)} \\\\'
      for _, r in sv.iterrows()]
(TABLES / 'si_tab_sensitivity.tex').write_text(r"""\begin{tabular}{@{}l rrrr@{}}
\toprule
case & winners & gapped winners & certified entries & certified compositions \\
\midrule
""" + '\n'.join(sl) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------- SI: all 80 compositions
allc = []
for c in sorted(rp.index, key=lambda k: (k)):
    a = rp.loc[c]; b = rn.loc[c] if c in rn.index else None
    allc.append(f'{chem(c)} & {a.family.replace("_", " ")} & {prov.get(c, "?").replace("+", "{+}")} & {a.gap:.2f} & {a.hull:.3f} & {"yes" if a.heavy else "no"} & '
                f'{a.S_uVK:.0f} & {a.PF / 1e12:.2f} & {"--" if b is None else f"{b.S_uVK:.0f}"} & {"--" if b is None else f"{b.PF / 1e12:.2f}"} \\\\')
(TABLES / 'si_tab_all_compositions.tex').write_text(r"""\begin{longtable}{@{}l l c cc c rr rr@{}}
\toprule
 & & & & & & \multicolumn{2}{c}{p-type} & \multicolumn{2}{c}{n-type} \\
composition & family & set & $E_g$ (eV) & $E_\mathrm{hull}$ & SOC & $S$ & PF/$\tau$ & $S$ & PF/$\tau$ \\
\midrule\endhead
""" + '\n'.join(allc) + r"""
\bottomrule
\end{longtable}
""")
print('tables done; Table 3 rows:', len(sel), sel)

# ---------------------------------------------------------------- SI: queue provenance (B15_queue_provenance.csv)
qp = pd.read_csv(ANALYSIS / 'si/B15_queue_provenance.csv')
qp = qp[~qp.queue.str.startswith('TOTAL')]
ql = []
for _, r in qp.iterrows():
    name = r.queue.split(' ', 1)
    # v5: the source label "gate pool 1,408" is reworded so the table does not suggest pooling the two sets
    name = [n.replace('gate pool 1,408', 'gate-ordered list (1,408)') for n in name]
    ql.append(f'{name[0]} & {name[1] if len(name) > 1 else ""} & {int(r.runs):,} & {int(r.training):,} & {int(r.prediction):,} & {int(r.ambiguous)} & '
              f'{int(r.dos_semiconductor):,} & {int(r.dft_gapped)} & {int(r.winners)} & {int(r.certified)} \\\\')
tot = qp[['runs', 'training', 'prediction', 'ambiguous', 'dos_semiconductor', 'dft_gapped', 'winners', 'certified']].sum()
ql.append('\\midrule\n' + f'all & & {int(tot.runs):,} & {int(tot.training):,} & {int(tot.prediction):,} & {int(tot.ambiguous)} & '
          f'{int(tot.dos_semiconductor):,} & {int(tot.dft_gapped)} & {int(tot.winners)} & {int(tot.certified)} \\\\')
(TABLES / 'si_tab_queues.tex').write_text(r"""\begin{tabular}{@{}l p{4.2cm} r rrr r rrr@{}}
\toprule
queue & description & runs & T & P & n/a & d-DOS SC & DFT gapped & winners & certified \\
\midrule
""" + '\n'.join(ql).replace('&', '&').replace('_', r'\_') + r"""
\bottomrule
\end{tabular}
""")
print('queue table done')

# ---------------------------------------------------------------- typographic minus for negative numbers in all tables
import re as _re
for f in TABLES.glob('*.tex'):
    t = f.read_text()
    f.write_text(_re.sub(r'([&\[,]\s*)-(\d)', r'\1$-$\2', t))
print('minus signs fixed')

# ---------------------------------------------------------------- SI: earlier best-over-all-conditions Pareto set (deck lead list)
old = pd.read_excel(ANALYSIS / 'ranking_fixed_n.xlsx', sheet_name='old_pm0.5eV_bestT')
op = old[old.pareto == True].sort_values('combined_rank')
ol = [f'{chem(r.composition)} & {r.family.replace("_", " ")} & {prov.get(r.composition, "?").replace("+", "{+}")} & {r.S_uVK:.0f} & '
      f'{r.PF / 1e12:.2f} & {int(r.combined_rank)} \\\\' for _, r in op.iterrows()]
(TABLES / 'si_tab_old_pareto.tex').write_text(r"""\begin{tabular}{@{}l l c r r r@{}}
\toprule
composition & family & set & $S$ & PF/$\tau$ & combined rank \\
\midrule
""" + '\n'.join(ol).replace('-', '$-$').replace('$-$$', '-$') + r"""
\bottomrule
\end{tabular}
""")
pd.Series(list(op.composition)).to_csv(DATA / 'si_old_pareto.csv', index=False)
print('old Pareto set:', list(op.composition))
