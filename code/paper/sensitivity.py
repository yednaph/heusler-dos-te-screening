#!/usr/bin/env python3
"""
sensitivity.py -- how the winner / certified counts depend on the three declared thresholds (paper SI table).

The thresholds were fixed before the winner analysis (PF bar 1.0857e12, dopable |S| > 60 uV/K, hull < 0.10 eV/atom);
the paper keeps them (decision 2026-10-08) and shows here how the result moves when each is varied, including the
bar recomputed on the corrected (transport_v2) basis from the same 1,401 runs that defined it.

INPUT   results/paper_analysis/paper_dataset.pkl, raw transport files (runs_local, soc_campaign/runs),
        results/dxmag_master/master_dataset.csv (the 1,401 runs that defined the bar)
OUTPUT  results/paper_analysis/sensitivity.xlsx, sensitivity.txt
"""
import os as _os_htr  # release v1: project paths come from the environment
_HTR = _os_htr.environ.get('HEUSLER_TE_ROOT', _os_htr.path.abspath(_os_htr.path.join(_os_htr.path.dirname(__file__), '../..')))
_HTE = _os_htr.environ.get('HEUSLER_TE_EXPORT', _os_htr.path.join(_HTR, 'DOS_EXPORT'))
import json, os
import numpy as np
import pandas as pd

P = _HTR   # project root
OUT = f"{P}/results/paper_analysis"
RL = f"{P}/CMS/anchor/runs_local"; SR = f"{P}/CMS/anchor/soc_campaign/runs"               # raw run directories
BAR = 1.0857214253081801e12                                                               # declared PF/tau bar


def best_pfs(path):
    """(best reopt-or-intrinsic PF/tau, best intrinsic PF/tau) over 300-1200 K from one transport file."""
    x = json.load(open(path))["per_temperature"]
    intr = max(r["intrinsic"]["PF_over_tau"] for r in x)
    return max(intr, max(r["reopt_doping"]["PF_over_tau"] for r in x)), intr


def main():
    d = pd.read_pickle(f"{OUT}/paper_dataset.pkl")
    pr, pi, sr, si = [], [], [], []
    for _, r in d.iterrows():                                            # PF with and without doping, per run
        a, b = best_pfs(f"{RL}/{r.run_dir}/{r.scalar_file}") if isinstance(r.scalar_file, str) else (np.nan, np.nan)
        pr.append(a); pi.append(b)
        a, b = best_pfs(f"{SR}/{r.run_dir}/{r.soc_file}") if isinstance(r.soc_file, str) else (np.nan, np.nan)
        sr.append(a); si.append(b)
    d["pf_dop"], d["pf_intr"], d["soc_pf_dop"], d["soc_pf_intr"] = pr, pi, sr, si

    def count(bar=BAR, s_min=60.0, hull_max=0.10):
        """Winners, gapped winners, certified entries and compounds for one threshold choice."""
        dop = d.max_S_intr > s_min
        best = np.where(dop, d.pf_dop, d.pf_intr)                         # doping only if dopable
        win = dop & (best > bar) & (d.hull < hull_max)
        gw = win & d.gapped
        sdop = d.soc_max_S_intr > s_min
        sbest = np.where(sdop, d.soc_pf_dop, d.soc_pf_intr)
        cert = gw & (~d.heavy | (d.soc_file.notna() & (sbest > bar)))
        return int(win.sum()), int(gw.sum()), int(cert.sum()), set(d[cert].composition)

    # bar recomputed on the corrected basis: median 900 K reopt PF/tau over the same 1,401 runs
    m = pd.read_csv(f"{P}/results/dxmag_master/master_dataset.csv", low_memory=False)
    m = m[m.reopt_900K_PF_over_tau.notna()][["run_dir"]]
    bar_v2 = float(d.merge(m, on="run_dir").reopt900_PF.median())

    base = count()
    rows = []
    cases = [("declared (baseline)", {}), ("bar on corrected basis (%.4g)" % bar_v2, dict(bar=bar_v2)),
             ("bar x0.8", dict(bar=0.8 * BAR)), ("bar x0.9", dict(bar=0.9 * BAR)), ("bar x1.1", dict(bar=1.1 * BAR)),
             ("bar x1.2", dict(bar=1.2 * BAR)), ("bar x1.5", dict(bar=1.5 * BAR)),
             ("dopable |S| > 40", dict(s_min=40)), ("dopable |S| > 80", dict(s_min=80)), ("dopable |S| > 100", dict(s_min=100)),
             ("hull < 0.025", dict(hull_max=0.025)), ("hull < 0.05", dict(hull_max=0.05)), ("hull < 0.15", dict(hull_max=0.15))]
    for lab, kw in cases:
        w, g, c, comps = count(**kw)
        rows.append(dict(case=lab, winners=w, gapped_winners=g, certified_entries=c, certified_compounds=len(comps),
                         lost_vs_baseline=", ".join(sorted(base[3] - comps)), gained_vs_baseline=", ".join(sorted(comps - base[3]))))
    t = pd.DataFrame(rows)
    t.to_excel(f"{OUT}/sensitivity.xlsx", index=False)
    txt = t[["case", "winners", "gapped_winners", "certified_entries", "certified_compounds"]].to_string(index=False)
    open(f"{OUT}/sensitivity.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
