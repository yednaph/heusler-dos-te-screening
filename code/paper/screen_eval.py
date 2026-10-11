#!/usr/bin/env python3
"""
screen_eval.py -- how well each pre-DFT screening score finds the DFT-certified compounds (paper Fig. 4 / Table).

Target: certified (rule-defined, see build_paper_dataset.py). Unit: one row per UUID (5 prediction-set UUIDs were
computed twice; the first run is kept) -- the survey showed the per-run vs per-UUID choice moves B by ~0.003.
Scores: A |Seebeck| (Cutler-Mott, uV/K), B PF proxy, C band-edge sharpness, D DOS-mass proxy, D34 (July composite,
prediction set only), and two baselines: VEC rule (1 if valence electron count is 18 for half/24 for full/inverse
Heuslers) and random. Each is evaluated per data set (prediction = ML DOS, training = DFT DOS), overall, within the
half-cubic family, and within VEC-18/24 materials (does the score add information beyond electron counting?).
Metrics: ROC-AUC, PR-AUC (average precision; base rate given), and enrichment = share of certified entries
captured when only the top 10/20/30 % of the computed set by that score is kept.

CAVEAT (state in the paper): the evaluated set is the 3,625 materials actually computed, which the screen itself
selected; AUCs are conditional on that selection, not population estimates.

INPUT   results/paper_analysis/paper_dataset.pkl
OUTPUT  results/paper_analysis/screen_eval.xlsx, screen_eval.txt
"""
import os as _os_htr  # release v1: project paths come from the environment
_HTR = _os_htr.environ.get('HEUSLER_TE_ROOT', _os_htr.path.abspath(_os_htr.path.join(_os_htr.path.dirname(__file__), '../..')))
_HTE = _os_htr.environ.get('HEUSLER_TE_EXPORT', _os_htr.path.join(_HTR, 'DOS_EXPORT'))
import re
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def roc_auc_score(y, s):
    """ROC-AUC as the Mann-Whitney statistic: probability a random positive outranks a random negative (ties = 1/2)."""
    y = np.asarray(y, bool); r = rankdata(np.asarray(s, float))           # average ranks handle ties
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def average_precision_score(y, s):
    """Average precision (PR-AUC) with tie handling: thresholds at DISTINCT score values,
    AP = sum_k (R_k - R_{k-1}) P_k (same definition as scikit-learn). Tied items enter together."""
    y = np.asarray(y, bool); s = np.asarray(s, float)
    thr = np.unique(s)[::-1]                                             # distinct scores, high to low
    tp = np.array([(y & (s >= t)).sum() for t in thr]); n = np.array([(s >= t).sum() for t in thr])
    prec, rec = tp / n, tp / y.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def captured_top(y, s, frac):
    """Expected share of positives captured by keeping the top `frac` of items by score; the tied group at the
    cut-off is taken fractionally (expected value under random tie-breaking)."""
    y = np.asarray(y, bool); s = np.asarray(s, float)
    k = frac * len(s)                                                    # number of items kept
    cut = np.sort(s)[::-1][int(np.ceil(k)) - 1]                          # score of the last item kept
    above = s > cut; tied = s == cut
    take = (k - above.sum()) / tied.sum()                                # fraction of the tied group kept
    return float((y[above].sum() + take * y[tied].sum()) / y.sum())

P = _HTR
OUT = f"{P}/results/paper_analysis"
VAL = {"Sc": 3, "Y": 3, "La": 3, "Lu": 3, "Ti": 4, "Zr": 4, "Hf": 4, "V": 5, "Nb": 5, "Ta": 5, "Cr": 6, "Mo": 6, "W": 6,
       "Mn": 7, "Re": 7, "Fe": 8, "Ru": 8, "Os": 8, "Co": 9, "Rh": 9, "Ir": 9, "Ni": 10, "Pd": 10, "Pt": 10, "Cu": 11,
       "Ag": 11, "Au": 11, "Zn": 12, "Cd": 12, "Hg": 12, "Al": 3, "Ga": 3, "In": 3, "Tl": 3, "Si": 4, "Ge": 4, "Sn": 4,
       "Pb": 4, "P": 5, "As": 5, "Sb": 5, "Bi": 5}                      # valence electrons (s+d for TM, s+p for main group)
SCORES = {"A_seebeck": "SCREEN_A_seebeck_uVK", "B_pf": "SCREEN_B_pf", "C_sharp": "SCREEN_C_sharp",
          "D_dosmass": "SCREEN_D_dosmass", "D34_july": "SCREEN_pred_D34_TE_score"}


def vec(formula):
    """Valence electron count per formula unit."""
    return sum(VAL[e] * (int(n) if n else 1) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula))


def evaluate(sub, y):
    """ROC-AUC, PR-AUC and top-k enrichment of every score on one subset."""
    rows = []
    for name, col in list(SCORES.items()) + [("VEC_rule", "vec_rule"), ("random", "random")]:
        s = sub[col].astype(float)
        ok = s.notna()
        if ok.sum() < 20 or y[ok].nunique() < 2:
            continue                                                       # score absent or degenerate here
        r = dict(score=name, n=int(ok.sum()), positives=int(y[ok].sum()), base_rate=float(y[ok].mean()),
                 ROC_AUC=roc_auc_score(y[ok], s[ok]), PR_AUC=average_precision_score(y[ok], s[ok]))
        for frac in (0.1, 0.2, 0.3):                                       # share of positives in the top fraction
            r[f"captured_top{int(frac*100)}pct"] = captured_top(y[ok].values, s[ok].values, frac)
        rows.append(r)
    return rows


def main():
    d = pd.read_pickle(f"{OUT}/paper_dataset.pkl")
    d = d[d.data_set.isin(["prediction", "training"])].drop_duplicates("UUID")   # per-UUID unit, ambiguous runs excluded
    d["vec"] = d.composition.map(vec)
    magic = np.where(d.family.str.startswith("half"), 18, 24)
    d["vec_rule"] = (d.vec == magic).astype(float)                          # Slater-Pauling closed-shell count
    d["random"] = np.random.default_rng(0).random(len(d))                   # fixed-seed random baseline
    y_all = d.certified.astype(int)
    out = []
    for ds in ("prediction", "training"):
        for lab, mask in (("all computed", d.data_set == ds),
                          ("half_cubic only", (d.data_set == ds) & (d.family == "half_cubic")),
                          ("VEC 18/24 only", (d.data_set == ds) & (d.vec_rule == 1))):
            sub = d[mask]
            for r in evaluate(sub, y_all[mask]):
                out.append(dict(data_set=ds, subset=lab, **r))
    t = pd.DataFrame(out)

    # ---- paired bootstrap of AUC differences (B2): resample UUIDs with replacement within each data set and
    # subset, recompute both AUCs on the SAME resample, report the 2.5/97.5 percentiles of the difference.
    rng = np.random.default_rng(0)                                         # fixed seed: reproducible intervals
    pairs = [("A_seebeck", "B_pf"), ("A_seebeck", "D34_july"), ("A_seebeck", "VEC_rule"), ("A_seebeck", "C_sharp")]
    boot = []
    for ds in ("prediction", "training"):
        for lab, mask in (("all computed", d.data_set == ds), ("VEC 18/24 only", (d.data_set == ds) & (d.vec_rule == 1))):
            sub = d[mask]; y = y_all[mask].values
            for s1, s2 in pairs:
                c1 = SCORES.get(s1, s1); c2 = SCORES.get(s2, "vec_rule" if s2 == "VEC_rule" else s2)
                ok = (sub[c1].notna() & sub[c2].notna()).values
                if ok.sum() < 20 or y[ok].min() == y[ok].max() or sub[c2][ok].nunique() < 2:
                    continue                                               # pair not defined here (e.g. D34 on training)
                a1, a2, yy = sub[c1].values[ok].astype(float), sub[c2].values[ok].astype(float), y[ok]
                diffs = []
                for _ in range(2000):                                      # 2,000 paired resamples
                    i = rng.integers(0, len(yy), len(yy))
                    if yy[i].min() == yy[i].max():
                        continue                                           # resample without both classes
                    diffs.append(roc_auc_score(yy[i], a1[i]) - roc_auc_score(yy[i], a2[i]))
                diffs = np.array(diffs)
                boot.append(dict(data_set=ds, subset=lab, comparison=f"{s1} - {s2}", n=int(ok.sum()), positives=int(yy.sum()),
                                 delta_AUC=roc_auc_score(yy, a1) - roc_auc_score(yy, a2),
                                 CI95_low=np.percentile(diffs, 2.5), CI95_high=np.percentile(diffs, 97.5),
                                 frac_le_0=float((diffs <= 0).mean()), resamples=len(diffs)))
    b = pd.DataFrame(boot)

    # ---- prime window: AUC of score A among runs in the d-DOS prime window (0.10-1.40 eV)
    prime = []
    for ds in ("prediction", "training"):
        m = (d.data_set == ds) & d.SCREEN_gap_eV.between(0.10, 1.40)
        prime += [dict(data_set=ds, subset="prime window 0.10-1.40 eV", **r) for r in evaluate(d[m], y_all[m])]
    pw = pd.DataFrame(prime)

    with pd.ExcelWriter(f"{OUT}/screen_eval.xlsx") as w:                  # first sheet unchanged (read by fig_3.py)
        t.to_excel(w, "screen_eval", index=False)
        b.to_excel(w, "bootstrap_dAUC", index=False)
        pw.to_excel(w, "prime_window", index=False)
    txt = (t.round(3).to_string(index=False) + "\n\n# paired bootstrap of AUC differences (2,000 resamples, seed 0)\n"
           + b.round(3).to_string(index=False) + "\n\n# prime window\n" + pw.round(3).to_string(index=False))
    open(f"{OUT}/screen_eval.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
