#!/usr/bin/env python3
"""
rank_certified.py -- rank the certified compounds under the conventions agreed for the paper (2026-10-08).

CONVENTIONS (agreed by the authors for the paper, 2026-10-08)
  * fixed carrier concentration: best PF/tau among doping levels with |n| <= 1e20 or <= 1e21 cm^-3,
    p-type (holes) and n-type (electrons) ranked separately -- a real sample is doped one way;
  * headline temperature 900 K; best over 300-1200 K kept only as an upper reference;
  * one row per distinct compound (composition); a compound certified in both DXMag entries
    (training + prediction twin) is flagged "confirmed twice"; its value is the max over its certified entries;
  * SOC values for heavy (Z >= 72) compounds, scalar values otherwise (as in the certification rule);
  * ranking only on computed Seebeck and power factor; nothing is compared with published values.

INPUT   results/paper_analysis/paper_dataset.pkl   (from code/paper/build_paper_dataset.py)
OUTPUT  results/paper_analysis/ranking_fixed_n.xlsx (one sheet per convention + summary)
        results/paper_analysis/ranking_summary.txt
"""
import os as _os_htr  # release v1: project paths come from the environment
_HTR = _os_htr.environ.get('HEUSLER_TE_ROOT', _os_htr.path.abspath(_os_htr.path.join(_os_htr.path.dirname(__file__), '../..')))
_HTE = _os_htr.environ.get('HEUSLER_TE_EXPORT', _os_htr.path.join(_HTR, 'DOS_EXPORT'))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

P = _HTR   # project root
OUT = f"{P}/results/paper_analysis"                                                       # output directory


def pareto(pf, s):
    """True where no other compound has both higher PF and higher |S|."""
    return np.array([not np.any((pf > a) & (s > b)) for a, b in zip(pf, s)])


def pareto_layers(pf, s):
    """Non-dominated sorting: tier 1 = Pareto front, tier 2 = front of the rest, ... (threshold-free tiers, B12)."""
    tier = np.zeros(len(pf), int); left = np.arange(len(pf)); k = 0
    while len(left):
        k += 1
        front = pareto(pf[left], s[left])
        tier[left[front]] = k
        left = left[~front]
    return tier


def rank_table(c, pf_col, s_col, n_col=None, carrier=None):
    """Per-compound ranking on one convention: PF rank, |S| rank, combined rank (mean of the two), Pareto flag,
    Pareto-layer tier. Rows whose Seebeck sign contradicts the carrier type (the compound never reaches that carrier
    type inside the doping window) are kept but flagged wrong_sign and left unranked (B10)."""
    sub = c[c[pf_col].notna()].copy()
    rep = (sub.sort_values(pf_col, ascending=False)                      # best certified entry per compound
              .groupby("composition").head(1).copy())
    rep["PF"] = rep[pf_col]; rep["S_uVK"] = rep[s_col]
    rep["n_cm3"] = rep[n_col] if n_col and n_col in rep else np.nan
    base = pf_col[: -len("_PF")]                                         # e.g. final_cap1e20_p_900
    for q in ("mu", "sigma", "kappa_e", "L_over_L0", "at_cap"):          # extra quantities at the same point (B8)
        rep[q] = rep[f"{base}_{q}"] if f"{base}_{q}" in rep else np.nan
    if n_col is None and f"{base}_n" in rep:                             # bestT sheets also carry n (B8)
        rep["n_cm3"] = rep[f"{base}_n"]
    rep["wrong_sign"] = ((carrier == "p") & (rep.S_uVK < 0)) | ((carrier == "n") & (rep.S_uVK > 0))
    ok = ~rep.wrong_sign
    for col in ("PF_rank", "S_rank", "combined_rank", "tier"):
        rep[col] = np.nan
    rep.loc[ok, "PF_rank"] = rep.PF[ok].rank(ascending=False, method="min")
    rep.loc[ok, "S_rank"] = rep.S_uVK[ok].abs().rank(ascending=False, method="min")
    rep.loc[ok, "combined_rank"] = ((rep.PF_rank[ok] + rep.S_rank[ok]) / 2).rank(method="min")
    rep["pareto"] = False
    rep.loc[ok, "pareto"] = pareto(rep.PF[ok].values, rep.S_uVK[ok].abs().values)
    rep.loc[ok, "tier"] = pareto_layers(rep.PF[ok].values, rep.S_uVK[ok].abs().values)
    keep = ["combined_rank", "composition", "PF", "PF_rank", "S_uVK", "S_rank", "n_cm3", "pareto", "tier",
            "wrong_sign", "mu", "sigma", "kappa_e", "L_over_L0", "at_cap",
            "confirmed_twice", "gap", "heavy", "family", "hull", "UUID", "data_set"]
    return rep.sort_values(["wrong_sign", "combined_rank", "PF_rank"])[keep].reset_index(drop=True)


def main():
    d = pd.read_pickle(f"{OUT}/paper_dataset.pkl")
    c = d[d.certified].copy()
    sets = c.groupby("composition").data_set.agg(lambda s: set(s))
    c["confirmed_twice"] = c.composition.map(lambda k: {"prediction", "training"} <= sets[k])   # both DXMag twins certified

    conventions = {                                                       # name -> (PF column, S column, n column)
        "old_pm0.5eV_bestT": ("final_best_PF", "final_S_best", None),     # campaign convention, for comparison
    }
    for cap in ("1e21", "1e20"):
        for typ in ("p", "n"):
            for tmode in ("900", "best"):
                base = f"final_cap{cap}_{typ}_{tmode}"
                conventions[f"n{cap}_{typ}_{tmode}K" if tmode == "900" else f"n{cap}_{typ}_bestT"] = (
                    f"{base}_PF", f"{base}_S", f"{base}_n" if tmode == "900" else None)

    tables, lines = {}, []
    for name, (pf, s, n) in conventions.items():
        if pf not in c:
            lines.append(f"SKIP {name}: column {pf} missing (re-extract with p/n caps first)"); continue
        carrier = "p" if "_p_" in name else ("n" if "_n_" in name else None)
        tables[name] = rank_table(c, pf, s, n, carrier)

    old = tables["old_pm0.5eV_bestT"].set_index("composition")
    lines.append(f"certified compounds: {c.composition.nunique()}  (entries {len(c)}; confirmed twice: "
                 f"{c.groupby('composition').confirmed_twice.first().sum()})")
    for name, t in tables.items():                                        # robustness vs the old convention
        j = t[~t.wrong_sign].set_index("composition").join(old[["PF_rank", "S_rank"]], rsuffix="_old", how="inner")
        rpf = spearmanr(j.PF_rank, j.PF_rank_old).correlation
        rs = spearmanr(j.S_rank, j.S_rank_old).correlation
        top = ", ".join(t[~t.wrong_sign].composition.head(5))
        cap_n = int(t.at_cap.fillna(False).astype(bool).sum())
        lines.append(f"{name:22s} n={len(t):3d}  wrong-sign {int(t.wrong_sign.sum())}  at-cap {cap_n}  "
                     f"Spearman vs old: PF {rpf:5.2f}  |S| {rs:5.2f}  pareto={int(t.pareto.sum())}  "
                     f"tier1-3 sizes {[int((t.tier == k).sum()) for k in (1, 2, 3)]}  top5: {top}")
    open(f"{OUT}/ranking_summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    with pd.ExcelWriter(f"{OUT}/ranking_fixed_n.xlsx") as w:
        pd.DataFrame({"README": [
            "Certified compounds ranked under the paper conventions (2026-10-08). One row per distinct compound.",
            "PF = PF/tau (W m^-1 K^-2 s^-1, CRTA); S in uV/K; n = net carriers at the optimum (+ holes, - electrons), cm^-3.",
            "Sheet names: n<cap>_<p|n>_<900K|bestT> = best PF/tau with |n| <= cap, p- or n-type only, at 900 K or best over 300-1200 K.",
            "old_pm0.5eV_bestT = campaign convention (best over +/-0.5 eV and 300-1200 K), kept for comparison only.",
            "combined_rank = rank of the mean of PF rank and |S| rank; pareto = no other compound has both higher PF and higher |S|.",
            "tier = Pareto layer (1 = Pareto front, 2 = front after removing tier 1, ...); the paper reports tiers, not a single #1.",
            "wrong_sign = Seebeck sign contradicts the carrier type of the sheet (never reaches it in the window): flagged, not ranked.",
            "mu (eV from E_F), sigma = sigma/tau, kappa_e = kappa_e/tau, L_over_L0 and at_cap are at the same point as PF and S.",
            "confirmed_twice = both DXMag twins (training + prediction entry) are certified. SOC values used for heavy (Z>=72) compounds.",
            "Built by code/paper/rank_certified.py from results/paper_analysis/paper_dataset.pkl."] + lines}).to_excel(w, "README", index=False)
        for name, t in tables.items():
            t.to_excel(w, name[:31], index=False)
    print("wrote", f"{OUT}/ranking_fixed_n.xlsx")


if __name__ == "__main__":
    main()
