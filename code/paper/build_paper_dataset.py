#!/usr/bin/env python3
"""
build_paper_dataset.py -- rebuild the DFT results chain for the paper from RAW run files.

WHY (2026-10-08 paper-prep survey, blocker B3): no committed script rebuilt the master workbooks
(FINAL_BoltzTraP_master_list.xlsx, ALL_DFT_3625_materials_DXMAG_style.xlsx), the certification
labels or the ranking. This script does, from first-hand data, and checks itself against them.

INPUTS
  index (inputs only, nothing derived from DFT): results/screening/ALL_DFT_3625_materials_DXMAG_style.xlsx
      sheet Materials -> UUID, composition, run_dir, data set, family, DXMag hull/phonon, SCREEN_* columns
  raw DFT/transport per run (CMS/anchor):
      runs_local/<run_dir>/nscf.out                     -> real PBE gap + electronic class (code/audit_dft_gaps.py)
      runs_local/<run_dir>/transport_v2.json            -> scalar transport (else transport.json; v2 = corrected mesh)
      soc_campaign/runs/<run_dir>/transport_v2.json     -> SOC transport (else transport.json)
      .../transport_w1.json                             -> +/-1.0 eV re-extraction with carrier concentrations (2026-10-08)

RULES (fixed before the winner analysis; see the master workbook README)
  dopable    : max over 300-1200 K of the undoped |S| > 60 uV/K
  PF bar     : best PF/tau > 1.0857214253081801e12 W m^-1 K^-2 s^-1 (median 900 K reopt PF/tau of the first 1,401 runs)
  stable     : DXMag hull distance < 0.10 eV/atom
  winner     : dopable AND above the bar AND stable
  certified  : winner with a real gap (class 'gapped', gap >= 0.136 eV) that is light (no Z>=72 element),
               or heavy and still above the bar in its SOC calculation
  best point : over T in 300..1200 K, the intrinsic point and (if dopable) the PF-reoptimised doping point

OUTPUTS (results/paper_analysis/)
  paper_dataset.pkl / .xlsx     -- one row per DFT run, every derived column recomputed here
  build_check.txt                -- agreement with the published workbook (must be exact)

Read-only on all inputs. Light: one process, ~1-3 min.
"""
import os as _os_htr  # release v1: project paths come from the environment
_HTR = _os_htr.environ.get('HEUSLER_TE_ROOT', _os_htr.path.abspath(_os_htr.path.join(_os_htr.path.dirname(__file__), '../..')))
_HTE = _os_htr.environ.get('HEUSLER_TE_EXPORT', _os_htr.path.join(_HTR, 'DOS_EXPORT'))
import importlib.util, json, os, re, sys
import numpy as np
import pandas as pd

P = _HTR   # project root
RL = f"{P}/CMS/anchor/runs_local"                                # scalar run directories
SR = f"{P}/CMS/anchor/soc_campaign/runs"                         # SOC run directories
OUT = f"{P}/results/paper_analysis"                              # output directory for paper tables
INDEX = f"{P}/results/screening/ALL_DFT_3625_materials_DXMAG_style.xlsx"   # index workbook (inputs only)

BAR = 1.0857214253081801e12                                       # fixed PF/tau winner bar (declared constant)
DOPABLE_UVK = 60.0                                               # dopability threshold on undoped |S|, uV/K
HULL_MAX = 0.10                                                  # stability threshold, eV/atom
TEMPS = list(range(300, 1201, 100))                              # temperature sweep of every transport file, K
L0 = 2.44e-8                                                     # Sommerfeld Lorenz number, W Ohm K^-2
HEAVY = {"Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi"}   # Z >= 72 elements present in DXMag

# the project's own gap auditor (same code that produced DFT_verdict), loaded by path
_spec = importlib.util.spec_from_file_location("adg", f"{P}/code/audit_dft_gaps.py")
adg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(adg)


def elements(formula):
    """Set of element symbols in a formula string."""
    return set(re.findall(r"[A-Z][a-z]?", formula))


def first_existing(directory, names):
    """First file of `names` that exists in `directory`, else None (v2 before the original file)."""
    for n in names:
        if os.path.exists(f"{directory}/{n}"):
            return n
    return None


def best_point(path):
    """Campaign convention: dopability, best PF/tau point over T (intrinsic, plus reopt if dopable), and derived
    quantities at that point. Identical logic to the published master list."""
    d = json.load(open(path))
    pt = {int(round(r["T_K"])): r for r in d["per_temperature"]}            # per-temperature records keyed by K
    max_S_intr = max(abs(pt[T]["intrinsic"]["S_uVK"]) for T in TEMPS)       # undoped |S| peak over 300-1200 K
    dopable = max_S_intr > DOPABLE_UVK
    best = None
    for T in TEMPS:                                                          # scan every temperature
        cands = [("intrinsic", pt[T]["intrinsic"])]
        if dopable:                                                          # doping only allowed if dopable
            cands.append(("reopt_doping", pt[T]["reopt_doping"]))
        for conv, rec in cands:
            if best is None or rec["PF_over_tau"] > best[2]["PF_over_tau"]:
                best = (T, conv, rec)
    T, conv, rec = best
    S = rec["S_uVK"] * 1e-6                                                  # V/K
    sig, ke = rec["sigma_over_tau"], rec["kappa_e_over_tau"]
    return dict(max_S_intr=max_S_intr, dopable=dopable, best_PF=rec["PF_over_tau"], best_T=T, best_conv=conv,
                S_best=rec["S_uVK"], sigma_best=sig, kappa_e_best=ke, mu_best=rec.get("mu_eV", 0.0),
                zT_ceiling=S ** 2 / L0,                                      # zT if kappa_L = 0 and L = L0
                zT_e=S ** 2 * sig * T / ke if ke else np.nan,                # tau-free electronic zT (upper bound)
                L_over_L0=ke / (sig * T) / L0 if sig else np.nan,            # Wiedemann-Franz sanity ratio
                reopt900_PF=pt[900]["reopt_doping"]["PF_over_tau"],          # quantity the bar was defined on
                railed=abs(rec.get("mu_eV", 0.0)) >= 0.49)                   # best mu at the +/-0.5 eV scan edge


def capped_points(path):
    """From transport_w1.json: fixed-carrier-concentration optima at 900 K and best over T, for each cap.
    Prefers sign-split fields (p/n, added 2026-10-08) when present; falls back to the |n| cap."""
    d = json.load(open(path))
    pt = {int(round(r["T_K"])): r for r in d["per_temperature"]}
    out = {}
    for cap in ("1e20", "1e21"):
        for typ in ("p", "n", "any"):
            key = f"reopt_cap_{cap}" + ("" if typ == "any" else f"_{typ}")
            if key not in pt[900]:
                continue                                                     # field not in this file version
            def put(tag, rec, T):                                            # all quantities at one capped optimum
                out[f"cap{cap}_{typ}_{tag}_PF"] = rec["PF_over_tau"]; out[f"cap{cap}_{typ}_{tag}_S"] = rec["S_uVK"]
                out[f"cap{cap}_{typ}_{tag}_n"] = rec["carrier_conc_cm3"]; out[f"cap{cap}_{typ}_{tag}_mu"] = rec["mu_eV"]
                out[f"cap{cap}_{typ}_{tag}_sigma"] = rec["sigma_over_tau"]          # sigma/tau, S m^-1 s^-1
                out[f"cap{cap}_{typ}_{tag}_kappa_e"] = rec["kappa_e_over_tau"]      # kappa_e/tau, W m^-1 K^-1 s^-1
                out[f"cap{cap}_{typ}_{tag}_L_over_L0"] = (rec["kappa_e_over_tau"] / (rec["sigma_over_tau"] * T) / L0
                                                          if rec["sigma_over_tau"] else np.nan)   # Wiedemann-Franz ratio
                out[f"cap{cap}_{typ}_{tag}_at_cap"] = rec.get("at_cap", np.nan)     # optimum pressed against the cap
            r9 = pt[900][key]
            if r9:                                                           # 900 K headline point
                put("900", r9, 900)
            bestT = max((T for T in TEMPS if pt[T].get(key)), key=lambda T: pt[T][key]["PF_over_tau"], default=None)
            if bestT:                                                        # best over 300-1200 K (upper reference)
                put("best", pt[bestT][key], bestT)
                out[f"cap{cap}_{typ}_best_T"] = bestT
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    idx_cols = ["UUID", "composition", "RUN_run_dir", "RUN_data_set", "RUN_family", "hull distance (eV/atom)",
                "phonon computed", "phonon stable (commensurate)", "space group symbol", "number of atoms in the cell"]
    m = pd.read_excel(INDEX, "Materials")
    screen_cols = [c for c in m.columns if c.startswith("SCREEN_")]          # screening inputs (pre-DFT)
    ref_cols = [c for c in m.columns if c.startswith(("DFT_verdict", "DFT_band_gap", "BT2_", "SOC_", "VERDICT_", "RANK_"))]
    idx = m[idx_cols + screen_cols].rename(columns={"RUN_run_dir": "run_dir", "RUN_data_set": "data_set",
                                                    "RUN_family": "family", "hull distance (eV/atom)": "hull"})
    rows = []
    for i, r in idx.iterrows():
        rd = r.run_dir
        o = {"row": i}
        g = adg.dft_gap(f"{RL}/{rd}/nscf.out")                               # (gap eV, E_F eV, class) from band energies
        if g:
            o["gap"], o["E_F"], o["dft_class"] = g
        f = first_existing(f"{RL}/{rd}", ["transport_v2.json", "transport.json"])
        o["scalar_file"] = f
        if f:
            o.update(best_point(f"{RL}/{rd}/{f}"))
        if os.path.exists(f"{RL}/{rd}/transport_w1.json"):                  # +/-1.0 eV capped re-extraction
            o.update({"w1_" + k: v for k, v in capped_points(f"{RL}/{rd}/transport_w1.json").items()})
        sf = first_existing(f"{SR}/{rd}", ["transport_v2.json", "transport.json"]) if os.path.isdir(f"{SR}/{rd}") else None
        o["soc_file"] = sf
        if sf:
            o.update({"soc_" + k: v for k, v in best_point(f"{SR}/{rd}/{sf}").items()})
            if os.path.exists(f"{SR}/{rd}/transport_w1.json"):
                o.update({"soc_w1_" + k: v for k, v in capped_points(f"{SR}/{rd}/transport_w1.json").items()})
        o["soc_failed"] = os.path.exists(f"{SR}/{rd}/FAILED")
        # explicit three-way SOC status (blocker B5): a run with no SOC directory at all (e.g. IrTaSn bccd322f) was
        # previously indistinguishable from "not needed"; the paper's SOC denominator counts only status "done"
        o["soc_status"] = ("done" if sf else
                           "failed" if (o["soc_failed"] or os.path.isdir(f"{SR}/{rd}")) else "not run")
        rows.append(o)
    d = pd.concat([idx, pd.DataFrame(rows).set_index("row")], axis=1)

    # ---- rule chain ---------------------------------------------------------------------------------
    d["heavy"] = d.composition.map(lambda s: bool(elements(s) & HEAVY))     # contains an element with Z >= 72
    d["stable"] = d.hull < HULL_MAX                                          # NaN hull (ambiguous UUID) -> False
    d["above_bar"] = d.best_PF > BAR
    d["winner"] = d.dopable.fillna(False).astype(bool) & d.above_bar & d.stable
    d["gapped"] = d.dft_class == "gapped"
    d["soc_above_bar"] = d.soc_best_PF > BAR
    d["certified"] = d.winner & d.gapped & (~d.heavy | d.soc_above_bar.fillna(False).astype(bool))
    # final (reported) values: SOC for heavy compounds, scalar otherwise
    for k in ["best_PF", "S_best", "best_T", "mu_best", "zT_ceiling", "zT_e", "L_over_L0", "sigma_best", "kappa_e_best"]:
        d["final_" + k] = np.where(d.heavy, d["soc_" + k], d[k])
    for c in [c for c in d.columns if c.startswith("w1_cap")]:               # capped values: SOC for heavy
        sc = "soc_" + c
        d["final_" + c[3:]] = np.where(d.heavy, d[sc] if sc in d else np.nan, d[c])

    # ---- self-check against the published workbook --------------------------------------------------
    ref = m[ref_cols]
    lines = []
    def chk(name, mine, theirs):
        ok = (mine.fillna(-1) == theirs.fillna(-1)).all() if mine.dtype != float else np.allclose(mine.fillna(0), theirs.fillna(0), rtol=1e-9)
        lines.append(f"{'OK  ' if ok else 'DIFF'} {name}")
        return ok
    chk("DFT class == DFT_verdict", d.dft_class, ref.DFT_verdict)
    chk("gap == DFT_band_gap_eV", d.gap.astype(float), ref.DFT_band_gap_eV.astype(float))
    chk("best PF/tau == BT2_best_PF_over_tau", d.best_PF.astype(float), ref.BT2_best_PF_over_tau.astype(float))
    chk("winner == VERDICT_PF_winner", d.winner, ref.VERDICT_PF_winner.astype(str).str.lower().isin(["yes", "true"]))
    chk("certified == VERDICT_certified", d.certified, ref.VERDICT_certified.astype(str).str.lower().isin(["yes", "true"]))
    lines.append(f"counts: runs {len(d)}, winners {int(d.winner.sum())}, gapped winners {int((d.winner & d.gapped).sum())}, "
                 f"certified entries {int(d.certified.sum())}, certified compounds {d[d.certified].composition.nunique()}")
    open(f"{OUT}/build_check.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    d.to_pickle(f"{OUT}/paper_dataset.pkl")                          # pickle (pyarrow not installed)
    d.to_excel(f"{OUT}/paper_dataset.xlsx", index=False)
    print("wrote", f"{OUT}/paper_dataset.pkl")


if __name__ == "__main__":
    main()
