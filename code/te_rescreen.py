#!/usr/bin/env python3
# =============================================================================
# te_rescreen.py  --  Stage 1.5 of the rigorous TE screening funnel
#
#   Re-rank the ML-predicted DOS of the prediction-set Heuslers with CORRECT transport
#   physics, replacing the physics-inverted D34 (which rewards a DOS peak AT E_F
#   = a metal, and buries the real degenerate semiconductors).
#
#   Everything is built from the Cutler-Mott / CRTA transport moments — the same
#   integrals BoltzTraP2 uses, with the band-velocity transport function replaced
#   by the DOS g(E). So this is "BoltzTraP2 without velocities"; Stage-3 DFT adds
#   the velocities. NO second Gaussian smoothing: the predicted DOS is already
#   smooth, and the -df/dE Fermi window is the only (physical) energy averaging.
#
#   Descriptors (all from L0,L1,L2 at 900 K, mu scanned +/-1.0 eV in 0.02 eV steps;
#   +/-0.5 eV before 2026-08-22, reproducible with --mu_max 0.5):
#     A  Seebeck            S = -(1/eT) L1/L0            -> max_mu |S|
#     B  Power factor       PF = S^2 * L0                -> max_mu PF   (primary)
#     C  Mahan-Sofo sharp   (kBT)^2 / Var(transport)     (reported)
#     D  DOS-mass proxy     L0 at the PF-optimal mu      (reported; coarse mobility axis)
#   Gates: semiconductor (gap in the Mahan-Sofo window 0.1-1.0 eV; exclude metals
#   and |spin_pol|>0.9 half-metals) + stability (ML_formE<0, ML_hull<0.10,
#   Phonon_ML) + reliability (ensemble std). kappa_L proxy = mean atomic mass.
#
#   LEAN START: the shortlist consensus uses A + B only; C, D, kappa_L are
#   reported so the first-principles calculations can decide if they add value.
#
#   Usage:
#     py code/te_rescreen.py --screening_root results/screening \
#        --dxmag data/DXMag_HeuslerDB_alld_2026-03-10.csv --gap_window 0.1 1.0 --top 15
# =============================================================================
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

# ---- DOS grid (confirmed from compute_metrics.py) ---------------------------
NE      = 2001                    # energy points per spin channel
EF_IDX  = 1000                    # index of E = 0 (Fermi level)
DE      = 0.01                    # eV per grid point
E_GRID  = (np.arange(NE) - EF_IDX) * DE          # energies -10..+10 eV, E_F at 0

# ---- physics constants ------------------------------------------------------
KB   = 8.617333e-5                # eV/K
T_OP = 900.0                      # operating temperature (K) -- locked decision
KBT  = KB * T_OP                  # 0.07756 eV
BETA = 1.0 / KBT

def fermi_window(x):              # -df/dE = (beta/4)/cosh^2(beta x /2), the transport window
    return (BETA / 4.0) / np.cosh(BETA * x / 2.0) ** 2

# mu-scan grid (accessible doping range around the intrinsic E_F)
# 2026-08-22 WIDENED 0.5 -> 1.0 eV.  Reason: with a +/-0.5 eV scan, 21.8% of all
# materials had their PF-optimal mu pinned at the scan edge (|mu*| >= 0.49), i.e.
# the optimum was never actually located -- it was clipped by the search window.
# A material whose transport gap exceeds 2*MU_MAX has BOTH band edges outside the
# scan, so mu can never reach a band and its Seebeck is a search artefact, not
# physics.  MU_MAX is module-level and overridable via --mu_max so the previous
# behaviour remains exactly reproducible (--mu_max 0.5).
MU_MAX  = 1.0                                     # eV, scan half-width
MU_STEP = 0.02                                    # eV, scan resolution
MU_GRID = np.arange(-MU_MAX, MU_MAX + 1e-9, MU_STEP)      # eV

FAMILIES = ["full_cubic", "full_tetra", "inverse_cubic",
            "inverse_tetra", "half_cubic", "half_tetra"]

# compact atomic masses (u) for every element that appears in the Heusler set
ATOMIC_MASS = {
 'H':1.008,'Li':6.94,'Be':9.012,'B':10.81,'C':12.011,'N':14.007,'O':15.999,'F':18.998,
 'Na':22.99,'Mg':24.305,'Al':26.982,'Si':28.085,'P':30.974,'S':32.06,'Cl':35.45,'K':39.098,
 'Ca':40.078,'Sc':44.956,'Ti':47.867,'V':50.942,'Cr':51.996,'Mn':54.938,'Fe':55.845,
 'Co':58.933,'Ni':58.693,'Cu':63.546,'Zn':65.38,'Ga':69.723,'Ge':72.63,'As':74.922,
 'Se':78.971,'Br':79.904,'Rb':85.468,'Sr':87.62,'Y':88.906,'Zr':91.224,'Nb':92.906,
 'Mo':95.95,'Tc':98.0,'Ru':101.07,'Rh':102.906,'Pd':106.42,'Ag':107.868,'Cd':112.414,
 'In':114.818,'Sn':118.71,'Sb':121.76,'Te':127.6,'I':126.904,'Cs':132.905,'Ba':137.327,
 'La':138.905,'Ce':140.116,'Hf':178.49,'Ta':180.948,'W':183.84,'Re':186.207,'Os':190.23,
 'Ir':192.217,'Pt':195.084,'Au':196.967,'Hg':200.592,'Tl':204.383,'Pb':207.2,'Bi':208.98,
 'Lu':174.967,'Yb':173.045,'Tm':168.934,'Gd':157.25}

def mean_atomic_mass(composition):
    """Mean atomic mass from a formula string like 'VAuTl' or 'Sc3Al'."""
    toks = re.findall(r'([A-Z][a-z]?)(\d*)', composition)
    tot_m, tot_n = 0.0, 0
    for el, cnt in toks:
        if not el:
            continue
        n = int(cnt) if cnt else 1
        tot_m += ATOMIC_MASS.get(el, 100.0) * n     # 100 u fallback for any unlisted element
        tot_n += n
    return tot_m / max(tot_n, 1)


# ---------------------------------------------------------------------------
# Cutler-Mott transport descriptors for one material's total DOS g(E)
# ---------------------------------------------------------------------------
def transport_descriptors(g):
    """
    g : total d-DOS on E_GRID (states/eV), shape (2001,).
    Returns dict with A_seebeck, B_pf, C_sharp, D_dosmass, mu_star, carrier_n.
    """
    best = {"B_pf": -1.0, "A_seebeck": 0.0, "C_sharp": 0.0, "D_dosmass": 0.0,
            "mu_star": 0.0, "S_at_star": 0.0}
    A_max = 0.0
    for mu in MU_GRID:
        w  = fermi_window(E_GRID - mu)              # transport window centred at mu
        gw = g * w
        L0 = np.sum(gw) * DE
        if L0 < 1e-12:
            continue
        L1 = np.sum(gw * (E_GRID - mu)) * DE
        L2 = np.sum(gw * (E_GRID - mu) ** 2) * DE
        S  = -1e6 * (L1 / L0) / T_OP                # Seebeck in uV/K  (S = -(1/eT)L1/L0)
        PF = S * S * L0                             # power-factor proxy (uV^2/K^2 * DOS)
        var = L2 / L0 - (L1 / L0) ** 2              # transport-function variance (eV^2)
        A_max = max(A_max, abs(S))
        if PF > best["B_pf"]:
            best.update(B_pf=PF, S_at_star=S, mu_star=mu, D_dosmass=L0,
                        C_sharp=(KBT ** 2) / var if var > 1e-9 else 0.0)
    best["A_seebeck"] = A_max
    # Railing diagnostic: if the PF-optimal mu sits on the last grid point, the
    # optimum was clipped by the scan window rather than found.  Such a row's
    # A/B/C/D are lower bounds only and must not be read as converged optima.
    best["mu_railed"] = bool(abs(best["mu_star"]) >= MU_MAX - MU_STEP / 2.0)
    # carrier density accumulated between E_F and mu* (states per cell), sign = doping type
    k0, k1 = EF_IDX, EF_IDX + int(round(best["mu_star"] / DE))
    lo, hi = (k0, k1) if k1 >= k0 else (k1, k0)
    best["carrier_n"] = float(np.sum(g[lo:hi]) * DE) * (1 if k1 >= k0 else -1)
    return best


# ---------------------------------------------------------------------------
# Electronic-type classification + transport-gap width (both spin channels)
# ---------------------------------------------------------------------------
def classify(g_up, g_dn, gap_thr=0.15):
    """metal / half_metal / semiconductor, plus gap width (eV) around E_F."""
    ef = EF_IDX
    up_ef = g_up[ef-2:ef+3].mean()                 # DOS at E_F, +/-0.02 eV
    dn_ef = g_dn[ef-2:ef+3].mean()
    pol   = (up_ef - dn_ef) / (up_ef + dn_ef + 1e-9)
    g     = g_up + g_dn
    # contiguous width around E_F where total DOS < gap_thr (transport gap)
    below = g < gap_thr
    # BUG FIX (2026-08-13): gap_w required DOS below threshold EXACTLY at ef while
    # etype only needed it ANYWHERE in +/-3 bins, so ~3,663 materials were typed
    # "semiconductor" yet assigned gap_w=0.0 and then rejected for "having no gap".
    # That fix repaired the GUARD (up_gap/dn_gap below) but NOT the WALK.
    #
    # BUG FIX (2026-08-23): the walk itself still started at `ef`. If below[ef] was
    # False both loops exited immediately, giving left=right=0 and therefore
    # gap_w = max(0, -DE) = 0.0 -- no matter what the repaired guard said. A single
    # bin of DOS sitting exactly on E_F (routine in an ML-predicted DOS, and present
    # in real DFT DOS too) erased the whole gap. Demonstrated on a synthetic DOS
    # with a clean +/-0.5 eV gap plus one spike at E_F: reported 0.000 eV instead of
    # 0.990 eV. It affected 1,240 prediction-set and 706 training-set materials, and
    # cost ~17% of the genuinely useful (0.10-1.40 eV) population on full_cubic.
    #
    # The walk now starts from the local DOS minimum inside the SAME +/-3 bin window
    # the etype test already uses, so guard and walk finally agree. When below[ef] is
    # already True the start is unchanged and the returned width is bit-identical to
    # before -- this fix can only ADD gaps, never alter an existing one.
    start = ef
    if not below[ef]:
        win = np.arange(ef - 3, ef + 4)
        cand = win[below[win]]
        if len(cand):
            start = int(cand[np.argmin(np.abs(cand - ef))])   # nearest qualifying bin
    left = right = 0
    if below[start]:
        k = start
        while k > 0 and below[k]:
            left += 1; k -= 1
        k = start
        while k < NE - 1 and below[k]:
            right += 1; k += 1
    up_gap = g_up[ef-3:ef+4].min() < gap_thr
    dn_gap = g_dn[ef-3:ef+4].min() < gap_thr
    gap_w = max(0.0, (left + right - 1) * DE) if (up_gap and dn_gap) else 0.0
    if up_gap and dn_gap:
        etype = "semiconductor"
    elif (up_gap or dn_gap) and abs(pol) > 0.9:
        etype = "half_metal"
    else:
        etype = "metal"
    return etype, gap_w, up_ef, dn_ef, pol


# ---------------------------------------------------------------------------
# Per-family processing
# ---------------------------------------------------------------------------
def process_family(fam, root, dxmag):
    base = root / fam / "novel_expA_ensemble"
    pf, idf = base / "predictions.npy", base / "ids.txt"
    if not (pf.exists() and idf.exists()):
        print(f"  [skip] {fam}: no predictions"); return None
    pred = np.load(pf)
    ids  = [l.strip() for l in idf.read_text().splitlines() if l.strip()]
    stdf = base / "predictions_std.npy"
    std  = np.load(stdf) if stdf.exists() else None
    rankf = base / "te_ranking.csv"
    d34 = pd.read_csv(rankf)[["material_id", "pred_D34_TE_score"]] if rankf.exists() else None

    rows = []
    for i, mid in enumerate(ids):
        up, dn = pred[i, 0:NE], pred[i, NE:2*NE]
        g = up + dn
        d = transport_descriptors(g)
        etype, gap_w, up_ef, dn_ef, pol = classify(up, dn)
        std_fw = float(std[i, 800:1201].mean() + std[i, 2801:3202].mean()) / 2 if std is not None else np.nan
        comp = mid.split("_")[0]
        rows.append({
            "family": fam, "material_id": mid, "composition": comp,
            "etype": etype, "gap_eV": gap_w, "dos_ef": up_ef + dn_ef, "spin_pol": pol,
            "A_seebeck_uVK": d["A_seebeck"], "B_pf": d["B_pf"], "C_sharp": d["C_sharp"],
            "D_dosmass": d["D_dosmass"], "mu_star_eV": d["mu_star"],
            "S_at_star_uVK": d["S_at_star"], "carrier_n": d["carrier_n"],
            "kL_proxy_mass": mean_atomic_mass(comp), "std_fermi": std_fw,
            "mu_railed": d["mu_railed"],
        })
    df = pd.DataFrame(rows)
    if d34 is not None:
        df = df.merge(d34, on="material_id", how="left")
    # --- merge DxMag stability by UUID (best-effort) ---
    if dxmag is not None:
        df["UUID"] = df["material_id"].str.split("_", n=1).str[1].str.lower()
        df = df.merge(dxmag, on="UUID", how="left")
    print(f"  {fam:14s}: {len(df):6d} materials  "
          f"({(df.etype=='semiconductor').sum()} sc / {(df.etype=='half_metal').sum()} hm / "
          f"{(df.etype=='metal').sum()} metal)")
    return df


def load_stability(path):
    """Read the Sept-2025 DxMag xlsx (the actual UUID source for the prediction CIFs)
       -> stability keyed by (lower-cased) UUID."""
    try:
        cols = {"UUID": "UUID",
                "formation energy (eV/atom)": "ML_formE",
                "hull distance (eV/atom)": "ML_hull",
                "phonon stable (commensurate)": "phonon_stable",
                "minimum phonon frequency commensurate (cm^-1)": "min_phonon_freq"}
        raw = pd.read_excel(path, usecols=list(cols.keys()), engine="openpyxl").rename(columns=cols)
        raw["UUID"] = raw["UUID"].astype(str).str.strip().str.lower()
        for c in ("ML_formE", "ML_hull", "min_phonon_freq"):
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        print(f"  stability: {len(raw):,} rows from {path.name}")
        return raw
    except Exception as e:
        print(f"  stability read failed ({e}) -- stability gate skipped")
        return None


def main():
    # Declared up-front: argparse reads MU_MAX for its default and the parsed
    # value may rebind it, so the declaration must precede any reference.
    global MU_MAX, MU_GRID
    ap = argparse.ArgumentParser()
    ap.add_argument("--screening_root", type=Path, default=Path("results/screening"))
    ap.add_argument("--stability", type=Path, default=Path("data/DXMag_HeuslerDB_2025-09-09.xlsx"))
    ap.add_argument("--gap_window", type=float, nargs=2, default=[0.1, 1.0])
    ap.add_argument("--mu_max", type=float, default=MU_MAX,
                    help="mu-scan half-width in eV (0.5 reproduces the pre-2026-08-22 runs)")
    ap.add_argument("--hull_max", type=float, default=0.10)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # Rebuild the module-level mu grid if the caller overrode the half-width.
    if args.mu_max != MU_MAX:
        MU_MAX  = args.mu_max
        MU_GRID = np.arange(-MU_MAX, MU_MAX + 1e-9, MU_STEP)
    print(f"mu-scan: +/-{MU_MAX} eV in {MU_STEP} eV steps ({len(MU_GRID)} points)")

    dxmag = load_stability(args.stability) if args.stability and args.stability.exists() else None
    print("=== Stage 1.5 re-rank (Cutler-Mott, 900 K) ===")
    frames = [d for fam in FAMILIES if (d := process_family(fam, args.screening_root, dxmag)) is not None]
    if not frames:
        raise SystemExit("no family predictions found under " + str(args.screening_root))
    allm = pd.concat(frames, ignore_index=True)
    print(f"\nPooled: {len(allm):,} materials, {allm.family.nunique()} families")

    gmin, gmax = args.gap_window
    # --- gates ---------------------------------------------------------------
    q75 = allm["std_fermi"].quantile(0.75) if allm["std_fermi"].notna().any() else np.inf
    allm["gate_semiconductor"] = (allm.etype == "semiconductor") & allm.gap_eV.between(gmin, gmax)
    allm["gate_reliable"]      = allm["std_fermi"] < q75
    if "ML_hull" in allm.columns:
        # NaN (material not matched / property not computed) = UNKNOWN -> do NOT block
        hull_ok = (allm.ML_hull < args.hull_max) | allm.ML_hull.isna()
        form_ok = (allm.ML_formE < 0) | allm.ML_formE.isna()
        ph_bad = pd.Series(False, index=allm.index)          # reject only CLEAR phonon instability
        if "phonon_stable" in allm.columns:
            ph_bad |= allm.phonon_stable.astype(str).str.lower().isin(["false", "no", "0", "unstable"])
        if "min_phonon_freq" in allm.columns:
            ph_bad |= (allm.min_phonon_freq < -5)
        allm["gate_stable"] = hull_ok & form_ok & ~ph_bad
        allm["stab_known"]  = allm.ML_hull.notna()
    else:
        allm["gate_stable"] = True
        allm["stab_known"]  = False
    allm["passes_all"] = allm.gate_semiconductor & allm.gate_reliable & allm.gate_stable

    # ------------------------------------------------------------------
    # RECALL-FIRST ANNOTATION (2026-08-22).  Nothing below deletes a row.
    # Measured against the first 1,401 BoltzTraP2-calculated materials, `passes_all`
    # rejects 65 of the top-100 by the BoltzTraP2 ranking metric then in use (a
    # tau-free CRTA quantity, not a measured zT).  The gates are therefore
    # emitted as GRADED ANNOTATIONS so the cut point is a downstream choice
    # rather than a hardcoded one.  `passes_all` is retained unchanged for
    # backward compatibility with earlier analyses.
    # ------------------------------------------------------------------

    # Stability tier.  Hull distance is orthogonal physics -- Spearman vs every
    # DOS descriptor is ~0, so no ranker can stand in for it and it must be
    # reported, not inferred.  But the single 0.10 eV/atom cut costs 54 of the
    # true top-100, so it is graded instead of binarised.  NaN = never computed,
    # which is not evidence of instability and is kept as its own tier.
    hull_edges  = [-np.inf, 0.05, 0.10, 0.20, np.inf]
    hull_labels = ["<=0.05", "0.05-0.10", "0.10-0.20", ">0.20"]
    allm["hull_tier"] = pd.cut(allm.get("ML_hull", pd.Series(np.nan, index=allm.index)),
                               bins=hull_edges, labels=hull_labels, right=False)
    allm["hull_tier"] = allm["hull_tier"].cat.add_categories("unknown").fillna("unknown")

    # Ensemble-spread percentile.  std_fermi is POSITIVELY correlated with that
    # BoltzTraP2 metric (Spearman +0.540, n=494): the model is least certain about the best
    # materials, because high-Seebeck DOS shapes lie furthest from the training
    # distribution.  Rejecting the top quartile therefore removes good materials
    # by construction, so this is exported as a percentile for triage priority
    # and never used to drop a row.
    allm["std_fermi_pctile"] = allm.std_fermi.rank(pct=True)

    # Transport-gap tier.  The BoltzTraP2 results show performance holding to ~1.4 eV, not
    # falling off at the 1.0 eV gate ceiling; and Chasmar-Stratton puts the
    # optimum at 6-10 kBT = 0.47-0.78 eV at 900 K.  Graded, not cut.
    gap_edges  = [-np.inf, 0.10, 0.40, 1.00, 1.40, np.inf]
    gap_labels = ["<0.10", "0.10-0.40", "0.40-1.00", "1.00-1.40", ">1.40"]
    allm["gap_tier"] = pd.cut(allm.gap_eV, bins=gap_edges, labels=gap_labels, right=False)

    # ---- per-ranker ranks over the FULL population, nothing filtered ----
    # Metal-contamination measured on the 90 gap-excluded materials that carry
    # BoltzTraP2 results (baseline share 6.0%), counting metals in the
    # top-100 when the gap gate is switched off entirely:
    #     A_seebeck 0%   C_sharp 1%   B_pf 18%   D_dosmass 60%
    # A_seebeck and C_sharp therefore reproduce the gap gate on their own;
    # B_pf and D_dosmass do not and must not be used without it.
    RANKERS = {
        "A_seebeck_uVK":     "metal-clean",
        "C_sharp":           "metal-clean",
        "B_pf":              "METAL-CONTAMINATED-needs-gap-gate",
        "D_dosmass":         "METAL-DETECTOR-never-use-alone",
        "pred_D34_TE_score": "metal-clean-but-check-coverage",
    }
    for col in RANKERS:
        if col not in allm.columns:
            continue
        # Global rank, 1 = best. NaN scores stay NaN (absent, not last).
        allm[f"rank_{col}"] = allm[col].rank(ascending=False, method="min")
        # Within-family rank, so a family is never buried by another family's scale.
        allm[f"famrank_{col}"] = allm.groupby("family")[col].rank(ascending=False,
                                                                 method="min")

    # --- reject-reason bookkeeping ------------------------------------------
    def reason(r):
        if r.passes_all: return "PASS"
        rs = []
        if not r.gate_semiconductor: rs.append(f"not-semiconductor({r.etype},gap={r.gap_eV:.2f})")
        if not r.gate_reliable:      rs.append("unreliable")
        if not r.gate_stable:        rs.append("unstable")
        return ";".join(rs)
    allm["reject_reason"] = allm.apply(reason, axis=1)

    # --- LEAN consensus over A + B among survivors --------------------------
    surv = allm[allm.passes_all].copy()
    if len(surv):
        surv["rank_A"] = surv.A_seebeck_uVK.rank(pct=True)
        surv["rank_B"] = surv.B_pf.rank(pct=True)
        # confidence-penalised primary score (rank by power factor, dock high-variance)
        surv["consensus"] = surv[["rank_A", "rank_B"]].mean(axis=1)
        surv["standout"]  = (surv.rank_A > 0.90) & (surv.rank_B > 0.90)   # top-decile in BOTH
        surv = surv.sort_values("B_pf", ascending=False)

    print("\n=== gate funnel (legacy filter view) ===")
    print(f"  semiconductor+gap : {allm.gate_semiconductor.sum():,}")
    print(f"  reliable          : {allm.gate_reliable.sum():,}")
    print(f"  stable            : {allm.gate_stable.sum():,}")
    print(f"  PASS all          : {allm.passes_all.sum():,}")

    # --- recall-first view: nothing is dropped, everything is graded ---------
    print("\n=== recall-first annotation (NOTHING DELETED) ===")
    print(f"  rows carried forward : {len(allm):,}")
    railed = int(allm.mu_railed.sum()) if "mu_railed" in allm.columns else -1
    print(f"  mu* railed at scan edge: {railed:,} ({railed/len(allm)*100:.1f}%) "
          f"[was 21.8% at +/-0.5 eV]")
    print("  hull_tier:")
    for k, v in allm.hull_tier.value_counts().sort_index().items():
        print(f"    {str(k):<12} {v:>7,}")
    print("  gap_tier:")
    for k, v in allm.gap_tier.value_counts().sort_index().items():
        print(f"    {str(k):<12} {v:>7,}")
    print("  ranker coverage (non-NaN scores):")
    for col, note in RANKERS.items():
        if col in allm.columns:
            n = int(allm[col].notna().sum())
            print(f"    {col:<20} {n:>7,}/{len(allm):,}  [{note}]")
    if len(surv):
        print(f"  standouts (A&B top-decile): {surv.standout.sum():,}")

    out = args.out or (args.screening_root / "rescreen_unified_ranked.csv")
    allm.to_csv(str(out).replace(".csv", "_ALL_annotated.csv"), index=False)
    if len(surv):
        cols = ["family", "composition", "material_id", "etype", "gap_eV", "dos_ef",
                "A_seebeck_uVK", "B_pf", "S_at_star_uVK", "mu_star_eV", "carrier_n",
                "C_sharp", "D_dosmass", "kL_proxy_mass", "std_fermi",
                "pred_D34_TE_score", "ML_hull", "ML_formE", "standout"]
        cols = [c for c in cols if c in surv.columns]
        surv[cols].to_csv(out, index=False)
        print(f"\n=== Top {args.top} survivors by Power-Factor potential (900 K) ===")
        show = ["family", "composition", "etype", "gap_eV", "dos_ef",
                "A_seebeck_uVK", "B_pf", "pred_D34_TE_score"]
        show = [c for c in show if c in surv.columns]
        with pd.option_context("display.width", 200, "display.max_colwidth", 22,
                               "display.float_format", lambda v: f"{v:.3g}"):
            print(surv[show].head(args.top).to_string(index=False))
    print(f"\nwrote: {out}")
    print(f"wrote: {str(out).replace('.csv', '_ALL_annotated.csv')}")

    # --- sanity: where did the old D34 top-10 go? ---------------------------
    if "pred_D34_TE_score" in allm.columns:
        d34top = allm.sort_values("pred_D34_TE_score", ascending=False).head(10)
        print("\n=== Fate of the OLD D34 top-10 (did we demote the metals?) ===")
        for _, r in d34top.iterrows():
            print(f"  {r.composition:10s} D34={r.pred_D34_TE_score:11.1f}  "
                  f"-> {r.etype:13s} gap={r.gap_eV:.2f}  {r.reject_reason}")


if __name__ == "__main__":
    main()
