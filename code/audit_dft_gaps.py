#!/usr/bin/env python3
# =============================================================================
# audit_dft_gaps.py -- extract the TRUE DFT band gap for every completed
# material and compare it against the gap the ML screen gated on.
#
# WHY THIS MATTERS: the screening pipeline gates "semiconductor" on the
# d-ORBITAL-PROJECTED DOS only (data['spd_dos']['d']['densities'] -- see
# dos_dataset.py / te_rescreen_training_local.py), and classify() looks for a
# gap in exactly that array. A gap in the d-DOS does NOT imply a band gap: s and
# p bands can still cross E_F. Confirmed on LuOsTl -- screen said semiconductor
# (gap 0.18 eV), real DFT band structure is METALLIC (band overlap -0.30 eV).
#
# This script quantifies how often that happens across the campaign, turning a
# known caveat into a measured number. It reads each material's nscf.out (kept
# on disk; note out/ is deleted by the run script's trap, but nscf.out survives)
# and decides metallicity PER SPIN CHANNEL with the assumption-free test
#     "does any band straddle E_F across k"  (min_k E_n < E_F < max_k E_n)
# which classifies metal / half_metal / gapped without needing an occupied-band
# index. See dft_gap()'s docstring for why the earlier nelec-based version was
# wrong for magnetic systems.
# =============================================================================
import csv, glob, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANCHOR = ROOT / "CMS/anchor"
RESULTS = ROOT / "results/screening"


def dft_gap(nscf_out):
    """Return (gap_eV, EF, verdict) for a finished nscf.out, or None.

    SPIN-AWARE. Metallicity is decided by the ROBUST test "does any band
    straddle E_F across k" (min_k E_n < E_F < max_k E_n), evaluated PER SPIN
    CHANNEL -- it needs no occupied-band-index assumption. An earlier version
    inferred the band index from nelec and pooled both spin channels together,
    which produced a meaningless -2.585 eV "gap" for the magnetic CrTiTl by
    comparing a spin-up level against a spin-down one. Verdicts:
        metal      - bands cross E_F in EVERY spin channel
        half_metal - gapped in one spin channel, metallic in the other
        gapped     - no band crosses E_F in any channel (gap = CBM-VBM)
    """
    try:
        t = open(nscf_out, errors="replace").read()
    except OSError:
        return None
    if "JOB DONE" not in t:
        return None
    m_ef = re.search(r"the Fermi energy is\s+([-\d.]+)", t)
    if not m_ef:
        return None
    EF = float(m_ef.group(1))
    spinpol = bool(re.search(r"-+ SPIN UP -+", t))
    sections = re.split(r"-+ SPIN (?:UP|DOWN) -+", t)[1:] if spinpol else [t]

    per_channel = []          # (crosses_EF, vbm, cbm) per spin channel
    for sec in sections:
        blocks = re.findall(r"k =.*?bands \(ev\):\s*\n\n(.*?)(?:\n\s*\n|\Z)", sec, re.S)
        kpts = []
        for b in blocks:
            vals = []
            for tok in b.split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
            if vals:
                kpts.append(sorted(vals))
        if not kpts:
            continue
        nb = min(len(k) for k in kpts)
        crosses = any(min(k[n] for k in kpts) < EF < max(k[n] for k in kpts)
                      for n in range(nb))
        below = [e for k in kpts for e in k if e <= EF]
        above = [e for k in kpts for e in k if e > EF]
        vbm = max(below) if below else None
        cbm = min(above) if above else None
        per_channel.append((crosses, vbm, cbm))

    if not per_channel:
        return None
    n_cross = sum(1 for c, _, _ in per_channel if c)
    if n_cross == len(per_channel):
        verdict, gap = "metal", 0.0
    elif n_cross > 0:
        verdict, gap = "half_metal", 0.0
    else:
        # gapped in every channel -- the true gap is the smallest channel gap
        gaps = [cbm - vbm for c, vbm, cbm in per_channel
                if vbm is not None and cbm is not None]
        gap = min(gaps) if gaps else 0.0
        verdict = "gapped" if gap >= 0.1361 else "narrow"
    return gap, EF, verdict


def load_ml_gaps():
    """Map material_id -> {pool: (etype, gap_eV)}.

    KEYED BY material_id (the UUID), NOT composition. Duplicate-formula/
    different-UUID entries genuinely exist in this data (a composition can
    appear as several structural variants with DIFFERENT etype/gap), so a
    composition-keyed lookup silently returns whichever row happened to be
    first/last and can report the wrong label -- exactly the mistake this
    project's standing rule ("match by exact material_id, never by formula")
    exists to prevent.
    """
    ml = {}
    for path, pool in ((RESULTS / "rescreen_unified_ranked_ALL_annotated.csv", "novel"),
                       (RESULTS / "train_rescreen_unified_ranked_ALL_annotated.csv", "train")):
        if not path.exists():
            continue
        for r in csv.DictReader(open(path, newline="")):
            try:
                g = float(r["gap_eV"])
            except (TypeError, ValueError, KeyError):
                continue
            ml.setdefault(r["material_id"], {})[pool] = (r.get("etype", "?"), g)
    return ml


def load_campaign_ids():
    """composition -> material_id for the batch-3 campaign (runs_local/ dirs are
    keyed by composition, but the ML lookup must go through the exact UUID)."""
    p = ANCHOR / "batch3_selection.csv"
    if not p.exists():
        return {}
    return {r["composition"]: r["material_id"]
            for r in csv.DictReader(open(p, newline=""))}


def main():
    ml = load_ml_gaps()
    ids = load_campaign_ids()
    rows = []
    for nscf in sorted(glob.glob(str(ANCHOR / "runs_local/*/nscf.out"))):
        comp = os.path.basename(os.path.dirname(nscf))
        res = dft_gap(nscf)
        if res is None:
            continue
        gap, EF, verdict = res
        mid = ids.get(comp)                    # exact UUID this campaign actually ran
        e = ml.get(mid, {}) if mid else {}
        nov = e.get("novel"); tra = e.get("train")
        rows.append({
            "composition": comp,
            "material_id": mid or "",
            "dft_gap_eV": round(gap, 4),
            "dft_verdict": verdict,
            "ml_novel_etype": nov[0] if nov else "",
            "ml_novel_gap_eV": round(nov[1], 4) if nov else "",
            "ml_train_etype": tra[0] if tra else "",
            "ml_train_gap_eV": round(tra[1], 4) if tra else "",
        })
    if not rows:
        print("no completed nscf.out found yet"); return
    out = ANCHOR / "dft_gap_audit.csv"
    with open(out, "w", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    n = len(rows)
    metals = sum(1 for r in rows if r["dft_verdict"] == "metal")
    halfm  = sum(1 for r in rows if r["dft_verdict"] == "half_metal")
    narrow = sum(1 for r in rows if r["dft_verdict"] == "narrow")
    print(f"=== TRUE DFT band gap vs ML screen gate  (n={n} completed) ===")
    print(f"{'material':11s} {'DFT gap':>9s} {'verdict':>8s}  {'ML(novel)':>22s}  {'ML(train)':>22s}")
    for r in rows:
        nv = f"{r['ml_novel_etype']}/{r['ml_novel_gap_eV']}" if r["ml_novel_etype"] else "-"
        tv = f"{r['ml_train_etype']}/{r['ml_train_gap_eV']}" if r["ml_train_etype"] else "-"
        print(f"{r['composition']:11s} {r['dft_gap_eV']:9.4f} {r['dft_verdict']:>8s}  {nv:>22s}  {tv:>22s}")
    print(f"\n  metal           : {metals}/{n} ({100*metals/n:.0f}%)")
    print(f"  half_metal      : {halfm}/{n} ({100*halfm/n:.0f}%)")
    print(f"  narrow (<smear) : {narrow}/{n}")
    print(f"  genuinely gapped: {n-metals-halfm-narrow}/{n}")
    print(f"  >> NOT a true semiconductor: {metals+halfm}/{n} ({100*(metals+halfm)/n:.0f}%)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
