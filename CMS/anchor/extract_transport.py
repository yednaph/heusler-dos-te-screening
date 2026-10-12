#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The authors listed in CITATION.cff; see CMS/anchor/COPYING.
# =============================================================================
# extract_transport.py  --  true BoltzTraP2 transport for ONE anchor material,
#                            across a TEMPERATURE SWEEP (not a single 900K point)
#
#   Reads interpolation.bt2 (in the cwd), computes the CRTA transport WITH band
#   velocities (the first-principles reference against which the DOS-derived
#   indicators are assessed), and prints one JSON with S / sigma-over-tau / kappa_e-over-tau /
#   PF-over-tau at EVERY temperature in the sweep, reported THREE ways per T:
#     - "intrinsic"      : at the undoped (mu = DFT Fermi level) point
#     - "reopt_doping"    : at THIS temperature's own PF-maximizing mu
#                          ("the best PF/tau reachable if the sample could be
#                          re-doped at every operating temperature")
#     - "fixed_doping"    : at the mu that maximises PF at --fixed_doping_T_K
#                          (default: the highest T in the sweep), same mu used
#                          for every T ("what a single real, fixed-doping
#                          sample would actually show as T changes")
#   Both conventions are standard in the TE literature (see the anchor pipeline
#   report/PR discussion, sources included there) -- reporting both rather than
#   picking one avoids re-running this expensive step to get the other later.
#
#   sigma and kappa_e are reported PER RELAXATION TIME TAU (CRTA) -- this
#   script does not, and cannot by itself, know tau. Multiply through by a
#   tau derived from an experimental reference conductivity (separate step)
#   to get absolute values. No figure of merit zT = S^2*sigma*T/kappa follows from
#   this output alone: tau and the lattice thermal conductivity are not computed.
# =============================================================================
import argparse, json, sys
import numpy as np
from BoltzTraP2 import serialization, fite
from BoltzTraP2.bandlib import BTPDOS, fermiintegrals, calc_Onsager_coefficients, BOLTZMANN

HA2EV = 27.211386245988


def iso(t):
    """Collapse a (...,3,3) tensor to its isotropic (trace/3) average."""
    return (t[..., 0, 0] + t[..., 1, 1] + t[..., 2, 2]) / 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True)
    # Default sweep: 300-1200 K in 100 K steps. 1200K (not 900K) LOCKED after
    # checking real half-Heusler TE literature: state-of-the-art compositions
    # (e.g. FeNbSb-type) report their measured peak zT near 1200 K -- most of
    # this project's current top-20 are half-Heuslers, so capping at 900K
    # risked understating exactly the candidates that dominate the shortlist.
    # Deliberately NOT extended further to 1500K: real decomposition data for
    # a closely analogous, well-studied family (TiNiSn decomposes at 1453K,
    # vs ZrNiSn/HfNiSn stable to 1700K+ -- a 250K+ swing from one element
    # change alone) means composition-specific stability can't be assumed by
    # analogy for the unstudied candidates, and measured zT literature
    # data itself thins out past 1200K. Adjust via these flags if a different
    # range/step is ever needed -- decide before running the full 20-material
    # sweep, since interpolation.bt2 is deleted after each material to
    # protect the disk quota, so adding temperatures later means re-running
    # relax+scf+nscf+interpolate from scratch, not just this script.
    ap.add_argument("--t_min", type=float, default=300.0)
    ap.add_argument("--t_max", type=float, default=1200.0)
    ap.add_argument("--t_step", type=float, default=100.0)
    ap.add_argument("--fixed_doping_t_k", type=float, default=None,
                    help="Temperature whose PF-optimal mu is used for the "
                         "fixed_doping convention at every T. Default: the "
                         "highest T in the sweep.")
    # Half-width of the chemical-potential (doping) scan around the DFT Fermi level, eV.
    # 0.5 is the campaign value (transport.json / transport_v2.json). 49 of the 80 certified
    # compounds had their PF optimum at the +/-0.5 eV edge, so a +/-1.0 eV rerun
    # (transport_w1.json, 2026-10-08) checks whether those optima lie outside the scan.
    ap.add_argument("--window_eV", type=float, default=0.5)
    a = ap.parse_args()

    Tr = np.arange(a.t_min, a.t_max + 1e-6, a.t_step)
    if len(Tr) == 0:
        sys.exit(f"  ERROR: empty temperature sweep (t_min={a.t_min}, "
                 f"t_max={a.t_max}, t_step={a.t_step})")
    fixed_t = a.fixed_doping_t_k if a.fixed_doping_t_k is not None else Tr.max()

    data, equiv, coeffs, meta = serialization.load_calculation("interpolation.bt2")
    lattvec = data.get_lattvec()
    eband, vvband = fite.getBTPbands(equiv, coeffs, lattvec)[:2]
    eps, dos, vvdos, _ = BTPDOS(eband, vvband, npts=4000)

    margin = 9.0 * BOLTZMANN * Tr.max()
    mask = (eps > eps.min() + margin) & (eps < eps.max() - margin)
    mur = eps[mask]
    win = np.abs((mur - data.fermi) * HA2EV) <= a.window_eV    # doping scan: |mu - E_F| <= window
    mur = mur[win]
    if len(mur) == 0:                                            # fallback: at least +/-1 eV (or the window, if wider)
        win = np.abs((eps[mask] - data.fermi) * HA2EV) <= max(1.0, a.window_eV)
        mur = eps[mask][win]
    if len(mur) == 0:
        sys.exit(f"  [extract] ERROR: no chemical potentials in +/-{max(1.0, a.window_eV)} eV window for {a.comp}")

    # N, L0, L1, L2 all have shape (nT, nmu); sig/seeb/kap have shape (nT, nmu, 3, 3)
    N, L0, L1, L2, Lm11 = fermiintegrals(eps, dos, vvdos, mur=mur, Tr=Tr, dosweight=data.dosweight)
    sig, seeb, kap, Hall = calc_Onsager_coefficients(L0, L1, L2, mur, Tr, data.get_volume(), Lm11)

    S_all  = iso(seeb)          # (nT, nmu)  V/K            -- absolute, tau-independent
    sg_all = iso(sig)           # (nT, nmu)  S/(m*s)        -- PER TAU
    ke_all = iso(kap)           # (nT, nmu)  W/(m*K*s)      -- PER TAU
    PF_all = S_all ** 2 * sg_all

    # Net carriers at each (T, mu), BoltzTraP2 convention (same as its 'dope' command):
    # fermiintegrals returns N = -(electrons in the interpolated bands), so N + nelect is the
    # excess HOLE count per cell (> 0 p-type, < 0 n-type). Added 2026-10-08 so the reported
    # optimum doping can be checked for physical reachability (it was not stored before).
    BOHR_CM = 0.529177210903e-8                            # bohr radius in cm (volume is in bohr^3)
    # The undoped electron count is taken from the data itself: -N at the intrinsic mu and the
    # lowest T, rounded to an integer. For gapped materials this is exact; it also stays correct
    # for SOC runs, where data.nelect does not match the spinor dosweight=1 band counting
    # (RhTaSn SOC test gave a constant 2.7e23 cm^-3 offset with data.nelect).
    i0_ = int(np.argmin(np.abs(mur - data.fermi)))         # intrinsic mu index (same as i0 below)
    nelect_eff = float(np.rint(-N[0, i0_]))                # undoped electrons per cell in the interpolated bands
    ncell = N + nelect_eff                                 # (nT, nmu) holes per cell
    ncm3 = ncell / (data.get_volume() * BOHR_CM ** 3)      # (nT, nmu) holes per cm^3

    i0 = int(np.argmin(np.abs(mur - data.fermi)))         # intrinsic (undoped) mu index

    # mu* used for the fixed_doping convention: the PF-optimal mu at fixed_t
    it_fixed = int(np.argmin(np.abs(Tr - fixed_t)))
    j_fixed = int(np.argmax(PF_all[it_fixed]))
    mu_fixed_eV = float((mur[j_fixed] - data.fermi) * HA2EV)

    def capped(iT, cap, carrier=None):                     # PF-optimal point among mu with |n| <= cap at temperature iT
        ok = np.abs(ncm3[iT]) <= cap
        if carrier == "p":                                 # p-type only: net holes (n >= 0)
            ok &= ncm3[iT] >= 0
        elif carrier == "n":                               # n-type only: net electrons (n <= 0)
            ok &= ncm3[iT] <= 0
        if not ok.any():
            return None                                    # no mu reaches a carrier density this low (e.g. metals)
        j = int(np.flatnonzero(ok)[np.argmax(PF_all[iT][ok])])
        return {
            "mu_eV": float((mur[j] - data.fermi) * HA2EV),
            "S_uVK": float(S_all[iT, j] * 1e6),
            "sigma_over_tau": float(sg_all[iT, j]),
            "kappa_e_over_tau": float(ke_all[iT, j]),
            "PF_over_tau": float(PF_all[iT, j]),
            "carriers_per_cell": float(ncell[iT, j]),
            "carrier_conc_cm3": float(ncm3[iT, j]),
            "at_cap": bool(abs(ncm3[iT, j]) > 0.9 * cap),  # optimum pressed against the cap
        }

    per_T = []
    for iT, T in enumerate(Tr):
        j_reopt = int(np.argmax(PF_all[iT]))               # THIS T's own PF-optimal mu
        per_T.append({
            "T_K": float(T),
            "intrinsic": {
                "S_uVK": float(S_all[iT, i0] * 1e6),
                "sigma_over_tau": float(sg_all[iT, i0]),
                "kappa_e_over_tau": float(ke_all[iT, i0]),
                "PF_over_tau": float(PF_all[iT, i0]),
                "carriers_per_cell": float(ncell[iT, i0]),          # + holes / - electrons per cell
                "carrier_conc_cm3": float(ncm3[iT, i0]),            # + holes / - electrons per cm^3
            },
            "reopt_doping": {
                "mu_eV": float((mur[j_reopt] - data.fermi) * HA2EV),
                "S_uVK": float(S_all[iT, j_reopt] * 1e6),
                "sigma_over_tau": float(sg_all[iT, j_reopt]),
                "kappa_e_over_tau": float(ke_all[iT, j_reopt]),
                "PF_over_tau": float(PF_all[iT, j_reopt]),
                "carriers_per_cell": float(ncell[iT, j_reopt]),          # + holes / - electrons per cell
                "carrier_conc_cm3": float(ncm3[iT, j_reopt]),            # + holes / - electrons per cm^3
            },
            # Best PF with the doping capped at a realistic carrier concentration. In CRTA (tau fixed)
            # PF keeps rising as mu moves into a band, so the uncapped optimum can sit at ~5e21 cm^-3,
            # beyond typical half-Heusler optimum doping (1e20-1e21). Caps added 2026-10-08.
            **{f"reopt_cap_{lab}": capped(iT, cap) for lab, cap in (("1e21", 1e21), ("1e20", 1e20))},
            # same caps, p- and n-type separately (a real sample is doped one way; added 2026-10-08 for fixed-n ranking)
            **{f"reopt_cap_{lab}_{c}": capped(iT, cap, c) for lab, cap in (("1e21", 1e21), ("1e20", 1e20)) for c in ("p", "n")},
            "fixed_doping": {
                "mu_eV": mu_fixed_eV,
                "S_uVK": float(S_all[iT, j_fixed] * 1e6),
                "sigma_over_tau": float(sg_all[iT, j_fixed]),
                "kappa_e_over_tau": float(ke_all[iT, j_fixed]),
                "PF_over_tau": float(PF_all[iT, j_fixed]),
                "carriers_per_cell": float(ncell[iT, j_fixed]),          # + holes / - electrons per cell
                "carrier_conc_cm3": float(ncm3[iT, j_fixed]),            # + holes / - electrons per cm^3
            },
        })

    print(json.dumps({
        "comp": a.comp,
        "temperatures_K": Tr.tolist(),
        "fixed_doping_reference_T_K": float(fixed_t),
        "fixed_doping_mu_eV": mu_fixed_eV,
        "per_temperature": per_T,
        "n_mur_pts": len(mur),
        "nelect_data": float(data.nelect),                 # electron count stored in the interpolation
        "nelect_eff": nelect_eff,                          # electron count used for carrier concentrations
        "window_eV": round(float(np.abs((mur[-1] - mur[0]) * HA2EV)), 3),
        "NOTE_units": "sigma_over_tau and kappa_e_over_tau are PER RELAXATION "
                      "TIME TAU (CRTA) -- multiply by a tau derived from an "
                      "experimental reference conductivity to get absolute "
                      "S/m and W/(m*K) before using in zT = S^2*sigma*T/kappa. "
                      "S_uVK is already absolute, no tau needed. No tau is "
                      "estimated anywhere in this script.",
        "NOTE_conventions": "reopt_doping = PF-optimal mu AT THIS temperature "
                            "(best achievable if re-doped per-T). fixed_doping "
                            "= PF-optimal mu at fixed_doping_reference_T_K, "
                            "held constant across all T (what one real fixed-"
                            "doping sample would show).",
    }))


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    main()
