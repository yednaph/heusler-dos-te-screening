#!/usr/bin/env python3
# =============================================================================
# Local patch enabling BoltzTraP2 24.7.2 to read Quantum ESPRESSO spin-orbit
# (non-collinear) output.
#
# WHY IT IS NEEDED
# Upstream refuses non-collinear runs outright in io.py:
#     if non_collinear: raise ValueError("... not supported yet")
# Since QE requires noncolin=.true. for lspinorb, that single line blocks EVERY
# spin-orbit calculation from reaching transport.
#
# WHAT ACTUALLY HAS TO CHANGE -- and what does NOT
# The band data needs no special handling. For a non-collinear run QE writes
# lsda=false and packs ALL spinor bands into ONE eigenvalue list per k-point, so
# the existing n_spin=1 code path already reshapes it correctly.
# The real difference is OCCUPANCY: each spinor band holds ONE electron, not two.
# So dosweight must be 1.0, not 2.0. That CANNOT be inferred from the array shape
# -- a non-magnetic SOC run looks shape-identical to a scalar unpolarised run --
# which is why the flag has to be threaded through explicitly. Left unfixed,
# BoltzTraP2 would double-count every electron, placing mu wrongly and corrupting
# every Seebeck, sigma and PF value while raising no error at all.
#
# DELIBERATE LIMITATION -- magnetic SOC is still rejected
# BoltzTraP2 reconstructs the full Brillouin zone from the irreducible wedge using
# the crystal's spatial symmetry. For NON-MAGNETIC spin-orbit that stays valid:
# time reversal is preserved, so the spatial point group survives intact (verified
# empirically -- our Ag2AuAs SOC run kept all 16 symmetry operations and 75
# k-points, unchanged from scalar). For MAGNETIC spin-orbit, spin and space become
# coupled and that assumption breaks. Rather than silently mis-symmetrise those,
# this patch raises a clear error for them.
# =============================================================================
import os as _os_htr  # release v1: project paths come from the environment
_HTR = _os_htr.environ.get('HEUSLER_TE_ROOT', _os_htr.path.abspath(_os_htr.path.join(_os_htr.path.dirname(__file__), '../../..')))
_HTE = _os_htr.environ.get('HEUSLER_TE_EXPORT', _os_htr.path.join(_HTR, 'DOS_EXPORT'))
import re, shutil, sys, os

B = "~/.local/lib/python3.8/site-packages/BoltzTraP2"
IO, DFT = f"{B}/io.py", f"{B}/dft.py"

OLD_IO = '''    non_collinear = xml_bands.find("./noncolin").text == "true"
    if non_collinear:
        raise ValueError(
            "non-collinear ESPRESSO calculations are not supported yet"
        )
'''

NEW_IO = '''    non_collinear = xml_bands.find("./noncolin").text == "true"
    # LOCAL PATCH: accept non-collinear (spin-orbit) runs. QE packs all spinor
    # bands into one eigenvalue list per k point with lsda=false, so the n_spin=1
    # path below reshapes them correctly with no change. What differs is that each
    # spinor band holds ONE electron, so dosweight must be 1.0 -- flagged here for
    # ESPRESSOLoader, since it is not inferable from the array shape.
    nruter["noncollinear"] = non_collinear
    if non_collinear:
        # Magnetic SOC is still refused: BoltzTraP2 rebuilds the full BZ from the
        # irreducible wedge using the SPATIAL point group, which stays valid for
        # non-magnetic SOC (time reversal preserved) but not once spin and space
        # are coupled by a net moment. Fail loudly rather than mis-symmetrise.
        _absmag = xml_tree.find("./output/magnetization/absolute")
        if _absmag is not None and abs(float(_absmag.text)) > 1.0e-3:
            # The refusal is the DEFAULT and stays that way for the production
            # queue. It can be lifted only by an explicit opt-in, which exists so
            # the assumption can be MEASURED rather than argued: with the bypass
            # set, check_interp_fit.py re-evaluates the fitted bands at the input
            # k-points. If the assumed (spatial) point group were too large the
            # star-function fit could not pass through the data and the residual
            # would blow up. Only a residual at the non-magnetic noise floor
            # justifies trusting transport from a magnetic SOC run.
            if os.environ.get("BT2_ALLOW_MAGNETIC_SOC") != "1":
                raise ValueError(
                    "magnetic non-collinear (spin-orbit) ESPRESSO calculations are "
                    "not supported: BZ reconstruction would use the spatial point "
                    "group, which is invalid when spin and space are coupled "
                    "(absolute magnetization = %s). Set BT2_ALLOW_MAGNETIC_SOC=1 "
                    "only after check_interp_fit.py shows an acceptable residual."
                    % _absmag.text
                )
            nruter["magnetic_noncollinear"] = True
'''

OLD_DFT = '''            if self.ebands.shape[1] > 1 or (
                r["magmom"] is not None and r["magmom"].ndim > 1
            ):
                self.dosweight = 1.0
            else:
                self.dosweight = 2.0
'''

NEW_DFT = '''            # LOCAL PATCH: r["noncollinear"] added. A non-magnetic spin-orbit
            # run has n_spin == 1 and magmom None, so it is shape-identical to a
            # scalar unpolarised run and would silently take dosweight = 2.0.
            # Spinor bands hold ONE electron each, so it must be 1.0; without
            # this every electron is double-counted and mu is placed wrongly.
            if (
                self.ebands.shape[1] > 1
                or r.get("noncollinear", False)
                or (r["magmom"] is not None and r["magmom"].ndim > 1)
            ):
                self.dosweight = 1.0
            else:
                self.dosweight = 2.0
'''

def patch(path, old, new, label):
    src = open(path).read()
    if new.strip().splitlines()[1].strip() in src:
        print(f"  {label}: already patched, skipping")
        return True
    n = src.count(old)
    if n != 1:
        print(f"  {label}: FAILED -- anchor found {n} times (expected exactly 1)")
        return False
    open(path, "w").write(src.replace(old, new, 1))
    print(f"  {label}: patched OK")
    return True

print("applying BoltzTraP2 SOC patch")
ok = patch(IO, OLD_IO, NEW_IO, "io.py") and patch(DFT, OLD_DFT, NEW_DFT, "dft.py")
if not ok:
    sys.exit(1)
# remove stale bytecode so the edit takes effect
for f in os.listdir(f"{B}/__pycache__"):
    if f.startswith(("io.", "dft.")):
        os.remove(f"{B}/__pycache__/{f}")
print("cleared __pycache__ for io/dft")
