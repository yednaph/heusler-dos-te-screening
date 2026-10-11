#!/bin/bash
# =============================================================================
# Spin-orbit BoltzTraP2 campaign driver.
#
# Independent of the main scalar campaign: its own directory, own queue, own log.
# It does NOT touch runs_local/, so the 3,610 scalar results stay untouched and
# the scalar/SOC comparison stays possible.
#
# Per material: scf -> nscf -> btp2 interpolate -> extract_transport -> CLEAN UP.
# The cleanup is mandatory, not tidiness: each SOC out/ is ~435 MB and the disk is
# at 92% (77 GB free). 15 materials without cleanup would be ~6.5 GB, and the full
# 40-material tier would not fit alongside a margin.
# =============================================================================
set -u
export HOME=$HOME
export PATH="$HOME/.local/bin:$PATH"
[ -z "${I_MPI_ROOT:-}" ] && [ -f /opt/intel/oneapi/setvars.sh ] && source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
QE=$HOME/code-installed/qe-7.1/bin
A=${HEUSLER_TE_ROOT:?set HEUSLER_TE_ROOT}/CMS/anchor
C=$A/soc_campaign
NR=${NR:-8}            # MPI ranks per material
NPOOL=${NPOOL:-2}

command -v btp2 >/dev/null || { echo "FATAL: btp2 missing"; exit 1; }
[ -x "$QE/pw.x" ]        || { echo "FATAL: pw.x missing"; exit 1; }

echo "=== SOC sweep start $(date) ===  ranks=$NR npool=$NPOOL"
echo "    queue: $(grep -vc '^#' $C/soc_queue.txt) materials"

while read -r rd; do
  case "$rd" in \#*|"") continue;; esac
  d=$C/runs/$rd
  [ -d "$d" ] || { echo "  [skip] $rd (no inputs)"; continue; }
  [ -s "$d/transport.json" ] && { echo "  [skip] $rd (done)"; continue; }
  [ -f "$d/FAILED" ] && { echo "  [skip] $rd (failed earlier: $(cat $d/FAILED))"; continue; }

  cd "$d" || continue
  echo "=== $rd start $(date +%T) ==="
  rm -rf out interpolation.bt2

  t0=$(date +%s)
  nice -n 10 mpirun -np $NR "$QE/pw.x" -npool $NPOOL -in scf.in > scf.out 2>&1 </dev/null
  if ! grep -aq "convergence has been achieved" scf.out; then
      echo "scf_no_converge" > FAILED; echo "  [FAIL] $rd scf"; rm -rf out; continue; fi
  echo "  scf ok ($(( $(date +%s)-t0 ))s)  $(grep -a -m1 'number of electrons' scf.out | sed 's/.*= *//')"

  t1=$(date +%s)
  nice -n 10 mpirun -np $NR "$QE/pw.x" -npool $NPOOL -in nscf.in > nscf.out 2>&1 </dev/null
  if ! grep -aq "End of band structure" nscf.out; then
      echo "nscf_failed" > FAILED; echo "  [FAIL] $rd nscf"; rm -rf out; continue; fi
  echo "  nscf ok ($(( $(date +%s)-t1 ))s)  $(grep -a -m1 'number of k points=' nscf.out | awk '{print $5}') kpts"

  nice -n 10 btp2 -n $NR interpolate -m 15 ./out/*.save > interp.out 2>&1 </dev/null
  [ -f interpolation.bt2 ] || { echo "interpolate_failed" > FAILED
      echo "  [FAIL] $rd interpolate:"; tail -3 interp.out; rm -rf out; continue; }
  # confirm the SOC path was taken: spinor bands must carry dosweight 1.0
  python3 -c "
from BoltzTraP2 import serialization
d,_,_,_=serialization.load_calculation('interpolation.bt2')
assert d.dosweight==1.0, 'dosweight=%s -- SOC NOT detected, refusing' % d.dosweight
print('  dosweight=%s nelect=%s ebands=%s' % (d.dosweight,d.nelect,d.ebands.shape))
" || { echo "dosweight_wrong" > FAILED; echo "  [FAIL] $rd dosweight"; rm -rf out; continue; }

  nice -n 10 python3 "$A/extract_transport.py" --comp "$rd" > transport.json 2>extract.err \
      || { rm -f transport.json; echo "extract_failed" > FAILED; echo "  [FAIL] $rd extract"; rm -rf out; continue; }

  rm -rf out interpolation.bt2          # reclaim ~435 MB immediately
  echo "  [DONE] $rd  total $(( $(date +%s)-t0 ))s"
done 3< /dev/null < $C/soc_queue.txt

n=$(find $C/runs -mindepth 2 -maxdepth 2 -name transport.json -size +0 | wc -l)
echo "=== SOC SWEEP COMPLETE $(date) -- $n done ==="
