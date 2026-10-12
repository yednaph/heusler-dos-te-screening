#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The authors listed in CITATION.cff; see CMS/anchor/COPYING.
# widen/drive.sh -- +/-1.0 eV doping-window re-extraction (transport_w1.json) for every scalar and SOC run,
# rebuilt from the saved nscf.out text (no new DFT). Restart-safe: finished materials are skipped.
# P parallel single-rank workers (default 4: the other 4 physical cores may be running a DFT job).
set -u
A=${HEUSLER_TE_ROOT:?set HEUSLER_TE_ROOT}/CMS/anchor
P=${1:-4}                                                  # number of parallel workers
cd "$A/widen" || exit 1
# One thread per worker: numpy/BLAS otherwise spawns a thread per core in every worker (load 39 on 8 cores, 2026-10-08)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "=== widen start $(date) -- $(wc -l < worklist_scalar.txt) scalar + $(wc -l < worklist_soc.txt) SOC, $P-way ==="
nice -n 10 xargs -a worklist_scalar.txt -P "$P" -n 1 bash "$A/widen/widen_one.sh"        # scalar runs
nice -n 10 xargs -a worklist_soc.txt    -P "$P" -n 1 bash "$A/widen/widen_soc_one.sh"    # SOC runs
echo "=== widen end $(date) ==="
ok=$(grep -l '^OK' logs/*.log 2>/dev/null | wc -l); bad=$(grep -l '^FAIL' logs/*.log 2>/dev/null | wc -l)
echo "OK: $ok   FAIL: $bad"
[ "$bad" -eq 0 ]                                           # non-zero exit if anything failed (re-running retries only the failed runs)
