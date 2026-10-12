#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The authors listed in CITATION.cff; see CMS/anchor/COPYING.
# widen copy (2026-10-08): identical to recovery/recover_one.sh except output transport_w1.json, +/-1.0 eV doping window, widen/ logs+scratch.
# Recover ONE material's transport from its nscf.out text.
# Writes runs_local/<rd>/transport_w1.json and NEVER touches transport.json.
# Each invocation works in its own scratch dir so 8 can run in parallel.
set -u
rd="$1"
A=${HEUSLER_TE_ROOT:?set HEUSLER_TE_ROOT}/CMS/anchor
export HOME=$HOME; export PATH="$HOME/.local/bin:$PATH"
R=$A/runs_local/$rd
OUT=$R/transport_w1.json
LOG=$A/widen/logs/$rd.log
mkdir -p "$A/widen/logs" "$A/widen/scratch"   # log and scratch folders (mktemp below needs the scratch parent)
[ -s "$OUT" ] && { echo "SKIP $rd (already recovered)"; exit 0; }

W=$(mktemp -d "$A/widen/scratch/${rd}.XXXXXX") || { echo "FAIL $rd mktemp"; exit 1; }
trap 'rm -rf "$W"' EXIT

{
  # 1. nscf.out -> GENE text (.structure + .energy)
  python3 "$A/soc_campaign/nscfout_to_gene.py" "$rd" --src "$A/runs_local" --dest "$W/gene" \
    || { echo "FAIL $rd convert"; exit 1; }
  # 2. interpolate. -n 1: interpolation is small and our own benchmark showed
  #    QE/FFT-style parallelism anti-scales here; 8 materials x 1 rank beats
  #    1 material x 8 ranks.
  cd "$W/gene" || exit 1
  btp2 -n 1 interpolate -m 15 . > interp.log 2>&1 \
    || { echo "FAIL $rd interpolate"; tail -3 interp.log; exit 1; }
  # 3. assert the loaded data matches what nscf.out says -- catches a silent
  #    mis-parse (wrong nspin, truncated k-list, dropped bands)
  #    The heredoc is quoted ('PY'), so the shell expands nothing inside it:
  #    the run name and the CMS/anchor path are passed as arguments instead.
  python3 - "$rd" "$A" <<'PY' || { echo "FAIL $rd assert"; exit 1; }
import sys, re
from BoltzTraP2 import serialization
rd = sys.argv[1]                       # run directory name under runs_local/
A = sys.argv[2]                        # absolute CMS/anchor path, resolved by the shell above
b = open(f"{A}/runs_local/{rd}/nscf.out", "rb").read()
nk = int(re.search(rb"number of k points=\s*(\d+)", b).group(1))
nspin = 2 if (b"SPIN UP" in b or b"total magnetization" in b) else 1
d, eq, c, m = serialization.load_calculation("interpolation.bt2")
assert d.kpoints.shape[0] == nk, f"kpoints {d.kpoints.shape[0]} != nscf {nk}"
want = 1.0 if nspin == 2 else 2.0
assert d.dosweight == want, f"dosweight {d.dosweight} != {want} for nspin={nspin}"
print(f"  ok kpts={nk} nspin={nspin} dosweight={d.dosweight} bands={d.ebands.shape[0]}")
PY
  # 4. transport, same extractor as the campaign
  python3 "$A/extract_transport.py" --comp "$rd" --window_eV 1.0 > "$W/t.json" 2> "$W/extract.err" \
    || { echo "FAIL $rd extract"; head -3 "$W/extract.err"; exit 1; }
  python3 -c "
import json,sys
j=json.load(open('$W/t.json'))
assert len(j.get('per_temperature',[]))>=10, 'too few temperatures'
" || { echo "FAIL $rd sanity"; exit 1; }
  mv "$W/t.json" "$OUT"
  echo "OK $rd"
} > "$LOG" 2>&1
tail -1 "$LOG"
