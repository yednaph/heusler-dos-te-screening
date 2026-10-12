#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The authors listed in CITATION.cff; see CMS/anchor/COPYING.
# widen copy (2026-10-08): identical to recovery/recover_soc_one.sh except output transport_w1.json, +/-1.0 eV doping window, widen/ logs+scratch.
# Same text-recovery, but for the SOC campaign runs (soc_campaign/runs/<rd>).
# Those were produced with the same stale-XML bug, so their transport.json used
# the 9x9x9 shifted SCF mesh too.
set -u
rd="$1"
A=${HEUSLER_TE_ROOT:?set HEUSLER_TE_ROOT}/CMS/anchor
export HOME=$HOME; export PATH="$HOME/.local/bin:$PATH"
R=$A/soc_campaign/runs/$rd
OUT=$R/transport_w1.json
[ -s "$OUT" ] && { echo "SKIP $rd"; exit 0; }
[ -s "$R/nscf.out" ] || { echo "FAIL $rd no nscf.out"; exit 1; }
mkdir -p "$A/widen/logs" "$A/widen/scratch"   # log and scratch folders (mktemp below needs the scratch parent)
W=$(mktemp -d "$A/widen/scratch/soc_${rd}.XXXXXX") || exit 1
trap 'rm -rf "$W"' EXIT
{
python3 "$A/soc_campaign/nscfout_to_gene.py" "$rd" --src "$A/soc_campaign/runs" --dest "$W/gene" || { echo "FAIL $rd convert"; exit 1; }
cd "$W/gene" || exit 1
btp2 -n 1 interpolate -m 15 . > i.log 2>&1 || { echo "FAIL $rd interp"; tail -3 i.log; exit 1; }
# GENE infers dosweight from its nspin field. A NONCOLLINEAR run prints no
# SPIN UP/DOWN markers, so the converter reads nspin=1 and GENE assigns
# dosweight=2.0 -- wrong: each spinor band holds ONE electron. Correct it on the
# serialized object, and verify the SOC run really was noncollinear first.
python3 -c "
import re
from BoltzTraP2 import serialization
b=open('$R/scf.out','rb').read()
assert b'spin-orbit' in b, 'scf.out does not report a spin-orbit run'
d,eq,c,m=serialization.load_calculation('interpolation.bt2')
if d.dosweight != 1.0:
    d.dosweight = 1.0                     # spinor bands: one electron each
    serialization.save_calculation('interpolation.bt2', d, eq, c, m)
    d,eq,c,m=serialization.load_calculation('interpolation.bt2')
assert d.dosweight==1.0
print(f'  ok kpts={d.kpoints.shape[0]} bands={d.ebands.shape[0]} dosweight={d.dosweight}')
" || { echo "FAIL $rd dosweight"; exit 1; }
python3 "$A/extract_transport.py" --comp "$rd" --window_eV 1.0 > "$W/t.json" 2>"$W/e.err" || { echo "FAIL $rd extract"; head -3 "$W/e.err"; exit 1; }
mv "$W/t.json" "$OUT"; echo "OK $rd"
} > "$A/widen/logs/soc_$rd.log" 2>&1
tail -1 "$A/widen/logs/soc_$rd.log"
