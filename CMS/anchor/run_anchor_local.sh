#!/bin/bash
# =============================================================================
# run_anchor_local.sh -- run the DFT->BoltzTraP2 anchor pipeline on THIS PC,
# autonomously, while HPC submission is suspended.
#
# Same pipeline and same hard-won bug fixes as run_anchor_batch3*.slurm:
#   relax -> post_relax (spglib symmetrisation) -> scf -> nscf (Davidson, with
#   automatic cg fallback for the Ga failure class) -> btp2 interpolate ->
#   extract_transport.py
#
# LOCAL-SPECIFIC CHOICES (this is a workstation, not a compute node):
#   * NRANKS=8 of 16 cores, and `nice` -- the machine must stay usable.
#   * MIN_FREE_GB guard: only ~85 GB free on this disk, and QE scratch per
#     material is GBs. out/ is deleted via trap after every material, and the
#     loop aborts cleanly rather than filling the disk.
#   * QE here is 7.1; the HPC ran 7.5. The queue therefore starts with 3
#     already-HPC-completed materials so the two versions can be compared
#     before any new local result is trusted (see code/build_local_queue.py).
#
# Usage:  cd CMS/anchor && nohup bash run_anchor_local.sh > local_run.log 2>&1 &
#         tail -f local_run.log
# Resume: just re-run it -- completed materials are skipped, interrupted ones
#         resume at whichever step they reached.
# =============================================================================
set -u

ANCHOR="$(cd "$(dirname "$0")" && pwd)"
cd "$ANCHOR" || exit 1

QE_BIN="$HOME/code-installed/qe-7.1/bin"
SSSP="$ANCHOR/SSSP"
JSON="$ANCHOR/SSSP_efficiency.json"
MATERIALS="$ANCHOR/materials_local.txt"
RUNS="$ANCHOR/runs_local"
NRANKS=8
NPOOL=2                 # 4 ranks/pool -- MATCHES the original 8-rank config's
                         # ratio (8:2=4 ranks/pool), not the HPC's 2 ranks/pool.
MPI="mpirun -np $NRANKS"
# NOTE (2026-07-23, 2nd revision): 16 ranks/npool=8 (2 ranks/pool) measured
# 2-4x SLOWER relax times than the original 8 ranks/npool=2 config on this
# desktop (likely memory-bandwidth contention, not a benefit from more ranks).
# That test changed TWO variables at once (rank count AND ranks-per-pool
# ratio), which confounds the result -- this 12-rank test deliberately KEEPS
# the proven 4-ranks/pool ratio to isolate rank count as the only variable.
MIN_FREE_GB=15
QUICK_FAIL_SECONDS=120

# RELAX toggle (2026-07-23). Default OFF: build scf/nscf DIRECTLY from the CIF
# (the ML-relaxed geometry), skipping DFT vc-relax entirely -- keeps transport
# on the SAME structure the DOS screen used, avoids relax-convergence failures,
# and is ~5-10x faster (relax was the bottleneck). Results are "transport at
# the ML-relaxed geometry", NOT full-DFT-relaxed ground state. Set RELAX=1 to
# restore the iterative Pulay vc-relax path (for absolute-zT work on a single
# champion material).
RELAX="${RELAX:-0}"

command -v mpirun >/dev/null || { echo "FATAL: mpirun missing"; exit 1; }
[ -x "$QE_BIN/pw.x" ] || { echo "FATAL: pw.x missing at $QE_BIN"; exit 1; }
command -v btp2 >/dev/null || { echo "FATAL: btp2 missing"; exit 1; }
python3 -c "import spglib" 2>/dev/null || { echo "FATAL: spglib missing"; exit 1; }
[ -d "$SSSP" ] && [ -f "$JSON" ] || { echo "FATAL: SSSP not extracted"; exit 1; }
[ -f "$MATERIALS" ] || { echo "FATAL: $MATERIALS missing"; exit 1; }
mkdir -p "$RUNS"

echo "=== LOCAL anchor sweep  $(date) ==="
echo "    QE: $($QE_BIN/pw.x -h </dev/null 2>&1 | grep -m1 -o 'v\.[0-9.]*')  ranks=$NRANKS  npool=$NPOOL"
echo "    queue: $(grep -cv '^\s*\(#\|$\)' "$MATERIALS") materials"

check_disk() {
    local avail_gb
    avail_gb=$(df --output=avail "$ANCHOR" 2>/dev/null | tail -1)
    avail_gb=$(( avail_gb / 1024 / 1024 ))
    if [ "$avail_gb" -lt "$MIN_FREE_GB" ]; then
        echo "  ABORT: only ${avail_gb} GB free (need >= ${MIN_FREE_GB}). Stopping cleanly."
        exit 1
    fi
}
check_disk

# Composition -> already-assigned-plain-name marker (2026-07-28). materials.txt
# is composition-keyed end to end (runs_local/<comp>/, transport.json's "comp"
# field), but duplicate-formula/different-UUID entries are known to exist in
# this project's data (see CLAUDE.md) -- extending the queue toward the full
# 1408-material pool means SOME newly appended lines legitimately share a
# formula with a material already queued/done. Giving both the SAME plain
# directory would make the driver's `[ -f "$d/transport.json" ] && skip` guard
# silently treat the new, genuinely different material as "already done" using
# the OLD material's result (this exact risk killed the composition-only
# design for the 2026-07-25 queue extension; see SESSION_CONTEXT.md).
# Fix: the FIRST line (top to bottom) with a given composition gets the plain
# name (so every material already run under the old scheme is untouched --
# none of the current 854 entries' formulas repeat among themselves). Any
# LATER line with the same composition gets its directory disambiguated by the
# UUID embedded in its cif filename. Deterministic and restart-safe: driven
# purely by (stable) file content and line order, no persisted state needed.
declare -A SEEN_COMP=()

while IFS=$'\t' read -r comp cif fam <&3; do
  comp="${comp%%$'\r'}"; cif="${cif%%$'\r'}"; fam="${fam%%$'\r'}"
  [ -z "${comp:-}" ] && continue
  [[ "$comp" == \#* ]] && continue

  if [ -n "${SEEN_COMP[$comp]:-}" ]; then
      uuid=$(printf '%s' "$cif" | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
      d="$RUNS/${comp}__${uuid:0:8}"
  else
      d="$RUNS/$comp"
  fi
  SEEN_COMP[$comp]=1
  [ -f "$d/transport.json" ] && { echo "[skip] $comp (done)"; continue; }
  [ -f "$d/FAILED" ] && { echo "[skip] $comp (failed: $(cat "$d/FAILED"))"; continue; }
  check_disk
  echo "=== $comp ($fam)  start $(date +%H:%M:%S) ==="
  mkdir -p "$d"

  (
    exec 3<&-
    cd "$d" || exit 0
    trap 'rm -rf out interpolation.bt2' EXIT      # scratch cleanup: disk is tight

    fail_material() { echo "  [FAIL: $1] $comp"; echo "$1" > FAILED; exit 0; }

    # ---- STEPS 1+2: build scf.in/nscf.in ----------------------------------
    if [ "$RELAX" != 1 ]; then
        # NO-RELAX (default): scf/nscf DIRECTLY from the CIF (ML-relaxed
        # geometry) via spglib symmetrisation + primitive reduction. Skips DFT
        # vc-relax entirely -- transport is computed on the SAME structure the
        # DOS screen scored. Idempotent, so no resume marker needed.
        nice -n 10 python3 "$ANCHOR/generate_qe_inputs.py" --stage cif_scf_nscf \
            --cif "$ANCHOR/cifs/$cif" --sssp_dir "$SSSP" --sssp_json "$JSON" \
            --outdir . --prefix "$comp" || fail_material "gen_cif_scf_nscf_error"
    else
        # ---- STEP 1: ITERATIVE vc-relax (Pulay-stress self-consistency) ----
        # A single vc-relax fixes the plane-wave basis to the STARTING cell, so
        # once the cell relaxes the basis is slightly incomplete for the new
        # cell (Pulay stress) and the reported "converged" cell is a bit off.
        # Cure: restart vc-relax from the relaxed structure (regenerates the
        # basis) and repeat until a restart converges WITHOUT MOVING, i.e. QE
        # prints "bfgs converged in ... 1 bfgs steps". Only enabled via RELAX=1.
        MAX_RELAX_ROUNDS=8
        if [ -f RELAX_CONVERGED ]; then
            echo "  [skip relax] already self-consistent (RELAX_CONVERGED present)"
        else
            round=1
            [ -f .relax_round ] && round=$(cat .relax_round)   # resume mid-iteration
            while : ; do
                if [ "$round" -gt "$MAX_RELAX_ROUNDS" ]; then
                    fail_material "relax_not_selfconsistent_after_${MAX_RELAX_ROUNDS}_rounds"
                fi
                echo "$round" > .relax_round
                if [ "$round" -eq 1 ]; then
                    nice -n 10 python3 "$ANCHOR/generate_qe_inputs.py" --stage relax \
                        --cif "$ANCHOR/cifs/$cif" --sssp_dir "$SSSP" --sssp_json "$JSON" \
                        --outdir . --prefix "$comp" || fail_material "gen_relax_input_error"
                else
                    nice -n 10 python3 "$ANCHOR/generate_qe_inputs.py" --stage rerelax \
                        --relax_out relax_prev.out --sssp_dir "$SSSP" --sssp_json "$JSON" \
                        --outdir . --prefix "$comp" || fail_material "gen_rerelax_input_error"
                fi
                T0=$(date +%s)
                nice -n 10 $MPI "$QE_BIN/pw.x" -npool $NPOOL -in relax.in > relax.out 2>&1 </dev/null
                RC=$?; EL=$(( $(date +%s) - T0 ))
                if ! grep -q "Begin final coordinates" relax.out; then
                    { [ "$RC" -eq 143 ] || [ "$RC" -eq 137 ]; } && { echo "  [relax round $round interrupted -- will resume]"; exit 0; }
                    grep -qE "MPI_ABORT|%%%%%%%%%%%%%%%%%%%%" relax.out && fail_material "relax_qe_error"
                    grep -q "JOB DONE" relax.out && fail_material "relax_did_not_converge_in_${EL}s"
                    [ "$EL" -lt "$QUICK_FAIL_SECONDS" ] && fail_material "relax_failed_fast_rc${RC}"
                    echo "  [relax round $round incomplete, ${EL}s -- will resume]"; exit 0
                fi
                NB=$(grep "bfgs converged in" relax.out | tail -1 | grep -oE "[0-9]+ bfgs steps" | grep -oE "^[0-9]+")
                VOL=$(grep "new unit-cell volume" relax.out | tail -1 | grep -oE "[0-9.]+ Ang" | grep -oE "[0-9.]+")
                echo "  [relax round $round: converged in ${NB:-?} bfgs steps, vol=${VOL:-?} Ang^3, ${EL}s]"
                if [ "${NB:-99}" -le 1 ]; then
                    echo "  [relax SELF-CONSISTENT after $round round(s) -- cell stable on restart]"
                    touch RELAX_CONVERGED
                    rm -f .relax_round relax_prev.out
                    break
                fi
                cp relax.out relax_prev.out
                round=$((round + 1))
            done
        fi
        # ---- STEP 2: scf/nscf inputs from the RELAXED structure ----
        nice -n 10 python3 "$ANCHOR/generate_qe_inputs.py" --stage post_relax --relax_out relax.out \
            --sssp_dir "$SSSP" --sssp_json "$JSON" --outdir . --prefix "$comp" \
            || fail_material "gen_post_relax_input_error"
    fi

    # ---- STEP 3: SCF, with automatic gentler-mixing retry ------------------
    # Magnetic Heuslers can charge-slosh: the SCF accuracy oscillates instead of
    # converging under the default mixing_beta=0.3 and burns all 300 iterations.
    # Real case: CoCrIn (5.37 muB) stalled at ~1e-3 Ry vs a 1e-8 target, failing
    # under BOTH the relax and no-relax pipelines. Oscillating (not diverging)
    # is the signature of too-aggressive mixing, so retry once with a smaller
    # beta before giving up -- same self-healing pattern as the nscf david->cg
    # fallback below.
    if [ -d out ] && [ -f scf.out ] && grep -q "convergence has been achieved" scf.out; then
        echo "  [skip scf] already converged"
    else
        # Retry ladder. Attempt 1 = default FERROMAGNETIC start (correct and fast
        # for most materials). Attempt 2 = FERRIMAGNETIC/AFM start: MAG3D gives
        # every magnetic atom a positive moment, but Mn2YZ Heuslers and Cr-Co
        # pairs are ferrimagnets whose sublattices point OPPOSITE ways -- an FM
        # start traps the SCF in the wrong spin topology and it oscillates.
        # PROVEN: Mn2TaBi burned 2793s failing under FM (and failed again at
        # beta=0.1), then converged in 23 iterations / 229s with --afm, ending at
        # exactly 0.00 muB -- the signature of a compensated ferrimagnet.
        # (A beta=0.1 tier was tried first and removed: it ran 6 times and
        # rescued 0 materials, while doubling the cost of every failure.)
        SCF_JUST_RAN=1
        SCF_OK=0
        for ATTEMPT in fm afm; do
            if [ "$ATTEMPT" = afm ]; then
                echo "  [scf: retrying with FERRIMAGNETIC (antiparallel) starting moments]"
                if [ "$RELAX" != 1 ]; then
                    nice -n 10 python3 "$ANCHOR/generate_qe_inputs.py" --stage cif_scf_nscf \
                        --cif "$ANCHOR/cifs/$cif" --sssp_dir "$SSSP" --sssp_json "$JSON" \
                        --outdir . --prefix "$comp" --afm \
                        || fail_material "gen_cif_scf_nscf_error"
                else
                    nice -n 10 python3 "$ANCHOR/generate_qe_inputs.py" --stage post_relax \
                        --relax_out relax.out --sssp_dir "$SSSP" --sssp_json "$JSON" \
                        --outdir . --prefix "$comp" --afm \
                        || fail_material "gen_post_relax_input_error"
                fi
            fi
            T0=$(date +%s)
            nice -n 10 $MPI "$QE_BIN/pw.x" -npool $NPOOL -in scf.in > scf.out 2>&1 </dev/null
            RC=$?; EL=$(( $(date +%s) - T0 ))
            if grep -q "convergence has been achieved" scf.out; then
                SCF_OK=1
                [ "$ATTEMPT" = afm ] && echo "  [scf CONVERGED with ferrimagnetic start (${EL}s)]"
                break
            fi
            { [ "$RC" -eq 143 ] || [ "$RC" -eq 137 ]; } && { echo "  [scf interrupted]"; exit 0; }
            grep -qE "MPI_ABORT|%%%%%%%%%%%%%%%%%%%%" scf.out && fail_material "scf_qe_error"
            if grep -q "JOB DONE" scf.out; then
                [ "$ATTEMPT" = afm ] && fail_material "scf_no_converge_fm_or_afm_in_${EL}s"
                echo "  [scf did not converge with FM start (${EL}s) -- retrying ferrimagnetic]"
                continue
            fi
            [ "$EL" -lt "$QUICK_FAIL_SECONDS" ] && fail_material "scf_not_converged"
            echo "  [scf incomplete, ${EL}s -- will resume]"; exit 0
        done
        [ "$SCF_OK" -eq 1 ] || fail_material "scf_not_converged"
    fi

    # ---- STEP 4: NSCF, Davidson with automatic cg fallback (Ga fix) -------
    if [ -z "${SCF_JUST_RAN:-}" ] && [ -d out ] && [ -f nscf.out ] && grep -q "JOB DONE" nscf.out; then
        echo "  [skip nscf] already done"
    else
        NSCF_OK=0
        for DIAG in david cg; do
            if [ "$DIAG" = cg ]; then
                echo "  [nscf: retrying with conjugate-gradient diagonalization]"
                # Regenerate the inputs with cg diagonalisation. This MUST branch
                # on RELAX exactly like the scf/afm step above: under the
                # no-relax DEFAULT there is no relax.out, so calling
                # --stage post_relax here died with FileNotFoundError and lost
                # the material (Fe2HfGe 23:06, Fe2TaGa 23:58 -- and Ga is the
                # very failure class this cg fallback exists to rescue).
                # It must ALSO carry the ferrimagnetic flag through: if the SCF
                # converged from a split-species AFM start (Mn/Mn1 with opposite
                # moments), regenerating without --afm would write an nscf.in
                # whose ATOMIC_SPECIES no longer match the charge density left
                # in out/. Detect that from the negative starting_magnetization
                # already sitting in the converged scf.in.
                AFM_FLAG=""
                grep -qE 'starting_magnetization\([0-9]+\)[[:space:]]*=[[:space:]]*-' scf.in 2>/dev/null \
                    && AFM_FLAG="--afm"
                if [ "$RELAX" != 1 ]; then
                    nice -n 10 python3 "$ANCHOR/generate_qe_inputs.py" --stage cif_scf_nscf \
                        --cif "$ANCHOR/cifs/$cif" --sssp_dir "$SSSP" --sssp_json "$JSON" \
                        --outdir . --prefix "$comp" --nscf_diag cg $AFM_FLAG \
                        || fail_material "gen_cif_scf_nscf_error"
                else
                    nice -n 10 python3 "$ANCHOR/generate_qe_inputs.py" --stage post_relax \
                        --relax_out relax.out --sssp_dir "$SSSP" --sssp_json "$JSON" \
                        --outdir . --prefix "$comp" --nscf_diag cg $AFM_FLAG \
                        || fail_material "gen_post_relax_input_error"
                fi
            fi
            T0=$(date +%s)
            nice -n 10 $MPI "$QE_BIN/pw.x" -npool $NPOOL -in nscf.in > nscf.out 2>&1 </dev/null
            RC=$?; EL=$(( $(date +%s) - T0 ))
            if grep -q "JOB DONE" nscf.out; then NSCF_OK=1; break; fi
            { [ "$RC" -eq 143 ] || [ "$RC" -eq 137 ]; } && { echo "  [nscf interrupted]"; exit 0; }
            if grep -qE "MPI_ABORT|%%%%%%%%%%%%%%%%%%%%" nscf.out; then
                [ "$DIAG" = cg ] && fail_material "nscf_qe_error_even_with_cg"
                echo "  [nscf Davidson error -- retrying with cg]"; continue
            fi
            if [ "$EL" -lt "$QUICK_FAIL_SECONDS" ]; then
                [ "$DIAG" = cg ] && fail_material "nscf_cg_failed_fast"
                echo "  [nscf failed fast -- retrying with cg]"; continue
            fi
            echo "  [nscf incomplete, ${EL}s -- will resume]"; exit 0
        done
        [ "$NSCF_OK" -eq 1 ] || fail_material "nscf_failed"
    fi

    # ---- STEP 5/6: BoltzTraP2 interpolate + transport ---------------------
    nice -n 10 btp2 -n $NRANKS interpolate -m 15 ./out/*.save > interp.out 2>&1 </dev/null
    [ -f interpolation.bt2 ] || { tail -5 interp.out; fail_material "interpolate_failed"; }
    nice -n 10 python3 "$ANCHOR/extract_transport.py" --comp "$comp" > transport.json 2>extract.err \
        || { rm -f transport.json; fail_material "extract_failed"; }
    echo "  [DONE] $comp"
  )
  echo "  === $comp finished  $(date +%H:%M:%S) ==="
done 3< "$MATERIALS"

n_done=$(find "$RUNS" -mindepth 2 -maxdepth 2 -name transport.json 2>/dev/null | wc -l)
n_fail=$(find "$RUNS" -mindepth 2 -maxdepth 2 -name FAILED 2>/dev/null | wc -l)
echo ""
echo "=== LOCAL SWEEP COMPLETE: ${n_done} done, ${n_fail} failed -- $(date) ==="
