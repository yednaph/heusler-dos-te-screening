# CMS/anchor: first-principles workflow (Quantum ESPRESSO + BoltzTraP2)

The 3,625 scalar-relativistic and 607 completed spin–orbit (SOC) calculations of the study were run with these scripts on one
workstation (Python 3.8.10, Quantum ESPRESSO 7.1, BoltzTraP2 24.7.2, spglib 2.0.2). Licence: GPL-3.0-or-later for everything
in this directory (`COPYING`, `LICENSE-GPL-3.0-NOTICE.md`). Settings are those of the manuscript (Methods): PBE, SSSP
efficiency potentials, Marzari–Vanderbilt smearing 0.01 Ry, DXMag cell reduced to its primitive cell with spglib (tolerance
10⁻³) and not relaxed, BoltzTraP2 interpolation multiplier 15, constant relaxation time, 300–1200 K.

The drivers read `HEUSLER_TE_ROOT` (the repository root) and find `pw.x` at `$HOME/code-installed/qe-7.1/bin` (variable `QE_BIN`
in `run_anchor_local.sh`, `QE` in `soc_campaign/run_soc.sh`; edit it for another installation) and `btp2` on `PATH`.

## Pipeline order

| Step | Script | Reads | Writes (per run directory) |
|---|---|---|---|
| 0 | `boltztrap2_patch/apply_patch.py [BoltzTraP2 package dir]` (once) | the installed BoltzTraP2 24.7.2 (`io.py`, `dft.py`) | patched package: accepts non-magnetic SOC output with one electron per spinor band; refuses magnetic SOC unless `BT2_ALLOW_MAGNETIC_SOC=1` |
| 1 | `run_anchor_local.sh` | `materials_local.txt`, `cifs/`, `SSSP/`, `SSSP_efficiency.json` | `runs_local/<run_dir>/`: `scf.in`, `nscf.in` (via `generate_qe_inputs.py --stage cif_scf_nscf`), `scf.out`, `nscf.out`, `transport.json`, or `FAILED` |
| 2 | mesh correction (see below) | `runs_local/<run_dir>/nscf.out` | `transport_v2.json` |
| 3 | `widen/drive.sh [workers]` → `widen/widen_one.sh`, `widen/widen_soc_one.sh` | `nscf.out` of each scalar and SOC run; `widen/worklist_scalar.txt`, `widen/worklist_soc.txt` | `transport_w1.json` (μ window ±1.0 eV, carrier-concentration-capped optima) |
| 4 | `soc_campaign/gen_soc_inputs.py <list.csv> <dest>` | a CSV with a `run_dir` column; the scalar `scf.in` of each run; `pseudos_rel/` with `pseudo_index.json` | `<dest>/<run_dir>/scf.in`, `nscf.in` (same cell, positions and k-meshes as the scalar run; fully relativistic PAW; cutoffs and band counts per compound) |
| 5 | `soc_campaign/run_soc.sh` | `soc_campaign/soc_queue.txt` (one `run_dir` per line), `soc_campaign/runs/<run_dir>/` | `scf.out`, `nscf.out`, `transport.json`, or `FAILED` |
| — | `soc_campaign/check_interp_fit.py <interpolation.bt2>[=label] ...` | BoltzTraP2 interpolation files | fit residual at the input k-points: the test that decides whether a magnetic SOC run may be trusted |

`code/audit_dft_gaps.py` (outside this directory, MIT) then reads every `nscf.out` and assigns the electronic class and the QE-PBE
gap, and `code/paper/build_paper_dataset.py` turns the run directories into `results/paper_analysis/paper_dataset.pkl`.

## File meanings

| File | Meaning |
|---|---|
| `materials_local.txt` | Queue of the scalar campaign: `name<TAB>cif file name<TAB>family` per line; `#` lines are skipped. The name is a composition or an explicit `<composition>__<first 8 hex of the UUID>`; when a plain composition repeats, the first line gets `runs_local/<composition>/` and later ones `<composition>__<first 8 hex of the UUID in the CIF name>` |
| `cifs/` | Input CIF of each queued entry: the DXMag cell with every atom listed. The campaign's files still carry the DXMag space-group line, which `generate_qe_inputs.py` ignores; the P 1 files now written by `code/generate_prediction_cifs_from_dxmag.py` give identical QE inputs |
| `scf.in`, `nscf.in`, `scf.out`, `nscf.out` | QE inputs and outputs. `nscf.out` is written with `verbosity='high'`, so it holds every k-point and eigenvalue |
| `transport.json` | BoltzTraP2 output of `extract_transport.py`: S, σ/τ, κₑ/τ and PF/τ at every temperature, at the undoped point, at the PF-optimal μ of each temperature (`reopt_doping`, window ±0.5 eV) and at a fixed doping |
| `transport_v2.json` | The same quantities recomputed from the NSCF eigenvalues printed in `nscf.out`. The original driver interpolated the eigenvalues of a stale SCF file (the shifted SCF mesh instead of the NSCF mesh); `soc_campaign/nscfout_to_gene.py` rebuilds BoltzTraP2 input from the text output, so no DFT step is repeated. The paper uses `transport_v2.json` wherever it exists (3,619 of 3,625 scalar runs) |
| `transport_w1.json` | As `transport_v2.json`, with the μ window widened to ±1.0 eV and optima at capped carrier concentration (\|n\| ≤ 10²⁰ and 10²¹ cm⁻³, p and n type separately) |
| `FAILED` | One-word failure reason; the run is skipped on restart |

**Mesh correction (step 2).** `transport_v2.json` was produced by a script identical to `widen/widen_one.sh` (and
`widen_soc_one.sh` for SOC runs) except that it wrote `transport_v2.json` and called `extract_transport.py` with its default
±0.5 eV window (no `--window_eV 1.0`). That copy is not included; to regenerate the files, run the widen scripts with those two
changes.

## Restarting and failures

Every driver skips finished runs (`transport*.json` present) and runs marked `FAILED`, so an interrupted sweep is resumed by
running it again. `run_anchor_local.sh` deletes the QE scratch folder `out/` and `interpolation.bt2` after each run and stops
when less than 15 GB of disk remain. `run_soc.sh` refuses an interpolation whose spinor bands do not carry one electron each.
