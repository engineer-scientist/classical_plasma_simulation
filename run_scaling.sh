#!/usr/bin/env bash
# ============================================================================
#  run_scaling.sh  --  drive the strong- and weak-scaling sweeps
# ============================================================================
#
#  Builds both simulators, then runs them across a range of core counts and
#  appends one timing row per run to plasma_scaling.csv (done by the binaries
#  themselves).  When it finishes, run  `python3 plot_scaling.py`  to turn that
#  CSV into scaling_strong.png and scaling_weak.png.
#
#  Every run uses PLASMA_TIMING_ONLY=1, so no trajectory/animation files are
#  written -- the sweeps are pure compute and fast to repeat.
#
#  Two studies are run:
#
#    STRONG scaling -- fixed total problem size, more cores.
#        N is held at N_STRONG for every core count p.  Ideal: time ~ 1/p,
#        speedup ~ p.
#
#    WEAK scaling -- fixed work PER core, more cores.
#        The force kernel is O(N^2), so "work per core = const" means N^2/p is
#        constant, i.e. N = round(N0_WEAK * sqrt(p)).  Ideal: time ~ constant.
#
#  Everything is tunable from the environment, e.g.:
#      THREADS="1 2 4 8 16 32 48" N_STRONG=4000 STEPS=500 N0_WEAK=1500 ./run_scaling.sh
#
#  Defaults are sized to finish in a few minutes on a 48-core GCE node.
# ----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

# ---- tunables --------------------------------------------------------------
N_STRONG="${N_STRONG:-3000}"     # particles/species for the strong-scaling runs
STEPS="${STEPS:-400}"            # time steps for every run
N0_WEAK="${N0_WEAK:-1500}"      # particles/species per core for weak scaling
SEED="${SEED:-12345}"

# Core counts to test.  Default: powers of two up to nproc, plus nproc itself.
MAXC="$(nproc)"
if [[ -n "${THREADS:-}" ]]; then
    THREAD_LIST="${THREADS}"
else
    THREAD_LIST=""
    p=1
    while (( p < MAXC )); do
        THREAD_LIST+="$p "
        p=$(( p * 2 ))
    done
    THREAD_LIST+="$MAXC"
fi
# De-duplicate and sort numerically (in case nproc is itself a power of two).
THREAD_LIST="$(echo "$THREAD_LIST" | tr ' ' '\n' | sort -n -u | tr '\n' ' ')"

# Stable placement improves timing reproducibility across the sweep.
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export PLASMA_TIMING_ONLY=1

echo "=============================================================="
echo " Plasma scaling sweep"
echo "   cores tested : ${THREAD_LIST}"
echo "   strong       : N=${N_STRONG}, steps=${STEPS}  (fixed size)"
echo "   weak         : N=${N0_WEAK}*sqrt(p), steps=${STEPS}  (fixed work/core)"
echo "   output       : ${PLASMA_OUTDIR:-output}/plasma_scaling.csv"
echo "=============================================================="

# ---- build -----------------------------------------------------------------
make serial parallel

# round(N0 * sqrt(p)), computed with awk (no bc dependency).
weak_N () {
    awk -v n0="$N0_WEAK" -v p="$1" 'BEGIN { printf "%d", int(n0*sqrt(p)+0.5) }'
}

# ---- strong scaling --------------------------------------------------------
echo
echo "----- STRONG scaling (fixed N=${N_STRONG}) --------------------"
# True serial baseline (the 1-core reference for absolute speedup).
echo "  [serial ] N=${N_STRONG} steps=${STEPS}"
./plasma_sim_serial "$N_STRONG" "$STEPS" "$SEED" >/dev/null 2>&1
for p in $THREAD_LIST; do
    echo "  [par ${p}c] N=${N_STRONG} steps=${STEPS}"
    OMP_NUM_THREADS="$p" ./plasma_sim_parallel "$N_STRONG" "$STEPS" "$SEED" "$p" \
        >/dev/null 2>&1
done

# ---- weak scaling ----------------------------------------------------------
echo
echo "----- WEAK scaling (N=${N0_WEAK}*sqrt(p)) --------------------"
# Serial baseline at the 1-core problem size.
Nser="$(weak_N 1)"
echo "  [serial ] N=${Nser} steps=${STEPS}"
./plasma_sim_serial "$Nser" "$STEPS" "$SEED" >/dev/null 2>&1
for p in $THREAD_LIST; do
    Np="$(weak_N "$p")"
    echo "  [par ${p}c] N=${Np} steps=${STEPS}  (work/core held constant)"
    OMP_NUM_THREADS="$p" ./plasma_sim_parallel "$Np" "$STEPS" "$SEED" "$p" \
        >/dev/null 2>&1
done

echo
echo "Done.  Rows appended to ${PLASMA_OUTDIR:-output}/plasma_scaling.csv."
echo "Next:  python3 plot_scaling.py   (writes ${PLASMA_OUTDIR:-output}/scaling_{strong,weak}.png)"
