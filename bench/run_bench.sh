#!/usr/bin/env bash
# ===========================================================================
# Benchmark harness for the MPI / hybrid prime search
#
# Runs the baseline programs (serial, POSIX Threads, OpenMP) and the MPI
# programs (task1 = Open MPI, task2 = hybrid MPI + OpenMP) over the sweeps
# needed for the seven required graphs, and appends one CSV row per run.
#
#   ./run_bench.sh all            full local sweep (~20-30 min on 8 threads)
#   ./run_bench.sh quick          tiny matrix to validate the harness
#   ./run_bench.sh nsweep|psweep|dist|sieve   one local sweep only
#   ./run_bench.sh review        extra runs: P = 1 baselines at every n (theoretical
#                                 speed-up vs n), n = 3-5e7, and over-subscription P > cores
#   ./run_bench.sh caas-serial|caas-threads|caas-mpi|caas-hybrid
#                                 fixed-allocation sweeps for SLURM jobs
#                                 (launcher = srun, P/T taken from SLURM env)
#
# Output: results/<SITE>.csv  and  results/<SITE>_ranks.csv
#
# Unified CSV schema (all implementations):
#   site,impl,n,procs,threads,dist,chunk,sched,sched_chunk,run,
#   total,bcast,comp_max,comp_min,comp_avg,gather,merge,write,count,wall
# `wall` is the wall-clock time of the whole process as seen by this script.
# For task1/task2 it additionally contains the mpirun/srun launch and
# MPI_Init/Finalize, which `total` (MPI_Wtime inside the program) excludes;
# the difference is the launch overhead the speed-up graphs leave out.
# The phase columns are only filled by task1/task2 (they come straight from
# the "CSV," line those programs print).  For the baseline programs `total` is
# the wall-clock time of the whole process (their own timer stops BEFORE the
# file is written, and pthread/OpenMP allocate their n-element result array
# before starting it); the time they report themselves goes into `comp_max`.
# Speed-up is defined on the overall time including file writing,
# so `total` is what plot.py uses.
#
# Environment overrides:
#   SITE      label for this machine / cluster config (default: hostname)
#   CORES     max processes/threads to sweep (default: nproc)
#   REPEATS   runs per configuration (default 3)
#   N_BIG     n for the process/thread sweeps (default 20000000)
#   N_SIEVE   n for the sieve-kernel sweep (default 500000000)
#   DIST      MPI distribution used in the n/P sweeps (default blockcyclic)
#   OUT_DIR   scratch directory for prime output files (default /tmp/lab2_bench)
#             -> keep this on local disk or tmpfs; a bind-mounted or network
#                FS makes the serial write phase look far slower than it is,
#                and the 250 MB files of the sieve sweep stall sporadically
#                on Docker's virtual disk (run docker with --tmpfs /tmp:rw,exec,size=3g).
# ===========================================================================
set -u
cd "$(dirname "$0")"
BENCH_DIR=$PWD
LAB2_DIR=$(cd .. && pwd)
LAB1_DIR=$(cd ../../parallel-primes-pthreads-openmp && pwd)

MODE=${1:-all}
SITE=${SITE:-$(hostname -s 2>/dev/null || echo local)}
CORES=${CORES:-$(nproc 2>/dev/null || echo 8)}
REPEATS=${REPEATS:-3}
N_BIG=${N_BIG:-20000000}
N_SIEVE=${N_SIEVE:-500000000}
DIST=${DIST:-blockcyclic}
CHUNK=${CHUNK:-1024}
OUT_DIR=${OUT_DIR:-/tmp/lab2_bench}
RESULTS_DIR=$BENCH_DIR/results

# 32 values of n (the spec asks for at least 30); weighted toward large n so
# that run times stay well above the ~1 s noise floor on the parallel runs.
N_VALUES=(
    200000 400000 600000 800000 1000000
    1500000 2000000 2500000 3000000 3500000
    4000000 4500000 5000000 5500000 6000000
    6500000 7000000 7500000 8000000 8500000
    9000000 9500000 10000000 11000000 12000000
    13000000 14000000 15000000 16000000 17000000
    18000000 20000000
)

# Local launcher.  --use-hwthread-cpus: count SMT threads as slots (a 4C/8T
# laptop otherwise refuses -np 8).  --bind-to none: Open MPI pins np<=2 runs
# to a single core by default, which cripples the hybrid 1 x T runs.
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
LAUNCHER=${LAUNCHER:-mpirun}      # "mpirun" locally, "srun" inside a SLURM job
launch() {                        # launch <procs> <command...>
    local p=$1; shift
    # stdin is redirected from /dev/null: mpirun/srun otherwise swallow the
    # stdin of an enclosing `while read` loop and the loop ends after one run.
    if [ "$LAUNCHER" = srun ]; then
        # newer SLURM does not pass --cpus-per-task from sbatch to srun
        srun -n "$p" --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" "$@" < /dev/null
    else
        mpirun --use-hwthread-cpus --oversubscribe --bind-to none -np "$p" "$@" < /dev/null
    fi
}

mkdir -p "$OUT_DIR" "$RESULTS_DIR"
CSV=$RESULTS_DIR/$SITE.csv
RANKS_CSV=$RESULTS_DIR/${SITE}_ranks.csv
HEADER="site,impl,n,procs,threads,dist,chunk,sched,sched_chunk,run,total,bcast,comp_max,comp_min,comp_avg,gather,merge,write,count,wall"
[ -f "$CSV" ] || echo "$HEADER" > "$CSV"
[ -f "$RANKS_CSV" ] || echo "site,impl,n,procs,threads,dist,chunk,run,rank,compute,count" > "$RANKS_CSV"

# ---------------------------------------------------------------- build ----
build() {
    (cd "$LAB2_DIR" && make -s all) || { echo "build of task1/task2 failed"; exit 1; }
    gcc -O2 -o "$OUT_DIR/serial"  "$LAB1_DIR/task1.c" -lm          || exit 1
    gcc -O2 -o "$OUT_DIR/pthread" "$LAB1_DIR/task2.c" -pthread -lm || exit 1
    gcc -O2 -fopenmp -o "$OUT_DIR/openmp" "$LAB1_DIR/task3.c" -lm  || exit 1
    gcc -O2 -o "$OUT_DIR/serial_sieve" "$BENCH_DIR/serial_sieve.c" -lm || exit 1
}

# ------------------------------------------------------------- runners ----
# The baseline programs write ./primes_output*.txt, so run them inside OUT_DIR.
run_baseline() {                       # run_baseline <impl> <n> <threads> <run>
    local impl=$1 n=$2 t=$3 run=$4 out total reported count sched="-" schunk=0 t0 t1
    t0=$(date +%s.%N)
    case $impl in
        serial)  out=$(cd "$OUT_DIR" && ./serial "$n") ;;
        pthread) out=$(cd "$OUT_DIR" && ./pthread "$n" "$t") ;;
        openmp)  out=$(cd "$OUT_DIR" && ./openmp "$n" "$t"); sched=static; schunk=1 ;;
        serial_sieve) out=$(cd "$OUT_DIR" && ./serial_sieve "$n" "$OUT_DIR/out.txt") ;;
    esac
    t1=$(date +%s.%N)
    total=$(awk "BEGIN { printf \"%.6f\", $t1 - $t0 }")
    reported=$(echo "$out" | sed -n 's/.*time taken = \([0-9.]*\) seconds.*/\1/p')
    count=$(echo "$out" | sed -n 's/.*primes found = \([0-9]*\).*/\1/p')
    echo "$SITE,$impl,$n,1,$t,-,0,$sched,$schunk,$run,${total:-NA},,${reported:-},,,,,,${count:-NA},${total:-NA}" >> "$CSV"
}

run_mpi() {                         # run_mpi <n> <procs> <dist> <chunk> <run> [verbose] [kernel]
    local n=$1 p=$2 dist=$3 chunk=$4 run=$5 verbose=${6:-} kernel=${7:-trial} out line t0 t1 wall
    t0=$(date +%s.%N)
    out=$(PRIMES_VERBOSE=$verbose launch "$p" "$LAB2_DIR/task1" "$n" "$OUT_DIR/out.txt" "$dist" "$chunk" "$kernel" 2>&1)
    t1=$(date +%s.%N)
    wall=$(awk "BEGIN { printf \"%.6f\", $t1 - $t0 }")
    line=$(echo "$out" | grep '^CSV,')
    if [ -z "$line" ]; then echo "  WARNING task1 n=$n np=$p $dist: $out" >&2; return; fi
    # CSV,impl,n,procs,threads,dist,chunk,sched,sched_chunk,total,... -> insert site and run, append wall
    echo "$line" | awk -F, -v site="$SITE" -v run="$run" -v wall="$wall" 'BEGIN{OFS=","}{ $1=site; $9=$9 OFS run; print $0, wall }' >> "$CSV"
    if [ -n "$verbose" ]; then
        echo "$out" | grep '^RANK,' | awk -F, -v s="$SITE,$(echo "$line" | cut -d, -f2),$n,$p,1,$dist,$chunk,$run" 'BEGIN{OFS=","}{ print s, $2, $3, $NF }' >> "$RANKS_CSV"
    fi
}

run_hybrid() {                      # run_hybrid <n> <procs> <threads> <dist> <chunk> <sched> <schunk> <run> [verbose] [kernel]
    local n=$1 p=$2 t=$3 dist=$4 chunk=$5 sched=$6 schunk=$7 run=$8 verbose=${9:-} kernel=${10:-trial} out line t0 t1 wall
    t0=$(date +%s.%N)
    out=$(PRIMES_VERBOSE=$verbose OMP_NUM_THREADS=$t launch "$p" "$LAB2_DIR/task2" "$n" "$OUT_DIR/out.txt" "$t" "$dist" "$chunk" "$sched" "$schunk" "$kernel" 2>&1)
    t1=$(date +%s.%N)
    wall=$(awk "BEGIN { printf \"%.6f\", $t1 - $t0 }")
    line=$(echo "$out" | grep '^CSV,')
    if [ -z "$line" ]; then echo "  WARNING task2 n=$n np=$p t=$t: $out" >&2; return; fi
    echo "$line" | awk -F, -v site="$SITE" -v run="$run" -v wall="$wall" 'BEGIN{OFS=","}{ $1=site; $9=$9 OFS run; print $0, wall }' >> "$CSV"
    if [ -n "$verbose" ]; then
        echo "$out" | grep '^RANK,' | awk -F, -v s="$SITE,$(echo "$line" | cut -d, -f2),$n,$p,$t,$dist,$chunk,$run" 'BEGIN{OFS=","}{ print s, $2, $3, $NF }' >> "$RANKS_CSV"
    fi
}

# Hybrid (P, T) combinations with P*T <= CORES, both powers of two.
hybrid_combos() {
    local p t
    for p in 1 2 4 8 16 32; do
        for t in 1 2 4 8 16 32; do
            [ $((p * t)) -le "$CORES" ] && echo "$p $t"
        done
    done
}

# -------------------------------------------------------------- sweeps ----
# Graphs 1, 2 and Task 3 "increasing n": every implementation at full core
# count, over all 32 values of n.
sweep_n() {
    echo "=== n sweep: ${#N_VALUES[@]} values, CORES=$CORES, REPEATS=$REPEATS ==="
    local n r hp=$(( CORES >= 2 ? CORES / 2 : 1 ))
    for n in "${N_VALUES[@]}"; do
        for r in $(seq 1 "$REPEATS"); do
            run_baseline serial  "$n" 1 "$r"
            run_baseline pthread "$n" "$CORES" "$r"
            run_baseline openmp  "$n" "$CORES" "$r"
            run_mpi "$n" "$CORES" "$DIST" "$CHUNK" "$r"
            run_hybrid "$n" "$hp" 2 "$DIST" "$CHUNK" dynamic 64 "$r"
        done
        echo "  n=$n done"
    done
}

# Graphs 3-7: fixed large n, increasing processes / threads.
sweep_p() {
    echo "=== P/T sweep at n=$N_BIG, 1..$CORES ==="
    local p t r combo
    for r in $(seq 1 "$REPEATS"); do
        run_baseline serial "$N_BIG" 1 "$r"
        for p in $(seq 1 "$CORES"); do
            run_baseline pthread "$N_BIG" "$p" "$r"
            run_baseline openmp  "$N_BIG" "$p" "$r"
            run_mpi "$N_BIG" "$p" "$DIST" "$CHUNK" "$r"
        done
        while read -r p t; do
            run_hybrid "$N_BIG" "$p" "$t" "$DIST" "$CHUNK" dynamic 64 "$r"
        done < <(hybrid_combos)
        echo "  repeat $r done"
    done
}

# Task 1/2 "explore different workload distributions": all four schemes at
# several process counts, chunk sensitivity, OpenMP schedule sensitivity,
# with per-rank compute times captured for the load-balance chart.
sweep_dist() {
    echo "=== distribution sweep at n=$N_BIG ==="
    local p d c s r
    for r in $(seq 1 "$REPEATS"); do
        for d in block cyclic blockcyclic wblock; do
            # cyclic's residue-class trap only shows at P = 3, 5, 6, 7: sweep
            # every P for the interleaved schemes, powers of two for the rest
            for p in $(seq 1 "$CORES"); do
                case "$d $p" in block\ [3567]|wblock\ [3567]|block\ 1[0-5]|wblock\ 1[0-5]) continue;; esac
                [ "$p" -le "$CORES" ] && run_mpi "$N_BIG" "$p" "$d" "$CHUNK" "$r" 1
            done
        done
        for c in 16 64 256 1024 4096 16384; do
            run_mpi "$N_BIG" "$CORES" blockcyclic "$c" "$r"
        done
        for s in static dynamic guided; do
            for c in 1 16 64 256; do
                run_hybrid "$N_BIG" 2 $(( CORES >= 2 ? CORES / 2 : 1 )) "$DIST" "$CHUNK" "$s" "$c" "$r"
            done
        done
        echo "  repeat $r done"
    done
}

# Sieve kernel (from earlier review feedback: "could be optimised with a segmented sieve").
# Compared against its own serial baseline (serial_sieve.c); with the sieve
# the per-candidate cost is uniform, so `block` (no merge) is the natural
# scheme.  n is much larger than N_BIG because the sieve is ~100x faster.
sweep_sieve() {
    echo "=== sieve sweep at n=$N_SIEVE (block) and trial-vs-sieve at n=$N_BIG ==="
    local p t r n
    for r in $(seq 1 "$REPEATS"); do
        run_baseline serial_sieve "$N_SIEVE" 1 "$r"
        for p in $(seq 1 "$CORES"); do
            run_mpi "$N_SIEVE" "$p" block "$CHUNK" "$r" 1 sieve
        done
        run_mpi "$N_SIEVE" "$CORES" blockcyclic "$CHUNK" "$r" 1 sieve
        run_mpi "$N_SIEVE" "$CORES" wblock "$CHUNK" "$r" 1 sieve
        while read -r p t; do
            run_hybrid "$N_SIEVE" "$p" "$t" block "$CHUNK" dynamic 4 "$r" "" sieve
        done < <(hybrid_combos)
        # same n as the trial-division sweeps, for the direct kernel comparison
        run_baseline serial_sieve "$N_BIG" 1 "$r"
        run_mpi "$N_BIG" "$CORES" block "$CHUNK" "$r" "" sieve
        run_hybrid "$N_BIG" $(( CORES >= 2 ? CORES / 2 : 1 )) 2 block "$CHUNK" dynamic 4 "$r" "" sieve
        # a few n values for the sieve, full core count
        for n in 50000000 100000000 200000000 1000000000; do
            [ "$n" -le "$N_SIEVE" ] || continue
            run_baseline serial_sieve "$n" 1 "$r"
            run_mpi "$n" "$CORES" block "$CHUNK" "$r" "" sieve
        done
        echo "  repeat $r done"
    done
}

# Extra measurements added after re-reading the spec:
#  * MPI P = 1 and hybrid 1 x 1 at every n -> serial fraction s(n) -> theoretical
#    speed-up with increasing n (Task 3 asks for all three axes)
#  * three larger n so the parallel runs are well above the 1 s noise floor
#  * P > cores (over-subscription): "what happens if you use more processes /
#    threads than cores?"
sweep_review() {
    echo "=== review sweep: P=1 baselines at every n, n=3-5e7, over-subscription ==="
    local n r p hp=$(( CORES >= 2 ? CORES / 2 : 1 ))
    for r in $(seq 1 "$REPEATS"); do
        for n in "${N_VALUES[@]}"; do
            run_mpi "$n" 1 "$DIST" "$CHUNK" "$r"
            run_hybrid "$n" 1 1 "$DIST" "$CHUNK" dynamic 64 "$r"
        done
        echo "  P=1 baselines, repeat $r done"
        for n in 30000000 40000000 50000000; do
            run_baseline serial  "$n" 1 "$r"
            run_baseline pthread "$n" "$CORES" "$r"
            run_baseline openmp  "$n" "$CORES" "$r"
            run_mpi "$n" "$CORES" "$DIST" "$CHUNK" "$r"
            run_hybrid "$n" "$hp" 2 "$DIST" "$CHUNK" dynamic 64 "$r"
            run_mpi "$n" 1 "$DIST" "$CHUNK" "$r"
            run_hybrid "$n" 1 1 "$DIST" "$CHUNK" dynamic 64 "$r"
        done
        echo "  large n, repeat $r done"
        for p in $(( CORES * 3 / 2 )) $(( CORES * 2 )); do
            run_baseline pthread "$N_BIG" "$p" "$r"
            run_baseline openmp  "$N_BIG" "$p" "$r"
            run_mpi "$N_BIG" "$p" "$DIST" "$CHUNK" "$r"
        done
        echo "  over-subscription, repeat $r done"
    done
}

# --------------------------------------------------------- CAAS sweeps ----
# These run inside one SLURM allocation: P and T are fixed by the job's
# --ntasks / --cpus-per-task, and the launcher is srun.  Each job must
# finish within 10 minutes, so the n list is shortened.
CAAS_N=(1000000 2000000 4000000 6000000 8000000 10000000 14000000 18000000 20000000)

caas_serial() {
    local n r
    for n in "${CAAS_N[@]}"; do for r in $(seq 1 "$REPEATS"); do run_baseline serial "$n" 1 "$r"; done; done
}
caas_threads() {                    # pthread / OpenMP at 1..cpus-per-task threads
    local n r t
    for n in "${CAAS_N[@]}"; do
        for t in 1 2 4 8 16; do
            [ "$t" -le "${SLURM_CPUS_PER_TASK:-$CORES}" ] || continue
            for r in $(seq 1 "$REPEATS"); do run_baseline pthread "$n" "$t" "$r"; run_baseline openmp "$n" "$t" "$r"; done
        done
    done
}
caas_mpi() {                        # task1 at the allocated process count
    local p=${SLURM_NTASKS:?run inside sbatch} n d r
    for n in "${CAAS_N[@]}"; do
        for d in block cyclic blockcyclic wblock; do
            for r in $(seq 1 "$REPEATS"); do run_mpi "$n" "$p" "$d" "$CHUNK" "$r" 1; done
        done
    done
    for r in $(seq 1 "$REPEATS"); do
        run_baseline serial_sieve "$N_SIEVE" 1 "$r"
        run_mpi "$N_SIEVE" "$p" block "$CHUNK" "$r" 1 sieve
    done
}
caas_hybrid() {                     # task2 at the allocated P x T
    local p=${SLURM_NTASKS:?run inside sbatch} t=${SLURM_CPUS_PER_TASK:?} n s r
    for n in "${CAAS_N[@]}"; do
        for s in static dynamic guided; do
            for r in $(seq 1 "$REPEATS"); do run_hybrid "$n" "$p" "$t" "$DIST" "$CHUNK" "$s" 64 "$r" 1; done
        done
    done
}

# ---------------------------------------------------------------- main ----
build
case $MODE in
    all)    sweep_n; sweep_p; sweep_dist; sweep_sieve; sweep_review ;;
    quick)  REPEATS=1; CORES=$(( CORES > 4 ? 4 : CORES )); N_VALUES=(200000 500000 1000000); N_BIG=1000000; N_SIEVE=20000000
            sweep_n; sweep_p; sweep_dist; sweep_sieve ;;
    nsweep) sweep_n ;;
    psweep) sweep_p ;;
    dist)   sweep_dist ;;
    sieve)  sweep_sieve ;;
    review) sweep_review ;;
    caas-serial)  LAUNCHER=srun; caas_serial ;;
    caas-threads) LAUNCHER=srun; caas_threads ;;
    caas-mpi)     LAUNCHER=srun; caas_mpi ;;
    caas-hybrid)  LAUNCHER=srun; caas_hybrid ;;
    *) echo "unknown mode: $MODE"; exit 1 ;;
esac
echo "Done. Rows in $CSV: $(($(wc -l < "$CSV") - 1))"
