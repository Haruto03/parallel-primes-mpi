#!/usr/bin/env bash
# Correctness check: every MPI / hybrid configuration must produce exactly the
# same output file as the serial baseline (../parallel-primes-pthreads-openmp/task1.c).
#
#   ./verify.sh [n]        (default n = 1000000)
set -euo pipefail
cd "$(dirname "$0")"

N=${1:-1000000}
SERIAL_SRC=../parallel-primes-pthreads-openmp/task1.c
WORK=${TMPDIR:-/tmp}/lab2_verify.$$
# Open MPI refuses to run as root by default (e.g. inside Docker); slots are
# counted per hardware thread so -np 8 works on a 4-core/8-thread laptop.
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
MPIRUN="mpirun --use-hwthread-cpus --oversubscribe --bind-to none"

mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

make -s all
gcc -O2 -o "$WORK/serial" "$SERIAL_SRC" -lm

( cd "$WORK" && ./serial "$N" > /dev/null )      # serial always writes ./primes_output.txt
mv "$WORK/primes_output.txt" "$WORK/expected.txt"
echo "serial: $(($(wc -l < "$WORK/expected.txt") - 1)) primes"

fail=0
check() {
    local label=$1; shift
    "$@" > /dev/null
    if cmp -s "$WORK/out.txt" "$WORK/expected.txt"; then
        echo "MATCH  $label"
    else
        echo "DIFF   $label"; fail=1
    fi
}

# Small n prints to stdout (should be 2, 3, 5, 7)
echo "--- n = 10 ---"
$MPIRUN -np 3 ./task1 10 "$WORK/out.txt" cyclic
$MPIRUN -np 3 ./task2 10 "$WORK/out.txt" 2 blockcyclic 2

echo "--- n = $N ---"
for p in 1 2 3 4 8; do
    for dist in block cyclic blockcyclic wblock; do
        check "task1 np=$p $dist"  $MPIRUN -np $p ./task1 "$N" "$WORK/out.txt" $dist 512
    done
    for dist in block blockcyclic wblock; do
        check "task1 np=$p $dist sieve"  $MPIRUN -np $p ./task1 "$N" "$WORK/out.txt" $dist 512 sieve
    done
done
# sieve segment boundaries: chunk sizes around the 32768-index segment
for c in 1 7 1000 32768 40000 100000; do
    check "task1 np=3 blockcyclic chunk=$c sieve"  $MPIRUN -np 3 ./task1 "$N" "$WORK/out.txt" blockcyclic $c sieve
done
for p in 1 2 4; do
    for t in 1 2 3; do
        for dist in block cyclic blockcyclic wblock; do
            for sched in static dynamic guided; do
                check "task2 np=$p t=$t $dist $sched" \
                    $MPIRUN -np $p ./task2 "$N" "$WORK/out.txt" $t $dist 512 $sched 32
            done
        done
    done
done

for p in 1 2 4; do
    for t in 1 3; do
        for dist in block blockcyclic wblock; do
            for sched in static dynamic; do
                check "task2 np=$p t=$t $dist $sched sieve"                     $MPIRUN -np $p ./task2 "$N" "$WORK/out.txt" $t $dist 512 $sched 4 sieve
            done
        done
    done
done

if [ $fail -eq 0 ]; then echo "ALL MATCH"; else echo "SOME DIFF"; exit 1; fi
