/*
 * Prime search using Open MPI
 *
 * Finds every prime strictly less than n and writes them, sorted, to an
 * output file (or stdout when n < 100), in exactly the format of the serial
 * serial program so the two outputs can be diffed.
 *
 * Usage:
 *   mpirun -np <P> ./task1 <n> [output_file] [dist] [chunk] [kernel]
 *     output_file : where the sorted list is written (default primes.txt).
 *                   Benchmarks should point this at fast local storage
 *                   (e.g. /tmp) so the serial write phase is not dominated
 *                   by a slow network / bind-mounted file system.
 *     dist        : blockcyclic (default) | block | cyclic | wblock
 *     chunk       : chunk size for blockcyclic (default 1024)
 *     kernel      : trial (default) | sieve
 *
 * Design
 * ------
 * Only the root process reads the arguments; n and the tuning parameters
 * are disseminated with a single MPI_Bcast.
 *
 * The candidate set is the odd numbers 3, 5, 7, ... < n.  Candidate j is the
 * number k = 3 + 2j, for j in [0, M) with M = (n - 2) / 2.  Even numbers are
 * never tested; the prime 2 is prepended by the root.  Four ways of
 * splitting the index range [0, M) over P processes are implemented:
 *
 *   block        rank r owns the contiguous slice [r*M/P, (r+1)*M/P).
 *                Cheapest to merge (concatenation) but, with trial division,
 *                unbalanced: is_prime(k) costs O(sqrt(k)) and the last rank
 *                gets the largest k.
 *   cyclic       rank r owns j = r, r+P, r+2P, ...  Every rank sees the same
 *                spread of small and large k, so the sqrt(k) cost looks
 *                balanced -- but this is a residue-class partition: rank r
 *                holds exactly the k = 3 + 2r (mod 2P).  Whenever an odd
 *                prime q divides P, one rank receives ONLY multiples of q
 *                (P = 3: rank 0 gets 3, 9, 15, ...), which is_prime rejects
 *                on its first division, while the other ranks get none of
 *                those cheap composites.  Measured: P = 3 is no faster than
 *                P = 2 and P = 6 no faster than P = 4.  Powers of two are
 *                safe because k is always odd.  Kept as an experiment.
 *   blockcyclic  (default) chunks of `chunk` consecutive indices are dealt
 *                round-robin.  A chunk spans 2*chunk consecutive integers and
 *                so contains every residue class: balanced for any P, with
 *                the same fine-grained interleave of small and large k as
 *                cyclic.  Needs the merge at the root.
 *   wblock       weighted block: contiguous slices whose boundaries follow a
 *                cost model.  Trial division costs O(sqrt(k)), so the work in
 *                [0, x] grows like x^(3/2) and equal-work cut points are
 *                x_r = n * (r/P)^(2/3).  Balanced like cyclic, but the
 *                slices are contiguous so no merge is needed at the root.
 *
 * Two primality kernels:
 *
 *   trial        trial division up to sqrt(k), identical to the serial baseline
 *                program, so the speed-up against it measures parallelisation
 *                only.
 *   sieve        segmented sieve of Eratosthenes (the optimisation suggested
 *                in earlier review feedback).  Every rank first sieves the odd
 *                primes up to sqrt(n) (tiny: 3401 primes for n = 1e9), then
 *                strikes their multiples out of its own candidate ranges in
 *                L1-sized segments of 32768 odd numbers.  Cost per candidate
 *                is O(log log n) and uniform, so for this kernel `block` is
 *                already balanced and needs no merge.  Not available with
 *                `cyclic`, which is not made of contiguous ranges.
 *
 * Every rank builds a sorted local list, pre-sized from the prime number
 * theorem for its own range so realloc essentially never runs (the number of
 * reallocs is reported).  Root collects the counts (MPI_Gather), then the
 * lists (MPI_Gatherv), and merges the P sorted runs with a min-heap in
 * O(N log P).  For the contiguous schemes (block, wblock) the concatenation
 * is already sorted and the merge is skipped.
 *
 * The output file is written with a hand-rolled integer-to-ASCII routine and
 * a 1 MiB buffer instead of one fprintf per prime: the write is a serial
 * phase, so its cost directly limits the achievable speed-up (Amdahl).
 *
 * Wall-clock time is measured with MPI_Wtime for each phase (broadcast,
 * compute, gather, merge, file write) so that the serial and parallel
 * fractions needed for Amdahl's / Gustafson's Law can be derived (Task 3).
 * The reported total includes communication, sorting and file writing.
 * Set PRIMES_VERBOSE=1 to also print one line per rank (compute time and
 * number of primes found) for load-balance analysis.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>

#define ROOT 0
#define DEFAULT_OUTPUT "primes.txt"

enum { DIST_BLOCK = 0, DIST_CYCLIC = 1, DIST_BLOCKCYCLIC = 2, DIST_WBLOCK = 3 };
static const char *DIST_NAMES[] = { "block", "cyclic", "blockcyclic", "wblock" };

enum { KERNEL_TRIAL = 0, KERNEL_SIEVE = 1 };
static const char *KERNEL_NAMES[] = { "trial", "sieve" };

/* Contiguous schemes produce globally sorted output by concatenation. */
static bool dist_is_contiguous(int dist) {
    return dist == DIST_BLOCK || dist == DIST_WBLOCK;
}

/* Growable array of primes owned by one rank. */
typedef struct {
    long long *data;
    long long  count;
    long long  capacity;
    int        reallocs;     /* how often the estimate was too small */
} PrimeList;

/* Trial division up to sqrt(k), skipping even divisors.
 * Identical to the serial baseline (parallel-primes-pthreads-openmp/task1.c) so timing
 * differences reflect parallelisation only. */
static bool is_prime(long long k) {
    if (k < 2) return false;
    if (k == 2) return true;
    if (k % 2 == 0) return false;
    long long limit = (long long) sqrt((double) k);
    for (long long i = 3; i <= limit; i += 2) {
        if (k % i == 0) return false;
    }
    return true;
}

static int list_init(PrimeList *l, long long capacity) {
    if (capacity < 16) capacity = 16;
    l->data = (long long *)malloc(capacity * sizeof(long long));
    l->count = 0;
    l->capacity = capacity;
    l->reallocs = 0;
    return l->data != NULL;
}

static int list_push(PrimeList *l, long long value) {
    if (l->count == l->capacity) {
        long long new_cap = l->capacity * 2;
        long long *tmp = (long long *)realloc(l->data, new_cap * sizeof(long long));
        if (tmp == NULL) return 0;
        l->data = tmp;
        l->capacity = new_cap;
        l->reallocs++;
    }
    l->data[l->count++] = value;
    return 1;
}

/* Prime number theorem, pi(x) ~ x / (ln x - 1), with a 30% margin.
 * Used to pre-size the local list so realloc is not hit during the timed
 * compute phase. */
static long long estimate_pi(long long x) {
    if (x < 100) return 30;
    return (long long)(1.3 * (double)x / (log((double)x) - 1.0)) + 16;
}

/* Slice [lo, hi) of the index range [0, M) owned by `rank`.
 * Plain block: equal-length slices.  Weighted block: boundaries at
 * M * (r/P)^(2/3), the equal-work cut points under the sqrt(k) cost model. */
static void block_bounds(long long M, int rank, int size, bool weighted,
                         long long *lo, long long *hi) {
    if (weighted) {
        *lo = (long long)((double)M * pow((double)rank / size, 2.0 / 3.0));
        *hi = (rank + 1 == size) ? M
            : (long long)((double)M * pow((double)(rank + 1) / size, 2.0 / 3.0));
    } else {
        *lo = (M * rank) / size;
        *hi = (M * (rank + 1)) / size;
    }
}

/* ---- Segmented sieve kernel --------------------------------------------- */

#define SEG_INDICES (1 << 15)   /* 32768 odd numbers per segment: 32 KB of marks, fits L1 */

/* Odd primes 3 <= p <= limit by a plain sieve.  limit ~ sqrt(n), so this is
 * tiny and every rank computes it redundantly rather than communicating. */
static long long *small_odd_primes(long long limit, int *count) {
    long long *primes = NULL;
    *count = 0;
    if (limit < 3) return primes;
    unsigned char *composite = (unsigned char *)calloc(limit + 1, 1);
    primes = (long long *)malloc((limit / 2 + 1) * sizeof(long long));
    if (composite == NULL || primes == NULL) { free(composite); free(primes); return NULL; }
    for (long long p = 3; p <= limit; p += 2) {
        if (composite[p]) continue;
        primes[(*count)++] = p;
        for (long long m = p * p; m <= limit; m += 2 * p) composite[m] = 1;
    }
    free(composite);
    return primes;
}

/* Sieve the candidate index range [lo, hi) (k = 3 + 2j), segment by segment,
 * appending the primes found to `out`.  `mark` must hold SEG_INDICES bytes. */
static int sieve_range(long long lo, long long hi, const long long *sp, int nsp,
                       unsigned char *mark, PrimeList *out) {
    for (long long seg_lo = lo; seg_lo < hi; seg_lo += SEG_INDICES) {
        long long seg_hi = seg_lo + SEG_INDICES;
        if (seg_hi > hi) seg_hi = hi;
        long long k_lo = 3 + 2 * seg_lo;          /* first candidate of the segment */
        long long k_hi = 3 + 2 * seg_hi;          /* one past the last */
        long long len = seg_hi - seg_lo;
        memset(mark, 0, (size_t)len);

        for (int i = 0; i < nsp; i++) {
            long long p = sp[i];
            if (p * p >= k_hi) break;
            long long start = ((k_lo + p - 1) / p) * p;   /* first multiple >= k_lo */
            if (start < p * p) start = p * p;
            if (start % 2 == 0) start += p;               /* only odd multiples are candidates */
            for (long long m = start; m < k_hi; m += 2 * p) {
                mark[(m - k_lo) / 2] = 1;
            }
        }
        for (long long i = 0; i < len; i++) {
            if (!mark[i] && !list_push(out, k_lo + 2 * i)) return 0;
        }
    }
    return 1;
}

/* ---- Per-rank computation ------------------------------------------------ */

/* Compute the primes among this rank's share of the candidates.
 * Candidates are indexed j in [0, M) with k = 3 + 2j.  The list produced is
 * sorted ascending because every scheme visits its indices in order.
 * Returns 0 on allocation failure. */
static int find_local_primes(long long n, int rank, int size, int dist,
                             long long chunk, int kernel, PrimeList *out) {
    long long M = (n >= 3) ? (n - 2) / 2 : 0;
    long long num_chunks = (M + chunk - 1) / chunk;
    long long lo = 0, hi = 0;
    if (dist_is_contiguous(dist)) block_bounds(M, rank, size, dist == DIST_WBLOCK, &lo, &hi);

    /* Pre-size from the prime number theorem: for a contiguous slice use the
     * primes expected in [k_lo, k_hi); otherwise an equal share of pi(n). */
    long long cap = dist_is_contiguous(dist)
        ? estimate_pi(3 + 2 * hi) - estimate_pi(3 + 2 * lo) + 16
        : estimate_pi(n) / size + 16;
    if (!list_init(out, cap)) return 0;

    /* Root owns the only even prime. */
    if (rank == ROOT && n > 2) {
        if (!list_push(out, 2)) return 0;
    }

    long long *sp = NULL;
    unsigned char *mark = NULL;
    int nsp = 0;
    if (kernel == KERNEL_SIEVE) {
        sp = small_odd_primes((long long)sqrt((double)n) + 1, &nsp);
        mark = (unsigned char *)malloc(SEG_INDICES);
        if ((sp == NULL && nsp > 0) || mark == NULL) { free(sp); free(mark); return 0; }
    }

    int ok = 1;
    switch (dist) {
    case DIST_BLOCK:
    case DIST_WBLOCK:
        if (kernel == KERNEL_SIEVE) {
            ok = sieve_range(lo, hi, sp, nsp, mark, out);
        } else {
            for (long long j = lo; j < hi && ok; j++) {
                long long k = 3 + 2 * j;
                if (is_prime(k)) ok = list_push(out, k);
            }
        }
        break;
    case DIST_CYCLIC:
        /* trial division only: the residue classes are not ranges */
        for (long long j = rank; j < M && ok; j += size) {
            long long k = 3 + 2 * j;
            if (is_prime(k)) ok = list_push(out, k);
        }
        break;
    case DIST_BLOCKCYCLIC:
        for (long long c = rank; c < num_chunks && ok; c += size) {
            long long c_lo = c * chunk;
            long long c_hi = c_lo + chunk;
            if (c_hi > M) c_hi = M;
            if (kernel == KERNEL_SIEVE) {
                ok = sieve_range(c_lo, c_hi, sp, nsp, mark, out);
            } else {
                for (long long j = c_lo; j < c_hi && ok; j++) {
                    long long k = 3 + 2 * j;
                    if (is_prime(k)) ok = list_push(out, k);
                }
            }
        }
        break;
    default:
        ok = 0;
    }
    free(sp);
    free(mark);
    return ok;
}

/* ---- k-way merge of sorted runs with a binary min-heap ------------------ */

typedef struct {
    long long value;
    int       run;
} HeapItem;

static void heap_sift_down(HeapItem *h, int size, int i) {
    for (;;) {
        int l = 2 * i + 1, r = l + 1, m = i;
        if (l < size && h[l].value < h[m].value) m = l;
        if (r < size && h[r].value < h[m].value) m = r;
        if (m == i) break;
        HeapItem t = h[i]; h[i] = h[m]; h[m] = t;
        i = m;
    }
}

/* Merge `runs` sorted sub-arrays of `src` (described by counts/displs, as
 * produced by MPI_Gatherv) into `dst`.  O(N log runs). */
static int merge_sorted_runs(const long long *src, const int *counts,
                             const int *displs, int runs, long long *dst) {
    HeapItem *heap = (HeapItem *)malloc(runs * sizeof(HeapItem));
    long long *pos = (long long *)malloc(runs * sizeof(long long));
    if (heap == NULL || pos == NULL) { free(heap); free(pos); return 0; }

    int hsize = 0;
    for (int r = 0; r < runs; r++) {
        pos[r] = 0;
        if (counts[r] > 0) {
            heap[hsize].value = src[displs[r]];
            heap[hsize].run = r;
            hsize++;
        }
    }
    for (int i = hsize / 2 - 1; i >= 0; i--) heap_sift_down(heap, hsize, i);

    long long out = 0;
    while (hsize > 0) {
        int r = heap[0].run;
        dst[out++] = heap[0].value;
        pos[r]++;
        if (pos[r] < counts[r]) {
            heap[0].value = src[displs[r] + pos[r]];
        } else {
            heap[0] = heap[--hsize];
        }
        heap_sift_down(heap, hsize, 0);
    }
    free(heap);
    free(pos);
    return 1;
}

/* ---- Output ------------------------------------------------------------- */

/* Write the primes one per line.  A hand-rolled integer-to-ASCII conversion
 * into a 1 MiB buffer is about twice as fast as calling fprintf("%lld\n") for
 * every prime (measured: 0.070 s -> 0.034 s for the 664,579 primes below 1e7),
 * which matters because this is a serial phase.  Returns 0 on I/O or
 * allocation failure. */
static int write_primes(const char *path, long long n, const long long *primes,
                        long long count) {
    enum { BUF_SIZE = 1 << 20, MAX_LINE = 24 };   /* 20 digits + '\n' + slack */
    FILE *file = fopen(path, "w");
    if (file == NULL) return 0;
    char *buf = (char *)malloc(BUF_SIZE);
    if (buf == NULL) { fclose(file); return 0; }

    int pos = snprintf(buf, BUF_SIZE, "Prime numbers less than %lld (%lld found):\n", n, count);
    for (long long i = 0; i < count; i++) {
        if (pos > BUF_SIZE - MAX_LINE) {
            fwrite(buf, 1, pos, file);
            pos = 0;
        }
        char digits[MAX_LINE];
        int len = 0;
        long long v = primes[i];
        do { digits[len++] = (char)('0' + v % 10); v /= 10; } while (v > 0);
        while (len > 0) buf[pos++] = digits[--len];
        buf[pos++] = '\n';
    }
    fwrite(buf, 1, pos, file);
    int ok = !ferror(file);
    free(buf);
    return fclose(file) == 0 && ok;
}

/* ------------------------------------------------------------------------- */

static int parse_dist(const char *s) {
    if (strcmp(s, "block") == 0)       return DIST_BLOCK;
    if (strcmp(s, "cyclic") == 0)      return DIST_CYCLIC;
    if (strcmp(s, "blockcyclic") == 0) return DIST_BLOCKCYCLIC;
    if (strcmp(s, "wblock") == 0)      return DIST_WBLOCK;
    return -1;
}

static int parse_kernel(const char *s) {
    if (strcmp(s, "trial") == 0) return KERNEL_TRIAL;
    if (strcmp(s, "sieve") == 0) return KERNEL_SIEVE;
    return -1;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* params = { n, dist, chunk, kernel }.  Only root parses the command
     * line; everyone else learns the values from the broadcast.  The output
     * path is only ever used by root, so it is not broadcast. */
    long long params[4] = { 0, DIST_BLOCKCYCLIC, 1024, KERNEL_TRIAL };
    const char *output_file = DEFAULT_OUTPUT;
    if (rank == ROOT) {
        if (argc < 2) {
            fprintf(stderr, "Usage: mpirun -np <P> %s <n> [output_file] "
                            "[block|cyclic|blockcyclic|wblock] [chunk] [trial|sieve]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        params[0] = atoll(argv[1]);
        if (argc >= 3) output_file = argv[2];
        if (argc >= 4) params[1] = parse_dist(argv[3]);
        if (argc >= 5) params[2] = atoll(argv[4]);
        if (argc >= 6) params[3] = parse_kernel(argv[5]);
        if (params[0] < 2 || params[1] < 0 || params[2] < 1 || params[3] < 0) {
            fprintf(stderr, "Invalid arguments: n >= 2, dist in {block,cyclic,blockcyclic,wblock}, "
                            "chunk >= 1, kernel in {trial,sieve}\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (params[3] == KERNEL_SIEVE && params[1] == DIST_CYCLIC) {
            fprintf(stderr, "The sieve kernel needs contiguous ranges: use block, wblock or blockcyclic\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* ---- Phase 1: disseminate n and the tuning parameters --------------- */
    MPI_Bcast(params, 4, MPI_LONG_LONG, ROOT, MPI_COMM_WORLD);
    long long n      = params[0];
    int       dist   = (int)params[1];
    long long chunk  = params[2];
    int       kernel = (int)params[3];
    double t_bcast = MPI_Wtime();

    /* ---- Phase 2: every rank (root included) computes its share --------- */
    PrimeList local = { NULL, 0, 0, 0 };
    if (!find_local_primes(n, rank, size, dist, chunk, kernel, &local)) {
        fprintf(stderr, "Rank %d: memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double t_compute = MPI_Wtime();
    double my_compute_time = t_compute - t_bcast;

    /* ---- Phase 3: gather the local lists at the root -------------------- */
    int local_count = (int)local.count;
    int *counts = NULL, *displs = NULL;
    long long *gathered = NULL;
    long long total = 0;
    double *compute_times = NULL;
    int total_reallocs = 0;

    if (rank == ROOT) {
        counts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        compute_times = (double *)malloc(size * sizeof(double));
        if (counts == NULL || displs == NULL || compute_times == NULL) {
            fprintf(stderr, "Root: memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Gather(&my_compute_time, 1, MPI_DOUBLE, compute_times, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Reduce(&local.reallocs, &total_reallocs, 1, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        for (int r = 0; r < size; r++) {
            displs[r] = (int)total;
            total += counts[r];
        }
        /* MPI_Gatherv addresses elements with int counts/displacements. */
        if (total > INT_MAX) {
            fprintf(stderr, "Root: %lld primes exceed the MPI_Gatherv int limit\n", total);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        gathered = (long long *)malloc((total > 0 ? total : 1) * sizeof(long long));
        if (gathered == NULL) {
            fprintf(stderr, "Root: memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gatherv(local.data, local_count, MPI_LONG_LONG,
                gathered, counts, displs, MPI_LONG_LONG, ROOT, MPI_COMM_WORLD);
    free(local.data);
    double t_gather = MPI_Wtime();

    /* ---- Phase 4 + 5 (root only): merge into sorted order, write file --- */
    if (rank == ROOT) {
        long long *primes = gathered;
        if (!dist_is_contiguous(dist) && size > 1) {
            primes = (long long *)malloc((total > 0 ? total : 1) * sizeof(long long));
            if (primes == NULL || !merge_sorted_runs(gathered, counts, displs, size, primes)) {
                fprintf(stderr, "Root: merge failed\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            free(gathered);
        }
        double t_merge = MPI_Wtime();

        if (n < 100) {
            printf("\nPrime numbers less than %lld (%lld found):\n", n, total);
            for (long long i = 0; i < total; i++) {
                printf("%lld%s", primes[i], (i == total - 1) ? "" : ", ");
            }
            printf("\n");
        } else {
            if (!write_primes(output_file, n, primes, total)) {
                perror("Failed to write output file");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            printf("Found %lld prime numbers. Output written to '%s'.\n", total, output_file);
        }
        double t_end = MPI_Wtime();

        /* Load-balance statistics across ranks. */
        double c_min = compute_times[0], c_max = compute_times[0], c_sum = 0.0;
        for (int r = 0; r < size; r++) {
            if (compute_times[r] < c_min) c_min = compute_times[r];
            if (compute_times[r] > c_max) c_max = compute_times[r];
            c_sum += compute_times[r];
        }

        printf("Execution Time (Open MPI, %d processes, dist=%s, chunk=%lld, kernel=%s): %.6f seconds\n",
               size, DIST_NAMES[dist], chunk, KERNEL_NAMES[kernel], t_end - t_start);
        printf("  broadcast : %.6f s\n", t_bcast - t_start);
        printf("  compute   : max %.6f s  min %.6f s  avg %.6f s  (imbalance %.1f%%, reallocs %d)\n",
               c_max, c_min, c_sum / size, c_max > 0 ? 100.0 * (c_max - c_min) / c_max : 0.0,
               total_reallocs);
        printf("  gather    : %.6f s (includes waiting for the slowest rank)\n",
               t_gather - t_compute);
        printf("  merge     : %.6f s\n", t_merge - t_gather);
        printf("  write     : %.6f s\n", t_end - t_merge);
        if (getenv("PRIMES_VERBOSE") != NULL) {
            for (int r = 0; r < size; r++) {
                printf("RANK,%d,%.6f,%d\n", r, compute_times[r], counts[r]);
            }
        }
        /* Machine-readable line for the benchmark scripts:
         * CSV,impl,n,procs,threads,dist,chunk,sched,sched_chunk,total,bcast,
         *     comp_max,comp_min,comp_avg,gather,merge,write,count
         * impl is "mpi" for trial division and "mpi_sieve" for the sieve. */
        printf("CSV,%s,%lld,%d,1,%s,%lld,-,0,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%lld\n",
               kernel == KERNEL_SIEVE ? "mpi_sieve" : "mpi",
               n, size, DIST_NAMES[dist], chunk,
               t_end - t_start, t_bcast - t_start, c_max, c_min, c_sum / size,
               t_gather - t_compute, t_merge - t_gather, t_end - t_merge, total);

        free(primes);
        free(counts);
        free(displs);
        free(compute_times);
    }

    MPI_Finalize();
    return 0;
}
