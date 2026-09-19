/*
 * FIT3143 Lab #2 (Week 8)
 *
 * Author
 * Name: Haruto Iriyama
 *
 * Author
 * Name: Hiew Jia Hao
 */

/*
 * FIT3143 Lab #2 - Task 2: Prime search using hybrid Open MPI + OpenMP
 *
 * Finds every prime strictly less than n and writes them, sorted, to an
 * output file (or stdout when n < 100), in exactly the format of the Week 4
 * serial program so the two outputs can be diffed.
 *
 * Usage:
 *   mpirun -np <P> ./task2 <n> [output_file] [threads] [dist] [chunk] [sched] [sched_chunk] [kernel]
 *     output_file : where the sorted list is written (default primes.txt);
 *                   point benchmarks at fast local storage such as /tmp
 *     threads     : OpenMP threads per MPI process (default OMP_NUM_THREADS)
 *     dist        : blockcyclic (default) | block | cyclic | wblock  (across processes)
 *     chunk       : chunk size for blockcyclic (default 1024)
 *     sched       : OpenMP loop schedule within a process:
 *                   dynamic (default) | static | guided
 *     sched_chunk : OpenMP schedule chunk (default 64)
 *     kernel      : trial (default) | sieve
 *
 * Design
 * ------
 * Two levels of parallelism:
 *
 *   1. Distributed memory (MPI).  The root reads the arguments and broadcasts
 *      n plus all tuning parameters.  The candidate indices j in [0, M)
 *      (k = 3 + 2j, M = (n-2)/2, odd numbers only) are split across the P
 *      processes with the same block / cyclic / block-cyclic / weighted-block
 *      schemes as Task 1 (see task1.c for the cost model behind wblock and
 *      for why plain cyclic degenerates when an odd prime divides P).
 *
 *   2. Shared memory (OpenMP).  Inside each process the loop over that
 *      process's indices is a worksharing `omp for` with a runtime-selected
 *      schedule.  Each thread appends to a private list (no locks, no false
 *      sharing on a shared array), and the T sorted thread lists are merged
 *      by the main thread into one sorted per-process list.
 *
 * Two primality kernels, as in Task 1: `trial` (the Week 4 trial division,
 * for a fair speed-up comparison) and `sieve` (segmented sieve of
 * Eratosthenes).  With the sieve the OpenMP work unit is one L1-sized
 * segment of 32768 odd numbers; each thread owns a private mark buffer.
 * `cyclic` is not made of contiguous ranges and therefore only supports
 * trial division.
 *
 * The per-process lists are then collected at the root with MPI_Gather /
 * MPI_Gatherv and merged with a min-heap exactly as in Task 1 (skipped for
 * the contiguous schemes).  Only the main thread of the root process writes
 * the output file, using the same buffered integer-to-ASCII writer as Task 1
 * because the write is a serial phase that bounds the speed-up.
 *
 * MPI is initialised with MPI_THREAD_FUNNELED: only the main thread makes
 * MPI calls, which is all this program needs.
 *
 * Phase timings (broadcast, compute, local merge, gather, global merge, file
 * write) are reported so the serial / parallel fractions for Amdahl's or
 * Gustafson's Law can be derived (Task 3).  The total includes communication,
 * sorting and file writing.  Set PRIMES_VERBOSE=1 to also print one line per
 * rank (compute time, thread max/min, number of primes) for load-balance
 * analysis.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

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

enum { SCHED_STATIC = 0, SCHED_DYNAMIC = 1, SCHED_GUIDED = 2 };
static const char *SCHED_NAMES[] = { "static", "dynamic", "guided" };
static const omp_sched_t SCHED_KINDS[] = { omp_sched_static, omp_sched_dynamic, omp_sched_guided };

/* Growable array of primes owned by one thread / one rank. */
typedef struct {
    long long *data;
    long long  count;
    long long  capacity;
    int        reallocs;     /* how often the estimate was too small */
} PrimeList;

/* Trial division up to sqrt(k), skipping even divisors.
 * Identical to the Week 4 serial version (fit3143-lab1/task1.c) so timing
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
 * Used to pre-size the local lists so realloc is not hit during the timed
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

/* Merge `runs` sorted arrays (srcs[r] of length counts[r]) into dst.
 * O(N log runs). */
static int merge_sorted_runs(const long long **srcs, const long long *counts,
                             int runs, long long *dst) {
    HeapItem *heap = (HeapItem *)malloc(runs * sizeof(HeapItem));
    long long *pos = (long long *)malloc(runs * sizeof(long long));
    if (heap == NULL || pos == NULL) { free(heap); free(pos); return 0; }

    int hsize = 0;
    for (int r = 0; r < runs; r++) {
        pos[r] = 0;
        if (counts[r] > 0) {
            heap[hsize].value = srcs[r][0];
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
            heap[0].value = srcs[r][pos[r]];
        } else {
            heap[0] = heap[--hsize];
        }
        heap_sift_down(heap, hsize, 0);
    }
    free(heap);
    free(pos);
    return 1;
}

static int cmp_ll(const void *a, const void *b) {
    long long x = *(const long long *)a, y = *(const long long *)b;
    return (x > y) - (x < y);
}

static bool is_sorted(const long long *a, long long n) {
    for (long long i = 1; i < n; i++) if (a[i - 1] > a[i]) return false;
    return true;
}

/* ---- Per-process computation (OpenMP) ----------------------------------- */

/* Each thread gets a private PrimeList (and, for the sieve, a private mark
 * buffer) and pushes the primes it finds.  The `omp for` loops below are in
 * canonical form, so the runtime schedule (static / dynamic / guided) decides
 * which thread tests which candidates.  With every OpenMP schedule the chunks
 * are handed out in increasing iteration order, so each thread's list is
 * naturally sorted; a cheap check plus qsort fallback guards against any
 * implementation that does not.
 *
 * On return `out` holds this rank's primes in ascending order (with
 * `reallocs` summed over the threads), and thread_times[t] holds thread t's
 * time inside the parallel region.  Returns 0 on allocation failure. */
static int find_local_primes(long long n, int rank, int size, int dist,
                             long long chunk, int kernel, int nthreads,
                             PrimeList *out, double *thread_times) {
    long long M = (n >= 3) ? (n - 2) / 2 : 0;
    long long num_chunks = (M + chunk - 1) / chunk;
    long long lo = 0, hi = 0;
    if (dist_is_contiguous(dist)) block_bounds(M, rank, size, dist == DIST_WBLOCK, &lo, &hi);
    long long num_segs = (hi - lo + SEG_INDICES - 1) / SEG_INDICES;

    /* Per-thread pre-size from the prime number theorem.  With dynamic or
     * guided scheduling a thread can end up with more than 1/T of the rank's
     * work, so each thread gets twice the even share (memory is not an
     * issue: the whole rank's list is only ~ n / ln n / P entries). */
    long long rank_est = dist_is_contiguous(dist)
                       ? estimate_pi(3 + 2 * hi) - estimate_pi(3 + 2 * lo)
                       : estimate_pi(n) / size;
    long long cap = (nthreads > 1 ? 2 * rank_est / nthreads : rank_est) + 16;

    PrimeList *tl = (PrimeList *)calloc(nthreads, sizeof(PrimeList));
    if (tl == NULL) return 0;

    long long *sp = NULL;
    int nsp = 0;
    if (kernel == KERNEL_SIEVE) {
        sp = small_odd_primes((long long)sqrt((double)n) + 1, &nsp);
        if (sp == NULL && nsp > 0) { free(tl); return 0; }
    }

    int alloc_error = 0;

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        double t0 = omp_get_wtime();
        PrimeList *mine = &tl[tid];
        unsigned char *mark = NULL;
        int ok = list_init(mine, cap);
        if (ok && kernel == KERNEL_SIEVE) {
            mark = (unsigned char *)malloc(SEG_INDICES);
            if (mark == NULL) ok = 0;
        }
        if (!ok) {
            #pragma omp atomic write
            alloc_error = 1;
        }

        switch (dist) {
        case DIST_BLOCK:
        case DIST_WBLOCK: {
            long long j, s;
            if (kernel == KERNEL_SIEVE) {
                /* work unit = one L1-sized segment */
                #pragma omp for schedule(runtime) nowait
                for (s = 0; s < num_segs; s++) {
                    if (!ok) continue;
                    long long s_lo = lo + s * SEG_INDICES;
                    long long s_hi = s_lo + SEG_INDICES;
                    if (s_hi > hi) s_hi = hi;
                    ok = sieve_range(s_lo, s_hi, sp, nsp, mark, mine);
                }
            } else {
                #pragma omp for schedule(runtime) nowait
                for (j = lo; j < hi; j++) {
                    if (!ok) continue;
                    long long k = 3 + 2 * j;
                    if (is_prime(k) && !list_push(mine, k)) ok = 0;
                }
            }
            break;
        }
        case DIST_CYCLIC: {
            long long j;
            #pragma omp for schedule(runtime) nowait
            for (j = rank; j < M; j += size) {
                if (!ok) continue;
                long long k = 3 + 2 * j;
                if (is_prime(k) && !list_push(mine, k)) ok = 0;
            }
            break;
        }
        case DIST_BLOCKCYCLIC: {
            long long c;
            #pragma omp for schedule(runtime) nowait
            for (c = rank; c < num_chunks; c += size) {
                if (!ok) continue;
                long long c_lo = c * chunk;
                long long c_hi = c_lo + chunk;
                if (c_hi > M) c_hi = M;
                if (kernel == KERNEL_SIEVE) {
                    ok = sieve_range(c_lo, c_hi, sp, nsp, mark, mine);
                } else {
                    for (long long j = c_lo; j < c_hi && ok; j++) {
                        long long k = 3 + 2 * j;
                        if (is_prime(k) && !list_push(mine, k)) ok = 0;
                    }
                }
            }
            break;
        }
        default:
            break;
        }

        free(mark);
        if (!ok) {
            #pragma omp atomic write
            alloc_error = 1;
        } else if (!is_sorted(mine->data, mine->count)) {
            qsort(mine->data, mine->count, sizeof(long long), cmp_ll);
        }
        thread_times[tid] = omp_get_wtime() - t0;
    }
    free(sp);

    if (alloc_error) {
        for (int t = 0; t < nthreads; t++) free(tl[t].data);
        free(tl);
        return 0;
    }

    /* Merge the T thread lists into one sorted list for this rank.  Root
     * also prepends 2, the only even prime. */
    long long total = 0;
    int reallocs = 0;
    for (int t = 0; t < nthreads; t++) { total += tl[t].count; reallocs += tl[t].reallocs; }
    long long offset = (rank == ROOT && n > 2) ? 1 : 0;

    int ok = list_init(out, total + offset);
    if (ok) {
        if (offset) out->data[0] = 2;
        if (nthreads == 1) {
            memcpy(out->data + offset, tl[0].data, total * sizeof(long long));
        } else {
            const long long **srcs = (const long long **)malloc(nthreads * sizeof(long long *));
            long long *counts = (long long *)malloc(nthreads * sizeof(long long));
            if (srcs == NULL || counts == NULL) {
                ok = 0;
            } else {
                for (int t = 0; t < nthreads; t++) { srcs[t] = tl[t].data; counts[t] = tl[t].count; }
                ok = merge_sorted_runs(srcs, counts, nthreads, out->data + offset);
            }
            free(srcs);
            free(counts);
        }
        out->count = total + offset;
        out->reallocs = reallocs;
    }

    for (int t = 0; t < nthreads; t++) free(tl[t].data);
    free(tl);
    if (!ok) { free(out->data); out->data = NULL; }
    return ok;
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

static int parse_sched(const char *s) {
    if (strcmp(s, "static") == 0)  return SCHED_STATIC;
    if (strcmp(s, "dynamic") == 0) return SCHED_DYNAMIC;
    if (strcmp(s, "guided") == 0)  return SCHED_GUIDED;
    return -1;
}

static int parse_kernel(const char *s) {
    if (strcmp(s, "trial") == 0) return KERNEL_TRIAL;
    if (strcmp(s, "sieve") == 0) return KERNEL_SIEVE;
    return -1;
}

int main(int argc, char *argv[]) {
    int provided, rank, size;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* params: n, threads, dist, chunk, sched, sched_chunk, kernel.  Only root
     * parses the command line; everyone else learns the values from the
     * broadcast and all threads of a process share them. */
    long long params[7] = { 0, omp_get_max_threads(), DIST_BLOCKCYCLIC, 1024, SCHED_DYNAMIC, 64, KERNEL_TRIAL };
    const char *output_file = DEFAULT_OUTPUT;   /* used by root only */
    if (rank == ROOT) {
        if (argc < 2) {
            fprintf(stderr,
                "Usage: mpirun -np <P> %s <n> [output_file] [threads] "
                "[block|cyclic|blockcyclic|wblock] [chunk] [static|dynamic|guided] [sched_chunk] "
                "[trial|sieve]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        params[0] = atoll(argv[1]);
        if (argc >= 3) output_file = argv[2];
        if (argc >= 4) params[1] = atoll(argv[3]);
        if (argc >= 5) params[2] = parse_dist(argv[4]);
        if (argc >= 6) params[3] = atoll(argv[5]);
        if (argc >= 7) params[4] = parse_sched(argv[6]);
        if (argc >= 8) params[5] = atoll(argv[7]);
        if (argc >= 9) params[6] = parse_kernel(argv[8]);
        if (params[0] < 2 || params[1] < 1 || params[2] < 0 || params[3] < 1 ||
            params[4] < 0 || params[5] < 1 || params[6] < 0) {
            fprintf(stderr, "Invalid arguments.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (params[6] == KERNEL_SIEVE && params[2] == DIST_CYCLIC) {
            fprintf(stderr, "The sieve kernel needs contiguous ranges: use block, wblock or blockcyclic\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (provided < MPI_THREAD_FUNNELED) {
            fprintf(stderr, "Warning: MPI library does not provide MPI_THREAD_FUNNELED\n");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* ---- Phase 1: disseminate n and all tuning parameters -------------- */
    MPI_Bcast(params, 7, MPI_LONG_LONG, ROOT, MPI_COMM_WORLD);
    long long n           = params[0];
    int       nthreads    = (int)params[1];
    int       dist        = (int)params[2];
    long long chunk       = params[3];
    int       sched       = (int)params[4];
    int       sched_chunk = (int)params[5];
    int       kernel      = (int)params[6];

    omp_set_dynamic(0);
    omp_set_num_threads(nthreads);
    omp_set_schedule(SCHED_KINDS[sched], sched_chunk);
    double t_bcast = MPI_Wtime();

    /* ---- Phase 2: every process computes its share with T threads ------ */
    double *thread_times = (double *)calloc(nthreads, sizeof(double));
    PrimeList local = { NULL, 0, 0, 0 };
    if (thread_times == NULL ||
        !find_local_primes(n, rank, size, dist, chunk, kernel, nthreads, &local, thread_times)) {
        fprintf(stderr, "Rank %d: memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double t_compute = MPI_Wtime();

    /* Per-rank statistics: total compute time (parallel region + thread
     * merge) and thread-level imbalance inside this process. */
    double th_min = thread_times[0], th_max = thread_times[0];
    for (int t = 1; t < nthreads; t++) {
        if (thread_times[t] < th_min) th_min = thread_times[t];
        if (thread_times[t] > th_max) th_max = thread_times[t];
    }
    double my_stats[3] = { t_compute - t_bcast, th_max, th_min };
    free(thread_times);

    /* ---- Phase 3: gather the per-process lists at the root ------------- */
    int local_count = (int)local.count;
    int *counts = NULL, *displs = NULL;
    long long *gathered = NULL;
    long long total = 0;
    double *all_stats = NULL;
    int total_reallocs = 0;

    if (rank == ROOT) {
        counts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        all_stats = (double *)malloc(3 * size * sizeof(double));
        if (counts == NULL || displs == NULL || all_stats == NULL) {
            fprintf(stderr, "Root: memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Gather(my_stats, 3, MPI_DOUBLE, all_stats, 3, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
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

    /* ---- Phase 4 + 5 (root main thread): merge across ranks, write ----- */
    if (rank == ROOT) {
        long long *primes = gathered;
        if (!dist_is_contiguous(dist) && size > 1) {
            primes = (long long *)malloc((total > 0 ? total : 1) * sizeof(long long));
            const long long **srcs = (const long long **)malloc(size * sizeof(long long *));
            long long *lcounts = (long long *)malloc(size * sizeof(long long));
            if (primes == NULL || srcs == NULL || lcounts == NULL) {
                fprintf(stderr, "Root: memory allocation failed\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            for (int r = 0; r < size; r++) { srcs[r] = gathered + displs[r]; lcounts[r] = counts[r]; }
            if (!merge_sorted_runs(srcs, lcounts, size, primes)) {
                fprintf(stderr, "Root: merge failed\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            free(srcs);
            free(lcounts);
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

        /* Load-balance statistics across ranks and threads. */
        double c_min = all_stats[0], c_max = all_stats[0], c_sum = 0.0, th_imb = 0.0;
        for (int r = 0; r < size; r++) {
            double c = all_stats[3 * r], tmax = all_stats[3 * r + 1], tmin = all_stats[3 * r + 2];
            if (c < c_min) c_min = c;
            if (c > c_max) c_max = c;
            c_sum += c;
            double imb = tmax > 0 ? (tmax - tmin) / tmax : 0.0;
            if (imb > th_imb) th_imb = imb;
        }

        printf("Execution Time (Hybrid, %d processes x %d threads, dist=%s, chunk=%lld, "
               "omp=%s/%d, kernel=%s): %.6f seconds\n",
               size, nthreads, DIST_NAMES[dist], chunk, SCHED_NAMES[sched], sched_chunk,
               KERNEL_NAMES[kernel], t_end - t_start);
        printf("  broadcast : %.6f s\n", t_bcast - t_start);
        printf("  compute   : max %.6f s  min %.6f s  avg %.6f s  "
               "(rank imbalance %.1f%%, worst thread imbalance %.1f%%, reallocs %d)\n",
               c_max, c_min, c_sum / size,
               c_max > 0 ? 100.0 * (c_max - c_min) / c_max : 0.0, 100.0 * th_imb, total_reallocs);
        printf("  gather    : %.6f s (includes waiting for the slowest rank)\n",
               t_gather - t_compute);
        printf("  merge     : %.6f s\n", t_merge - t_gather);
        printf("  write     : %.6f s\n", t_end - t_merge);
        if (getenv("PRIMES_VERBOSE") != NULL) {
            for (int r = 0; r < size; r++) {
                printf("RANK,%d,%.6f,%.6f,%.6f,%d\n", r, all_stats[3 * r],
                       all_stats[3 * r + 1], all_stats[3 * r + 2], counts[r]);
            }
        }
        /* Machine-readable line for the benchmark scripts:
         * CSV,impl,n,procs,threads,dist,chunk,sched,sched_chunk,total,bcast,
         *     comp_max,comp_min,comp_avg,gather,merge,write,count
         * impl is "hybrid" for trial division and "hybrid_sieve" for the sieve. */
        printf("CSV,%s,%lld,%d,%d,%s,%lld,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%lld\n",
               kernel == KERNEL_SIEVE ? "hybrid_sieve" : "hybrid",
               n, size, nthreads, DIST_NAMES[dist], chunk, SCHED_NAMES[sched], sched_chunk,
               t_end - t_start, t_bcast - t_start, c_max, c_min, c_sum / size,
               t_gather - t_compute, t_merge - t_gather, t_end - t_merge, total);

        free(primes);
        free(counts);
        free(displs);
        free(all_stats);
    }

    MPI_Finalize();
    return 0;
}
