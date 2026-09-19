/*
 * Serial segmented sieve of Eratosthenes - the fair baseline for the `sieve`
 * kernel of task1 / task2 (comparing a parallel sieve against the Week 4
 * trial-division program would measure the algorithm, not the parallelism).
 *
 * Same algorithm, segment size and output format as the sieve kernel in
 * task1.c, and the same summary line format as the Week 4 serial program so
 * run_bench.sh can parse it:
 *   n = N | primes found = C | time taken = T seconds
 * The timer covers the sieve only (like the Week 4 program); run_bench.sh
 * records the wall-clock time of the whole process including the write.
 *
 *   gcc -O2 -o serial_sieve serial_sieve.c -lm
 *   ./serial_sieve <n> [output_file]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define SEG_INDICES (1 << 15)

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char *argv[]) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <n> [output_file]\n", argv[0]); return 1; }
    long long n = atoll(argv[1]);
    const char *output_file = argc >= 3 ? argv[2] : "primes_output_sieve.txt";
    if (n < 2) { printf("There are no prime numbers less than %lld.\n", n); return 0; }

    double t0 = now();

    /* odd primes up to sqrt(n) */
    long long limit = (long long)sqrt((double)n) + 1;
    unsigned char *composite = calloc(limit + 1, 1);
    long long *sp = malloc((limit / 2 + 1) * sizeof(long long));
    int nsp = 0;
    for (long long p = 3; p <= limit; p += 2) {
        if (composite[p]) continue;
        sp[nsp++] = p;
        for (long long m = p * p; m <= limit; m += 2 * p) composite[m] = 1;
    }
    free(composite);

    long long M = (n >= 3) ? (n - 2) / 2 : 0;           /* odd candidates 3 + 2j, j < M */
    long long cap = (long long)(1.3 * (double)n / (log((double)(n > 100 ? n : 100)) - 1.0)) + 32;
    long long *primes = malloc(cap * sizeof(long long));
    unsigned char *mark = malloc(SEG_INDICES);
    long long count = 0;
    if (n > 2) primes[count++] = 2;

    for (long long seg_lo = 0; seg_lo < M; seg_lo += SEG_INDICES) {
        long long seg_hi = seg_lo + SEG_INDICES;
        if (seg_hi > M) seg_hi = M;
        long long k_lo = 3 + 2 * seg_lo, k_hi = 3 + 2 * seg_hi, len = seg_hi - seg_lo;
        memset(mark, 0, (size_t)len);
        for (int i = 0; i < nsp; i++) {
            long long p = sp[i];
            if (p * p >= k_hi) break;
            long long start = ((k_lo + p - 1) / p) * p;
            if (start < p * p) start = p * p;
            if (start % 2 == 0) start += p;
            for (long long m = start; m < k_hi; m += 2 * p) mark[(m - k_lo) / 2] = 1;
        }
        for (long long i = 0; i < len; i++) {
            if (!mark[i]) {
                if (count == cap) { cap *= 2; primes = realloc(primes, cap * sizeof(long long)); }
                primes[count++] = k_lo + 2 * i;
            }
        }
    }
    double elapsed = now() - t0;

    if (n < 100) {
        printf("\nPrime numbers less than %lld (%lld found):\n", n, count);
        for (long long i = 0; i < count; i++) printf("%lld%s", primes[i], i == count - 1 ? "" : ", ");
        printf("\n");
    } else {
        FILE *fp = fopen(output_file, "w");
        if (!fp) { perror("fopen"); return 1; }
        char *buf = malloc(1 << 20);
        int pos = snprintf(buf, 1 << 20, "Prime numbers less than %lld (%lld found):\n", n, count);
        for (long long i = 0; i < count; i++) {
            if (pos > (1 << 20) - 24) { fwrite(buf, 1, pos, fp); pos = 0; }
            char d[24]; int len = 0; long long v = primes[i];
            do { d[len++] = (char)('0' + v % 10); v /= 10; } while (v > 0);
            while (len > 0) buf[pos++] = d[--len];
            buf[pos++] = '\n';
        }
        fwrite(buf, 1, pos, fp);
        fclose(fp);
        free(buf);
        printf("\n%lld primes written to \"%s\"\n", count, output_file);
    }
    printf("n = %lld | primes found = %lld | time taken = %.6f seconds\n", n, count, elapsed);

    free(primes); free(mark); free(sp);
    return 0;
}
