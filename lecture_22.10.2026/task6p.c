
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

#define NUM_THREADS 8
#define INCREMENTS 1000000LL
#define TOTAL_OPS (NUM_THREADS * INCREMENTS)

static uint64_t mutex_counter = 0;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

static _Atomic uint64_t atomic_counter = 0;

double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

void *mutex_worker(void *arg) {
    (void)arg;

    for (long long i = 0; i < INCREMENTS; i++) {
        pthread_mutex_lock(&mutex);
        mutex_counter++;
        pthread_mutex_unlock(&mutex);
    }

    return NULL;
}

void *cas_worker(void *arg) {
    (void)arg;

    for (long long i = 0; i < INCREMENTS; i++) {
        uint64_t expected = atomic_load(&atomic_counter);

        while (!atomic_compare_exchange_weak(
            &atomic_counter,
            &expected,
            expected + 1
        )) {
        }
    }

    return NULL;
}

double run_benchmark(void *(*worker)(void *)) {
    pthread_t threads[NUM_THREADS];
    double start = now_seconds();

    for (int i = 0; i < NUM_THREADS; i++) {
        if (pthread_create(&threads[i], NULL, worker, NULL) != 0) {
            perror("pthread_create");
            return -1.0;
        }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            perror("pthread_join");
            return -1.0;
        }
    }

    return now_seconds() - start;
}

int main(void) {
    printf("=== TASK 6: Mutex vs Atomic CAS ===\n");
    printf("Threads: %d\n", NUM_THREADS);
    printf("Increments per thread: %lld\n", INCREMENTS);
    printf("Total operations per benchmark: %lld\n\n",
           (long long)TOTAL_OPS);

    printf("64-bit atomic is lock-free: %s\n\n",
           atomic_is_lock_free(&atomic_counter) ? "YES" : "NO");

    double mutex_time = run_benchmark(mutex_worker);
    if (mutex_time < 0) return 1;

    double cas_time = run_benchmark(cas_worker);
    if (cas_time < 0) return 1;

    double mutex_throughput = (double)TOTAL_OPS / mutex_time;
    double cas_throughput = (double)TOTAL_OPS / cas_time;
    double ratio = cas_throughput / mutex_throughput;

    printf("Mutex final counter: %llu\n",
           (unsigned long long)mutex_counter);
    printf("CAS final counter:   %llu\n\n",
           (unsigned long long)atomic_load(&atomic_counter));

    printf("Mutex Execution Time: %.2f ms\n", mutex_time * 1000.0);
    printf("Mutex Throughput:     %.2f ops/sec\n\n",
           mutex_throughput);

    printf("Atomic CAS Execution Time: %.2f ms\n",
           cas_time * 1000.0);
    printf("Atomic CAS Throughput:     %.2f ops/sec\n\n",
           cas_throughput);

    printf("Throughput Ratio (CAS / Mutex): %.4f\n", ratio);

    if (mutex_counter != (uint64_t)TOTAL_OPS ||
        atomic_load(&atomic_counter) != (uint64_t)TOTAL_OPS) {
        fprintf(stderr, "ERROR: Counter verification failed!\n");
        return 1;
    }

    printf("\nVerification: PASSED\n");
    return 0;
}
