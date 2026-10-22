
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/sysctl.h>
#include <time.h>

#define ITERATIONS 50000000ULL
#define CACHE_LINE 64

typedef struct {
    volatile uint64_t count;
} SharedCounter;

typedef struct {
    int id;
    int mode;
    volatile uint64_t *shared;
    uint64_t local_result;
} ThreadArgs;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void *worker(void *arg) {
    ThreadArgs *a = (ThreadArgs *)arg;

    if (a->mode == 0) {
        for (uint64_t j = 0; j < ITERATIONS; j++) {
            a->shared[a->id]++;
        }
    } else {
        uint64_t local_count = 0;

        for (uint64_t j = 0; j < ITERATIONS; j++) {
            local_count++;
            __asm__ __volatile__("" : "+r"(local_count));
        }

        a->local_result = local_count;
        a->shared[a->id] = local_count;
    }

    return NULL;
}

static int physical_core_count(void) {
    int cores = 0;
    size_t size = sizeof(cores);

    if (sysctlbyname("hw.physicalcpu", &cores, &size, NULL, 0) != 0 ||
        cores < 1) {
        return 1;
    }

    return cores;
}

static double run_test(int n, int mode, uint64_t *total) {
    pthread_t *threads = calloc((size_t)n, sizeof(*threads));
    ThreadArgs *args = calloc((size_t)n, sizeof(*args));

    SharedCounter *counters = NULL;
    if (posix_memalign((void **)&counters, CACHE_LINE,
                       (size_t)n * sizeof(SharedCounter)) != 0) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++) {
        counters[i].count = 0;
        args[i].id = i;
        args[i].mode = mode;
        args[i].shared = &counters[0].count;
        args[i].local_result = 0;
    }

    double start = now_seconds();

    for (int i = 0; i < n; i++) {
        if (pthread_create(&threads[i], NULL, worker, &args[i]) != 0) {
            perror("pthread_create");
            exit(EXIT_FAILURE);
        }
    }

    *total = 0;
    for (int i = 0; i < n; i++) {
        pthread_join(threads[i], NULL);
        *total += counters[i].count;
    }

    double elapsed = now_seconds() - start;

    free(counters);
    free(args);
    free(threads);

    return elapsed * 1000.0;
}

int main(void) {
    int n = physical_core_count();
    uint64_t shared_total = 0, local_total = 0;

    printf("=== TASK 3: CACHE LINE CONTENTION ===\n");
    printf("Student ID: 230103077\n");
    printf("Physical CPU cores: %d\n", n);
    printf("Increments per thread: %llu\n", ITERATIONS);
    printf("Threads: %d\n\n", n);

    double shared_ms = run_test(n, 0, &shared_total);
    double local_ms = run_test(n, 1, &local_total);

    printf("%-22s %12s\n", "Method", "Wall Time (ms)");
    printf("-----------------------------------\n");
    printf("%-22s %12.3f\n", "Unpadded Shared", shared_ms);
    printf("%-22s %12.3f\n", "Local Variables", local_ms);

    printf("\nShared total: %llu\n", (unsigned long long)shared_total);
    printf("Local total:  %llu\n", (unsigned long long)local_total);

    if (local_ms > 0.0) {
        printf("Execution ratio (shared/local): %.3fx\n",
               shared_ms / local_ms);
    }

    if (shared_total != (uint64_t)n * ITERATIONS ||
        local_total != (uint64_t)n * ITERATIONS) {
        fprintf(stderr, "Verification FAILED\n");
        return EXIT_FAILURE;
    }

    printf("Verification: PASSED\n");
    printf("Status: COMPLETED\n");

    return EXIT_SUCCESS;
}
