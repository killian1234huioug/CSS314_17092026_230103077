
import os
import time
import math
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103077
TOTAL_POINTS = 40_000_000
NUM_THREADS = 4
CHUNK_SIZE = 250_000
CHUNKS_PER_THREAD = TOTAL_POINTS // (NUM_THREADS * CHUNK_SIZE)


def count_inside(points):
    x = points[:, 0]
    y = points[:, 1]
    return int(np.count_nonzero(x * x + y * y <= 1.0))


def run_shared_prng():
    """Four threads share one PRNG protected by a lock."""
    rng = np.random.default_rng(STUDENT_ID)
    rng_lock = threading.Lock()

    def worker(_):
        hits = 0

        for _ in range(CHUNKS_PER_THREAD):
            with rng_lock:
                points = rng.random((CHUNK_SIZE, 2))

            hits += count_inside(points)

        return hits

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(executor.map(worker, range(NUM_THREADS)))

    elapsed = time.perf_counter() - start
    pi_estimate = 4.0 * sum(results) / TOTAL_POINTS

    return pi_estimate, elapsed


def run_independent_prng():
    """Each thread uses its own PRNG and required seed."""
    def worker(thread_index):
        seed = (STUDENT_ID * 10) + thread_index
        rng = np.random.default_rng(seed)
        hits = 0

        for _ in range(CHUNKS_PER_THREAD):
            points = rng.random((CHUNK_SIZE, 2))
            hits += count_inside(points)

        return hits

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(executor.map(worker, range(NUM_THREADS)))

    elapsed = time.perf_counter() - start
    pi_estimate = 4.0 * sum(results) / TOTAL_POINTS

    return pi_estimate, elapsed


def main():
    print("=== TASK 2: PSEUDO-RANDOM GENERATOR STATE BOTTLENECK ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical CPU count: {os.cpu_count()}")
    print(f"Monte Carlo sample points: {TOTAL_POINTS:,}")
    print(f"Threads: {NUM_THREADS}")
    print(f"Chunk size: {CHUNK_SIZE:,}")
    print()

    print("Running shared PRNG with lock...")
    shared_pi, shared_time = run_shared_prng()

    print("Running independent PRNGs...")
    independent_pi, independent_time = run_independent_prng()

    shared_error = abs(shared_pi - math.pi)
    independent_error = abs(independent_pi - math.pi)
    speedup = (
        shared_time / independent_time
        if independent_time > 0 else float("nan")
    )

    print()
    print("--- Shared PRNG (with lock) ---")
    print(f"Computed pi:       {shared_pi:.10f}")
    print(f"Absolute error:    {shared_error:.10f}")
    print(f"Wall time:         {shared_time * 1000:.3f} ms")

    print()
    print("--- Independent PRNG (one per thread) ---")
    print(f"Computed pi:       {independent_pi:.10f}")
    print(f"Absolute error:    {independent_error:.10f}")
    print(f"Wall time:         {independent_time * 1000:.3f} ms")

    print()
    print("--- Performance Comparison ---")
    print(f"Speedup: {speedup:.3f}x")
    print("Verification: PASSED (both estimates are within 0.01 of pi)"
          if shared_error < 0.01 and independent_error < 0.01
          else "Verification: CHECK (at least one estimate differs by 0.01 or more)")
    print("Status: COMPLETED")


if __name__ == "__main__":
    main()
