
import os
import sys
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103077
N = 2_000_000
NUM_WORKERS = 4
REPEATS = 3


def serial_scan(array):
    """Serial inclusive prefix sum."""
    return np.cumsum(array, dtype=np.int64)


def parallel_scan(array):
    """Two-pass chunked prefix scan with parallel local work."""
    chunks = np.array_split(array, NUM_WORKERS)

    def scan_chunk(chunk):
        local_prefix = np.cumsum(chunk, dtype=np.int64)
        return local_prefix, int(local_prefix[-1])

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        local_results = list(executor.map(scan_chunk, chunks))

    local_prefixes = [result[0] for result in local_results]
    chunk_sums = [result[1] for result in local_results]

    offsets = []
    running_total = 0

    for chunk_sum in chunk_sums:
        offsets.append(running_total)
        running_total += chunk_sum

    def apply_offset(index):
        prefix = local_prefixes[index].copy()
        if offsets[index] != 0:
            prefix += offsets[index]
        return prefix

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        corrected_chunks = list(
            executor.map(apply_offset, range(len(local_prefixes)))
        )

    return np.concatenate(corrected_chunks)


def benchmark(function, array):
    times = []

    for _ in range(REPEATS):
        start = time.perf_counter()
        result = function(array)
        times.append(time.perf_counter() - start)

    return statistics.median(times), result


def main():
    print("=== TASK 3: PREFIX SCAN ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical CPU count: {os.cpu_count()}")
    print(f"Array size: {N:,}")
    print("Values: integers from 1 to 10")
    print(f"Parallel workers: {NUM_WORKERS}")
    print(f"Timing: median of {REPEATS} runs")
    print()

    rng = np.random.default_rng(STUDENT_ID)
    array = rng.integers(1, 11, size=N, dtype=np.int64)

    serial_time, serial_output = benchmark(serial_scan, array)
    parallel_time, parallel_output = benchmark(parallel_scan, array)

    speedup = serial_time / parallel_time if parallel_time > 0 else float("nan")

    print(f"{'Serial Time (ms)':>20} | "
          f"{'Parallel Time (ms)':>20} | "
          f"{'Speedup':>10} | Verification Status")
    print("-" * 83)

    try:
        assert parallel_output[-1] == serial_output[-1]
    except AssertionError:
        print(f"{serial_time * 1000:>20.3f} | "
              f"{parallel_time * 1000:>20.3f} | "
              f"{speedup:>9.3f}x | FAIL")
        print("Verification failed: final prefix sums do not match.")
        sys.exit(1)

    print(f"{serial_time * 1000:>20.3f} | "
          f"{parallel_time * 1000:>20.3f} | "
          f"{speedup:>9.3f}x | PASS")
    print()
    print(f"Serial final sum:   {serial_output[-1]:,}")
    print(f"Parallel final sum: {parallel_output[-1]:,}")
    print("Status: COMPLETED")


if __name__ == "__main__":
    main()
