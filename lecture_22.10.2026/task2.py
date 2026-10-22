
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from threadpoolctl import threadpool_limits

STUDENT_ID = 230103077
ARRAY_SIZE = 20_000_000
P = 0.95
CHUNK_COUNT = 64


def sum_chunk(chunk):
    return np.sum(chunk, dtype=np.float64)


def amdahl_speedup(workers):
    return 1.0 / ((1.0 - P) + P / workers)


def benchmark(data, workers):
    chunks = np.array_split(data, CHUNK_COUNT)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(sum_chunk, chunks))

        start = time.perf_counter()
        partial_sums = list(executor.map(sum_chunk, chunks))
        total = np.sum(partial_sums, dtype=np.float64)
        elapsed_ms = (time.perf_counter() - start) * 1000

    return total, elapsed_ms


def main():
    max_cores = os.cpu_count() or 1
    worker_counts = sorted(set([
        1, 2, 4, max_cores, max_cores * 2
    ]))

    print("=== TASK 2: MULTI-CORE SCALING VS AMDAHL'S LAW ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU: {platform.processor() or platform.machine()}")
    print(f"Logical Cores: {max_cores}")
    print(f"Array Size: {ARRAY_SIZE:,}")
    print(f"Data Type: float64")
    print(f"Parallelizable Fraction (P): {P:.2f}")
    print("Execution: ThreadPoolExecutor + NumPy")
    print()

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(ARRAY_SIZE, dtype=np.float64)

    results = []

    with threadpool_limits(limits=1):
        for workers in worker_counts:
            total, elapsed_ms = benchmark(data, workers)
            results.append({
                "workers": workers,
                "time_ms": elapsed_ms,
                "sum": total,
                "amdahl": amdahl_speedup(workers),
            })

    baseline_time = next(
        row["time_ms"] for row in results if row["workers"] == 1
    )
    baseline_sum = next(
        row["sum"] for row in results if row["workers"] == 1
    )

    print(
        f"{'Cores':>7} | {'Wall Time (ms)':>14} | "
        f"{'Speedup':>8} | {'Amdahl Upper Bound':>18}"
    )
    print("-" * 58)

    for row in results:
        speedup = baseline_time / row["time_ms"]
        print(
            f"{row['workers']:>7} | "
            f"{row['time_ms']:>14.3f} | "
            f"{speedup:>8.3f} | "
            f"{row['amdahl']:>18.3f}"
        )

        if not np.isclose(row["sum"], baseline_sum, rtol=1e-12, atol=1e-8):
            raise RuntimeError(
                f"Sum mismatch for {row['workers']} workers"
            )

    print()
    print(f"Mathematical Sum: {baseline_sum:.10f}")
    print("Verification: PASSED (all worker counts agree)")
    print("Status: COMPLETED")


if __name__ == "__main__":
    main()
