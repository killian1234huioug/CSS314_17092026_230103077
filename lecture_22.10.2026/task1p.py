
import os
import time
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = "230103077"
MATRIX_SIZE = 8_000
NUM_CORES = os.cpu_count() or 1
REPEATS = 3


def row_major_sum(matrix, workers):
    """Workers process contiguous rows."""
    chunks = np.array_split(matrix, workers, axis=0)

    def sum_chunk(chunk):
        return np.sum(chunk, dtype=np.float64)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        partial_sums = list(executor.map(sum_chunk, chunks))

    return sum(partial_sums)


def column_major_sum(matrix, workers):
    """Workers process columns, accessing strided memory."""
    column_chunks = np.array_split(
        np.arange(matrix.shape[1]), workers
    )

    def sum_columns(columns):
        return np.sum(matrix[:, columns], dtype=np.float64)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        partial_sums = list(executor.map(sum_columns, column_chunks))

    return sum(partial_sums)


def measure(function, matrix):
    times = []
    result = None

    for _ in range(REPEATS):
        start = time.perf_counter()
        result = function(matrix, NUM_CORES)
        times.append(time.perf_counter() - start)

    return statistics.median(times), result


def main():
    print("=== TASK 1: ROW-MAJOR VS COLUMN-MAJOR CACHE TRAVERSAL ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical core count: {NUM_CORES}")
    print(f"Physical core count (reported by OS may differ): "
          f"{os.cpu_count()}")
    print(f"Matrix: {MATRIX_SIZE:,} x {MATRIX_SIZE:,}")
    print("Data type: float64")
    print(f"Parallel workers: {NUM_CORES}")
    print(f"Timing: median of {REPEATS} runs")
    print()

    rng = np.random.default_rng(int(STUDENT_ID))
    matrix = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float64)

    print("Running Mode A: Row-Major...")
    time_a, sum_a = measure(row_major_sum, matrix)

    print("Running Mode B: Column-Major / Strided...")
    time_b, sum_b = measure(column_major_sum, matrix)

    if not np.isclose(sum_a, sum_b, rtol=1e-10, atol=1e-6):
        raise RuntimeError(
            f"Verification failed: sums differ ({sum_a}, {sum_b})"
        )

    penalty = time_b / time_a if time_a > 0 else float("nan")

    print()
    print(f"Mode A (Row-Major) runtime: {time_a * 1000:.3f} ms")
    print(f"Mode B (Column-Major) runtime: {time_b * 1000:.3f} ms")
    print(f"Cache penalty factor (B / A): {penalty:.3f}x")
    print(f"Matrix sum: {sum_a:.6f}")
    print("Verification: PASSED (both modes produce matching sums)")
    print("Status: COMPLETED")


if __name__ == "__main__":
    main()
