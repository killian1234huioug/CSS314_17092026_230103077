
import os
import time
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = "230103077"
ARRAY_SIZES = [1, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
REPEATS = 3
A = 2.5
B = 1.5
WORKERS = os.cpu_count() or 1


def serial_operation(x, y):
    np.multiply(x, A, out=y)
    np.add(y, B, out=y)


def parallel_operation(x, y):
    n = len(x)
    chunk_size = (n + WORKERS - 1) // WORKERS

    ranges = [
        (start, min(start + chunk_size, n))
        for start in range(0, n, chunk_size)
    ]

    def process(start, end):
        np.multiply(x[start:end], A, out=y[start:end])
        np.add(y[start:end], B, out=y[start:end])

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        list(executor.map(lambda r: process(*r), ranges))


def measure(function, x, y):
    times = []

    for _ in range(REPEATS):
        start = time.perf_counter()
        function(x, y)
        times.append(time.perf_counter() - start)

    return statistics.median(times)


def main():
    print("=== TASK 5: THE OVERHEAD CROSSOVER POINT ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"CPU: {os.uname().machine}")
    print(f"Available logical CPUs: {WORKERS}")
    print(f"Operation: y[i] = {A} * x[i] + {B}")
    print(f"Timing: median of {REPEATS} runs")
    print()

    results = []

    print(f"{'N':>12} {'Serial (ms)':>14} "
          f"{'Parallel (ms)':>16} {'Speedup':>10} {'Result':>12}")
    print("-" * 68)

    for n in ARRAY_SIZES:
        rng = np.random.default_rng(230103077 + n)
        x = rng.random(n, dtype=np.float64)
        y_serial = np.empty_like(x)
        y_parallel = np.empty_like(x)

        serial_operation(x, y_serial)

        serial_time = measure(serial_operation, x, y_serial)
        parallel_time = measure(parallel_operation, x, y_parallel)

        if not np.allclose(y_serial, y_parallel, rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"Verification failed for N={n}")

        speedup = serial_time / parallel_time
        result = "FASTER" if speedup > 1 else (
            "SLOWER" if speedup < 1 else "EQUAL"
        )

        results.append((n, serial_time, parallel_time, speedup, result))

        print(f"{n:>12,} {serial_time * 1000:>14.3f} "
              f"{parallel_time * 1000:>16.3f} "
              f"{speedup:>10.3f} {result:>12}")

    print()
    print("=== OVERHEAD CROSSOVER ANALYSIS ===")

    crossovers = []
    for previous, current in zip(results, results[1:]):
        if previous[4] == "SLOWER" and current[4] == "FASTER":
            crossovers.append((previous[0], current[0]))

    if crossovers:
        for low, high in crossovers:
            print(
                f"Crossover observed between N={low:,} and N={high:,}: "
                f"parallel changes from slower to faster."
            )
        print("Note: the exact point is bounded by tested array sizes.")
    else:
        print("No slower-to-faster crossover was observed "
              "between the tested array sizes.")
        fastest = next((r for r in results if r[4] == "FASTER"), None)
        if fastest:
            print(f"Smallest tested N with parallel faster: {fastest[0]:,}")
        else:
            print("Parallel execution was not faster at any tested size.")

    print()
    print("Verification: PASSED (serial and parallel outputs match)")
    print("Status: COMPLETED")


if __name__ == "__main__":
    main()
