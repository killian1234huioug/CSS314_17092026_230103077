
import os
import time
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = "230103077"
N = 30_000_000
REPEATS = 3
MAX_CORES = os.cpu_count() or 1
WORKER_COUNTS = sorted(set([1, 2, MAX_CORES]))


def make_chunks(n, workers):
    chunk_size = (n + workers - 1) // workers
    return [
        (start, min(start + chunk_size, n))
        for start in range(0, n, chunk_size)
    ]


def run_kernel_a(a, b, workers):
    """Memory-bound: copy A into B."""
    chunks = make_chunks(len(a), workers)

    def process(start, end):
        np.copyto(b[start:end], a[start:end])

    if workers == 1:
        for start, end in chunks:
            process(start, end)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(lambda r: process(*r), chunks))


def run_kernel_b(a, b, temp1, temp2, workers):
    """Compute-bound: B[i] = sin(A[i])*cos(A[i]) + tan(A[i])."""
    chunks = make_chunks(len(a), workers)

    def process(start, end):
        x = a[start:end]
        out = b[start:end]
        t1 = temp1[start:end]
        t2 = temp2[start:end]

        np.sin(x, out=out)
        np.cos(x, out=t1)
        np.tan(x, out=t2)
        np.multiply(out, t1, out=out)
        np.add(out, t2, out=out)

    if workers == 1:
        for start, end in chunks:
            process(start, end)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(lambda r: process(*r), chunks))


def measure(function):
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def report_results(name, times):
    baseline = times[1]
    print(f"\n{name}")
    print(f"{'Cores':>8} {'Time (s)':>12} "
          f"{'Speedup':>12} {'Efficiency':>13}")
    print("-" * 47)

    rows = []
    for cores in WORKER_COUNTS:
        elapsed = times[cores]
        speedup = baseline / elapsed
        efficiency = speedup / cores
        rows.append((cores, elapsed, speedup, efficiency))
        print(f"{cores:>8} {elapsed:>12.3f} "
              f"{speedup:>12.3f} {efficiency:>12.1%}")

    return rows


def main():
    print("=== TASK 7: ARITHMETIC INTENSITY VS MEMORY BOTTLENECKS ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"CPU architecture: {os.uname().machine}")
    print(f"Logical CPU count: {MAX_CORES}")
    print(f"Array size: {N:,} float64 elements")
    print(f"Workers tested: {WORKER_COUNTS}")
    print(f"Timing: median of {REPEATS} runs")
    print()

    rng = np.random.default_rng(230103077)
    a = rng.uniform(0.1, 1.0, size=N).astype(np.float64)
    b = np.empty_like(a)
    temp1 = np.empty_like(a)
    temp2 = np.empty_like(a)

    print("Running Kernel A: vector copy...")
    times_a = {}
    for workers in WORKER_COUNTS:
        times_a[workers] = measure(
            lambda w=workers: run_kernel_a(a, b, w)
        )

    if not np.array_equal(a, b):
        raise RuntimeError("Kernel A verification failed")

    print("Running Kernel B: sin*cos + tan...")
    times_b = {}
    for workers in WORKER_COUNTS:
        times_b[workers] = measure(
            lambda w=workers: run_kernel_b(a, b, temp1, temp2, w)
        )

    sample = a[:1000]
    expected = np.sin(sample) * np.cos(sample) + np.tan(sample)
    if not np.allclose(b[:1000], expected, rtol=1e-12, atol=1e-12):
        raise RuntimeError("Kernel B verification failed")

    rows_a = report_results("Kernel A: Memory-Bound Copy", times_a)
    rows_b = report_results("Kernel B: Compute-Intensive Math", times_b)

    max_a = next(row for row in rows_a if row[0] == MAX_CORES)
    max_b = next(row for row in rows_b if row[0] == MAX_CORES)

    print("\n=== SCALING ANALYSIS ===")
    print(f"Kernel A efficiency at {MAX_CORES} cores: {max_a[3]:.1%}")
    print(f"Kernel B efficiency at {MAX_CORES} cores: {max_b[3]:.1%}")

    if max_a[2] < max_b[2] and max_a[3] < max_b[3]:
        print(
            "Diagnosis: Kernel A showed weaker scaling than Kernel B, "
            "consistent with a memory-bandwidth bottleneck."
        )
        print(
            "Conclusion: Kernel A is the stronger candidate for "
            "memory-channel saturation on this run."
        )
    elif max_a[3] < 0.5:
        print(
            "Diagnosis: Kernel A efficiency fell below 50% at maximum "
            "cores, consistent with scaling saturation."
        )
        print(
            "Conclusion: Kernel A showed a likely memory-bandwidth "
            "bottleneck; compare Kernel B to distinguish compute limits."
        )
    else:
        print(
            "Diagnosis: This run did not clearly isolate memory-channel "
            "saturation from the timing results."
        )
        print(
            "Conclusion: Use the measured speedups and efficiencies; "
            "low efficiency alone does not prove RAM saturation."
        )

    print("\nVerification: PASSED")
    print("Status: COMPLETED")


if __name__ == "__main__":
    main()
