
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import time
import statistics
import math

from harness import emit_verify_tag


STUDENT_ID = 230103077

P = 8

# Phase 3A
BUFFER_SIZE = 64 * 1024 * 1024
FLOPS_PER_WORKER = 60_000_000
COMPUTE_RUNS = 3
MEMORY_RUNS = 5

# Phase 3B
N = 256
TILE_SIZES = [8, 16, 32, 64, 96, 128]
GEMM_RUNS = 3

MASK = 0xFFFFFFFFFFFFFFFF


# =========================================================
# GENERAL HELPERS
# =========================================================

def get_median(samples):
    return statistics.median(samples)


def get_ranges(total, workers):

    ranges = []

    base = total // workers
    remainder = total % workers

    start = 0

    for i in range(workers):

        size = base + (1 if i < remainder else 0)

        end = start + size

        ranges.append((start, end))

        start = end

    return ranges


def reset_shm(shm, size):

    ctypes.memset(
        ctypes.addressof(
            ctypes.c_char.from_buffer(shm.buf)
        ),
        0,
        size
    )


# =========================================================
# PHASE 3A: MEMORY BANDWIDTH
# =========================================================

def memory_copy_worker(name_a, name_b, start, end):

    shm_a = shared_memory.SharedMemory(name=name_a)
    shm_b = shared_memory.SharedMemory(name=name_b)

    mv_a = shm_a.buf.cast("B")
    mv_b = shm_b.buf.cast("B")

    mv_a[start:end] = mv_b[start:end]

    del mv_a
    del mv_b

    shm_a.close()
    shm_b.close()


def run_memory_copy(shm_a, shm_b):

    ranges = get_ranges(BUFFER_SIZE, P)

    workers = []

    start = time.perf_counter_ns()

    for s, e in ranges:

        proc = mp.Process(
            target=memory_copy_worker,
            args=(shm_a.name, shm_b.name, s, e)
        )

        workers.append(proc)
        proc.start()

    for proc in workers:

        proc.join()

        if proc.exitcode != 0:
            raise RuntimeError("Memory copy worker failed")

    end = time.perf_counter_ns()

    return (end - start) / 1e9


def benchmark_memory():

    print("\n--- PHASE 3A: DRAM MEMORY BANDWIDTH ---")

    shm_a = shared_memory.SharedMemory(
        create=True,
        size=BUFFER_SIZE
    )

    shm_b = shared_memory.SharedMemory(
        create=True,
        size=BUFFER_SIZE
    )

    try:

        reset_shm(shm_a, BUFFER_SIZE)

        # Initialize source buffer.
        ctypes.memset(
            ctypes.addressof(
                ctypes.c_char.from_buffer(shm_b.buf)
            ),
            1,
            BUFFER_SIZE
        )

        samples = []

        for trial in range(MEMORY_RUNS):

            elapsed = run_memory_copy(shm_a, shm_b)

            samples.append(elapsed)

            print(
                f"Memory trial {trial + 1}: "
                f"{elapsed:.4f} seconds"
            )

        median_time = get_median(samples)

        bandwidth = (
            2 * BUFFER_SIZE
        ) / median_time / 1e9

        print(f"\nMedian time: {median_time:.6f} seconds")
        print(f"Sustained bandwidth: {bandwidth:.4f} GB/s")

        return bandwidth

    finally:

        shm_a.close()
        shm_a.unlink()

        shm_b.close()
        shm_b.unlink()


# =========================================================
# PHASE 3A: COMPUTE CEILING
# =========================================================

def compute_worker():

    # Sixteen independent scalar accumulators.
    a0 = 0.1
    a1 = 0.2
    a2 = 0.3
    a3 = 0.4
    a4 = 0.5
    a5 = 0.6
    a6 = 0.7
    a7 = 0.8

    a8 = 0.9
    a9 = 1.0
    a10 = 1.1
    a11 = 1.2
    a12 = 1.3
    a13 = 1.4
    a14 = 1.5
    a15 = 1.6

    b = 1.0000001
    c = 0.0000001

    iterations = FLOPS_PER_WORKER // 32

    for _ in range(iterations):

        a0 = a0 * b + c
        a1 = a1 * b + c
        a2 = a2 * b + c
        a3 = a3 * b + c
        a4 = a4 * b + c
        a5 = a5 * b + c
        a6 = a6 * b + c
        a7 = a7 * b + c

        a8 = a8 * b + c
        a9 = a9 * b + c
        a10 = a10 * b + c
        a11 = a11 * b + c
        a12 = a12 * b + c
        a13 = a13 * b + c
        a14 = a14 * b + c
        a15 = a15 * b + c

    return (
        a0 + a1 + a2 + a3 +
        a4 + a5 + a6 + a7 +
        a8 + a9 + a10 + a11 +
        a12 + a13 + a14 + a15
    )


def compute_process_worker(queue):

    result = compute_worker()

    queue.put(result)


def run_compute():

    queue = mp.Queue()
    workers = []

    start = time.perf_counter_ns()

    for _ in range(P):

        proc = mp.Process(
            target=compute_process_worker,
            args=(queue,)
        )

        workers.append(proc)
        proc.start()

    for proc in workers:

        proc.join()

        if proc.exitcode != 0:
            raise RuntimeError("Compute worker failed")

    results = [queue.get() for _ in range(P)]

    end = time.perf_counter_ns()

    return (end - start) / 1e9, results


def benchmark_compute():

    print("\n--- PHASE 3A: COMPUTE CEILING ---")

    samples = []

    for trial in range(COMPUTE_RUNS):

        elapsed, results = run_compute()

        samples.append(elapsed)

        print(
            f"Compute trial {trial + 1}: "
            f"{elapsed:.4f} seconds"
        )

    median_time = get_median(samples)

    total_flops = FLOPS_PER_WORKER * P

    performance = (
        total_flops / median_time / 1e9
    )

    print(f"\nMedian time: {median_time:.6f} seconds")
    print(f"Total FLOPs: {total_flops:,}")
    print(f"Compute ceiling: {performance:.4f} GFLOP/s")

    return performance


# =========================================================
# PHASE 3B: GEMM WORKER
# =========================================================

def gemm_worker(
    name_a,
    name_b,
    name_c,
    row_start,
    row_end,
    variant,
    block_size
):

    shm_a = shared_memory.SharedMemory(name=name_a)
    shm_b = shared_memory.SharedMemory(name=name_b)
    shm_c = shared_memory.SharedMemory(name=name_c)

    A = (ctypes.c_double * (N * N)).from_buffer(shm_a.buf)
    B = (ctypes.c_double * (N * N)).from_buffer(shm_b.buf)
    C = (ctypes.c_double * (N * N)).from_buffer(shm_c.buf)

    if variant == "naive":

        # Variant 1: i-j-k
        for i in range(row_start, row_end):

            for j in range(N):

                total = 0.0

                for k in range(N):

                    total += A[i * N + k] * B[k * N + j]

                C[i * N + j] = total

    elif variant == "reordered":

        # Variant 2: i-k-j
        for i in range(row_start, row_end):

            for k in range(N):

                a_ik = A[i * N + k]

                for j in range(N):

                    C[i * N + j] += (
                        a_ik * B[k * N + j]
                    )

    elif variant == "blocked":

        # Variant 3: Cache-blocked i-k-j
        BS = block_size

        for ii in range(row_start, row_end, BS):

            i_end = min(ii + BS, row_end)

            for jj in range(0, N, BS):

                j_end = min(jj + BS, N)

                for kk in range(0, N, BS):

                    k_end = min(kk + BS, N)

                    for i in range(ii, i_end):

                        for k in range(kk, k_end):

                            a_ik = A[i * N + k]

                            for j in range(jj, j_end):

                                C[i * N + j] += (
                                    a_ik * B[k * N + j]
                                )

    else:

        raise ValueError("Unknown GEMM variant")

    del A
    del B
    del C

    shm_a.close()
    shm_b.close()
    shm_c.close()


# =========================================================
# GEMM EXECUTION
# =========================================================

def run_gemm(shm_a, shm_b, shm_c, variant, block_size=32):

    reset_shm(shm_c, N * N * 8)

    ranges = get_ranges(N, P)

    workers = []

    start = time.perf_counter_ns()

    for row_start, row_end in ranges:

        proc = mp.Process(
            target=gemm_worker,
            args=(
                shm_a.name,
                shm_b.name,
                shm_c.name,
                row_start,
                row_end,
                variant,
                block_size
            )
        )

        workers.append(proc)
        proc.start()

    for proc in workers:

        proc.join()

        if proc.exitcode != 0:
            raise RuntimeError(
                f"GEMM worker failed: {proc.exitcode}"
            )

    end = time.perf_counter_ns()

    return (end - start) / 1e9


def measure_gemm(shm_a, shm_b, shm_c, variant, block_size=32):

    samples = []

    for _ in range(GEMM_RUNS):

        elapsed = run_gemm(
            shm_a,
            shm_b,
            shm_c,
            variant,
            block_size
        )

        samples.append(elapsed)

    return get_median(samples)


def calculate_gflops(elapsed):

    return (
        2 * N ** 3
    ) / elapsed / 1e9


# =========================================================
# INITIALIZE MATRICES
# =========================================================

def initialize_matrices(shm_a, shm_b):

    A = (ctypes.c_double * (N * N)).from_buffer(shm_a.buf)
    B = (ctypes.c_double * (N * N)).from_buffer(shm_b.buf)

    for i in range(N * N):

        A[i] = ((i * 17) % 101) / 101.0
        B[i] = ((i * 13 + 7) % 103) / 103.0

    del A
    del B


# =========================================================
# VERIFY GEMM RESULTS
# =========================================================

def compare_matrices(shm_a, shm_b):

    C2 = (ctypes.c_double * (N * N)).from_buffer(shm_a.buf)
    C3 = (ctypes.c_double * (N * N)).from_buffer(shm_b.buf)

    max_error = 0.0

    for i in range(N * N):

        error = abs(C2[i] - C3[i])

        if error > max_error:
            max_error = error

    del C2
    del C3

    return max_error


# =========================================================
# PHASE 3B: GEMM BENCHMARK
# =========================================================

def benchmark_gemm():

    print("\n--- PHASE 3B: GEMM BENCHMARK ---")

    size = N * N * 8

    shm_a = shared_memory.SharedMemory(
        create=True,
        size=size
    )

    shm_b = shared_memory.SharedMemory(
        create=True,
        size=size
    )

    shm_c = shared_memory.SharedMemory(
        create=True,
        size=size
    )

    shm_check = shared_memory.SharedMemory(
        create=True,
        size=size
    )

    try:

        initialize_matrices(shm_a, shm_b)

        # Variant 1
        print("\nVariant 1: Naive i-j-k")

        t1 = measure_gemm(
            shm_a,
            shm_b,
            shm_c,
            "naive"
        )

        gflops1 = calculate_gflops(t1)

        print(f"Median: {t1:.6f} seconds")
        print(f"Performance: {gflops1:.6f} GFLOP/s")

        # Variant 2
        print("\nVariant 2: Reordered i-k-j")

        t2 = measure_gemm(
            shm_a,
            shm_b,
            shm_c,
            "reordered"
        )

        gflops2 = calculate_gflops(t2)

        print(f"Median: {t2:.6f} seconds")
        print(f"Performance: {gflops2:.6f} GFLOP/s")

        # Save Variant 2 output for verification.
        reset_shm(shm_check, size)

        # Recalculate Variant 2 into check buffer.
        ranges = get_ranges(N, P)
        workers = []

        for row_start, row_end in ranges:

            proc = mp.Process(
                target=gemm_worker,
                args=(
                    shm_a.name,
                    shm_b.name,
                    shm_check.name,
                    row_start,
                    row_end,
                    "reordered",
                    32
                )
            )

            workers.append(proc)
            proc.start()

        for proc in workers:
            proc.join()

            if proc.exitcode != 0:
                raise RuntimeError("Verification worker failed")

        # Tile sweep
        print("\n--- CACHE TILE SWEEP ---")

        tile_results = {}

        for BS in TILE_SIZES:

            print(f"\nTesting BS={BS}")

            elapsed = measure_gemm(
                shm_a,
                shm_b,
                shm_c,
                "blocked",
                BS
            )

            gflops = calculate_gflops(elapsed)

            footprint = 3 * BS * BS * 8
            intensity = BS / 12.0

            tile_results[BS] = {
                "time": elapsed,
                "gflops": gflops,
                "footprint": footprint,
                "intensity": intensity
            }

            print(f"Median: {elapsed:.6f} seconds")
            print(f"GFLOP/s: {gflops:.6f}")
            print(f"Tile footprint: {footprint:,} bytes")
            print(f"Operational intensity: {intensity:.4f} FLOP/byte")

        # Select best measured tile.
        best_bs = max(
            tile_results,
            key=lambda bs: tile_results[bs]["gflops"]
        )

        best = tile_results[best_bs]

        print("\n--- BEST TILE ---")
        print("Best BS:", best_bs)
        print(f"Best GFLOP/s: {best['gflops']:.6f}")
        print(f"Tile footprint: {best['footprint']:,} bytes")
        print(f"Operational intensity: {best['intensity']:.4f}")

        # Run the best blocked variant once for verification.
        run_gemm(
            shm_a,
            shm_b,
            shm_c,
            "blocked",
            best_bs
        )

        error = compare_matrices(
            shm_check,
            shm_c
        )

        print("\n--- NUMERICAL VERIFICATION ---")
        print(f"Maximum absolute error: {error:.12e}")

        if error < 1e-9:
            print("GEMM verification PASSED")
        else:
            print("GEMM verification FAILED")

        return {
            "gflops1": gflops1,
            "gflops2": gflops2,
            "best_bs": best_bs,
            "best_gflops": best["gflops"],
            "error": error
        }

    finally:

        for shm in [shm_a, shm_b, shm_c, shm_check]:

            shm.close()
            shm.unlink()


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 75)
    print("PHASE 3: EMPIRICAL ROOFLINE & CACHE-BLOCKED GEMM")
    print("=" * 75)

    print("CPU logical cores:", mp.cpu_count())
    print("Physical workers:", P)
    print("Cache line:", 128, "bytes")
    print("L1d cache:", 65536, "bytes")
    print("L2 cache:", 4194304, "bytes")

    bandwidth = benchmark_memory()

    compute_performance = benchmark_compute()

    ridge_point = compute_performance / bandwidth

    print("\n--- ROOFLINE CEILINGS ---")
    print(f"Memory bandwidth: {bandwidth:.4f} GB/s")
    print(f"Compute ceiling: {compute_performance:.4f} GFLOP/s")
    print(f"Ridge point: {ridge_point:.4f} FLOP/byte")

    gemm = benchmark_gemm()

    print("\n--- ROOFLINE SUMMARY ---")

    print(
        f"Naive GEMM: {gemm['gflops1']:.6f} GFLOP/s"
    )

    print(
        f"Reordered GEMM: {gemm['gflops2']:.6f} GFLOP/s"
    )

    print(
        f"Blocked GEMM: {gemm['best_gflops']:.6f} GFLOP/s"
    )

    print(f"Best tile size: {gemm['best_bs']}")

    print("\n--- VERIFICATION ---")

    tag = emit_verify_tag(
        "PHASE3",
        STUDENT_ID,
        gemm["gflops1"],
        gemm["best_gflops"]
    )

    print(tag)

    print("\nPHASE 3 COMPLETED")


if __name__ == "__main__":

    mp.freeze_support()
    main()
