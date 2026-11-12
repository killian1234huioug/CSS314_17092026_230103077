
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import statistics
import hashlib
import gc

# =========================
# CONFIGURATION
# =========================

N = 384
P = 8
RUNS = 5
BLOCK_SIZES = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]

A_SHM = None
B_SHM = None
C1_SHM = None
C2_SHM = None
C3_SHM = None

A = None
B = None
C1 = None
C2 = None
C3 = None


# =========================
# WORKER INITIALIZATION
# =========================

def init_worker(a_name, b_name, c1_name, c2_name, c3_name):
    global A_SHM, B_SHM, C1_SHM, C2_SHM, C3_SHM
    global A, B, C1, C2, C3

    A_SHM = shared_memory.SharedMemory(name=a_name)
    B_SHM = shared_memory.SharedMemory(name=b_name)
    C1_SHM = shared_memory.SharedMemory(name=c1_name)
    C2_SHM = shared_memory.SharedMemory(name=c2_name)
    C3_SHM = shared_memory.SharedMemory(name=c3_name)

    A = A_SHM.buf.cast("d")
    B = B_SHM.buf.cast("d")
    C1 = C1_SHM.buf.cast("d")
    C2 = C2_SHM.buf.cast("d")
    C3 = C3_SHM.buf.cast("d")


# =========================
# MATRIX MULTIPLICATION
# =========================

def gemm_worker(task):
    start_row, end_row, variant, block_size = task

    if variant == 1:
        # Variant 1: Naive i-j-k
        for i in range(start_row, end_row):
            for j in range(N):
                total = 0.0
                for k in range(N):
                    total += A[i * N + k] * B[k * N + j]
                C1[i * N + j] = total

    elif variant == 2:
        # Variant 2: Loop reordering i-k-j
        for i in range(start_row, end_row):
            row_start = i * N

            for j in range(N):
                C2[row_start + j] = 0.0

            for k in range(N):
                aik = A[row_start + k]
                b_row = k * N

                for j in range(N):
                    C2[row_start + j] += aik * B[b_row + j]

    elif variant == 3:
        # Variant 3: Cache blocking
        bs = block_size

        for i in range(start_row, end_row):
            row_start = i * N
            for j in range(N):
                C3[row_start + j] = 0.0

        for ii in range(start_row, end_row, bs):
            i_end = min(ii + bs, end_row)

            for kk in range(0, N, bs):
                k_end = min(kk + bs, N)

                for jj in range(0, N, bs):
                    j_end = min(jj + bs, N)

                    for i in range(ii, i_end):
                        row_start = i * N

                        for k in range(kk, k_end):
                            aik = A[row_start + k]
                            b_row = k * N

                            for j in range(jj, j_end):
                                C3[row_start + j] += (
                                    aik * B[b_row + j]
                                )

    return end_row - start_row


# =========================
# BENCHMARK FUNCTION
# =========================

def benchmark(pool, variant, block_size=None):
    tasks = []

    rows_per_worker = N // P

    for worker in range(P):
        start = worker * rows_per_worker
        end = (worker + 1) * rows_per_worker

        if worker == P - 1:
            end = N

        tasks.append((start, end, variant, block_size))

    # Warm-up
    pool.map(gemm_worker, tasks)

    times = []

    for run in range(RUNS):
        start_time = time.perf_counter()

        pool.map(gemm_worker, tasks)

        elapsed = time.perf_counter() - start_time
        times.append(elapsed)

        print(
            "Run {}: {:.4f} seconds".format(
                run + 1, elapsed
            ),
            flush=True
        )

    median_time = statistics.median(times)

    total_flops = 2 * (N ** 3)
    gflops = total_flops / median_time / 1e9

    return median_time, gflops


# =========================
# MAIN
# =========================

def main():
    global A, B, C1, C2, C3

    mp.freeze_support()

    matrix_bytes = N * N * 8

    shm_objects = []

    try:
        # Create shared memory
        for _ in range(5):
            shm_objects.append(
                shared_memory.SharedMemory(
                    create=True,
                    size=matrix_bytes
                )
            )

        shm_a, shm_b, shm_c1, shm_c2, shm_c3 = shm_objects

        # Initialize matrices
        a_np = np.ndarray(
            (N, N), dtype=np.float64, buffer=shm_a.buf
        )

        b_np = np.ndarray(
            (N, N), dtype=np.float64, buffer=shm_b.buf
        )

        c1_np = np.ndarray(
            (N, N), dtype=np.float64, buffer=shm_c1.buf
        )

        c2_np = np.ndarray(
            (N, N), dtype=np.float64, buffer=shm_c2.buf
        )

        c3_np = np.ndarray(
            (N, N), dtype=np.float64, buffer=shm_c3.buf
        )

        for i in range(N):
            for j in range(N):
                a_np[i, j] = ((i * 3 + j * 7) % 100) / 100.0
                b_np[i, j] = ((i * 5 + j * 11) % 100) / 100.0

        c1_np.fill(0.0)
        c2_np.fill(0.0)
        c3_np.fill(0.0)

        print("\n========== PHASE 3B: GEMM ==========")
        print("Matrix size: {} x {}".format(N, N))
        print("Processes:", P)
        print("Timed runs per configuration:", RUNS)

        results = {}

        with mp.Pool(
            processes=P,
            initializer=init_worker,
            initargs=(
                shm_a.name,
                shm_b.name,
                shm_c1.name,
                shm_c2.name,
                shm_c3.name
            )
        ) as pool:

            # -------------------------
            # VARIANT 1
            # -------------------------

            print("\n========== VARIANT 1: NAIVE i-j-k ==========")

            t1, g1 = benchmark(pool, 1)

            results["Variant 1"] = (t1, g1)

            print("Median: {:.4f} seconds".format(t1))
            print("Performance: {:.6f} GFLOP/s".format(g1))

            # -------------------------
            # VARIANT 2
            # -------------------------

            print("\n========== VARIANT 2: LOOP REORDERING i-k-j ==========")

            t2, g2 = benchmark(pool, 2)

            results["Variant 2"] = (t2, g2)

            print("Median: {:.4f} seconds".format(t2))
            print("Performance: {:.6f} GFLOP/s".format(g2))

            # -------------------------
            # VARIANT 3
            # -------------------------

            print("\n========== VARIANT 3: CACHE BLOCKING ==========")

            for bs in BLOCK_SIZES:
                print("\nBlock size BS =", bs)

                t3, g3 = benchmark(pool, 3, bs)

                results[bs] = (t3, g3)

                tile_footprint = 3 * bs * bs * 8

                print("Median: {:.4f} seconds".format(t3))
                print("Performance: {:.6f} GFLOP/s".format(g3))
                print(
                    "Tile footprint: {} bytes".format(
                        tile_footprint
                    )
                )

            # -------------------------
            # BEST BLOCK
            # -------------------------

            best_bs = max(
                BLOCK_SIZES,
                key=lambda bs: results[bs][1]
            )

            best_time, best_gflops = results[best_bs]

            print("\n========== BEST BLOCK SIZE ==========")
            print("Best BS:", best_bs)
            print("Best median: {:.4f} seconds".format(best_time))
            print("Best performance: {:.6f} GFLOP/s".format(best_gflops))

            # Run best block again for verification
            tasks = []

            for worker in range(P):
                start = worker * (N // P)
                end = (worker + 1) * (N // P)
                tasks.append((start, end, 3, best_bs))

            pool.map(gemm_worker, tasks)

        # -------------------------
        # NUMERICAL VERIFICATION
        # -------------------------

        print("\n========== NUMERICAL VERIFICATION ==========")

        c2_result = np.array(c2_np, copy=True)
        c3_result = np.array(c3_np, copy=True)

        max_error = float(
            np.max(np.abs(c3_result - c2_result))
        )

        print("Maximum absolute error:", max_error)

        if max_error < 1e-9:
            print("VERIFICATION PASSED")
        else:
            print("VERIFICATION FAILED")

        # -------------------------
        # ROOFLINE COMPARISON
        # -------------------------

        print("\n========== ROOFLINE COMPARISON ==========")

        print("Variant 1: {:.6f} GFLOP/s".format(g1))
        print("Variant 2: {:.6f} GFLOP/s".format(g2))
        print("Best Variant 3: {:.6f} GFLOP/s".format(best_gflops))

        print("\nTile footprint for best BS:")
        print("{} bytes".format(3 * best_bs * best_bs * 8))

        print("\n========== PHASE 3B SUMMARY ==========")

        print("Best block size:", best_bs)
        print("Best Variant 3 GFLOP/s:", best_gflops)
        print("Max numerical error:", max_error)

        verify_text = (
            "PHASE3|V1={:.6f}|V3={:.6f}|BS={}|ERROR={:.12e}"
        ).format(g1, best_gflops, best_bs, max_error)

        verify_hash = hashlib.sha256(
            verify_text.encode()
        ).hexdigest()[:16].upper()

        print("VERIFY_HASH[PHASE3]:", verify_hash)

    finally:
        # Release parent memory views before closing shared memory
        A = B = C1 = C2 = C3 = None
        gc.collect()

        for shm in shm_objects:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
