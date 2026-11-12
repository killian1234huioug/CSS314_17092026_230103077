
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import statistics

P_PHYS = 8

# Memory benchmark: two buffers of 96 MB each
BUFFER_SIZE = 96_000_000
CHUNK_SIZE = BUFFER_SIZE // P_PHYS

# Compute benchmark: 80 million FLOPs per worker
FLOPS_PER_WORKER = 80_000_000
ITERATIONS = FLOPS_PER_WORKER // 32

REPEATS = 5
WARMUPS = 1

_shared_a = None
_shared_b = None
_view_a = None
_view_b = None


def init_memory_worker(name_a, name_b):
    global _shared_a, _shared_b, _view_a, _view_b

    _shared_a = shared_memory.SharedMemory(name=name_a)
    _shared_b = shared_memory.SharedMemory(name=name_b)

    _view_a = _shared_a.buf
    _view_b = _shared_b.buf


def copy_worker(bounds):
    start, end = bounds
    _view_a[start:end] = _view_b[start:end]
    return end - start


def compute_worker(worker_id):
    a = 1.0000000001
    b = 0.00000001

    x0 = 1.0
    x1 = 1.1
    x2 = 1.2
    x3 = 1.3
    x4 = 1.4
    x5 = 1.5
    x6 = 1.6
    x7 = 1.7
    x8 = 1.8
    x9 = 1.9
    x10 = 2.0
    x11 = 2.1
    x12 = 2.2
    x13 = 2.3
    x14 = 2.4
    x15 = 2.5

    for _ in range(ITERATIONS):
        x0 = x0 * a + b
        x1 = x1 * a + b
        x2 = x2 * a + b
        x3 = x3 * a + b
        x4 = x4 * a + b
        x5 = x5 * a + b
        x6 = x6 * a + b
        x7 = x7 * a + b
        x8 = x8 * a + b
        x9 = x9 * a + b
        x10 = x10 * a + b
        x11 = x11 * a + b
        x12 = x12 * a + b
        x13 = x13 * a + b
        x14 = x14 * a + b
        x15 = x15 * a + b

    return (
        x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
        + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15
    )


def benchmark_memory():
    print("\n========== PHASE 3A: MEMORY BANDWIDTH ==========")

    shm_a = shared_memory.SharedMemory(
        create=True, size=BUFFER_SIZE
    )
    shm_b = shared_memory.SharedMemory(
        create=True, size=BUFFER_SIZE
    )


    chunks = [
        (i * CHUNK_SIZE,
         (i + 1) * CHUNK_SIZE if i < P_PHYS - 1 else BUFFER_SIZE)
        for i in range(P_PHYS)
    ]

    times = []

    try:
        ctx = mp.get_context("spawn")

        with ctx.Pool(
            processes=P_PHYS,
            initializer=init_memory_worker,
            initargs=(shm_a.name, shm_b.name)
        ) as pool:

            for _ in range(WARMUPS):
                pool.map(copy_worker, chunks)

            for run in range(REPEATS):
                start = time.perf_counter()
                pool.map(copy_worker, chunks)
                elapsed = time.perf_counter() - start

                times.append(elapsed)
                print(
                    f"Memory run {run + 1}: "
                    f"{elapsed:.4f} seconds"
                )

        median_time = statistics.median(times)

        # Each copied byte causes one read and one write
        total_bytes = 2 * BUFFER_SIZE
        bandwidth = total_bytes / median_time / 1e9

        print(f"\nMedian memory time: {median_time:.6f} seconds")
        print(f"Memory bandwidth: {bandwidth:.6f} GB/s")

        return bandwidth

    finally:
        shm_a.close()
        shm_b.close()
        shm_a.unlink()
        shm_b.unlink()


def benchmark_compute():
    print("\n========== PHASE 3A: PYTHON COMPUTE CEILING ==========")

    ctx = mp.get_context("spawn")
    times = []

    with ctx.Pool(processes=P_PHYS) as pool:

        for _ in range(WARMUPS):
            pool.map(compute_worker, range(P_PHYS), chunksize=1)

        for run in range(REPEATS):
            start = time.perf_counter()

            results = pool.map(
                compute_worker,
                range(P_PHYS),
                chunksize=1
            )

            elapsed = time.perf_counter() - start
            times.append(elapsed)

            print(
                f"Compute run {run + 1}: "
                f"{elapsed:.4f} seconds"
            )

    median_time = statistics.median(times)

    total_flops = FLOPS_PER_WORKER * P_PHYS
    performance = total_flops / median_time / 1e9

    print(f"\nMedian compute time: {median_time:.6f} seconds")
    print(f"Total FLOPs: {total_flops}")
    print(f"Compute performance: {performance:.6f} GFLOP/s")

    return performance


def main():
    bandwidth = benchmark_memory()
    performance = benchmark_compute()

    ridge_point = performance / bandwidth

    print("\n========== PHASE 3A SUMMARY ==========")
    print(f"Physical processes: {P_PHYS}")
    print(f"BW_sustained: {bandwidth:.6f} GB/s")
    print(f"P_sustained: {performance:.6f} GFLOP/s")
    print(f"Ridge Point: {ridge_point:.6f} FLOPs/Byte")


if __name__ == "__main__":
    main()
