
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import time
import statistics
import os

# ==============================
# PHASE 1: FALSE SHARING
# ==============================

M = 24_000_000
SHM_SIZE = 4096
ARRAY_SIZE = 512

P_PHYS = 8
P_LOG = 8

STRIDES = [1, 2, 4, 8, 16, 32]
PROCESS_COUNTS = [1, 2, 4, 8]

WARMUP_RUNS = 2
TIMED_RUNS = 7

MASK = 0xFFFFFFFFFFFFFFFF
MULT = 6364136223846793005


# ==============================
# WORKER
# ==============================

def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:

    shm = shared_memory.SharedMemory(name=shm_name)

    arr = (ctypes.c_uint64 * ARRAY_SIZE).from_buffer(shm.buf)

    acc = arr[slot_idx]

    mask = MASK
    mult = MULT

    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc

    del arr
    shm.close()


# ==============================
# ONE EXPERIMENT
# ==============================

def run_once(p: int, stride: int) -> float:

    shm = shared_memory.SharedMemory(
        create=True,
        size=SHM_SIZE
    )

    arr = (ctypes.c_uint64 * ARRAY_SIZE).from_buffer(shm.buf)

    # Reset shared memory before each run
    for i in range(ARRAY_SIZE):
        arr[i] = 0

    del arr

    ctx = mp.get_context("spawn")

    iters = M // p

    processes = []

    start = time.perf_counter()

    for t in range(p):

        slot_idx = t * stride

        process = ctx.Process(
            target=fs_worker,
            args=(shm.name, slot_idx, iters)
        )

        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    end = time.perf_counter()

    for process in processes:
        if process.exitcode != 0:
            shm.close()
            shm.unlink()
            raise RuntimeError(
                f"Worker failed: exit code {process.exitcode}"
            )

    elapsed_ms = (end - start) * 1000

    shm.close()
    shm.unlink()

    return elapsed_ms


# ==============================
# BENCHMARK
# ==============================

def percentile(sorted_data, percent):

    position = (len(sorted_data) - 1) * percent

    lower = int(position)
    upper = min(lower + 1, len(sorted_data) - 1)

    fraction = position - lower

    return (
        sorted_data[lower] * (1 - fraction)
        + sorted_data[upper] * fraction
    )


def run_benchmark(p, stride,
                  warmup_runs=2,
                  timed_runs=7):

    print(
        f"Running: p={p}, stride={stride}",
        flush=True
    )

    # Warm-up runs
    for _ in range(warmup_runs):
        run_once(p, stride)

    # Timed runs
    results = []

    for run in range(timed_runs):
        elapsed = run_once(p, stride)
        results.append(elapsed)

        print(
            f"  Timed run {run + 1}/{timed_runs}: "
            f"{elapsed:.2f} ms",
            flush=True
        )

    results.sort()

    median_ms = statistics.median(results)

    q1 = percentile(results, 0.25)
    q3 = percentile(results, 0.75)

    iqr_ms = q3 - q1

    throughput = M / (median_ms * 1000)

    return {
        "median": median_ms,
        "iqr": iqr_ms,
        "throughput": throughput
    }


# ==============================
# TABLE 1A: STRIDE SENSITIVITY
# ==============================

def stride_sweep():

    print("\n========== TABLE 1A ==========")

    results = {}

    for stride in STRIDES:

        result = run_benchmark(
            p=P_PHYS,
            stride=stride,
            warmup_runs=WARMUP_RUNS,
            timed_runs=TIMED_RUNS
        )

        results[stride] = result

    print("\nTABLE 1A: STRIDE SENSITIVITY")

    print(
        f"{'Stride':<10}"
        f"{'Bytes':<10}"
        f"{'Median ms':<15}"
        f"{'IQR ms':<15}"
        f"{'Mops/s':<15}"
    )

    for stride, result in results.items():

        print(
            f"{stride:<10}"
            f"{stride * 8:<10}"
            f"{result['median']:<15.2f}"
            f"{result['iqr']:<15.2f}"
            f"{result['throughput']:<15.3f}"
        )

    return results


# ==============================
# KARp-FLATT METRIC
# ==============================

def karp_flatt(speedup, p):

    if p < 2:
        return None

    return (
        (1 / speedup - 1 / p)
        / (1 - 1 / p)
    )


# ==============================
# TABLE 1B: PROCESS SCALING
# ==============================

def scaling_sweep(stride):

    print(
        f"\n========== SCALING: STRIDE {stride} =========="
    )

    results = {}

    for p in PROCESS_COUNTS:

        result = run_benchmark(
            p=p,
            stride=stride,
            warmup_runs=WARMUP_RUNS,
            timed_runs=TIMED_RUNS
        )

        results[p] = result

    T1 = results[1]["median"]

    print(f"\nTABLE 1B: STRIDE = {stride}")

    print(
        f"{'p':<8}"
        f"{'Median ms':<15}"
        f"{'Speedup':<15}"
        f"{'Efficiency':<15}"
        f"{'Karp-Flatt e(p)':<18}"
    )

    for p in PROCESS_COUNTS:

        Tp = results[p]["median"]

        speedup = T1 / Tp

        efficiency = speedup / p

        e = karp_flatt(speedup, p)

        e_text = "N/A" if e is None else f"{e:.6f}"

        print(
            f"{p:<8}"
            f"{Tp:<15.2f}"
            f"{speedup:<15.3f}"
            f"{efficiency:<15.3f}"
            f"{e_text:<18}"
        )

        results[p]["speedup"] = speedup
        results[p]["efficiency"] = efficiency
        results[p]["karp_flatt"] = e

    return results


# ==============================
# MAIN
# ==============================

def main():

    print("PHASE 1: FALSE SHARING DIAGNOSIS")
    print("CPU: Apple M3")
    print(f"Physical cores: {P_PHYS}")
    print(f"Logical cores: {P_LOG}")
    print(f"Total iterations: {M:,}")

    # Table 1A
    stride_results = stride_sweep()

    # Table 1B - severe false sharing
    scaling_1 = scaling_sweep(stride=1)

    # Table 1B - isolated cache lines
    scaling_16 = scaling_sweep(stride=16)

    # Verification output
    median_1 = stride_results[1]["median"]
    median_16 = stride_results[16]["median"]

    print("\n========== VERIFICATION ==========")

    print(
        f"VERIFY_HASH[PHASE1] "
        f"passingmedian_ms(stride=1)={median_1:.2f} "
        f"median_ms(stride=16)={median_16:.2f} "
        f"atp={P_PHYS}"
    )

    print("\nPHASE 1 COMPLETED.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
