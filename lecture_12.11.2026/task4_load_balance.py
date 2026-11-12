
import multiprocessing as mp
import time
import statistics
import hashlib

# =====================================
# CONFIGURATION
# =====================================

N = 12000
P = 8

CHUNK_SIZES = [1, 4, 16, 64, 256, 1024]
RUNS = 3


# =====================================
# WORKLOAD FUNCTION
# =====================================

def workload(i):
    """
    Variable workload:
    K(i) = 1 + floor((i / 60)^2)
    """

    k = 1 + (i // 60) ** 2

    x = i + 1

    for _ in range(k):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF

    return x


# =====================================
# WORKER FUNCTION
# =====================================

def worker(worker_id, strategy, chunk_size,
           active_times, checksums, counter):

    active_start = time.perf_counter()
    checksum = 0

    if strategy == "static":
        # Static contiguous scheduling
        start = worker_id * N // P
        end = (worker_id + 1) * N // P

        for i in range(start, end):
            checksum ^= workload(i)

    elif strategy == "cyclic":
        # Cyclic round-robin scheduling
        for i in range(worker_id, N, P):
            checksum ^= workload(i)

    elif strategy == "dynamic":
        # Dynamic scheduling using a shared counter
        while True:

            with counter.get_lock():
                start = counter.value

                if start >= N:
                    break

                end = min(start + chunk_size, N)
                counter.value = end

            for i in range(start, end):
                checksum ^= workload(i)

    active_end = time.perf_counter()

    active_times[worker_id] = active_end - active_start
    checksums[worker_id] = checksum


# =====================================
# RUN ONE CONFIGURATION
# =====================================

def run_once(strategy, chunk_size=None):

    active_times = mp.Array("d", P, lock=False)
    checksums = mp.Array("q", P, lock=False)

    counter = mp.Value("i", 0)

    processes = []

    start_time = time.perf_counter()

    for worker_id in range(P):

        process = mp.Process(
            target=worker,
            args=(
                worker_id,
                strategy,
                chunk_size,
                active_times,
                checksums,
                counter
            )
        )

        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    wall_time = time.perf_counter() - start_time

    for process in processes:
        if process.exitcode != 0:
            raise RuntimeError(
                "Worker process failed: exit code {}".format(
                    process.exitcode
                )
            )

    times = list(active_times)
    total_checksum = 0

    for value in checksums:
        total_checksum ^= value

    average_time = sum(times) / P

    if average_time > 0:
        imbalance = max(times) / average_time
    else:
        imbalance = 0.0

    return wall_time, times, imbalance, total_checksum


# =====================================
# BENCHMARK CONFIGURATION
# =====================================

def benchmark(strategy, chunk_size=None):

    wall_results = []
    worker_results = [[] for _ in range(P)]
    imbalance_results = []
    checksum_results = []

    for run in range(RUNS):

        wall, times, imbalance, checksum = run_once(
            strategy,
            chunk_size
        )

        wall_results.append(wall)
        imbalance_results.append(imbalance)
        checksum_results.append(checksum)

        for worker_id in range(P):
            worker_results[worker_id].append(
                times[worker_id]
            )

        print(
            "Run {}: Wall time = {:.4f} s, Imbalance = {:.4f}".format(
                run + 1, wall, imbalance
            ),
            flush=True
        )

    median_wall = statistics.median(wall_results)

    average_worker_times = [
        statistics.mean(worker_results[i])
        for i in range(P)
    ]

    average_imbalance = statistics.mean(imbalance_results)

    return (
        median_wall,
        average_worker_times,
        average_imbalance,
        checksum_results
    )


# =====================================
# MAIN
# =====================================

def main():

    mp.freeze_support()

    print("\n========== PHASE 4: LOAD BALANCING ==========")
    print("Number of tasks:", N)
    print("Number of processes:", P)
    print("Timed runs:", RUNS)

    results = {}

    # ---------------------------------
    # STRATEGY 1: STATIC CONTIGUOUS
    # ---------------------------------

    print("\n========== STRATEGY 1: STATIC CONTIGUOUS ==========")

    wall, times, imbalance, checksums = benchmark("static")

    results["static"] = (wall, times, imbalance, checksums)

    print("\nMedian wall time: {:.6f} seconds".format(wall))
    print("Worker active times:")

    for i, value in enumerate(times):
        print("Worker {}: {:.6f} seconds".format(i, value))

    print("Load imbalance lambda: {:.6f}".format(imbalance))

    # ---------------------------------
    # STRATEGY 2: CYCLIC ROUND-ROBIN
    # ---------------------------------

    print("\n========== STRATEGY 2: CYCLIC ROUND-ROBIN ==========")

    wall, times, imbalance, checksums = benchmark("cyclic")

    results["cyclic"] = (wall, times, imbalance, checksums)

    print("\nMedian wall time: {:.6f} seconds".format(wall))
    print("Worker active times:")

    for i, value in enumerate(times):
        print("Worker {}: {:.6f} seconds".format(i, value))

    print("Load imbalance lambda: {:.6f}".format(imbalance))

    # ---------------------------------
    # STRATEGY 3: DYNAMIC SCHEDULING
    # ---------------------------------

    print("\n========== STRATEGY 3: DYNAMIC SCHEDULING ==========")

    for chunk in CHUNK_SIZES:

        print("\n---------- CHUNK SIZE C = {} ----------".format(chunk))

        wall, times, imbalance, checksums = benchmark(
            "dynamic",
            chunk
        )

        results[chunk] = (wall, times, imbalance, checksums)

        print("\nMedian wall time: {:.6f} seconds".format(wall))
        print("Worker active times:")

        for i, value in enumerate(times):
            print("Worker {}: {:.6f} seconds".format(i, value))

        print("Load imbalance lambda: {:.6f}".format(imbalance))

    # ---------------------------------
    # BEST DYNAMIC CHUNK
    # ---------------------------------

    best_chunk = min(
        CHUNK_SIZES,
        key=lambda c: results[c][0]
    )

    best_wall, best_times, best_imbalance, _ = results[best_chunk]

    print("\n========== BEST DYNAMIC CONFIGURATION ==========")
    print("Best chunk size C*:", best_chunk)
    print("Best median wall time: {:.6f} seconds".format(best_wall))
    print("Best load imbalance lambda: {:.6f}".format(best_imbalance))

    # ---------------------------------
    # COMPARISON
    # ---------------------------------

    print("\n========== FINAL COMPARISON ==========")

    print(
        "{:<25} {:>15} {:>15}".format(
            "Strategy", "Wall time (s)", "Lambda"
        )
    )

    for name in ["static", "cyclic"]:
        wall, times, imbalance, _ = results[name]

        print(
            "{:<25} {:>15.6f} {:>15.6f}".format(
                name, wall, imbalance
            )
        )

    for chunk in CHUNK_SIZES:
        wall, times, imbalance, _ = results[chunk]

        print(
            "{:<25} {:>15.6f} {:>15.6f}".format(
                "dynamic C={}".format(chunk),
                wall,
                imbalance
            )
        )

    # ---------------------------------
    # VERIFICATION
    # ---------------------------------

    static_wall = results["static"][0]

    verify_text = (
        "PHASE4|STATIC={:.6f}|DYNAMIC={:.6f}|C={}"
    ).format(
        static_wall,
        best_wall,
        best_chunk
    )

    verify_hash = hashlib.sha256(
        verify_text.encode()
    ).hexdigest()[:16].upper()

    print("\n========== PHASE 4 SUMMARY ==========")
    print("Static wall time:", static_wall)
    print("Best dynamic wall time:", best_wall)
    print("Best chunk C*:", best_chunk)
    print("VERIFY_HASH[PHASE4]:", verify_hash)


if __name__ == "__main__":
    main()
