
import multiprocessing as mp
import math
import time
import statistics

from harness import emit_verify_tag


STUDENT_ID = 230103077

N = 8000
P = 8

CHUNK_SIZES = [1, 8, 64, 512]

# One full run per scheduling strategy.
# The workload is intentionally computationally expensive.
RUNS = 1


# =========================================================
# WORKLOAD
# =========================================================

def process_item(i):

    # Triangular skew:
    # K(i) = 1 + floor((i / 40)^2)

    iterations = 1 + int((i / 40) ** 2)

    x = (i + 1) * 0.001

    for _ in range(iterations):

        x = math.sin(x)
        x = x + math.cos(x)
        x = math.sqrt(abs(x) + 1.0)

    return x


# =========================================================
# WORKER: STATIC BLOCK / CYCLIC
# =========================================================

def static_worker(worker_id, p, strategy, runtimes):

    start_time = time.perf_counter_ns()

    checksum = 0.0

    if strategy == "block":

        start = worker_id * N // p
        end = (worker_id + 1) * N // p

        for i in range(start, end):
            checksum += process_item(i)

    elif strategy == "cyclic":

        for i in range(worker_id, N, p):
            checksum += process_item(i)

    else:

        raise ValueError("Unknown static strategy")

    end_time = time.perf_counter_ns()

    runtimes[worker_id] = (
        end_time - start_time
    ) / 1e9

    # Keep a result from the workload.
    return checksum


# =========================================================
# WORKER: DYNAMIC SHARED QUEUE
# =========================================================

def dynamic_worker(
    worker_id,
    p,
    next_index,
    chunk_size,
    runtimes
):

    start_time = time.perf_counter_ns()

    checksum = 0.0

    while True:

        # Synchronize access to the shared index.
        with next_index.get_lock():

            start = next_index.value

            if start >= N:
                break

            end = min(
                start + chunk_size,
                N
            )

            next_index.value = end

        # Process the assigned chunk outside the lock.
        for i in range(start, end):

            checksum += process_item(i)

    end_time = time.perf_counter_ns()

    runtimes[worker_id] = (
        end_time - start_time
    ) / 1e9

    return checksum


# =========================================================
# EXECUTE ONE SCHEDULING EXPERIMENT
# =========================================================

def run_experiment(strategy, chunk_size=None):

    runtimes = mp.Array("d", P)

    next_index = mp.Value("i", 0)

    workers = []

    start_time = time.perf_counter_ns()

    for worker_id in range(P):

        if strategy == "dynamic":

            proc = mp.Process(
                target=dynamic_worker,
                args=(
                    worker_id,
                    P,
                    next_index,
                    chunk_size,
                    runtimes
                )
            )

        else:

            proc = mp.Process(
                target=static_worker,
                args=(
                    worker_id,
                    P,
                    strategy,
                    runtimes
                )
            )

        workers.append(proc)
        proc.start()

    for proc in workers:

        proc.join()

        if proc.exitcode != 0:

            raise RuntimeError(
                f"Worker failed: exit code {proc.exitcode}"
            )

    end_time = time.perf_counter_ns()

    wall_time = (
        end_time - start_time
    ) / 1e9

    worker_times = [
        runtimes[i] for i in range(P)
    ]

    average_worker_time = statistics.mean(
        worker_times
    )

    max_worker_time = max(worker_times)

    imbalance = (
        max_worker_time / average_worker_time
        if average_worker_time > 0
        else 0
    )

    return {
        "wall": wall_time,
        "worker_times": worker_times,
        "imbalance": imbalance,
        "average": average_worker_time,
        "maximum": max_worker_time
    }


# =========================================================
# PRINT RESULTS
# =========================================================

def print_result(name, result):

    print(f"\n{name}")

    print(
        f"Wall time: {result['wall']:.4f} seconds"
    )

    print(
        f"Average worker time: "
        f"{result['average']:.4f} seconds"
    )

    print(
        f"Maximum worker time: "
        f"{result['maximum']:.4f} seconds"
    )

    print(
        f"Load Imbalance Factor: "
        f"{result['imbalance']:.4f}"
    )

    print("Individual worker runtimes:")

    for worker_id, elapsed in enumerate(
        result["worker_times"]
    ):

        print(
            f"Worker {worker_id}: "
            f"{elapsed:.4f} seconds"
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 75)
    print("PHASE 4: SKEWED WORKLOADS & DYNAMIC CHUNK QUEUE")
    print("=" * 75)

    print("Logical cores:", mp.cpu_count())
    print("Processes:", P)
    print("Total items:", N)

    print(
        "Workload: K(i) = 1 + floor((i/40)^2)"
    )

    print("Chunk sizes:", CHUNK_SIZES)

    results = {}

    # -----------------------------------------------------
    # PART 1: CONTIGUOUS BLOCK STATIC
    # -----------------------------------------------------

    print("\n--- PART 1: CONTIGUOUS BLOCK STATIC ---")

    block_result = run_experiment("block")

    results["block"] = block_result

    print_result(
        "BLOCK STATIC",
        block_result
    )

    # -----------------------------------------------------
    # PART 2: CYCLIC ROUND-ROBIN STATIC
    # -----------------------------------------------------

    print("\n--- PART 2: CYCLIC ROUND-ROBIN STATIC ---")

    cyclic_result = run_experiment("cyclic")

    results["cyclic"] = cyclic_result

    print_result(
        "CYCLIC STATIC",
        cyclic_result
    )

    # -----------------------------------------------------
    # PART 3: DYNAMIC QUEUE
    # -----------------------------------------------------

    print("\n--- PART 3: DYNAMIC CHUNK QUEUE ---")

    dynamic_results = {}

    for chunk_size in CHUNK_SIZES:

        print(
            f"\nRunning Dynamic Queue with C={chunk_size}"
        )

        result = run_experiment(
            "dynamic",
            chunk_size
        )

        dynamic_results[chunk_size] = result

        results[f"dynamic_{chunk_size}"] = result

        print_result(
            f"DYNAMIC C={chunk_size}",
            result
        )

    # -----------------------------------------------------
    # PART 4: SUMMARY TABLE
    # -----------------------------------------------------

    print("\n--- PART 4: SCHEDULING SUMMARY ---")

    print(
        f"{'Strategy':<22}"
        f"{'Wall (s)':<15}"
        f"{'Imbalance':<15}"
    )

    print(
        f"{'Block Static':<22}"
        f"{block_result['wall']:<15.4f}"
        f"{block_result['imbalance']:<15.4f}"
    )

    print(
        f"{'Cyclic Static':<22}"
        f"{cyclic_result['wall']:<15.4f}"
        f"{cyclic_result['imbalance']:<15.4f}"
    )

    for chunk_size in CHUNK_SIZES:

        result = dynamic_results[chunk_size]

        print(
            f"{('Dynamic C=' + str(chunk_size)):<22}"
            f"{result['wall']:<15.4f}"
            f"{result['imbalance']:<15.4f}"
        )

    # -----------------------------------------------------
    # PART 5: SELECT OPTIMAL CHUNK
    # -----------------------------------------------------

    best_chunk = min(
        CHUNK_SIZES,
        key=lambda c: dynamic_results[c]["wall"]
    )

    best_result = dynamic_results[best_chunk]

    print("\n--- OPTIMAL CHUNK SIZE ---")

    print("Best chunk size:", best_chunk)

    print(
        f"Dynamic wall time: "
        f"{best_result['wall']:.4f} seconds"
    )

    print(
        f"Dynamic imbalance: "
        f"{best_result['imbalance']:.4f}"
    )

    # -----------------------------------------------------
    # PART 6: VERIFICATION
    # -----------------------------------------------------

    print("\n--- PART 6: VERIFICATION ---")

    tag = emit_verify_tag(
        "PHASE4",
        STUDENT_ID,
        block_result["wall"] * 1000,
        best_result["wall"] * 1000
    )

    print(tag)

    print("\nPHASE 4 COMPLETED")


if __name__ == "__main__":

    mp.freeze_support()
    main()
