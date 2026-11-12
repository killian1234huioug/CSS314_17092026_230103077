
import multiprocessing as mp
import math
import time
import statistics

# Apple M3
P_PHYS = 8
P_LOG = 8

# Experiment parameters
N_STRONG = 1_600_000
N0_WEAK = 250_000
REPEATS = 3
WARMUPS = 1
ITERATIONS = 35


def stage_a_serial(n):
    """Serial stage: process the first 5% of elements."""
    count = max(1, int(n * 0.05))
    result = 0.0

    for i in range(count):
        x = (i + 1) * 0.0001

        for j in range(12):
            x = math.sin(x + j * 0.001)
            x = math.sqrt(abs(x) + 1.0)

        result += x

    return result


def stage_b_worker(bounds):
    """Parallel compute kernel with no shared writes."""
    start, end = bounds
    result = 0.0

    for i in range(start, end):
        x = (i + 1) * 0.0001

        for j in range(ITERATIONS):
            x = x * 1.0000001 + math.sin(x + j * 0.000001)

        result += x

    return result


def make_chunks(n, p):
    """Divide N elements into p*4 chunks."""
    number_of_chunks = p * 4
    chunk_size = (n + number_of_chunks - 1) // number_of_chunks

    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunks.append((start, end))

    return chunks


def run_trial(pool, n, p):
    """Run both stages and measure total execution time."""
    start_time = time.perf_counter()

    # Stage A: serial work in the parent process
    stage_a_result = stage_a_serial(n)

    # Stage B: parallel work using a persistent process pool
    chunks = make_chunks(n, p)
    results = pool.map(stage_b_worker, chunks)

    total_time = time.perf_counter() - start_time

    # Keep the computed results available
    checksum = stage_a_result + sum(results)

    return total_time, checksum


def benchmark_configuration(p, n):
    """Create the pool outside the timed region."""
    ctx = mp.get_context("spawn")

    total_times = []
    stage_a_times = []
    checksum = 0.0

    with ctx.Pool(processes=p) as pool:

        # Warm-up
        for _ in range(WARMUPS):
            run_trial(pool, n, p)

        # Timed trials
        for trial in range(REPEATS):
            start = time.perf_counter()

            stage_a_start = time.perf_counter()
            stage_a_serial(n)
            stage_a_elapsed = time.perf_counter() - stage_a_start

            chunks = make_chunks(n, p)
            results = pool.map(stage_b_worker, chunks)

            elapsed = time.perf_counter() - start

            checksum = sum(results)
            total_times.append(elapsed)
            stage_a_times.append(stage_a_elapsed)

            print(
                f"p={p}, N={n}, trial={trial + 1}, "
                f"time={elapsed:.3f}s"
            )

    median_total = statistics.median(total_times)
    median_stage_a = statistics.median(stage_a_times)

    return median_total, median_stage_a, checksum


def main():
    strong_results = {}
    weak_results = {}

    print("\n========== PHASE 2A: STRONG SCALING ==========")

    for p in range(1, P_LOG + 1):
        print(f"\nRunning Strong Scaling: p={p}")

        total, stage_a, checksum = benchmark_configuration(
            p, N_STRONG
        )

        strong_results[p] = {
            "time": total,
            "stage_a": stage_a,
            "checksum": checksum
        }

    t1_strong = strong_results[1]["time"]
    serial_fraction = strong_results[1]["stage_a"] / t1_strong

    amdahl_max = 1.0 / serial_fraction
    amdahl_p_log = 1.0 / (
        serial_fraction + (1.0 - serial_fraction) / P_LOG
    )

    print("\nSTRONG SCALING RESULTS")
    print("p | N | Median(s) | Speedup | Efficiency")

    previous_speedup = None
    marginal_gains = {}

    for p in range(1, P_LOG + 1):
        t = strong_results[p]["time"]
        speedup = t1_strong / t
        efficiency = speedup / p

        marginal = None
        if previous_speedup is not None:
            marginal = speedup - previous_speedup
            marginal_gains[p] = marginal

        print(
            f"{p} | {N_STRONG} | {t:.4f} | "
            f"{speedup:.4f} | {efficiency:.4f}"
        )

        previous_speedup = speedup

    print("\nAMDAHL ANALYSIS")
    print(f"Serial fraction s = {serial_fraction:.6f}")
    print(f"Amdahl maximum speedup = {amdahl_max:.4f}")
    print(f"Amdahl speedup at p={P_LOG} = {amdahl_p_log:.4f}")

    observed_strong = t1_strong / strong_results[P_LOG]["time"]
    print(f"Observed strong speedup at p={P_LOG} = {observed_strong:.4f}")

    print("\nMARGINAL SPEEDUP GAINS")
    for p, gain in marginal_gains.items():
        print(f"p={p}: additional speedup = {gain:.4f}")

    if 4 in marginal_gains and 5 in marginal_gains:
        if marginal_gains[4] != 0:
            drop = (
                (marginal_gains[4] - marginal_gains[5])
                / marginal_gains[4]
            ) * 100
            print(
                "Change in marginal gain from p=4 to p=5: "
                f"{drop:.2f}%"
            )

    print("\n========== PHASE 2B: WEAK SCALING ==========")

    for p in range(1, P_LOG + 1):
        n = p * N0_WEAK
        print(f"\nRunning Weak Scaling: p={p}, N={n}")

        total, stage_a, checksum = benchmark_configuration(p, n)

        weak_results[p] = {
            "time": total,
            "stage_a": stage_a,
            "checksum": checksum,
            "n": n
        }

    t1_weak = weak_results[1]["time"]

    print("\nWEAK SCALING RESULTS")
    print("p | N | Median(s) | Weak Efficiency | Alpha | Gustafson Speedup")

    gustafson_last = None

    for p in range(1, P_LOG + 1):
        t = weak_results[p]["time"]
        n = weak_results[p]["n"]

        efficiency = t1_weak / t
        alpha = weak_results[p]["stage_a"] / t
        gustafson = p - alpha * (p - 1)

        if p == P_LOG:
            gustafson_last = gustafson

        print(
            f"{p} | {n} | {t:.4f} | "
            f"{efficiency:.4f} | {alpha:.6f} | {gustafson:.4f}"
        )

    print("\n========== PHASE 2 SUMMARY ==========")
    print(f"Pphys = {P_PHYS}")
    print(f"Plog = {P_LOG}")
    print(f"Strong speedup at Plog = {observed_strong:.6f}")
    print(f"Gustafson speedup at Plog = {gustafson_last:.6f}")

    print(
        f"\nVERIFY_HASH[PHASE2] "
        f"Sstrong_Plog={observed_strong:.6f} "
        f"S_Gustafson_Plog={gustafson_last:.6f}"
    )


if __name__ == "__main__":
    main()
