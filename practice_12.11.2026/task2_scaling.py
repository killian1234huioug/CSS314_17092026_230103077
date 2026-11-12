
import multiprocessing as mp
import math
import statistics
import time

from harness import emit_verify_tag


STUDENT_ID = 230103077

STRONG_N = 1_200_000
WEAK_N0 = 200_000

RUNS = 3
STAGE_A_FRACTION = 0.05

# Stage A: 10 math operations per element
STAGE_A_ROUNDS = 5

# Stage B: 30 math operations per element
STAGE_B_ROUNDS = 15


# ---------------------------------------------------------
# STAGE A: SERIAL WORK IN THE PARENT PROCESS
# ---------------------------------------------------------

def stage_a_work(count):

    checksum = 0.0

    for i in range(count):

        x = (i + 1) * 0.000001

        for _ in range(STAGE_A_ROUNDS):
            x = math.sin(x)
            x = math.sqrt(abs(x) + 1.0)

        checksum += x

    return checksum


# ---------------------------------------------------------
# STAGE B: PARALLEL WORKER
# ---------------------------------------------------------

def stage_b_worker(task):

    start, end = task

    checksum = 0.0

    for i in range(start, end):

        x = (i + 1) * 0.000001

        for _ in range(STAGE_B_ROUNDS):
            x = math.sin(x)
            x = math.sqrt(abs(x) + 1.0)

        checksum += x

    return checksum


# ---------------------------------------------------------
# PARTITION DATA INTO p WORKER TASKS
# ---------------------------------------------------------

def make_tasks(n, p):

    tasks = []

    base = n // p
    remainder = n % p

    start = 0

    for worker in range(p):

        size = base + (1 if worker < remainder else 0)

        end = start + size

        tasks.append((start, end))

        start = end

    return tasks


# ---------------------------------------------------------
# RUN ONE TWO-STAGE PIPELINE TRIAL
# ---------------------------------------------------------

def run_pipeline(pool, n, p):

    total_start = time.perf_counter_ns()

    # Stage A: first 5% processed serially
    stage_a_count = int(n * STAGE_A_FRACTION)

    stage_a_start = time.perf_counter_ns()

    stage_a_work(stage_a_count)

    stage_a_end = time.perf_counter_ns()

    stage_a_ms = (
        stage_a_end - stage_a_start
    ) / 1e6

    # Stage B: all N elements processed by the Pool
    tasks = make_tasks(n, p)

    pool.map(stage_b_worker, tasks)

    total_end = time.perf_counter_ns()

    total_ms = (
        total_end - total_start
    ) / 1e6

    return total_ms, stage_a_ms


# ---------------------------------------------------------
# MEASURE MULTIPLE TRIALS
# ---------------------------------------------------------

def measure_pipeline(pool, n, p):

    total_samples = []
    stage_a_samples = []

    for _ in range(RUNS):

        total_ms, stage_a_ms = run_pipeline(
            pool,
            n,
            p
        )

        total_samples.append(total_ms)
        stage_a_samples.append(stage_a_ms)

    return {
        "median": statistics.median(total_samples),
        "stage_a": statistics.median(stage_a_samples),
        "min": min(total_samples),
        "max": max(total_samples)
    }


# ---------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------

def print_results(label, p, result):

    print(
        f"{label:8s} "
        f"p={p:<2d} "
        f"median={result['median']:10.3f} ms "
        f"Stage_A={result['stage_a']:9.3f} ms "
        f"range={result['min']:.3f}-{result['max']:.3f} ms"
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    logical_cores = mp.cpu_count()

    process_counts = list(
        range(1, logical_cores + 1)
    )

    print("=" * 75)
    print("PHASE 2: STRONG SCALING VS WEAK SCALING")
    print("=" * 75)

    print("Logical cores:", logical_cores)
    print("Strong scaling N:", STRONG_N)
    print("Weak scaling N0:", WEAK_N0)
    print("Benchmark runs:", RUNS)

    strong_results = {}
    weak_results = {}

    # -----------------------------------------------------
    # PART 1: STRONG SCALING
    # -----------------------------------------------------

    print("\n--- PART 1: STRONG SCALING ---")

    for p in process_counts:

        print(f"\nCreating persistent Pool with p={p}")

        # Pool is created outside the timed trials.
        with mp.Pool(processes=p) as pool:

            result = measure_pipeline(
                pool,
                STRONG_N,
                p
            )

        strong_results[p] = result

        print_results("STRONG", p, result)

    # -----------------------------------------------------
    # PART 2: STRONG SPEEDUP AND EFFICIENCY
    # -----------------------------------------------------

    strong_baseline = strong_results[1]["median"]

    print("\n--- STRONG SCALING ANALYSIS ---")

    print(
        f"{'p':<8}{'Time (ms)':<16}"
        f"{'Speedup':<15}{'Efficiency':<15}"
    )

    for p in process_counts:

        tp = strong_results[p]["median"]

        speedup = strong_baseline / tp
        efficiency = speedup / p

        print(
            f"{p:<8}"
            f"{tp:<16.3f}"
            f"{speedup:<15.4f}"
            f"{efficiency:<15.4f}"
        )

    # Amdahl's theoretical serial fraction
    stage_a_1 = strong_results[1]["stage_a"]

    serial_fraction = stage_a_1 / strong_baseline

    amdahl_cap = 1.0 / serial_fraction

    strong_final_speedup = (
        strong_baseline /
        strong_results[logical_cores]["median"]
    )

    print("\n--- AMDAHL'S LAW ---")

    print(f"Serial fraction s: {serial_fraction:.6f}")
    print(f"Amdahl theoretical cap: {amdahl_cap:.4f}")
    print(f"Observed speedup at p={logical_cores}: "
          f"{strong_final_speedup:.4f}")

    # -----------------------------------------------------
    # PART 3: WEAK SCALING
    # -----------------------------------------------------

    print("\n--- PART 3: WEAK SCALING ---")

    for p in process_counts:

        n = p * WEAK_N0

        print(
            f"\nCreating persistent Pool with p={p}, N={n}"
        )

        with mp.Pool(processes=p) as pool:

            result = measure_pipeline(
                pool,
                n,
                p
            )

        weak_results[p] = result

        print_results("WEAK", p, result)

    # -----------------------------------------------------
    # PART 4: WEAK EFFICIENCY AND GUSTAFSON SPEEDUP
    # -----------------------------------------------------

    weak_baseline = weak_results[1]["median"]

    print("\n--- WEAK SCALING ANALYSIS ---")

    print(
        f"{'p':<8}{'Time (ms)':<16}"
        f"{'Efficiency':<15}{'Alpha':<15}"
        f"{'Gustafson':<15}"
    )

    gustafson_final = None

    for p in process_counts:

        tp = weak_results[p]["median"]

        stage_a_p = weak_results[p]["stage_a"]

        weak_efficiency = weak_baseline / tp

        alpha = stage_a_p / tp

        gustafson = p - alpha * (p - 1)

        if p == logical_cores:
            gustafson_final = gustafson

        print(
            f"{p:<8}"
            f"{tp:<16.3f}"
            f"{weak_efficiency:<15.4f}"
            f"{alpha:<15.6f}"
            f"{gustafson:<15.4f}"
        )

    # -----------------------------------------------------
    # PART 5: APPLE M3 ARCHITECTURAL INFLECTION
    # -----------------------------------------------------

    print("\n--- APPLE M3 ARCHITECTURAL INFLECTION ---")

    # M3: 4 performance cores + 4 efficiency cores.
    # Compare the transition from 4 to 5 processes.

    if logical_cores >= 5:

        t3 = strong_results[3]["median"]
        t4 = strong_results[4]["median"]
        t5 = strong_results[5]["median"]

        gain_3_to_4 = t3 / t4
        gain_4_to_5 = t4 / t5

        print("M3 P-core/E-core transition: p=4 -> p=5")

        print(f"Marginal gain 3 -> 4: {gain_3_to_4:.4f}")
        print(f"Marginal gain 4 -> 5: {gain_4_to_5:.4f}")

        if gain_3_to_4 != 0:

            drop = (
                (gain_3_to_4 - gain_4_to_5)
                / gain_3_to_4
            ) * 100

            print(
                f"Marginal gain change: {drop:.2f}%"
            )

    # -----------------------------------------------------
    # PART 6: VERIFICATION
    # -----------------------------------------------------

    print("\n--- PART 6: VERIFICATION ---")

    tag = emit_verify_tag(
        "PHASE2",
        STUDENT_ID,
        strong_final_speedup,
        gustafson_final
    )

    print(tag)

    print("\nPHASE 2 COMPLETED")


if __name__ == "__main__":

    mp.freeze_support()
    main()

