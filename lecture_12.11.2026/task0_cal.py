import multiprocessing as mp
import time
from harness import emit_verify_tag


def main():
    # 1. Number of logical CPU cores
    cpu_count = mp.cpu_count()
    print("Logical CPU cores:", cpu_count)

    # 2. Measure timer call latency
    iterations = 2_000_000

    start = time.perf_counter_ns()

    for _ in range(iterations):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    total_time = end - start
    average_latency = total_time / iterations

    print("Total timer measurement (ns):", total_time)
    print("Average timer call latency (ns):", average_latency)

    # 3. Generate verification hash
    verify = emit_verify_tag(
        "PHASE0",
        230103077,
        average_latency,
        cpu_count
    )

    print(verify)


if __name__ == "__main__":
    main()